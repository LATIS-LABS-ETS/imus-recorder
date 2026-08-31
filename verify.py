import argparse
import json
import os
import sys

import cv2
import pandas as pd


def verify_recording(rec_dir):
    if not os.path.isdir(rec_dir):
        print(f"[ERROR] Directory does not exist: {rec_dir}")
        return 1

    meta_path = os.path.join(rec_dir, "metadata.json")
    imu_path = os.path.join(rec_dir, "imu.csv")
    frames_path = os.path.join(rec_dir, "frames.csv")
    video_path = os.path.join(rec_dir, "video.avi")

    # 1. Check file existence
    missing_required_files = []
    for fp in [meta_path, frames_path, video_path]:
        if not os.path.isfile(fp):
            missing_required_files.append(os.path.basename(fp))

    if missing_required_files:
        print(f"[ERROR] Missing required files: {', '.join(missing_required_files)}")
        return 1

    imu_file_missing = not os.path.isfile(imu_path)

    errors = 0
    warnings = 0

    def log_err(msg):
        nonlocal errors
        print(f"[ERROR] {msg}")
        errors += 1

    def log_warn(msg):
        nonlocal warnings
        print(f"[WARN]  {msg}")
        warnings += 1

    def log_ok(msg):
        print(f"[OK]    {msg}")

    # 2. Load and check Metadata
    with open(meta_path, "r") as f:
        try:
            metadata = json.load(f)
        except json.JSONDecodeError:
            print("[ERROR] Failed to parse metadata.json.")
            return 1

    aborted = metadata.get("aborted", False)
    if aborted:
        log_warn("Metadata indicates this scan was ABORTED.")
        if imu_file_missing:
            log_warn("This aborted scan does not have an imu.csv file.")

    if not aborted and imu_file_missing:
        print("[ERROR] Missing required file: imu.csv")
        return 1

    expected_length_s = metadata.get("length_s")
    if expected_length_s <= 0:
        log_err(f"Invalid length_s in metadata: {expected_length_s}")

    # 3. Check IMU CSV
    if not imu_file_missing:
        try:
            imu_df = pd.read_csv(imu_path)
        except Exception as e:
            log_err(f"Failed to read imu.csv: {e}")
            imu_df = pd.DataFrame()

        if not imu_df.empty:
            imu_duration_s = (imu_df["imu_timestamp_ms"].max() - imu_df["imu_timestamp_ms"].min()) / 1000.0
            if abs(imu_duration_s - expected_length_s) > 0.5:
                log_err(
                    f"IMU duration ({imu_duration_s:.2f}s) deviates significantly from metadata length_s ({expected_length_s:.2f}s)."
                )
            else:
                log_ok(f"IMU duration matches metadata ({imu_duration_s:.2f}s).")

            sensors_meta = metadata.get("sensors", {})
            sensor_mappings = {
                "ACC": sensors_meta.get("accelerometer", {}).get("frequency_hz"),
                "GYRO": sensors_meta.get("gyroscope", {}).get("frequency_hz"),
                "MAG": sensors_meta.get("magnetometer", {}).get("frequency_hz"),
            }

            for sensor, expected_hz in sensor_mappings.items():
                if not expected_hz:
                    continue

                sensor_df = imu_df[imu_df["sensor"] == sensor].copy()
                if sensor_df.empty:
                    log_err(f"No data found for IMU sensor: {sensor}")
                    continue

                # Calculate actual frequency and check for dropped packets
                sensor_df = sensor_df.sort_values(by="imu_timestamp_ms")
                diffs_ms = sensor_df["imu_timestamp_ms"].diff().dropna()

                if diffs_ms.empty:
                    log_warn(f"Not enough data points to calculate frequency for {sensor}.")
                    continue

                avg_diff_ms = diffs_ms.mean()
                actual_hz = 1000.0 / avg_diff_ms if avg_diff_ms > 0 else 0
                expected_diff_ms = 1000.0 / expected_hz

                # Frequency tolerance check (10% margin)
                if abs(actual_hz - expected_hz) / expected_hz > 0.10:
                    log_err(f"{sensor} actual frequency ({actual_hz:.1f}Hz) deviates from expected ({expected_hz}Hz).")
                else:
                    log_ok(f"{sensor} frequency is consistent ({actual_hz:.1f}Hz).")

                # Dropped packet check (if a gap is more than 1.8x the expected delta)
                max_diff_ms = diffs_ms.max()
                if max_diff_ms > (expected_diff_ms * 1.8):
                    log_warn(
                        f"{sensor} shows signs of dropped packets. Max gap: {max_diff_ms:.1f}ms (Expected ~{expected_diff_ms:.1f}ms)."
                    )

    # 4. Check Frames CSV
    try:
        frames_df = pd.read_csv(frames_path)
    except Exception as e:
        log_err(f"Failed to read frames.csv: {e}")
        frames_df = pd.DataFrame()

    if not frames_df.empty:
        frames_duration_s = (frames_df["pc_timestamp_ms"].max() - frames_df["pc_timestamp_ms"].min()) / 1000.0
        if abs(frames_duration_s - expected_length_s) > 0.5:
            log_err(
                f"Frames duration ({frames_duration_s:.2f}s) deviates significantly from metadata length_s ({expected_length_s:.2f}s)."
            )
        else:
            log_ok(f"Frames duration matches metadata ({frames_duration_s:.2f}s).")

        expected_fps = metadata.get("video", {}).get("fps")

        frames_df = frames_df.sort_values(by="frame_number")
        diffs_ms = frames_df["pc_timestamp_ms"].diff().dropna()

        if not diffs_ms.empty:
            avg_diff_ms = diffs_ms.mean()
            actual_fps = 1000.0 / avg_diff_ms if avg_diff_ms > 0 else 0

            if abs(actual_fps - expected_fps) / expected_fps > 0.10:
                log_warn(f"Frames CSV actual FPS ({actual_fps:.1f}) deviates from expected ({expected_fps}).")
            else:
                log_ok(f"Frames CSV timestamp delta is consistent (~{actual_fps:.1f} FPS).")

    # 5. Check Video File
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log_err(f"Failed to open video file: {video_path}")
    else:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Check Dimensions
        bbox = metadata.get("video", {}).get("bbox", {})
        expected_w, expected_h = bbox["width"], bbox["height"]
        dims_diff = max(abs(video_w - expected_w), abs(video_h - expected_h))
        if dims_diff == 0:
            log_ok(f"Video dimensions match metadata ({video_w}x{video_h}).")
        elif dims_diff < 16:
            log_warn(
                f"Video dimensions ({video_w}x{video_h}) differ slightly from metadata bbox width/height ({expected_w}x{expected_h})."
            )
        else:
            log_err(
                f"Video dimensions ({video_w}x{video_h}) differ significantly from metadata bbox width/height ({expected_w}x{expected_h})."
            )

        # Check FPS
        expected_fps = metadata.get("video", {}).get("fps")
        if abs(video_fps - expected_fps) > 1.0:
            log_warn(f"Video container FPS ({video_fps}) does not match metadata expected FPS ({expected_fps}).")
        else:
            log_ok(f"Video container FPS matches metadata ({video_fps}).")

        # Check Frame Count vs frames.csv
        if not frames_df.empty:
            csv_frame_count = len(frames_df)
            if video_frame_count != csv_frame_count:
                log_err(
                    f"Video frame count ({video_frame_count}) does not match frames.csv row count ({csv_frame_count})."
                )
            else:
                log_ok(f"Video frame count matches frames.csv ({video_frame_count} frames).")

    # 6. Final Report
    print(f"[INFO] Verification summary: {errors} errors, {warnings} warnings.")

    if errors > 0:
        print("[FAIL] Recording failed verification.")
        return 1
    else:
        print("[SUCCESS] Recording verified successfully.")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Verify the integrity of a recording.")
    parser.add_argument("dir", type=str, help="path to the directory of a single recording")
    args = parser.parse_args()

    sys.exit(verify_recording(args.dir))


if __name__ == "__main__":
    main()
