import argparse
import json
import os
import sys
from datetime import datetime

import cv2
import pandas as pd


def verify_recording(rec_dir):
    errors = 0
    warnings = 0

    def log_err(msg):
        nonlocal errors
        print(f"[ERROR]   {msg}")
        errors += 1

    def log_warn(msg):
        nonlocal warnings
        print(f"[WARN]    {msg}")
        warnings += 1

    def log_ok(msg):
        print(f"[OK]      {msg}")

    if not os.path.isdir(rec_dir):
        log_err(f"Directory does not exist: {rec_dir}")
        return 1

    meta_path = os.path.join(rec_dir, "metadata.json")
    imu_path = os.path.join(rec_dir, "imu.csv")
    frames_path = os.path.join(rec_dir, "frames.csv")

    # 1. Check file existence
    missing_required_files = []
    for fp in [meta_path, frames_path]:
        if not os.path.isfile(fp):
            missing_required_files.append(os.path.basename(fp))

    if missing_required_files:
        log_err(f"Missing required files: {', '.join(missing_required_files)}")
        return 1

    imu_file_missing = not os.path.isfile(imu_path)

    # 2. Load and check Metadata
    with open(meta_path, "r") as f:
        try:
            metadata = json.load(f)
        except json.JSONDecodeError:
            log_err("Failed to parse metadata.json.")
            return 1

    container = metadata.get("video", {}).get("container")
    if container is None:
        log_warn("Metadata is missing video.container. Assuming AVI.")
        container = "AVI"
    video_path = os.path.join(rec_dir, "video." + container.lower())
    if not os.path.isfile(video_path):
        log_err(f"Missing required file: {os.path.basename(video_path)}")
        return 1

    null_fields = [k for k in ["acquisition_view", "operator_id", "subject_id", "info"] if metadata.get(k) is None]
    if null_fields:
        log_warn(f"Metadata fields are null: {', '.join(null_fields)}.")
    else:
        log_ok(
            f"Metadata fields are set (view: {metadata['acquisition_view']}, "
            f"operator: {metadata['operator_id']}, subject: {metadata['subject_id']})."
        )

    aborted = metadata.get("aborted", False)
    if aborted:
        log_warn("Metadata indicates this scan was ABORTED.")
        if imu_file_missing:
            log_warn("This aborted scan does not have an imu.csv file.")

    if not aborted and imu_file_missing:
        log_err("Missing required file: imu.csv")
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

    expected_columns = ["frame_number", "pc_timestamp_ms", "pc_monotonic_ms"]
    if not frames_df.empty and list(frames_df.columns) != expected_columns:
        log_err(f"frames.csv columns {list(frames_df.columns)} do not match expected {expected_columns}.")
        frames_df = pd.DataFrame()

    if not frames_df.empty:
        if frames_df["frame_number"].tolist() != list(range(len(frames_df))):
            log_err("frames.csv frame_number is not 0, 1, 2, ...")
        else:
            log_ok("frames.csv frame_number is sequential.")

        mono_ms = frames_df["pc_monotonic_ms"]
        wall_ms = frames_df["pc_timestamp_ms"]
        mono_diffs_ms = mono_ms.diff().dropna()
        if mono_ms.iloc[0] != 0 or (mono_diffs_ms <= 0).any():
            log_err("frames.csv pc_monotonic_ms does not start at 0 and/or is not strictly increasing.")
        else:
            log_ok("frames.csv pc_monotonic_ms starts at 0 and is strictly increasing.")

        frames_duration_s = mono_ms.iloc[-1] / 1000.0
        if abs(frames_duration_s - expected_length_s) > 0.5:
            log_err(
                f"Frames duration ({frames_duration_s:.2f}s) deviates significantly from metadata length_s ({expected_length_s:.2f}s)."
            )
        else:
            log_ok(f"Frames duration matches metadata ({frames_duration_s:.2f}s).")

        # Wall clock should advance with the monotonic clock (small deviations from NTP slewing are normal)
        max_clock_diff_ms = ((wall_ms - wall_ms.iloc[0]) - mono_ms).abs().max()
        if max_clock_diff_ms > 50:
            log_warn(f"frames.csv wall clock deviates from monotonic clock by up to {max_clock_diff_ms:.1f}ms.")
        else:
            log_ok(f"frames.csv wall clock agrees with monotonic clock (max deviation {max_clock_diff_ms:.1f}ms).")

        # The recorder sleeps ~1s between creating the scan (metadata timestamp) and capturing the first frame
        scan_start_ms = datetime.fromisoformat(metadata["timestamp"]).timestamp() * 1000.0
        first_frame_delay_ms = wall_ms.iloc[0] - scan_start_ms
        if not 900 <= first_frame_delay_ms <= 1500:
            log_warn(f"First frame is {first_frame_delay_ms:.1f}ms after metadata timestamp (expected ~1000ms).")
        else:
            log_ok(f"First frame is {first_frame_delay_ms:.1f}ms after metadata timestamp.")

        expected_fps = metadata.get("video", {}).get("fps")
        avg_diff_ms = mono_diffs_ms.mean()
        actual_fps = 1000.0 / avg_diff_ms if avg_diff_ms > 0 else 0

        if abs(actual_fps - expected_fps) / expected_fps > 0.10:
            log_warn(f"Frames CSV actual FPS ({actual_fps:.1f}) deviates from expected ({expected_fps}).")
        else:
            log_ok(f"Frames CSV timestamp delta is consistent (~{actual_fps:.1f} FPS).")

        # Capture stall check (if a gap is more than 1.8x the expected delta)
        expected_diff_ms = 1000.0 / expected_fps
        max_diff_ms = mono_diffs_ms.max()
        if max_diff_ms > (expected_diff_ms * 1.8):
            log_warn(
                f"Frame capture shows signs of stalls. Max gap: {max_diff_ms:.1f}ms (Expected ~{expected_diff_ms:.1f}ms)."
            )

    # 5. Check Video Files (the recorded one, plus the grayscale FFV1 archival copy if present)
    archive_path = os.path.join(rec_dir, "video.mkv")
    video_paths = [video_path]
    if archive_path != video_path and os.path.isfile(archive_path):
        video_paths.append(archive_path)

    for path in video_paths:
        name = os.path.basename(path)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            log_err(f"Failed to open video file: {path}")
            continue

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Count frames by decoding, since the frame count in the container header can be estimated (e.g., MKV)
        video_frame_count = 0
        while cap.grab():
            video_frame_count += 1
        cap.release()

        # Check Dimensions
        bbox = metadata.get("video", {}).get("bbox", {})
        expected_w, expected_h = bbox["width"], bbox["height"]
        dims_diff = max(abs(video_w - expected_w), abs(video_h - expected_h))
        if dims_diff == 0:
            log_ok(f"{name} dimensions match metadata ({video_w}x{video_h}).")
        elif dims_diff < 16:
            log_warn(
                f"{name} dimensions ({video_w}x{video_h}) differ slightly from metadata bbox width/height ({expected_w}x{expected_h})."
            )
        else:
            log_err(
                f"{name} dimensions ({video_w}x{video_h}) differ significantly from metadata bbox width/height ({expected_w}x{expected_h})."
            )

        # Check FPS
        expected_fps = metadata.get("video", {}).get("fps")
        if abs(video_fps - expected_fps) > 1.0:
            log_warn(f"{name} container FPS ({video_fps}) does not match metadata expected FPS ({expected_fps}).")
        else:
            log_ok(f"{name} container FPS matches metadata ({video_fps}).")

        # Check Frame Count vs frames.csv
        if not frames_df.empty:
            csv_frame_count = len(frames_df)
            if video_frame_count != csv_frame_count:
                log_err(
                    f"{name} frame count ({video_frame_count}) does not match frames.csv row count ({csv_frame_count})."
                )
            else:
                log_ok(f"{name} frame count matches frames.csv ({video_frame_count} frames).")

    # 6. Final Report
    print(f"[INFO]    Verification summary: {errors} errors, {warnings} warnings.")

    if errors > 0:
        print("[FAIL]    Recording failed verification.")
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
