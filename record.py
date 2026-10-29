import argparse
import json
import os
import sys
import threading
import time
import traceback
from datetime import datetime

import cv2
import mbientlab.metawear as mw
import mbientlab.metawear.cbindings as mwc
import mss
import numpy as np
import pandas as pd
from pynput import keyboard

from verify import verify_recording


class ImusRecorder:
    def __init__(
        self,
        base_dir,
        mac_address,
        mode,
        timeout,
        skip_aborted,
        bbox,
        fps,
        codec,
        container,
        hotkey,
        acc_freq,
        acc_range,
        gyro_freq,
        gyro_range,
        mag_preset,
        acquisition_view,
        operator_id,
        subject_id,
        info,
    ):
        self.base_dir = base_dir
        self.mac_address = mac_address
        self.mode = mode
        self.timeout = timeout
        self.skip_aborted = skip_aborted
        self.bbox = bbox
        self.fps = fps
        self.codec = codec
        self.container = container
        self.hotkey = hotkey
        self.acc_freq = acc_freq
        self.acc_range = acc_range
        self.gyro_freq = gyro_freq
        self.gyro_range = gyro_range
        self.mag_preset = mag_preset
        self.acquisition_view = acquisition_view
        self.operator_id = operator_id
        self.subject_id = subject_id
        self.info = info

        # Internal state
        self.board = None
        self.connected = False
        self.sct = None
        self.video_writer = None
        self.video_thread = None
        self.recording_active = False
        self.aborted = False
        self.session_timestamp = None
        self.scan_timestamp = None
        self.recording_start_mono = None
        self.recording_end_mono = None
        self.output_dir = ""

        # Data buffering
        self.frames_data = []  # Stores dicts: {"frame_number": int, "pc_timestamp_ms": float, "pc_monotonic_ms": float}
        self.imu_data = []  # Stores dicts: {"imu_timestamp_ms": int, "sensor": str, "x": float, "y": float, "z": float}
        self.frame_count = 0

        # Sync primitives
        self.state_event = threading.Event()
        self.timeout_timer = None
        self.keyboard_listener = None

        # Warn if bbox width and/or height not divisible by 16
        if self.bbox["width"] % 16 != 0 or self.bbox["height"] % 16 != 0:
            print(
                "[WARN] Bounding box width and/or height not divisible by 16. Recorded video might have different dimensions."
            )

    def run(self):
        try:
            if self.initialize():
                self.run_recording_loop()
            else:
                return 1
        except KeyboardInterrupt:
            print("\n[INFO] KeyboardInterrupt received. Safely exiting...")
            return 0
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred: {e}")
            traceback.print_exc()
            return 1
        finally:
            self.cleanup()
            print("[INFO] Exiting.")
        return 1

    def initialize(self):
        self.session_timestamp = datetime.now()
        print(f"[INFO] Connecting to board at {self.mac_address}...")
        try:
            self.board = mw.MetaWear(self.mac_address)
            self.board.connect()
            self.connected = True
            print("[INFO] Connected to board.")
        except Exception as e:
            print(f"[ERROR] Failed to connect to board: {e}")
            return False

        print("[INFO] Clearing board configurations...")
        mw.libmetawear.mbl_mw_metawearboard_tear_down(self.board.board)
        mw.libmetawear.mbl_mw_logging_stop(self.board.board)
        mw.libmetawear.mbl_mw_logging_clear_entries(self.board.board)
        # Let the commands register to the board
        time.sleep(1.0)

        # Configure sensors
        print("[INFO] Configuring sensors...")
        # Accelerometer
        mw.libmetawear.mbl_mw_acc_set_odr(self.board.board, self.acc_freq)
        mw.libmetawear.mbl_mw_acc_set_range(self.board.board, self.acc_range)
        mw.libmetawear.mbl_mw_acc_write_acceleration_config(self.board.board)
        # Gyroscope
        gyro_odr = getattr(mwc.GyroBoschOdr, f"_{self.gyro_freq}Hz")
        gyro_range = getattr(mwc.GyroBoschRange, f"_{self.gyro_range}dps")
        mw.libmetawear.mbl_mw_gyro_bmi160_set_odr(self.board.board, gyro_odr)
        mw.libmetawear.mbl_mw_gyro_bmi160_set_range(self.board.board, gyro_range)
        mw.libmetawear.mbl_mw_gyro_bmi160_write_config(self.board.board)
        # Magnetometer
        mag_preset = getattr(mwc.MagBmm150Preset, self.mag_preset)
        mw.libmetawear.mbl_mw_mag_bmm150_set_preset(self.board.board, mag_preset)
        # Get signals
        self.acc_signal = mw.libmetawear.mbl_mw_acc_get_acceleration_data_signal(self.board.board)
        self.gyro_signal = mw.libmetawear.mbl_mw_gyro_bmi160_get_rotation_data_signal(self.board.board)
        self.mag_signal = mw.libmetawear.mbl_mw_mag_bmm150_get_b_field_data_signal(self.board.board)
        # Let the commands register to the board
        time.sleep(1.0)

        if self.mode == "log":
            print("[INFO] Initializing loggers...")
            self._setup_loggers()
            self.log_acc_cb = self._imu_log_callback("ACC")
            self.log_gyro_cb = self._imu_log_callback("GYRO")
            self.log_mag_cb = self._imu_log_callback("MAG")
            if "ACC" in self.loggers:
                mw.libmetawear.mbl_mw_logger_subscribe(self.loggers["ACC"], None, self.log_acc_cb)
            if "GYRO" in self.loggers:
                mw.libmetawear.mbl_mw_logger_subscribe(self.loggers["GYRO"], None, self.log_gyro_cb)
            if "MAG" in self.loggers:
                mw.libmetawear.mbl_mw_logger_subscribe(self.loggers["MAG"], None, self.log_mag_cb)
            # Let the commands register to the board
            time.sleep(1.0)

        # Keep declaration
        self.sct = None

        print("[INFO] Initialization complete.")
        return True

    def _imu_callback(self, sensor_name):
        def callback(context, data):
            vec = mw.parse_value(data)
            self.imu_data.append(
                {"imu_timestamp_ms": data.contents.epoch, "sensor": sensor_name, "x": vec.x, "y": vec.y, "z": vec.z}
            )

        return mwc.FnVoid_VoidP_DataP(callback)

    def _imu_log_callback(self, sensor_name):
        def callback(context, data):
            vec = mw.parse_value(data)
            self.imu_data.append(
                {"imu_timestamp_ms": data.contents.epoch, "sensor": sensor_name, "x": vec.x, "y": vec.y, "z": vec.z}
            )

        return mwc.FnVoid_VoidP_DataP(callback)

    def _setup_loggers(self):
        self.loggers = {}
        e_acc = threading.Event()
        e_gyro = threading.Event()
        e_mag = threading.Event()

        def acc_logger_created(context, logger):
            if logger:
                self.loggers["ACC"] = logger
            e_acc.set()

        def gyro_logger_created(context, logger):
            if logger:
                self.loggers["GYRO"] = logger
            e_gyro.set()

        def mag_logger_created(context, logger):
            if logger:
                self.loggers["MAG"] = logger
            e_mag.set()

        self._acc_logger_cb = mwc.FnVoid_VoidP_VoidP(acc_logger_created)
        self._gyro_logger_cb = mwc.FnVoid_VoidP_VoidP(gyro_logger_created)
        self._mag_logger_cb = mwc.FnVoid_VoidP_VoidP(mag_logger_created)

        mw.libmetawear.mbl_mw_datasignal_log(self.acc_signal, None, self._acc_logger_cb)
        if not e_acc.wait(timeout=3.0):
            print("[WARN] ACC logger timeout")

        mw.libmetawear.mbl_mw_datasignal_log(self.gyro_signal, None, self._gyro_logger_cb)
        if not e_gyro.wait(timeout=3.0):
            print("[WARN] GYRO logger timeout")

        mw.libmetawear.mbl_mw_datasignal_log(self.mag_signal, None, self._mag_logger_cb)
        if not e_mag.wait(timeout=3.0):
            print("[WARN] MAG logger timeout")

    def _set_led(self, color):
        pattern = mwc.LedPattern(repeat_count=mwc.Const.LED_REPEAT_INDEFINITELY)
        mw.libmetawear.mbl_mw_led_load_preset_pattern(mwc.byref(pattern), mwc.LedPreset.SOLID)
        mw.libmetawear.mbl_mw_led_stop_and_clear(self.board.board)
        if color == "BLUE":
            mw.libmetawear.mbl_mw_led_write_pattern(self.board.board, mwc.byref(pattern), mwc.LedColor.BLUE)
        elif color == "RED":
            mw.libmetawear.mbl_mw_led_write_pattern(self.board.board, mwc.byref(pattern), mwc.LedColor.RED)
        elif color == "GREEN":
            mw.libmetawear.mbl_mw_led_write_pattern(self.board.board, mwc.byref(pattern), mwc.LedColor.GREEN)
        elif color == "OFF":
            return
        mw.libmetawear.mbl_mw_led_play(self.board.board)

    def _screen_capture_loop(self):
        codec = self.codec if self.codec != "RAW" else "RAW "
        fourcc = cv2.VideoWriter_fourcc(*codec)
        video_filename = "video." + self.container.lower()
        video_path = os.path.join(self.output_dir, video_filename)
        self.video_writer = cv2.VideoWriter(video_path, fourcc, self.fps, (self.bbox["width"], self.bbox["height"]))

        with mss.mss() as sct:
            frame_duration = 1.0 / self.fps
            first_loop_start_mono = None
            while self.recording_active:
                loop_start_mono = time.monotonic()
                loop_start_time = time.time()
                if first_loop_start_mono is None:
                    first_loop_start_mono = loop_start_mono

                # Capture frame
                img = sct.grab(self.bbox)
                # Convert to numpy for OpenCV (MSS returns BGRA, need BGR)
                frame = np.array(img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                self.video_writer.write(frame)
                self.frames_data.append(
                    {
                        "frame_number": self.frame_count,
                        "pc_timestamp_ms": loop_start_time * 1000.0,
                        "pc_monotonic_ms": (loop_start_mono - first_loop_start_mono) * 1000.0,
                    }
                )
                self.frame_count += 1

                # Sleep to maintain FPS
                elapsed = time.monotonic() - loop_start_mono
                sleep_time = frame_duration - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        self.video_writer.release()

    def on_activate_hotkey(self):
        # Triggered by hotkey
        print("[DEBUG] Hotkey triggered.")
        self.state_event.set()

    def run_recording_loop(self):
        # Map the configured hotkey globally
        self.keyboard_listener = keyboard.GlobalHotKeys({self.hotkey: self.on_activate_hotkey})
        self.keyboard_listener.start()

        while True:
            # State: Idle
            self._set_led("BLUE")
            print(f"[INFO] Waiting for {self.hotkey} to start recording...")
            while not self.state_event.wait(0.2):
                pass
            self.state_event.clear()

            # State: Preflight
            self._set_led("RED")

            # Create output directory
            self.scan_timestamp = datetime.now()
            dir_name = f"scan_{self.scan_timestamp.strftime('%Y%m%d_%H%M%S')}"
            self.output_dir = os.path.join(self.base_dir, dir_name)
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"[INFO] Created directory: {self.output_dir}")

            # Setup streaming or logging
            if self.mode == "stream":
                self.stream_acc_cb = self._imu_callback("ACC")
                self.stream_gyro_cb = self._imu_callback("GYRO")
                self.stream_mag_cb = self._imu_callback("MAG")
                mw.libmetawear.mbl_mw_datasignal_subscribe(self.acc_signal, None, self.stream_acc_cb)
                mw.libmetawear.mbl_mw_datasignal_subscribe(self.gyro_signal, None, self.stream_gyro_cb)
                mw.libmetawear.mbl_mw_datasignal_subscribe(self.mag_signal, None, self.stream_mag_cb)
            elif self.mode == "log":
                mw.libmetawear.mbl_mw_logging_stop(self.board.board)
                mw.libmetawear.mbl_mw_logging_clear_entries(self.board.board)
                # Let the commands register to the board
                time.sleep(1.0)
                mw.libmetawear.mbl_mw_logging_start(self.board.board, 0)

            # State: Active recording
            self.frame_count = 0
            self.frames_data = []
            self.imu_data = []
            self.aborted = False

            # Start video recording
            self.recording_active = True
            self.video_thread = threading.Thread(target=self._screen_capture_loop)
            self.video_thread.start()

            # Start IMU sampling
            mw.libmetawear.mbl_mw_acc_enable_acceleration_sampling(self.board.board)
            mw.libmetawear.mbl_mw_acc_start(self.board.board)
            mw.libmetawear.mbl_mw_gyro_bmi160_enable_rotation_sampling(self.board.board)
            mw.libmetawear.mbl_mw_gyro_bmi160_start(self.board.board)
            mw.libmetawear.mbl_mw_mag_bmm150_enable_b_field_sampling(self.board.board)
            mw.libmetawear.mbl_mw_mag_bmm150_start(self.board.board)

            self.recording_start_mono = time.monotonic()
            self._set_led("GREEN")
            print("[INFO] Recording started.")

            # Wait for trigger or timeout
            if self.mode == "log":
                print(f"[INFO] Waiting for {self.hotkey} or {self.timeout}s timeout to stop recording...")
                trigger_happened = False
                while True:
                    if self.state_event.wait(0.2):
                        trigger_happened = True
                        break
                    if time.monotonic() - self.recording_start_mono >= self.timeout:
                        break
                if not trigger_happened:
                    print("[WARN] Log timeout reached. Aborting.")
                    self.aborted = True
            else:
                print(f"[INFO] Waiting for {self.hotkey} to stop recording...")
                while not self.state_event.wait(0.2):
                    pass

            self.state_event.clear()

            # State: Finalization
            self._set_led("RED")
            print("[INFO] Recording stopped.")

            # Stop IMU sampling
            mw.libmetawear.mbl_mw_acc_stop(self.board.board)
            mw.libmetawear.mbl_mw_acc_disable_acceleration_sampling(self.board.board)
            mw.libmetawear.mbl_mw_gyro_bmi160_stop(self.board.board)
            mw.libmetawear.mbl_mw_gyro_bmi160_disable_rotation_sampling(self.board.board)
            mw.libmetawear.mbl_mw_mag_bmm150_stop(self.board.board)
            mw.libmetawear.mbl_mw_mag_bmm150_disable_b_field_sampling(self.board.board)

            # Stop video recording
            self.recording_active = False

            self.recording_end_mono = time.monotonic()

            # Stop video thread
            if self.video_thread:
                self.video_thread.join()

            # Write frames data
            frames_df = pd.DataFrame(self.frames_data)
            frames_df.to_csv(os.path.join(self.output_dir, "frames.csv"), index=False)

            # Convenience function for JSON formatting
            def convert_number(val):
                if int(val) == float(val):
                    return int(val)
                return float(val)

            # Write metadata
            metadata = {
                "timestamp": self.scan_timestamp.astimezone().isoformat(),
                "mode": self.mode,
                "length_s": self.recording_end_mono - self.recording_start_mono,
                "aborted": self.aborted,
                "sensors": {
                    "accelerometer": {
                        "frequency_hz": convert_number(self.acc_freq),
                        "range_g": convert_number(self.acc_range),
                    },
                    "gyroscope": {
                        "frequency_hz": convert_number(self.gyro_freq),
                        "range_dps": convert_number(self.gyro_range),
                    },
                    "magnetometer": {
                        "preset": self.mag_preset,
                        "frequency_hz": 20 if self.mag_preset == "HIGH_ACCURACY" else 10,
                    },
                },
                "video": {
                    "fps": convert_number(self.fps),
                    "codec": self.codec,
                    "container": self.container,
                    "bbox": self.bbox,
                },
                "session_timestamp": self.session_timestamp.astimezone().isoformat(),
                "imu_mac_address": self.mac_address,
                "acquisition_view": self.acquisition_view,
                "operator_id": self.operator_id,
                "subject_id": self.subject_id,
                "info": self.info,
            }
            with open(os.path.join(self.output_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

            # Write IMU data
            if self.mode == "stream":
                mw.libmetawear.mbl_mw_datasignal_unsubscribe(self.acc_signal)
                mw.libmetawear.mbl_mw_datasignal_unsubscribe(self.gyro_signal)
                mw.libmetawear.mbl_mw_datasignal_unsubscribe(self.mag_signal)

                imu_df = pd.DataFrame(self.imu_data)
                if not imu_df.empty:
                    imu_df.to_csv(os.path.join(self.output_dir, "imu.csv"), index=False)
                else:
                    print("[WARN] No IMU data collected.")

            elif self.mode == "log":
                mw.libmetawear.mbl_mw_logging_stop(self.board.board)
                if not trigger_happened and self.skip_aborted:
                    print("[INFO] Skipping aborted log download.")
                else:
                    print("[INFO] Log download initiated...")

                    e = threading.Event()

                    def progress_update(context, entries_left, total_entries):
                        print(f"[DEBUG] Log download progress: {total_entries - entries_left}/{total_entries}")
                        if entries_left == 0:
                            e.set()

                    def unhandled_entry(context, data):
                        print("[WARN] Log download: Unhandled entry received.")

                    def unknown_entry(context, id, epoch, data, length):
                        print(f"[WARN] Log download: Unknown entry received: ID: {id}, Epoch: {epoch}, Len: {length}")

                    self._dl_fn_wrapper = mwc.FnVoid_VoidP_UInt_UInt(progress_update)
                    self._dl_err_wrapper_unhandled = mwc.FnVoid_VoidP_DataP(unhandled_entry)
                    self._dl_err_wrapper_unknown = mwc.FnVoid_VoidP_UByte_Long_UByteP_UByte(unknown_entry)

                    self._dl_handler = mwc.LogDownloadHandler(
                        context=None,
                        received_progress_update=self._dl_fn_wrapper,
                        received_unhandled_entry=self._dl_err_wrapper_unhandled,
                        received_unknown_entry=self._dl_err_wrapper_unknown,
                    )

                    mw.libmetawear.mbl_mw_logging_download(self.board.board, 100, mwc.byref(self._dl_handler))
                    e.wait(timeout=300.0)
                    print("[INFO] Log download complete or timed out.")

                    imu_df = pd.DataFrame(self.imu_data)
                    if not imu_df.empty:
                        imu_df.to_csv(os.path.join(self.output_dir, "imu.csv"), index=False)
                    else:
                        print("[WARN] No IMU data collected.")

            print(f"[INFO] Scan saved at: {self.output_dir}")
            print("[INFO] Verifying recording...")
            verify_recording(self.output_dir)

    def cleanup(self):
        self.recording_active = False
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join()

        if self.keyboard_listener:
            self.keyboard_listener.stop()

        if self.board and self.connected:
            print("[INFO] Resetting board state...")
            try:
                # Stop any potentially running sensors
                mw.libmetawear.mbl_mw_acc_stop(self.board.board)
                mw.libmetawear.mbl_mw_acc_disable_acceleration_sampling(self.board.board)
                mw.libmetawear.mbl_mw_gyro_bmi160_stop(self.board.board)
                mw.libmetawear.mbl_mw_gyro_bmi160_disable_rotation_sampling(self.board.board)
                mw.libmetawear.mbl_mw_mag_bmm150_stop(self.board.board)
                mw.libmetawear.mbl_mw_mag_bmm150_disable_b_field_sampling(self.board.board)

                # Stop and clear logs
                mw.libmetawear.mbl_mw_logging_stop(self.board.board)
                mw.libmetawear.mbl_mw_logging_clear_entries(self.board.board)

                # Turn off LED
                self._set_led("OFF")

                # Let the commands register to the board
                time.sleep(1.0)

                # Disconnect from the board
                self.board.disconnect()
            except Exception as e:
                print(f"[WARN] Error during cleanup: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Record IMU data and US video.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dir", "-d", type=str, default="data", help="base directory for recordings")
    parser.add_argument("--mac", "-a", type=str, default="D6:E8:E4:D4:C2:02", help="MAC address of MetaMotionRL")
    parser.add_argument("--mode", "-m", type=str, choices=["log", "stream"], default="log", help="data collection mode")
    parser.add_argument("--timeout", "-t", type=int, default=20, help="timeout in seconds for log mode")
    parser.add_argument(
        "--skip-aborted",
        "-r",
        action="store_true",
        help="skip downloading log data if the recording times out and aborts",
    )
    parser.add_argument(
        "--bbox",
        "-b",
        type=str,
        default="384,206,1152,624",
        help="ultrasound bounding box on the primary screen as comma-separated left,top,width,height where width and height are divisible by 16",
    )
    parser.add_argument("--fps", "-f", type=float, default=30.0, help="video capture frames per second")
    parser.add_argument(
        "--codec", "-c", type=str, choices=["XVID", "FFV1", "HFYU", "RAW"], default="HFYU", help="video codec"
    )
    parser.add_argument("--container", "-n", type=str, choices=["AVI", "MKV"], default="AVI", help="video container")
    parser.add_argument(
        "--hotkey", "-k", type=str, default="<ctrl>+<f12>", help="keyboard hotkey to trigger recording state"
    )
    parser.add_argument("--acc-freq", type=float, default=200.0, help="accelerometer sampling frequency in Hz")
    parser.add_argument("--acc-range", type=float, default=2.0, help="accelerometer range in g")
    parser.add_argument(
        "--gyro-freq",
        type=str,
        choices=["25", "50", "100", "200", "400", "800", "1600", "3200"],
        default="200",
        help="gyroscope sampling frequency in Hz",
    )
    parser.add_argument(
        "--gyro-range",
        type=str,
        choices=["125", "250", "500", "1000", "2000"],
        default="250",
        help="gyroscope range in dps",
    )
    parser.add_argument(
        "--mag-preset",
        type=str,
        choices=["LOW_POWER", "REGULAR", "ENHANCED_REGULAR", "HIGH_ACCURACY"],
        default="HIGH_ACCURACY",
        help="magnetometer preset; HIGH_ACCURACY: 20Hz, others: 10Hz",
    )
    parser.add_argument("--view", "-v", type=str, default=None, help="acquisition view")
    parser.add_argument("--operator", "-p", type=str, default=None, help="operator ID")
    parser.add_argument("--subject", "-s", type=str, default=None, help="subject ID")
    parser.add_argument("--info", "-i", type=str, default=None, help="additional info")
    args = parser.parse_args()

    bbox_list = [int(x) for x in args.bbox.split(",")]
    bbox_dict = {"left": bbox_list[0], "top": bbox_list[1], "width": bbox_list[2], "height": bbox_list[3]}
    recorder = ImusRecorder(
        base_dir=args.dir,
        mac_address=args.mac,
        mode=args.mode,
        timeout=args.timeout,
        skip_aborted=args.skip_aborted,
        bbox=bbox_dict,
        fps=args.fps,
        codec=args.codec,
        container=args.container,
        hotkey=args.hotkey,
        acc_freq=args.acc_freq,
        acc_range=args.acc_range,
        gyro_freq=args.gyro_freq,
        gyro_range=args.gyro_range,
        mag_preset=args.mag_preset,
        acquisition_view=args.view,
        operator_id=args.operator,
        subject_id=args.subject,
        info=args.info,
    )

    sys.exit(recorder.run())


if __name__ == "__main__":
    main()
