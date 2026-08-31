import argparse
import sys
import time

import mbientlab.metawear as mw


def clear_device(mac_address):
    print(f"[INFO] Connecting to board at {mac_address} to clear state...")
    try:
        board = mw.MetaWear(mac_address)
        board.connect()
        print("[INFO] Connected successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to connect: {e}")
        return 1

    try:
        print("[INFO] Stopping sensors and data sampling...")
        # Accelerometer
        mw.libmetawear.mbl_mw_acc_stop(board.board)
        mw.libmetawear.mbl_mw_acc_disable_acceleration_sampling(board.board)
        # Gyroscope
        mw.libmetawear.mbl_mw_gyro_bmi160_stop(board.board)
        mw.libmetawear.mbl_mw_gyro_bmi160_disable_rotation_sampling(board.board)
        # Magnetometer
        mw.libmetawear.mbl_mw_mag_bmm150_stop(board.board)
        mw.libmetawear.mbl_mw_mag_bmm150_disable_b_field_sampling(board.board)

        print("[INFO] Stopping and clearing logs...")
        mw.libmetawear.mbl_mw_logging_stop(board.board)
        mw.libmetawear.mbl_mw_logging_clear_entries(board.board)

        print("[INFO] Clearing LED...")
        mw.libmetawear.mbl_mw_led_stop_and_clear(board.board)

        # Let the commands register to the board
        time.sleep(1.0)

        print("[INFO] Disconnecting...")
        board.disconnect()
        print("[INFO] Device state cleared successfully.")

    except Exception as e:
        print(f"[ERROR] An error occurred during state clearing: {e}")
        try:
            board.disconnect()
        except Exception:
            pass
        return 1

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Clear MetaMotionRL device state.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--mac", "-a", type=str, default="D6:E8:E4:D4:C2:02", help="MAC address of MetaMotionRL")
    args = parser.parse_args()

    sys.exit(clear_device(args.mac))


if __name__ == "__main__":
    main()
