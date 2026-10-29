# IMUS Recorder

Record IMU data and US video.

## Usage

Use [uv](https://docs.astral.sh/uv/) to run the scripts.
Use the `--help` flag for available options:

```
uv run python record.py --help
uv run python verify.py --help
uv run python clear.py --help
```

You can also see the help messages below.

### Recording

Run the `record.py` script to start a recording session.
This script records IMU data from a [MetaMotionRL](https://mbientlab.com/metamotionrl/) device
and captures video from a bounding box on the primary monitor, which should display the
[EchoWave II](https://www.pcultrasound.com/software/software_ewii/) software.

```
usage: record.py [-h] [--dir DIR] [--mac MAC] [--mode {log,stream}]
                 [--timeout TIMEOUT] [--skip-aborted] [--bbox BBOX]
                 [--fps FPS] [--codec {XVID,FFV1,HFYU,RAW}]
                 [--container {AVI,MKV}] [--hotkey HOTKEY]
                 [--acc-freq ACC_FREQ] [--acc-range ACC_RANGE]
                 [--gyro-freq {25,50,100,200,400,800,1600,3200}]
                 [--gyro-range {125,250,500,1000,2000}]
                 [--mag-preset {LOW_POWER,REGULAR,ENHANCED_REGULAR,HIGH_ACCURACY}]
                 [--view VIEW] [--operator OPERATOR] [--subject SUBJECT]
                 [--info INFO]

Record IMU data and US video.

options:
  -h, --help            show this help message and exit
  --dir, -d DIR         base directory for recordings (default: data)
  --mac, -a MAC         MAC address of MetaMotionRL (default:
                        D6:E8:E4:D4:C2:02)
  --mode, -m {log,stream}
                        data collection mode (default: log)
  --timeout, -t TIMEOUT
                        timeout in seconds for log mode (default: 20)
  --skip-aborted, -r    skip downloading log data if the recording times out
                        and aborts (default: False)
  --bbox, -b BBOX       ultrasound bounding box on the primary screen as
                        comma-separated left,top,width,height where width and
                        height are divisible by 16 (default: 384,206,1152,624)
  --fps, -f FPS         video capture frames per second (default: 30.0)
  --codec, -c {XVID,FFV1,HFYU,RAW}
                        video codec (default: HFYU)
  --container, -n {AVI,MKV}
                        video container (default: AVI)
  --hotkey, -k HOTKEY   keyboard hotkey to trigger recording state (default:
                        <ctrl>+<f12>)
  --acc-freq ACC_FREQ   accelerometer sampling frequency in Hz (default:
                        200.0)
  --acc-range ACC_RANGE
                        accelerometer range in g (default: 2.0)
  --gyro-freq {25,50,100,200,400,800,1600,3200}
                        gyroscope sampling frequency in Hz (default: 200)
  --gyro-range {125,250,500,1000,2000}
                        gyroscope range in dps (default: 250)
  --mag-preset {LOW_POWER,REGULAR,ENHANCED_REGULAR,HIGH_ACCURACY}
                        magnetometer preset; HIGH_ACCURACY: 20Hz, others: 10Hz
                        (default: HIGH_ACCURACY)
  --view, -v VIEW       acquisition view (default: None)
  --operator, -p OPERATOR
                        operator ID (default: None)
  --subject, -s SUBJECT
                        subject ID (default: None)
  --info, -i INFO       additional info (default: None)
```

#### Modes

In log mode (default), MetaMotionRL saves the data on its onboard flash memory, and the script downloads the data over Bluetooth once scanning is finished.
Therefore, you have a limited time to complete a scan in this mode, enforced by `--timeout`.
See [MbientLab's documentation on Logging Memory Capacity](https://docs.mbientlab.com/api-specification/#logging-memory-capacity) for more information.

In stream mode (`--mode stream`), MetaMotionRL directly streams the data to the PC over Bluetooth.
In this mode, you have unlimited time for scanning, but the device might not be able to maintain a consistent frequency.

#### Recording States

The color of the LED on MetaMotionRL indicates the current recording state.
Press the hotkey (set by `--hotkey`) to switch the state.

- **🔴 Red:** Busy. Do not press the hotkey.
- **🔵 Blue:** Ready. Press the hotkey to start recording.
- **🟢 Green:** Recording. Press the hotkey to stop recording.

#### Video Codecs and Containers

You can select the codec and container of the recorded videos using `--codec` and `--container`.
The table below shows a comparison of the available codecs.

| Codec | Compression  | File Size        | Encoding Speed   |
| ----- | ------------ | ---------------- | ---------------- |
| RAW   | Uncompressed | Massive (100%)   | Instant          |
| HFYU  | Lossless     | Large (~30–40%)  | Extremely Fast   |
| FFV1  | Lossless     | Medium (~20–30%) | Slow to Moderate |
| XVID  | Lossy        | Tiny (~1–5%)     | Moderate to Fast |

It is recommended to use the HFYU codec and the AVI container for recording to maintain a consistent FPS.
For archival, it is recommended to use the FFV1 codec, the MKV container, and the grayscale color space for smaller sizes.
You can use FFmpeg to convert the videos:

```bash
ffmpeg -i video.avi -c:v ffv1 -pix_fmt gray -an video.mkv
```

### Verifying a Recording

Run the `verify.py` script to verify the integrity of a saved recording.
The `record.py` script runs this automatically after each recording.

```
usage: verify.py [-h] dir

Verify the integrity of a recording.

positional arguments:
  dir         path to the directory of a single recording

options:
  -h, --help  show this help message and exit
```

### Clearing the Device

You can run the `clear.py` script in case the MetaMotionRL device is stuck in a bad state.
The `record.py` script clears the device on exit, so you normally don't need to run `clear.py` separately.

```
usage: clear.py [-h] [--mac MAC]

Clear MetaMotionRL device state.

options:
  -h, --help     show this help message and exit
  --mac, -a MAC  MAC address of MetaMotionRL (default: D6:E8:E4:D4:C2:02)
```

## Bluetooth Troubleshooting

If `record.py` or `clear.py` cannot connect to the MetaMotionRL device, follow these troubleshooting steps.

### Windows

Scan Bluetooth devices until a *MetaWear* device appears, then exit (do *not* connect to the device).
Follow the steps below:

Settings > Bluetooth and devices > Add device > Bluetooth > Show all devices > *Wait for MetaWear to appear* > Cancel.

### Ubuntu

In the Bluetooth config file (`/etc/bluetooth/main.conf`), make sure that `ControllerMode` is *not* set to `bredr`.
The default `dual` mode should work, but if it does not, try explicitly setting it to `le`.

When you edit the Bluetooth config file, make sure to restart the Bluetooth service for changes to take effect:

```bash
sudo systemctl restart bluetooth
```

## EchoWave II Settings

By default, EchoWave II freezes the scan after a certain amount of time.
You can disable this feature for an uninterrupted long recording session.
Follow the steps below:

Menu > Tools > Options... > Scanning Control > Uncheck "Enable Auto Freeze".
