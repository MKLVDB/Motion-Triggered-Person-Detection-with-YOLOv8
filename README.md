# Motion-Triggered Person Detection with YOLOv8

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A **motion-triggered surveillance system** using a **YOLOv8 object detection model**.  
The system continuously monitors an RTSP video stream and works in **two steps** to minimize CPU usage:

1. Detect motion within a selectable region of interest (ROI).  
2. Only if motion is detected, run YOLOv8 to check for people.  

Snapshots are saved locally, and optional **MQTT notifications** can be sent when a person is detected.

> **Note:** The ROI must be selected via a GUI window at startup. After ROI selection, the script runs **headless** (without display windows).

---

## Features

- **Two-step detection** to save CPU:
  1. Motion detection first.
  2. YOLOv8 person detection only if motion occurs.
- ROI selection via GUI at startup.
- Person detection using YOLOv8.
- Automatic snapshot saving in a local `snapshots` folder.
- Optional MQTT notifications when a person is detected.
- Fully configurable: RTSP stream URL, motion sensitivity, snapshot cooldown, and MQTT credentials.
- Headless operation after ROI selection for minimal resource usage.
- Suitable for general motion- and person-detection applications with any camera.

---

## Requirements

- Python 3.10+  
- OpenCV (`cv2`)  
- NumPy (`numpy`)  
- Ultralytics YOLOv8 (`ultralytics`)  
- Paho MQTT client (`paho-mqtt`) (optional)

Install dependencies with:

```bash
pip install opencv-python-headless numpy ultralytics paho-mqtt
```

---

## Usage

1. **Configure the script**:
   - Set your RTSP stream URL.
   - Set the local folder for snapshots (`snapshots` by default).
   - Adjust motion sensitivity, snapshot cooldown, and MQTT settings if needed.

2. **Run the script**:

```bash
python motion_detection.py
```

3. **Select the ROI via the GUI window** on the first frame.

4. The system then continues **headless**:
   - Detects motion in the selected ROI.
   - If motion occurs, checks for people using YOLOv8.
   - Saves snapshots locally in the `snapshots` folder.
   - Optionally sends MQTT notifications.

---

## Notes

- Snapshots are **always saved locally**, regardless of MQTT.
- MQTT notifications are **optional** and do not affect snapshot saving.
- The ROI must be selected via the GUI each time the script starts.
- The **two-step detection** ensures low CPU usage.
- This project is generic and can be used for any camera-based motion and person detection application.

---

## License

This project is licensed under the MIT License.  

> **Important:** Do not include sensitive credentials (such as MQTT passwords) in public repositories.
