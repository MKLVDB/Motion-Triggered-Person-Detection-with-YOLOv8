# Motion-Triggered Person Detection with YOLOv8

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A **motion-triggered surveillance system** using a **YOLOv8 object detection model**.  
The system continuously monitors an RTSP video stream, detects motion first to save CPU, and then runs YOLOv8 to detect people **only when motion occurs**. Snapshots are saved locally, and optional **MQTT notifications** can be sent when a person is detected.

> **Note:** The region of interest (ROI) must be selected via a GUI window at startup. After ROI selection, the script runs **headless** (without display windows).

---

## Features

- **Two-step detection**:
  1. Detect motion within the ROI.
  2. Run YOLOv8 to detect people only if motion is detected, keeping CPU usage low.
- ROI selection via GUI at startup.
- Person detection using YOLOv8.
- Automatic snapshot saving in a local `snapshots` folder.
- Optional MQTT notifications when a person is detected.
- Fully configurable: RTSP stream URL, motion sensitivity, snapshot cooldown, and MQTT credentials.
- Runs headless after ROI selection for minimal resource usage.

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

