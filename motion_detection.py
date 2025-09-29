import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import threading
import queue
import paho.mqtt.client as mqtt

# -------- CONFIG --------
rtsp_url = "<rtsp url>"
snapshot_dir = "snapshots"
os.makedirs(snapshot_dir, exist_ok=True)

motion_threshold = 500
snapshot_cooldown = 5
delay_before_snapshot = 1.5  # seconds

mqtt_broker = "<mqtt broker ip/url"
mqtt_port = 1883
mqtt_topic = "<mqtt topic>"
mqtt_user = "<mqtt username>"
mqtt_password = "<mqtt password>"  # Consider removing before GitHub

# -------- MQTT Setup --------
mqtt_client = mqtt.Client()
mqtt_client.username_pw_set(mqtt_user, mqtt_password)
mqtt_client.connect(mqtt_broker, mqtt_port, 60)
mqtt_client.loop_start()

# -------- YOLO Model --------
model = YOLO("yolov8n.pt")

# -------- Frame Queue --------
frame_queue = queue.Queue(maxsize=5)  # buffer for frames

# -------- Functions --------
def resize_frame_preserve_aspect(frame, target_height=480):
    h, w = frame.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(frame, (new_w, target_height))

def open_rtsp_stream(url):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    return cap

def stream_reader():
    global frame_queue
    while True:
        cap = open_rtsp_stream(rtsp_url)
        if not cap.isOpened():
            print("Cannot open RTSP stream, retrying in 5s")
            time.sleep(5)
            continue

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame not read, reopening stream")
                break
            if not frame_queue.full():
                frame_queue.put(frame)
        cap.release()
        time.sleep(1)

# -------- Main Detection --------
def main():
    # Select ROI from first frame
    print("Waiting for first frame to select ROI...")
    while frame_queue.empty():
        time.sleep(0.1)
    frame = frame_queue.get()
    frame_resized = resize_frame_preserve_aspect(frame)
    roi = cv2.selectROI("Select ROI", frame_resized, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")
    x, y, w, h = roi
    print(f"ROI set: {(x, y, x + w, y + h)}")

    # Motion detection setup
    prev_gray = cv2.cvtColor(frame_resized[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)
    last_snapshot_time = 0
    person_detected_time = None

    while True:
        if frame_queue.empty():
            time.sleep(0.05)
            continue
        frame = frame_queue.get()
        frame_resized = resize_frame_preserve_aspect(frame)
        roi_frame = frame_resized[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Motion detection
        frame_diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(frame_diff, 10, 255, cv2.THRESH_BINARY)
        motion_pixels = np.sum(thresh > 0)

        current_time = time.time()
        person_in_frame = False

        if motion_pixels > motion_threshold:
            results = model.predict(roi_frame, conf=0.5, verbose=False)
            for result in results:
                for box in result.boxes:
                    if int(box.cls[0]) == 0:  # person
                        person_in_frame = True
                        break

        # Snapshot + MQTT logic
        if person_in_frame:
            if person_detected_time is None:
                person_detected_time = current_time
            elif (current_time - person_detected_time) >= delay_before_snapshot:
                if (current_time - last_snapshot_time) > snapshot_cooldown:
                    snapshot_frame = frame.copy()
                    snapshot_resized = resize_frame_preserve_aspect(snapshot_frame)
                    roi_snapshot = snapshot_resized[y:y+h, x:x+w]

                    # Draw bounding boxes
                    results = model.predict(roi_snapshot, conf=0.5, verbose=False)
                    for result in results:
                        for box in result.boxes:
                            if int(box.cls[0]) == 0:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                cv2.rectangle(snapshot_resized, (x + x1, y + y1), (x + x2, y + y2), (0, 255, 0), 2)
                                cv2.putText(snapshot_resized, "Person", (x + x1, y + y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    timestamp = int(current_time)
                    snapshot_path = os.path.join(snapshot_dir, f"person_{timestamp}.jpg")
                    cv2.imwrite(snapshot_path, snapshot_resized)
                    print(f"Snapshot saved: {snapshot_path}")

                    # Optional MQTT notification
                    try:
                        mqtt_client.publish(mqtt_topic, "Snapshot taken")
                    except Exception as e:
                        print(f"MQTT publish error: {e}")

                    last_snapshot_time = current_time

                person_detected_time = None
        else:
            person_detected_time = None

        prev_gray = gray.copy()

# -------- Start Threads --------
threading.Thread(target=stream_reader, daemon=True).start()
main()
