# ======================================
#  Bridge Crack Detection (YOLO + MQTT + Streamlit)
# ======================================
import streamlit as st
st.set_page_config(page_title="Bridge Crack Detection", layout="wide")

import cv2
import time
import tempfile
import os
import json
from datetime import datetime
from ultralytics import YOLO
from pymongo import MongoClient
import gridfs
import paho.mqtt.client as mqtt
import ssl

# =====================================================
# 🔐 LOAD CONFIG FROM STREAMLIT SECRETS
# =====================================================
MONGO_URI = st.secrets["MONGO_URI"]
MQTT_BROKER = st.secrets["MQTT_BROKER"]
MQTT_PORT = int(st.secrets["MQTT_PORT"])
MQTT_USER = st.secrets["MQTT_USER"]
MQTT_PASS = st.secrets["MQTT_PASS"]
MQTT_TOPIC = st.secrets["MQTT_TOPIC_CRACK"]

# --------------------------------------
# 🔹 MongoDB Setup
# --------------------------------------
client = MongoClient(MONGO_URI)
db = client["bridge_monitoring"]
collection = db["detections"]
fs = gridfs.GridFS(db)

# --------------------------------------
# 🔹 MQTT Setup
# --------------------------------------

mqtt_client = mqtt.Client()
mqtt_client.username_pw_set(MQTT_USER, MQTT_PASS)
mqtt_client.tls_set(cert_reqs=ssl.CERT_NONE)
mqtt_client.tls_insecure_set(True)

try:
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start()
    print(f"✅ Connected securely to HiveMQ Cloud at {MQTT_BROKER}:{MQTT_PORT}")
except Exception as e:
    print(f"❌ Could not connect to MQTT broker: {e}")
    mqtt_client = None
# --------------------------------------
# 🔹 YOLO Model Load
# --------------------------------------
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    model.overrides["verbose"] = False
    return model

model = load_model()

# --------------------------------------
# 🔹 Streamlit UI Setup
# --------------------------------------
st.title("🧠 Real-Time Bridge Crack Detection (MQTT)")
st.markdown("YOLOv8 + MQTT Integration • Live Feed + MongoDB Logging")

mode = st.sidebar.radio("Video Source", ["Webcam", "Upload Video"])
uploaded_file = None
if mode == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

start_btn = st.sidebar.button(" Start Detection")
stop_btn = st.sidebar.button(" Stop Detection")

frame_placeholder = st.empty()
status_placeholder = st.sidebar.empty()

# --------------------------------------
# 🔹 Detection Logic
# --------------------------------------
if start_btn:
    # Setup video source
    if mode == "Upload Video" and uploaded_file is not None:
        tmpfile = tempfile.NamedTemporaryFile(delete=False)
        tmpfile.write(uploaded_file.read())
        video_source = tmpfile.name
        video_source_name = uploaded_file.name
    else:
        video_source = 0
        video_source_name = f"webcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        st.error("Failed to open video source.")
        st.stop()

    status_placeholder.success(f" Using MQTT Broker at {MQTT_BROKER}:{MQTT_PORT}")

    store_interval = 20
    frame_counter = 0
    send_interval = 1.0
    last_send_time = 0.0

    st.info("Detection running... press **Stop Detection** in sidebar to end.")
    run_flag = True

    while run_flag:
        ret, frame = cap.read()
        if not ret:
            st.warning("End of video or camera error.")
            break

        frame_counter += 1
        results = model(frame, stream=True)

        crack_detected = False
        severe_crack_detected = False

        # Loop through YOLO detections
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls].lower()

                if label in ["cracks", "severe crack"] and conf > 0.6:
                    color = (0, 255, 0) if label == "cracks" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {conf:.2f}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, color, 2)

                    if label == "severe crack":
                        severe_crack_detected = True
                    else:
                        crack_detected = True

                    # Store data in MongoDB
                    if frame_counter % store_interval == 0:
                        doc = {
                            "timestamp": datetime.utcnow().isoformat(),
                            "label": label,
                            "confidence": conf,
                            "frame_id": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                            "video_source": video_source_name,
                        }
                        if label == "severe crack":
                            _, buf = cv2.imencode(".jpg", frame)
                            image_id = fs.put(buf.tobytes(),
                                              filename=f"severe_crack_{time.time()}.jpg",
                                              content_type="image/jpeg")
                            doc["image_id"] = image_id
                        collection.insert_one(doc)

        # --- Publish Crack Alert via MQTT ---
        if (crack_detected or severe_crack_detected) and (time.time() - last_send_time > send_interval):
            if mqtt_client:
                try:
                    mqtt_client.publish(MQTT_TOPIC, "1")
                    last_send_time = time.time()
                    print("📡 Published crack alert via MQTT")
                except Exception as e:
                    print(" MQTT publish error:", e)

        # Display frame in Streamlit UI
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", caption="Live YOLO Detection Feed")

        # Stop detection if user clicks stop
        if stop_btn:
            run_flag = False

    cap.release()
    if mqtt_client:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
    st.success("Detection stopped and MQTT disconnected.")
