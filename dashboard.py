# ======================================
#  Unified Bridge Monitoring Dashboard (Streamlit + MQTT + MongoDB)
# ======================================
import streamlit as st
import pandas as pd
import json
import threading
import time
import paho.mqtt.client as mqtt
from pymongo import MongoClient
from datetime import datetime
from bson import ObjectId
from PIL import Image
import io
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
import ssl


# ---------------- LOAD CONFIG FROM SECRETS ----------------
MONGO_URI = st.secrets["MONGO_URI"]
MQTT_BROKER = st.secrets["MQTT_BROKER"]
MQTT_PORT = int(st.secrets["MQTT_PORT"])
MQTT_TOPIC = st.secrets["MQTT_TOPIC"]
DEVICE_ID = st.secrets.get("DEVICE_ID", "esp32-mainboard-01")
REFRESH_INTERVAL = int(st.secrets.get("REFRESH_INTERVAL", 60000))

# ---------------- DATABASE INIT ----------------
client = MongoClient(MONGO_URI)
db = client["bridge_monitoring"]
detections = db["detections"]
alerts = db["alerts"]
fs_files = db["fs.files"]
fs_chunks = db["fs.chunks"]

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="Bridge Monitoring Dashboard", layout="wide")
st.title(" Bridge Health & Crack Detection Dashboard")
st.markdown("#### Real-time monitoring of cracks, vibration, and water levels via MQTT + MongoDB")
st.markdown("---")

# auto refresh
st_autorefresh(interval=REFRESH_INTERVAL, key="data_refresh")

# ---------------- MQTT BACKGROUND LISTENER ----------------
def store_alert(alert_type, alert_value):
    doc = {
        "timestamp": datetime.utcnow().isoformat(),
        "alert_type": alert_type,
        "value": alert_value,
        "device_id": DEVICE_ID,
    }
    alerts.insert_one(doc)
    print(f"[MongoDB] Stored Alert → {alert_type}: {alert_value}")

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[MQTT] Connected → {MQTT_BROKER}:{MQTT_PORT}")
        client.subscribe(MQTT_TOPIC)
        print(f"[MQTT] Subscribed to topic: {MQTT_TOPIC}")
    else:
        print(f"[MQTT] Connection failed with code {rc}")

def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode()
        print(f"[MQTT] Received message: {payload}")
        # Expect JSON payload like {"type":"WATER","value":123} or {"type":"VIBRATION","value":1}
        data = json.loads(payload)
        alert_type = data.get("type", "UNKNOWN")
        alert_value = data.get("value", None)
        if alert_type and alert_value is not None:
            store_alert(alert_type, alert_value)
    except Exception as e:
        print(f"[MQTT] Error processing message: {e}")

def start_mqtt_listener():
    mqtt_client = mqtt.Client()
    mqtt_client.username_pw_set(st.secrets["MQTT_USER"], st.secrets["MQTT_PASS"])
    mqtt_client.tls_set(cert_reqs=ssl.CERT_NONE)
    mqtt_client.tls_insecure_set(True)
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message

    try:
        print(f"[MQTT] Connecting securely to HiveMQ Cloud → {MQTT_BROKER}:{MQTT_PORT}")
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_forever()
    except Exception as e:
        print(f"[MQTT] Could not connect to HiveMQ Cloud: {e}")

# Start MQTT listener thread once per session
if "mqtt_thread_started" not in st.session_state:
    mqtt_thread = threading.Thread(target=start_mqtt_listener, daemon=True)
    mqtt_thread.start()
    st.session_state["mqtt_thread_started"] = True
    st.sidebar.success(" MQTT listener started in background")

# ----------------- Load detections -----------------
data = list(detections.find())
if not data:
    st.warning("No crack detection data found in MongoDB yet.")
    # don't stop: alerts may exist so continue to show alerts area
    df = pd.DataFrame([])
else:
    df = pd.DataFrame(data)
    # normalize columns if missing
    for col in ["label", "confidence", "timestamp", "video_source", "image_id"]:
        if col not in df.columns:
            df[col] = None
    df["label"] = df["label"].astype(str).str.lower()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["video_source"] = df["video_source"].fillna("unknown_source")
    df["date"] = df["timestamp"].dt.date
    df["time"] = df["timestamp"].dt.time

# ---------------- Sidebar filters ----------------
st.sidebar.header(" Filters")
if not df.empty:
    labels = sorted(df["label"].dropna().unique())
    sources = sorted(df["video_source"].dropna().unique())
else:
    labels = []
    sources = []
selected_labels = st.sidebar.multiselect("Select Crack Type", labels, default=labels if labels else [])
selected_sources = st.sidebar.multiselect("Select Video Source", sources, default=sources if sources else [])
if not df.empty:
    start_date = st.sidebar.date_input("Start Date", df["date"].min())
    end_date = st.sidebar.date_input("End Date", df["date"].max())
else:
    start_date = st.sidebar.date_input("Start Date", datetime.utcnow().date())
    end_date = st.sidebar.date_input("End Date", datetime.utcnow().date())

if not df.empty and (len(selected_labels) == 0):
    # if user deselects all, set default back to all
    selected_labels = labels
if not df.empty and (len(selected_sources) == 0):
    selected_sources = sources

if not df.empty:
    filtered_df = df[
        (df["label"].isin(selected_labels))
        & (df["video_source"].isin(selected_sources))
        & (df["date"] >= start_date)
        & (df["date"] <= end_date)
    ]
else:
    filtered_df = pd.DataFrame([])

# ---------------- CRACK ANALYTICS ----------------
if filtered_df.empty:
    st.info("No crack records match your filters or no detections yet.")
else:
    st.subheader(" Crack Detection Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric(" Total Detections", len(filtered_df))
    col2.metric(" Severe Cracks", len(filtered_df[filtered_df["label"] == "severe crack"]))
    avg_conf = filtered_df["confidence"].mean() if not filtered_df["confidence"].isnull().all() else 0.0
    col3.metric("🎯 Avg Confidence", f"{avg_conf:.2f}")

    st.markdown("---")
    st.subheader(" Crack Detections Over Time (by Source)")
    grouped = (
        filtered_df.groupby(["video_source", "date", "label"])
        .size()
        .reset_index(name="count")
        .sort_values("date")
    )
    if not grouped.empty:
        fig_time = px.line(
            grouped,
            x="date",
            y="count",
            color="label",
            facet_col="video_source",
            facet_col_wrap=2,
            markers=True,
            title="Crack Detections per Source Over Time",
        )
        st.plotly_chart(fig_time, use_container_width=True)

    st.markdown("---")
    st.subheader(" Confidence Distribution by Source")
    fig_conf = px.histogram(
        filtered_df,
        x="confidence",
        color="label",
        facet_col="video_source",
        nbins=20,
        title="Model Confidence Distribution",
    )
    st.plotly_chart(fig_conf, use_container_width=True)

    st.markdown("---")
    st.subheader(" Detection Class Breakdown")
    fig_pie = px.pie(filtered_df, names="label", title="Crack vs Severe Crack Ratio")
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    st.subheader(" Recent Severe Crack Detections")
    severe_df = (
        filtered_df[filtered_df["label"].str.lower() == "severe crack"]
        .sort_values("timestamp", ascending=False)
        .head(6)
    )

    if severe_df.empty:
        st.info("No severe crack images found for selected filters.")
    else:
        cols = st.columns(3)
        for i, (_, row) in enumerate(severe_df.iterrows()):
            image_id = row.get("image_id")
            caption = f"{row['video_source']} | Conf: {row['confidence']:.2f} | {row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
            if not image_id:
                cols[i % 3].warning(f"No image stored for {row['timestamp']}")
                continue
            try:
                # image_id in DB may be ObjectId or string
                _id = ObjectId(image_id) if not isinstance(image_id, ObjectId) else image_id
                file_doc = fs_files.find_one({"_id": _id})
                if not file_doc:
                    cols[i % 3].warning(" Image not found in GridFS.")
                    continue

                chunks = fs_chunks.find({"files_id": _id}).sort("n", 1)
                image_bytes = b"".join(bytes(chunk["data"]) for chunk in chunks)
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                cols[i % 3].image(
                    image,
                    caption=caption,
                    use_column_width=True,
                )
            except Exception as e:
                cols[i % 3].error(f" Error loading image: {e}")

    st.markdown("---")
    st.subheader(" Recent 10 Crack Detections")
    recent = filtered_df.sort_values("timestamp", ascending=False).head(10)
    st.dataframe(
        recent[["timestamp", "video_source", "label", "confidence", "frame_id"]],
        use_container_width=True,
    )

# ---------------- ALERTS SECTION ----------------
st.markdown("---")
st.header(" Real-Time Alerts (Water Level & Vibration)")

alert_data = list(alerts.find().sort("timestamp", -1).limit(100))
if not alert_data:
    st.info("No alerts logged yet. System stable.")
else:
    alert_df = pd.DataFrame(alert_data)
    alert_df["timestamp"] = pd.to_datetime(alert_df["timestamp"], errors="coerce", utc=True)
    alert_df["date"] = alert_df["timestamp"].dt.date

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Alerts", len(alert_df))
    col2.metric("Last Alert", alert_df["alert_type"].iloc[0])
    col3.metric("Last Seen", alert_df["timestamp"].iloc[0].strftime("%Y-%m-%d %H:%M:%S"))

    st.subheader(" Latest Alerts")
    st.dataframe(alert_df[["timestamp", "alert_type", "value", "device_id"]], use_container_width=True)

    st.markdown("---")
    st.subheader(" Alert Frequency Over Time")
    grouped_alerts = (
        alert_df.groupby(["date", "alert_type"])
        .size()
        .reset_index(name="count")
        .sort_values("date")
    )
    fig_alerts = px.bar(
        grouped_alerts,
        x="date",
        y="count",
        color="alert_type",
        title="Alert Frequency by Type Over Time",
    )
    st.plotly_chart(fig_alerts, use_container_width=True)

# ---------------- FUSED TIMELINE ----------------
if ("grouped" in locals() and not grouped.empty) and (not alert_data == []):
    st.markdown("---")
    st.subheader(" Combined Crack & Sensor Alert Timeline")

    grouped_combined = pd.concat([
        grouped.rename(columns={"label": "event_type", "count": "events"})[["date", "event_type", "events"]],
        grouped_alerts.rename(columns={"alert_type": "event_type", "count": "events"})[["date", "event_type", "events"]],
    ])
    grouped_combined = grouped_combined.groupby(["date", "event_type"]).sum().reset_index()

    fig_combined = px.line(
        grouped_combined, x="date", y="events", color="event_type",
        title="Crack and Sensor Alerts Over Time", markers=True
    )
    st.plotly_chart(fig_combined, use_container_width=True)

st.markdown("---")
st.caption(" Data Source: MongoDB `bridge_monitoring` database")
