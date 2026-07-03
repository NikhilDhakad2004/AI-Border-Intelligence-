import streamlit as st
from datetime import datetime
from zoneinfo import ZoneInfo
import cv2
import numpy as np
from PIL import Image
import sqlite3
import pandas as pd
from ultralytics import YOLO
from collections import Counter
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time

IST = ZoneInfo("Asia/Kolkata")

def now_ist():
    """Server clocks (e.g. Streamlit Cloud) run in UTC, so convert explicitly."""
    return datetime.now(IST)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Change-Detection Monitor",
    page_icon="🛰️",
    layout="wide"
)

# ---------------- YOLO MODEL (loaded once, not on every rerun) ----------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# COCO classes yolov8n.pt actually knows. No "tank" class exists in this model -
# any alert logic must be built from real classes only.
VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle"]

# ---------------- GEOLOCATOR ----------------
# Nominatim requires a real, descriptive user agent and respects a 1 req/sec limit.
geolocator = Nominatim(user_agent="ai-change-detection-monitor-demo")

def search_location(place):
    """Returns (lat, lon, error_message). error_message is None on success."""
    if not place or not place.strip():
        return None, None, "Please enter a location name."
    try:
        location = geolocator.geocode(place, timeout=10)
        if location is None:
            return None, None, f"No results found for '{place}'."
        return location.latitude, location.longitude, None
    except GeocoderTimedOut:
        return None, None, "Geocoding service timed out. Try again."
    except GeocoderServiceError as e:
        return None, None, f"Geocoding service error: {e}"
    except Exception as e:
        return None, None, f"Unexpected error: {e}"

# ---------------- DETECTION ----------------
def detect_objects(image_array):
    results = model(image_array)
    detected = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            detected.append(label)
    return detected, results

def filter_relevant_objects(counts):
    """Only classes actually relevant to a border/perimeter monitoring use case."""
    relevant = {}
    for obj in VEHICLE_CLASSES + ["person"]:
        if obj in counts:
            relevant[obj] = counts[obj]
    return relevant

# ---------------- SMART ALERT (honest version, no fake "tank" detection) ----------------
def generate_smart_alert(counts, change_percent):
    person = counts.get("person", 0)
    vehicles = sum(counts.get(v, 0) for v in VEHICLE_CLASSES)

    reasons = []
    risk_score = 0

    if vehicles > 3:
        risk_score += 2
        reasons.append(f"{vehicles} vehicles detected")
    elif vehicles > 0:
        risk_score += 1
        reasons.append(f"{vehicles} vehicle(s) detected")

    if person > 5:
        risk_score += 2
        reasons.append(f"{person} people detected (crowd)")
    elif person > 0:
        risk_score += 1
        reasons.append(f"{person} person(s) detected")

    if change_percent > 25:
        risk_score += 2
        reasons.append(f"large scene change ({change_percent:.1f}%)")
    elif change_percent > 10:
        risk_score += 1
        reasons.append(f"moderate scene change ({change_percent:.1f}%)")

    if not reasons:
        return "Low", "No significant activity or change detected"

    reason_text = ", ".join(reasons)
    if risk_score >= 4:
        return "High", reason_text
    elif risk_score >= 2:
        return "Medium", reason_text
    else:
        return "Low", reason_text

# ---------------- DATABASE ----------------
@st.cache_resource
def get_connection():
    conn = sqlite3.connect("border_intelligence.db", check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_date TEXT,
            change_percent REAL,
            risk_level TEXT,
            latitude REAL,
            longitude REAL,
            detected_objects TEXT
        )
    """)
    conn.commit()

    # Migration: older deployments may already have a "scans" table without
    # the detected_objects column. CREATE TABLE IF NOT EXISTS won't add it,
    # so check and patch the schema here.
    existing_cols = [row[1] for row in conn.execute("PRAGMA table_info(scans)").fetchall()]
    if "detected_objects" not in existing_cols:
        conn.execute("ALTER TABLE scans ADD COLUMN detected_objects TEXT")
        conn.commit()

    return conn

conn = get_connection()
cursor = conn.cursor()

# NOTE: On Streamlit Community Cloud the filesystem is ephemeral - this SQLite
# file resets whenever the app restarts or redeploys. For real persistence,
# swap this for a hosted DB (e.g. Postgres via st.connection, Supabase, etc).

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🛰️ Control Panel")

    page = st.radio("Navigation", [
        "Dashboard",
        "Image Scan",
        "Analytics",
        "Reports",
        "Settings"
    ])

    st.divider()
    st.write("🕒", now_ist().strftime("%d-%m-%Y"))
    st.write("⏰", now_ist().strftime("%H:%M:%S"))

# ==========================================================
# DASHBOARD (now reflects real DB state, not random numbers)
# ==========================================================
if page == "Dashboard":

    st.title("🛰️ AI Change-Detection Monitor")
    st.success("System ONLINE")

    df_dash = pd.read_sql_query("SELECT * FROM scans", conn)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Scans Logged", len(df_dash))
    col2.metric("High Risk Scans", int((df_dash["risk_level"] == "High").sum()) if len(df_dash) else 0)

    if len(df_dash):
        last_scan = df_dash.iloc[-1]["scan_date"]
    else:
        last_scan = "No scans yet"
    col3.metric("Last Scan", last_scan)

    st.caption(
        "This app detects everyday objects (people, cars, trucks, buses, motorcycles) "
        "using a standard YOLOv8 model and flags scene changes between two uploaded "
        "images. It does not use live satellite feeds and cannot detect military "
        "vehicles - the underlying model was never trained on that data."
    )

# ==========================================================
# IMAGE SCAN
# ==========================================================
elif page == "Image Scan":

    st.title("🛰️ Image Scan System")

    # ================= LOCATION SEARCH =================
    st.subheader("🔍 Search Location (City / Country)")

    location_name = st.text_input("Enter Location Name")

    if st.button("Search Location"):
        lat, lon, err = search_location(location_name)
        if err:
            st.error(err)
        else:
            st.session_state.lat = lat
            st.session_state.lon = lon
            st.success(f"Found: {lat:.4f}, {lon:.4f}")

    # ================= DEFAULT LOCATION =================
    if "lat" not in st.session_state:
        st.session_state.lat = 28.6139
    if "lon" not in st.session_state:
        st.session_state.lon = 77.2090

    # ================= MAP =================
    st.subheader("🌍 Live Map")

    m = folium.Map(
        location=[st.session_state.lat, st.session_state.lon],
        zoom_start=12,
        tiles="Esri.WorldImagery"
    )
    folium.Marker([st.session_state.lat, st.session_state.lon]).add_to(m)

    map_data = st_folium(m, width=900, height=500)

    if map_data and map_data.get("last_clicked"):
        st.session_state.lat = map_data["last_clicked"]["lat"]
        st.session_state.lon = map_data["last_clicked"]["lng"]

    st.write(f"📍 {st.session_state.lat:.4f}, {st.session_state.lon:.4f}")

    st.divider()

    # ================= MULTI LOCATION AUTO SCAN =================
    st.subheader("🌐 Multi Location Lookup")
    st.caption(
        "Geocodes each location below and logs a placeholder entry (no image "
        "analysis happens here - this only resolves coordinates, it does not "
        "generate real risk scores)."
    )

    locations_text = st.text_area(
        "Enter multiple locations (comma separated)",
        "Delhi, Mumbai, Chennai"
    )

    if st.button("Start Location Lookup"):
        locations = [l.strip() for l in locations_text.split(",") if l.strip()]

        if not locations:
            st.warning("Enter at least one location.")
        else:
            progress = st.progress(0)
            found, failed = 0, []

            for i, loc in enumerate(locations):
                lat, lon, err = search_location(loc)
                if err:
                    failed.append(f"{loc} ({err})")
                else:
                    found += 1
                    st.session_state[f"loc_{loc}"] = (lat, lon)

                progress.progress((i + 1) / len(locations))
                time.sleep(1)  # respect Nominatim's 1 req/sec usage policy

            st.success(f"Resolved {found}/{len(locations)} locations.")
            if failed:
                st.warning("Could not resolve: " + "; ".join(failed))

    st.divider()

    # ================= IMAGE UPLOAD =================
    st.subheader("🖼️ Upload Images")

    before_file = st.file_uploader("Before Image", type=["jpg", "jpeg", "png"])
    after_file = st.file_uploader("After Image", type=["jpg", "jpeg", "png"])

    if before_file and after_file:
        try:
            before_img = Image.open(before_file).convert("RGB")
            after_img = Image.open(after_file).convert("RGB")
        except Exception as e:
            st.error(f"Could not read one of the images: {e}")
            st.stop()

        before = np.array(before_img)
        after = np.array(after_img)

        col1, col2 = st.columns(2)
        col1.image(before, caption="Before")
        col2.image(after, caption="After")

        # ================= DETECTION =================
        if st.button("Run Detection"):
            with st.spinner("Running detection..."):
                try:
                    detected, results = detect_objects(after)
                except Exception as e:
                    st.error(f"Detection failed: {e}")
                    st.stop()

                counts = Counter(detected)
                st.write("All detected objects:", dict(counts))

                relevant = filter_relevant_objects(counts)
                st.write("Relevant objects (person / vehicles):", relevant)

                annotated = results[0].plot()
                st.image(annotated, caption="Detections")

                # ================= CHANGE DETECTION =================
                before_gray = cv2.cvtColor(cv2.resize(before, (600, 400)), cv2.COLOR_RGB2GRAY)
                after_gray = cv2.cvtColor(cv2.resize(after, (600, 400)), cv2.COLOR_RGB2GRAY)

                diff = cv2.absdiff(before_gray, after_gray)
                _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                change_percent = float(np.count_nonzero(thresh) / thresh.size * 100)

                st.write(f"Change %: {change_percent:.2f}")

                # ================= ALERT =================
                risk, reason = generate_smart_alert(counts, change_percent)

                if risk == "High":
                    st.error(f"HIGH RISK 🚨 | {reason}")
                elif risk == "Medium":
                    st.warning(f"MEDIUM RISK ⚠️ | {reason}")
                else:
                    st.success(f"LOW RISK ✅ | {reason}")

                # ================= SAVE =================
                cursor.execute(
                    """INSERT INTO scans
                    (scan_date, change_percent, risk_level, latitude, longitude, detected_objects)
                    VALUES (?,?,?,?,?,?)""",
                    (
                        now_ist().strftime("%d-%m-%Y %H:%M:%S"),
                        change_percent,
                        risk,
                        float(st.session_state.lat),
                        float(st.session_state.lon),
                        str(dict(counts)),
                    )
                )
                conn.commit()
                st.info("Scan saved to the log.")

# ==========================================================
# ANALYTICS
# ==========================================================
elif page == "Analytics":

    st.title("📊 Scan Analytics")

    df = pd.read_sql_query("SELECT * FROM scans", conn)

    if len(df) == 0:
        st.warning("No data available yet. Run a scan first.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Scans", len(df))
        col2.metric("High Risk", int((df["risk_level"] == "High").sum()))
        col3.metric("Avg Change %", round(df["change_percent"].mean(), 2))

        st.divider()

        st.subheader("🚨 Risk Distribution")
        st.bar_chart(df["risk_level"].value_counts())

        st.subheader("📈 Change Detection Trend")
        st.line_chart(df["change_percent"])

        st.subheader("🌍 Scan Locations")
        st.map(df.rename(columns={"latitude": "lat", "longitude": "lon"}))

        st.subheader("🔥 Change-Intensity Heatmap")
        from folium.plugins import HeatMap

        heat_data = df[["latitude", "longitude", "change_percent"]].values.tolist()
        heat_map = folium.Map(location=[20, 0], zoom_start=2)
        HeatMap(heat_data).add_to(heat_map)
        st_folium(heat_map, width=1000, height=500)

        st.caption("Red zones = larger detected change between before/after images at that location.")

# ==========================================================
# REPORTS
# ==========================================================
elif page == "Reports":

    st.title("📁 Scan Reports")

    df = pd.read_sql_query("SELECT * FROM scans", conn)

    if len(df) == 0:
        st.warning("No reports available")
    else:
        st.subheader("📋 All Scan Records")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Report (CSV)",
            data=csv,
            file_name="scan_report.csv",
            mime="text/csv"
        )

        st.subheader("🔍 Filter Reports")
        risk_filter = st.selectbox("Select Risk Level", ["All", "Low", "Medium", "High"])

        if risk_filter != "All":
            st.dataframe(df[df["risk_level"] == risk_filter])

# ==========================================================
# SETTINGS
# ==========================================================
elif page == "Settings":

    st.title("⚙️ System Settings & Limitations")

    st.subheader("Current capabilities")
    st.write("""
    - Object detection: standard YOLOv8n model (person, car, truck, bus, motorcycle, and ~75 other COCO classes)
    - Change detection: grayscale pixel-diff between two uploaded images
    - Location tools: OpenStreetMap/Nominatim geocoding + folium maps
    - Storage: local SQLite (resets on redeploy if hosted on Streamlit Community Cloud)
    """)

    st.subheader("Known limitations")
    st.write("""
    - No live satellite feed - image analysis only works on files you upload
    - Cannot detect military vehicles, weapons, or aircraft types - the model was never trained on that data
    - Dashboard/analytics numbers are computed from your own logged scans, not external intelligence sources
    - Nominatim geocoding is rate-limited to ~1 request/second; large batch lookups will be slow
    """)

    st.subheader("Possible future upgrades")
    st.write("""
    - Custom-trained detector for domain-specific object classes (would require a labeled dataset)
    - Persistent hosted database instead of local SQLite
    - Scheduled/automated re-scanning of fixed locations
    - Alerting via email/Telegram webhook
    """)

# ==========================================================
# FOOTER
# ==========================================================
st.divider()
st.caption("🛰️ AI Change-Detection Monitor | Built by Nikhil Dhakad")
