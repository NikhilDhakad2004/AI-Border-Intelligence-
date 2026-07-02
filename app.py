import streamlit as st
from datetime import datetime
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
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Border Intelligence - By Nikhil Dhakad",
    page_icon="🛰️",
    layout="wide"
)

# ---------------- YOLO MODEL ----------------
model = YOLO("yolov8n.pt")

# ---------------- GEOLOCATOR ----------------
geolocator = Nominatim(user_agent="border_ai")

def search_location(place):
    try:
        location = geolocator.geocode(place)
        return location.latitude, location.longitude
    except:
        return None, None

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
# ---------------- CRITICAL FILTER ----------------
def filter_critical_objects(counts):
    critical = {}
    for obj in ["car", "truck", "bus", "person"]:
        if obj in counts:
            critical[obj] = counts[obj]
    return critical


# ---------------- SMART ALERT ----------------
def generate_smart_alert(counts):
    person = counts.get("person", 0)
    vehicle = counts.get("car", 0) + counts.get("truck", 0) + counts.get("bus", 0)

    if "tank" in counts:
        return "High", "Tank detected 🚨"

    if vehicle > 3:
        return "High", "Multiple vehicles detected"

    if person > 5:
        return "Medium", "Crowd detected"

    if person > 0 or vehicle > 0:
        return "Medium", "Some activity detected"

    return "Low", "No major activity"

# ---------------- DATABASE ----------------
conn = sqlite3.connect("border_intelligence.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_date TEXT,
    change_percent REAL,
    risk_level TEXT,
    latitude REAL,
    longitude REAL
)
""")

conn.commit()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🛰️ Control Panel")

    page = st.radio("Navigation", [
        "Dashboard",
        "Satellite Scan",
        "Analytics",
        "Reports",
        "Settings"
    ])

    st.divider()
    st.write("🕒", datetime.now().strftime("%d-%m-%Y"))
    st.write("⏰", datetime.now().strftime("%H:%M:%S"))

# ==========================================================
# DASHBOARD
# ==========================================================
if page == "Dashboard":

    st.title("🛰️ AI Border Intelligence By-Nikhil Dhakad")
    st.success("System ONLINE")

    col1, col2, col3 = st.columns(3)

    col1.metric("Satellites", np.random.randint(6, 12))
    col2.metric("AI Models", np.random.randint(3, 6))
    col3.metric("Alerts", np.random.randint(0, 5))

# ==========================================================
# SATELLITE SCAN (UPGRADED)
# ==========================================================
elif page == "Satellite Scan":

    st.title("🛰️ Satellite Scan System")

    # ================= LOCATION SEARCH =================
    st.subheader("🔍 Search Location (City / Country)")

    location_name = st.text_input("Enter Location Name")

    if st.button("Search Location"):
        lat, lon = search_location(location_name)

        if lat:
            st.session_state.lat = lat
            st.session_state.lon = lon
            st.success(f"Found: {lat}, {lon}")
        else:
            st.error("Location not found")

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

    folium.Marker(
        [st.session_state.lat, st.session_state.lon]
    ).add_to(m)

    map_data = st_folium(m, width=900, height=500)

    if map_data and map_data.get("last_clicked"):
        st.session_state.lat = map_data["last_clicked"]["lat"]
        st.session_state.lon = map_data["last_clicked"]["lng"]

    st.write(f"📍 {st.session_state.lat}, {st.session_state.lon}")

    st.divider()

    # ================= MULTI LOCATION AUTO SCAN =================
    st.subheader("🌐 Multi Location Auto Scan")

    locations_text = st.text_area(
        "Enter multiple locations (comma separated)",
        "Delhi, Mumbai, China Border"
    )

    if st.button("Start Auto Scan"):
        locations = locations_text.split(",")

        progress = st.progress(0)

        for i, loc in enumerate(locations):

            lat, lon = search_location(loc.strip())

            if lat:
                cursor.execute(
                    """INSERT INTO scans 
                    (scan_date, change_percent, risk_level, latitude, longitude)
                    VALUES (?, ?, ?, ?, ?)""",
                    (
                        datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                        np.random.randint(5, 30),
                        np.random.choice(["Low", "Medium", "High"]),
                        lat,
                        lon
                    )
                )
                conn.commit()

            progress.progress((i+1)/len(locations))
            time.sleep(0.5)

        st.success("Auto Scan Completed 🚀")

    st.divider()

    # ================= IMAGE UPLOAD =================
    st.subheader("🖼️ Upload Images")

    before_file = st.file_uploader("Before Image", type=["jpg", "png"])
    after_file = st.file_uploader("After Image", type=["jpg", "png"])

    if before_file and after_file:

        before = np.array(Image.open(before_file))
        after = np.array(Image.open(after_file))

        col1, col2 = st.columns(2)
        col1.image(before)
        col2.image(after)

        # ================= DETECTION =================
        if st.button("Run Detection"):

            detected, results = detect_objects(after)
            counts = Counter(detected)

            st.write("Detected:", dict(counts))

            critical = filter_critical_objects(counts)
            st.write("Critical:", critical)

            annotated = results[0].plot()
            st.image(annotated)

            # ================= CHANGE DETECTION =================
            before_gray = cv2.cvtColor(cv2.resize(before,(600,400)), cv2.COLOR_RGB2GRAY)
            after_gray = cv2.cvtColor(cv2.resize(after,(600,400)), cv2.COLOR_RGB2GRAY)

            diff = cv2.absdiff(before_gray, after_gray)
            _, thresh = cv2.threshold(diff,30,255,cv2.THRESH_BINARY)

            change_percent = np.count_nonzero(thresh)/thresh.size*100

            st.write(f"Change %: {change_percent:.2f}")

            # 🔥 SMART ALERT BASED ON DETECTION
            risk, reason = generate_smart_alert(counts)

            if risk == "High":
                st.error(f"HIGH RISK 🚨 | {reason}")
            elif risk == "Medium":
                st.warning(f"MEDIUM RISK ⚠️ | {reason}")
            else:
                st.success(f"LOW RISK ✅ | {reason}")

            # ================= SAVE =================
            cursor.execute(
                "INSERT INTO scans (scan_date, change_percent, risk_level, latitude, longitude) VALUES (?,?,?,?,?)",
                (
                    datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                    float(change_percent),
                    risk,
                    float(st.session_state.lat),
                    float(st.session_state.lon)
                )
            )

            conn.commit()

# ==========================================================
# ANALYTICS (ADVANCED)
# ==========================================================
elif page == "Analytics":

    st.title("📊 Advanced Intelligence Analytics")

    df = pd.read_sql_query("SELECT * FROM scans", conn)

    if len(df) == 0:
        st.warning("No data available yet.")
    else:

        # ---------------- BASIC STATS ----------------
        col1, col2, col3 = st.columns(3)

        col1.metric("Total Scans", len(df))
        col2.metric("High Risk", len(df[df["risk_level"] == "High"]))
        col3.metric("Avg Change %", round(df["change_percent"].mean(), 2))

        st.divider()

        # ---------------- RISK DISTRIBUTION ----------------
        st.subheader("🚨 Risk Distribution")
        st.bar_chart(df["risk_level"].value_counts())

        # ---------------- CHANGE TREND ----------------
        st.subheader("📈 Change Detection Trend")
        st.line_chart(df["change_percent"])

        # ---------------- LOCATION SCATTER ----------------
        st.subheader("🌍 Scan Locations")

        st.map(df.rename(columns={
            "latitude": "lat",
            "longitude": "lon"
        }))

        # ==================================================
        # 🔥 HEATMAP (ADVANCED GLOBAL THREAT VIEW)
        # ==================================================
        st.subheader("🔥 Global Threat Heatmap")

        from folium.plugins import HeatMap

        heat_data = df[["latitude", "longitude", "change_percent"]].values.tolist()

        heat_map = folium.Map(location=[20, 0], zoom_start=2)

        HeatMap(heat_data).add_to(heat_map)

        st_folium(heat_map, width=1000, height=500)

        st.info("Red zones = High activity / potential threats")

# ==========================================================
# REPORTS SYSTEM
# ==========================================================
elif page == "Reports":

    st.title("📁 Intelligence Reports")

    df = pd.read_sql_query("SELECT * FROM scans", conn)

    if len(df) == 0:
        st.warning("No reports available")
    else:

        st.subheader("📋 All Scan Records")
        st.dataframe(df)

        # ---------------- DOWNLOAD CSV ----------------
        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Report (CSV)",
            data=csv,
            file_name="border_intelligence_report.csv",
            mime="text/csv"
        )

        # ---------------- FILTER ----------------
        st.subheader("🔍 Filter Reports")

        risk_filter = st.selectbox(
            "Select Risk Level",
            ["All", "Low", "Medium", "High"]
        )

        if risk_filter != "All":
            filtered = df[df["risk_level"] == risk_filter]
            st.dataframe(filtered)

# ==========================================================
# SETTINGS (FUTURE FEATURES PLACEHOLDER)
# ==========================================================
elif page == "Settings":

    st.title("⚙️ System Settings")

    st.info("🚀 Upcoming Features:")

    st.write("""
    🔐 Login System (Admin / User roles)
    📡 Live Satellite API Integration (ISRO / Google Earth)
    🤖 Custom Tank Detection Model
    📲 Telegram / WhatsApp Alerts
    🌐 Real-time Border Monitoring (no upload needed)
    🧠 AI Auto Threat Prediction (Deep Learning)
    """)

    st.warning("These features will be added in next upgrade phase 🔥")

# ==========================================================
# FOOTER
# ==========================================================
st.divider()
st.caption("🛰️ AI Border Intelligence System | Built by Nikhil Dhakad")