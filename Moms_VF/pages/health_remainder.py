import streamlit as st
from datetime import datetime, date, timedelta
from transformers import pipeline
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
hf_api_key = os.getenv("HF_API_KEY")

# Hide sidebar
hide_sidebar = """
    <style>
        [data-testid="stSidebar"] {
            display: none;
        }
        [data-testid="collapsedControl"] {
            display: none;
        }
    </style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

st.title("💊 Medicine Voice Reminder App")

# Collect tablet information in a list
tablets = st.number_input("No of Tablets per day", min_value=1, max_value=10, value=2, step=1)

tablet_data = []
for i in range(tablets):
    name = st.text_input(f"Tablet {i+1} Name", placeholder=f"e.g., Tablet {i+1}")
    dosage = st.number_input(f"Tablet {i+1} Dosage (mg)", min_value=1, max_value=1000, value=500, step=1, key=f"dosage_{i}")
    morning = st.selectbox(f"Tablet {i+1} Morning Time", ["8:00 AM", "9:00 AM", "10:00 AM", "NA"], index=1, key=f"morning_{i}")
    evening = st.selectbox(f"Tablet {i+1} Evening Time", ["7:00 PM", "8:00 PM", "9:00 PM", "NA"], index=1, key=f"evening_{i}")

    # Save each tablet's details in a list of dicts
    tablet_data.append({
        "name": name,
        "dosage": dosage,
        "morning": morning,
        "evening": evening
    })

# Select end date
end_date = st.date_input("End Date for Reminders", value=date.today())
current_dt = datetime.now()
st.markdown(f"🕒 Current Date & Time: *{current_dt.strftime('%Y-%m-%d %H:%M:%S')}*")

# Load TTS model
@st.cache_resource
def load_tts_pipeline():
    return pipeline(
        "text-to-speech",
        model="facebook/mms-tts-eng",
        token=hf_api_key
    )

tts = load_tts_pipeline()

# Generate audio
if st.button("🔊 Generate Reminder Audio"):
    for tab in tablet_data:
        name = tab["name"]
        dosage = tab["dosage"]
        morning = tab["morning"]
        evening = tab["evening"]

        if not name or not dosage:
            st.warning("Please enter both tablet name and dosage.")
            continue

        if morning != "NA":
            msg = f"Reminder: Take {name} of {dosage} mg in the morning at {morning}."
            with st.spinner(f"Generating morning reminder for {name}..."):
                output = tts(msg)
                st.audio(output["audio"], format="audio/wav", sample_rate=output["sampling_rate"])
                st.success(f"✅ Morning reminder for {name} is ready!")

        if evening != "NA":
            msg = f"Reminder: Take {name} of {dosage} mg in the evening at {evening}."
            with st.spinner(f"Generating evening reminder for {name}..."):
                output = tts(msg)
                st.audio(output["audio"], format="audio/wav", sample_rate=output["sampling_rate"])
                st.success(f"✅ Evening reminder for {name} is ready!")