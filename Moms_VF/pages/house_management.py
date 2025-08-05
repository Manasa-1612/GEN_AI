import streamlit as st
from datetime import datetime, date
from transformers import pipeline
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
hf_api_key = os.getenv("HF_API_KEY")

# Hide Sidebar
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

# Page Title
st.title("🏠 House Management Planner")

# 🕒 Get current day
today_day = datetime.today().strftime('%A')
st.markdown(f"📅 **Today is:** `{today_day}`")

# User Input: Select Days and Tasks
days = st.number_input("Days in a week for specific tasks", min_value=1, max_value=7, value=1)

# Store all selected days/tasks
task_schedule = []

for i in range(days):
    day = st.selectbox(f"Day {i+1}", 
                       ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                       key=f"day_{i}")
    task = st.selectbox(f"Task for {day}", 
                        ["Laundry", "Gardening", "Washing vehicles", "Kitchen deep cleaning"],
                        key=f"task_{i}")
    task_schedule.append((day, task))

# Load TTS Model
@st.cache_resource
def load_tts_pipeline():
    return pipeline(
        "text-to-speech",
        model="facebook/mms-tts-eng",
        token=hf_api_key
    )

tts = load_tts_pipeline()

# Generate Reminder Audio
if st.button("🔊 Generate Reminder Audio"):
    if not task_schedule:
        st.warning("Please select at least one day and task.")
    else:
        for day, task in task_schedule:
            reminder_text = f"Task Reminder: Today is {day} and your task is to do {task}."
            with st.spinner(f"Generating audio for task '{task}' on {day}..."):
                output = tts(reminder_text)
                st.audio(
                    output["audio"],
                    format="audio/wav",
                    sample_rate=output["sampling_rate"]
                )
                st.success(f"✅ Voice reminder for {day} - {task} is ready!")