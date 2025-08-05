import os
from dotenv import load_dotenv

import streamlit as st
from groq import Groq
from transformers import pipeline
st.set_page_config(page_title="Voice Answer Generator", layout="centered")
# -------- Hide Streamlit Sidebar --------
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

# --- Load environment variables ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_API_KEY   = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# --- Streamlit UI ---

st.title("🎙 Mom’s Virtual Friend - Voice Answer Generator")

user_input = st.text_area("📝 Ask a question:")

if st.button("🎤 Generate Voice Answer"):
    if not user_input.strip():
        st.warning("Please enter a question.")
    else:
        # Step 1: Generate text with Groq
        with st.spinner("Generating answer with Groq..."):
            client = Groq(api_key=GROQ_API_KEY)
            groq_response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": user_input}],
                temperature=0.7,
                max_tokens=512
            )
            answer_text = groq_response.choices[0].message.content.strip()
        st.success("Text answer generated.")
        st.markdown(f"💬 **Answer:** {answer_text}")

        # Step 2: Convert to speech
        with st.spinner("Converting text to speech..."):
            tts = pipeline(
                "text-to-speech",
                model="facebook/mms-tts-eng",
                token=HF_API_KEY
            )
            output = tts(answer_text)

        # Step 3: Play inline
        st.audio(
            output["audio"],
            format="audio/wav",
            sample_rate=output["sampling_rate"]
        )
        st.success("✅ Voice answer ready!")
