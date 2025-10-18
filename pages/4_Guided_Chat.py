import streamlit as st
from guided_chat_module import launch_basic_chat
from groq import Groq
import os

st.set_page_config(page_title="💬 Guided Chat | Edis Analytics", layout="wide")

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Custom sidebar header ---
st.sidebar.image("assets/logo.png", width=60)
st.sidebar.markdown("<div class='sidebar-title'>Edis Analytics</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-subtitle'>Data Intelligence Suite</div>", unsafe_allow_html=True)
st.sidebar.markdown("---")


st.markdown("<h2 class='section-header'>💬 Guided Dataset Chat</h2>", unsafe_allow_html=True)

if "ai_summary" not in st.session_state or "eda_df" not in st.session_state:
    st.warning("⚠️ Please complete the previous steps first.")
    st.stop()

api_key = os.getenv("GROQ_API_KEY", st.secrets.get("GROQ_API_KEY", ""))
client = Groq(api_key=api_key)

launch_basic_chat(
    client,
    st.session_state["ai_summary"],
    st.session_state["insights"],
    st.session_state["eda_df"],
)
