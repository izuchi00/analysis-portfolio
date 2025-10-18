import streamlit as st
from ai_summary_module import generate_ai_summary
from detect_category import detect_dataset_category
from groq import Groq
import os

st.set_page_config(page_title="🧠 AI Summary | Edis Analytics", layout="wide")

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Custom sidebar header ---
st.sidebar.image("assets/logo.png", width=60)
st.sidebar.markdown("<div class='sidebar-title'>Edis Analytics</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-subtitle'>Data Intelligence Suite</div>", unsafe_allow_html=True)
st.sidebar.markdown("---")


st.markdown("<h2 class='section-header'>🧠 AI Dataset Summary</h2>", unsafe_allow_html=True)

if "clean_df" not in st.session_state:
    st.warning("⚠️ Please complete cleaning first.")
    st.stop()

api_key = os.getenv("GROQ_API_KEY", st.secrets.get("GROQ_API_KEY", ""))
if not api_key:
    st.error("❌ Missing `GROQ_API_KEY` in secrets.toml.")
    st.stop()

client = Groq(api_key=api_key)

df_clean = st.session_state["clean_df"]
sector = detect_dataset_category(df_clean)
ai_summary, insights = generate_ai_summary(client, df_clean, sector)

st.session_state["ai_summary"] = ai_summary
st.session_state["insights"] = insights
st.success("✅ AI Summary generated! Continue to *Guided Chat*.")
