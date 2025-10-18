import streamlit as st
from eda_module import run_eda

st.set_page_config(page_title="📈 Exploratory Analysis | Edis Analytics", layout="wide")

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Custom sidebar header ---
st.sidebar.image("assets/logo.png", width=60)
st.sidebar.markdown("<div class='sidebar-title'>Edis Analytics</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-subtitle'>Data Intelligence Suite</div>", unsafe_allow_html=True)
st.sidebar.markdown("---")


st.markdown("<h2 class='section-header'>📈 Exploratory Data Analysis</h2>", unsafe_allow_html=True)

if "clean_df" not in st.session_state:
    st.warning("⚠️ Please run data cleaning first.")
    st.stop()

df_clean = st.session_state["clean_df"]
df_for_ai = run_eda(df_clean)
st.session_state["eda_df"] = df_for_ai

st.success("✅ EDA complete! Continue to *AI Summary*.")
