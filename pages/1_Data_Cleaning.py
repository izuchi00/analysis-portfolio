import streamlit as st
from clean_module import auto_data_clean

st.set_page_config(page_title="🧹 Data Cleaning | Edis Analytics", layout="wide")

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Custom sidebar header ---
st.sidebar.image("assets/logo.png", width=60)
st.sidebar.markdown("<div class='sidebar-title'>Edis Analytics</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-subtitle'>Data Intelligence Suite</div>", unsafe_allow_html=True)
st.sidebar.markdown("---")


st.markdown("<h2 class='section-header'>🧹 Data Cleaning & Quality Check</h2>", unsafe_allow_html=True)

if "uploaded_df" not in st.session_state:
    st.warning("⚠️ Please upload a dataset first on the Home page.")
    st.stop()

df = st.session_state["uploaded_df"]
df_clean = auto_data_clean(df)
st.session_state["clean_df"] = df_clean

st.success("✅ Data cleaning complete! Proceed to *Exploratory Analysis* page.")
