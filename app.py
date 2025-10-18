# ============================================================
# 📊 EDIS ANALYTICS — HOME / UPLOAD PAGE
# ============================================================

import streamlit as st
import pandas as pd
import os

# --- Page setup ---
st.set_page_config(
    page_title="Edis Analytics | Data Analysis Portfolio",
    layout="wide",
    page_icon="📊"
)

# --- Load Custom CSS ---
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Hero Section ---
st.markdown("""
<div class="hero-banner fade-scroll">
  <div class="hero-content">
    <img src="assets/logo.png" class="hero-logo" alt="Edis Analytics Logo">
    <div class="hero-text">
      <h1>Edis Analytics</h1>
      <p>Automated Data Insights Powered by AI</p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# --- File upload ---
st.markdown("<h2 class='section-header'>📂 Upload Your Dataset</h2>", unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Drag and drop your file or click to browse",
    type=["csv", "xlsx", "xls", "pdf"],
    label_visibility="collapsed"
)

# --- Safe loader (light version) ---
def safe_load_file(uploaded_file):
    import importlib, io
    if not uploaded_file:
        return None
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        elif name.endswith(".xlsx"):
            import openpyxl
            return pd.read_excel(uploaded_file, engine="openpyxl")
        elif name.endswith(".xls"):
            import xlrd
            return pd.read_excel(uploaded_file, engine="xlrd")
        elif name.endswith(".pdf"):
            import pdfplumber
            all_tables = []
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    t = page.extract_table()
                    if t:
                        dfp = pd.DataFrame(t[1:], columns=t[0])
                        all_tables.append(dfp)
            if all_tables:
                return pd.concat(all_tables, ignore_index=True)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# --- Preview + Session save ---
if uploaded:
    df = safe_load_file(uploaded)
    if df is not None:
        st.success(f"✅ {uploaded.name} loaded successfully!")
        st.dataframe(df.head(), use_container_width=True)
        st.session_state["uploaded_df"] = df
        st.info("Proceed to the next page ➡️ *Data Cleaning*")
    else:
        st.warning("⚠️ Could not load file. Please try again.")
else:
    st.info("👆 Upload a dataset to begin your analysis.")

# --- Footer ---
st.markdown("""
<div class='footer'>
  <p>© 2025 <strong>Edis Analytics</strong> • Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)
