# ============================================================
# 📊 INTERACTIVE DATA ANALYSIS PORTFOLIO APP
# ============================================================

import streamlit as st
import pandas as pd
from groq import Groq
import os

# --- Import custom modules ---
from clean_module import auto_data_clean          # ✅ corrected name
from detect_category import detect_dataset_category
from eda_module import run_eda
from ai_summary_module import generate_ai_summary
from guided_chat_module import launch_basic_chat


# --- Page setup ---
st.set_page_config(page_title="Data Analysis Portfolio", layout="wide")
st.title("📊 Interactive Data Analysis Portfolio")


# --- Secure API key ---
api_key = os.getenv("GROQ_API_KEY", st.secrets.get("GROQ_API_KEY", ""))
if not api_key:
    st.error("❌ Missing `GROQ_API_KEY`. Add it to `.streamlit/secrets.toml`.")
    st.stop()

client = Groq(api_key=api_key)


# --- File upload ---
uploaded = st.file_uploader("📁 Upload your CSV or Excel dataset", type=["csv", "xlsx", "xls", "pdf"])

if uploaded:
    try:
        # --- Load dataset automatically ---
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)

        st.success(f"✅ File **'{uploaded.name}'** uploaded successfully!")
        st.dataframe(df.head())

    except Exception as e:
        st.error(f"⚠️ Error reading file: {e}")
        st.stop()

    # --- 1️⃣ Data cleaning ---
    st.subheader("🧹 Data Cleaning")
    df_clean = auto_data_clean(df)   # ✅ updated call + receives log

    if df_clean is not None:

        # --- 2️⃣ Category detection ---
        sector = detect_dataset_category(df_clean)
        st.info(f"🧭 Detected dataset category: **{sector}**")

        # --- 3️⃣ Exploratory Data Analysis ---
        st.subheader("📈 Exploratory Data Analysis")
        df_for_ai = run_eda(df_clean)

        # --- 4️⃣ AI Summary ---
        st.subheader("🧠 AI Dataset Summary")
        ai_summary, insights = generate_ai_summary(client, df_clean, sector)
        st.markdown(ai_summary)

        # --- 5️⃣ Guided Chat ---
        st.subheader("💬 Basic Dataset Chat")
        launch_basic_chat(client, ai_summary, insights, df_for_ai, sector)

        # --- 6️⃣ Call-to-action ---
        st.markdown("""
        ---
        ### 🚀 Next Steps
        For deeper **AI-driven analytics**, **predictive modeling**, or **custom dashboards**,  
        please **[contact or hire me](#)** to unlock the advanced modules.
        ---
        """)

    else:
        st.error("❌ Cleaning failed. Please check your dataset format and retry.")

else:
    st.info("👆 Upload a dataset to begin your analysis.")
