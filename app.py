import streamlit as st
import pandas as pd
from groq import Groq
import os

from clean_module import auto_data_clean
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
    st.error("❌ Missing GROQ_API_KEY. Add it to .streamlit/secrets.toml.")
    st.stop()

client = Groq(api_key=api_key)

# --- File upload ---
uploaded = st.file_uploader("📁 Upload your CSV or Excel dataset", type=["csv", "xlsx"])
if uploaded:
    try:
        df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
        st.success(f"✅ File '{uploaded.name}' uploaded successfully!")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # --- Data cleaning ---
    st.subheader("🧹 Data Cleaning")
    df = auto_data_clean(df)

    # --- Category detection ---
    sector = detect_dataset_category(df)

    # --- EDA ---
    st.subheader("📈 Exploratory Data Analysis")
    df_for_ai = run_eda(df)

    # --- AI Summary ---
    st.subheader("🧠 AI Dataset Summary")
    ai_summary, insights = generate_ai_summary(client, df, sector)
    st.markdown(ai_summary)

    # --- Guided Chat ---
    st.subheader("💬 Basic Dataset Chat")
    launch_basic_chat(client, ai_summary, insights, df_for_ai, sector)

    # --- Next Steps CTA ---
    st.markdown("""
    ---
    ### 🚀 Next Steps
    For deeper **AI-driven analytics**, **predictive modeling**, or **custom dashboards**,  
    please **[contact or hire me](#)** to unlock the advanced modules.
    ---
    """)

else:
    st.info("👆 Upload a dataset to begin your analysis.")
