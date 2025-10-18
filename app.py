# ============================================================
# 📊 INTERACTIVE DATA ANALYSIS PORTFOLIO APP
# ============================================================

import streamlit as st
import pandas as pd
from groq import Groq
import os

# --- Import custom modules ---
from clean_module import auto_data_clean
from detect_category import detect_dataset_category
from eda_module import run_eda
from ai_summary_module import generate_ai_summary
from guided_chat_module import launch_basic_chat


# ============================================================
# ⚙️ PAGE SETUP
# ============================================================
st.set_page_config(page_title="Data Analysis Portfolio", layout="wide")
st.title("📊 Interactive Data Analysis Portfolio")


# ============================================================
# 🔐 SECURE API KEY
# ============================================================
api_key = os.getenv("GROQ_API_KEY", st.secrets.get("GROQ_API_KEY", ""))
if not api_key:
    st.error("❌ Missing `GROQ_API_KEY`. Add it to `.streamlit/secrets.toml`.")
    st.stop()

client = Groq(api_key=api_key)


# ============================================================
# 📂 FILE UPLOAD
# ============================================================
st.markdown("### 📂 Upload your CSV, Excel, or PDF dataset")

uploaded = st.file_uploader(
    "Drag and drop your file here",
    type=["csv", "xlsx", "xls", "pdf"],
    accept_multiple_files=False
)


# ============================================================
# 📥 SAFE FILE LOADER
# ============================================================
def safe_load_file(uploaded_file):
    """Safely load CSV, XLSX, XLS, or PDF with dependency checks and clear feedback."""
    import importlib
    import pandas as pd

    file_name = uploaded_file.name.lower()

    try:
        # --- CSV ---
        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ CSV file **{uploaded_file.name}** loaded successfully!")
            return df

        # --- XLSX ---
        elif file_name.endswith(".xlsx"):
            if importlib.util.find_spec("openpyxl") is None:
                st.error("⚠️ Missing dependency: `openpyxl` is required for .xlsx files.\nRun `pip install openpyxl`.")
                return None
            df = pd.read_excel(uploaded_file, engine="openpyxl")
            st.success(f"✅ Excel file **{uploaded_file.name}** loaded successfully!")
            return df

        # --- XLS ---
        elif file_name.endswith(".xls"):
            if importlib.util.find_spec("xlrd") is None:
                st.error("⚠️ Missing dependency: `xlrd>=2.0.1` is required for .xls files.\nRun `pip install xlrd`.")
                return None
            df = pd.read_excel(uploaded_file, engine="xlrd")
            st.success(f"✅ Legacy Excel file **{uploaded_file.name}** loaded successfully!")
            return df

        # --- PDF ---
        elif file_name.endswith(".pdf"):
            if importlib.util.find_spec("pdfplumber") is None:
                st.error("⚠️ Missing dependency: `pdfplumber` is required for PDFs.\nRun `pip install pdfplumber`.")
                return None

            import pdfplumber
            all_tables = []
            with pdfplumber.open(uploaded_file) as pdf:
                for i, page in enumerate(pdf.pages):
                    table = page.extract_table()
                    if table:
                        df_page = pd.DataFrame(table[1:], columns=table[0])
                        all_tables.append(df_page)

            if all_tables:
                df = pd.concat(all_tables, ignore_index=True)
                st.success(f"✅ Extracted {len(all_tables)} table(s) from **{uploaded_file.name}** successfully!")
                return df
            else:
                st.warning("⚠️ No readable tables were found in this PDF.")
                return None

        # --- Unsupported ---
        else:
            st.error("❌ Unsupported file type. Please upload a CSV, XLSX, XLS, or PDF file.")
            return None

    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        return None


# ============================================================
# 🚀 MAIN APP LOGIC
# ============================================================
if uploaded:
    df = safe_load_file(uploaded)

    if df is not None:
        st.dataframe(df.head())

        # --- 1️⃣ Data Cleaning ---
        st.divider()
        st.subheader("🧹 Data Cleaning")
        df_clean = auto_data_clean(df)

        if df_clean is not None:
            # --- 2️⃣ Detect Dataset Category ---
            sector = detect_dataset_category(df_clean)
            st.info(f"🧭 Detected dataset category: **{sector}**")

            # --- 3️⃣ Exploratory Data Analysis ---
            st.subheader("📈 Exploratory Data Analysis")
            df_for_ai = run_eda(df_clean)

            # --- 4️⃣ AI Dataset Summary ---
            st.subheader("🧠 AI Dataset Summary")

            # 🧩 Cache summary to avoid re-running every refresh
            @st.cache_data(show_spinner="🤖 Generating AI Summary...")
            def cached_ai_summary(_client, _df, _sector):
                return generate_ai_summary(_client, _df, _sector)

            ai_summary, insights = cached_ai_summary(client, df_clean, sector)

            # Display summary (already formatted inside the module)
            st.markdown("---")

            # --- 5️⃣ AI Chat Assistant ---
            st.subheader("💬 Basic Dataset Chat")
            launch_basic_chat(client, ai_summary, insights, df_for_ai, sector)

            # --- 6️⃣ Wrap-up ---
            st.markdown("""
            ---
            ### 🚀 Next Steps
            For deeper **AI-driven analytics**, **predictive modeling**, or **custom dashboards**,  
            please **[contact or hire me](#)** to unlock advanced modules.
            ---
            """)

    else:
        st.stop()

else:
    st.info("👆 Upload a dataset to begin your analysis.")
