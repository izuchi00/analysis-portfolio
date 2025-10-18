# ============================================================
# 📊 INTERACTIVE DATA ANALYSIS PORTFOLIO APP (UI-Enhanced)
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
# 🏗️ PAGE SETUP
# ============================================================
st.set_page_config(page_title="AI Data Analysis Portfolio", layout="wide")

st.markdown("""
<h1 style='text-align: center; color: #2563EB;'>
📊 Interactive Data Analysis Portfolio
</h1>
<p style='text-align: center; color: gray;'>
Upload, clean, explore, and understand your dataset — powered by AI insights.
</p>
""", unsafe_allow_html=True)

st.divider()


# ============================================================
# 🔐 Secure API Key
# ============================================================
api_key = os.getenv("GROQ_API_KEY", st.secrets.get("GROQ_API_KEY", ""))
if not api_key:
    st.error("❌ Missing `GROQ_API_KEY`. Add it to `.streamlit/secrets.toml`.")
    st.stop()

client = Groq(api_key=api_key)


# ============================================================
# 📂 File Upload Section
# ============================================================
with st.container():
    st.markdown("### 📂 Upload Your Dataset")
    st.caption("Supports **CSV**, **Excel (.xlsx/.xls)**, and **PDF tables**.")
    uploaded = st.file_uploader(
        "Drag and drop or browse your file",
        type=["csv", "xlsx", "xls", "pdf"],
        accept_multiple_files=False
    )


# ============================================================
# 🧩 Safe File Loader
# ============================================================
def safe_load_file(uploaded_file):
    """Safely load CSV, XLSX, XLS, or PDF with dependency checks and clear feedback."""
    import importlib
    import pandas as pd
    import io

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
                st.error("⚠️ Missing dependency: `openpyxl` required. Run `pip install openpyxl`.")
                return None
            df = pd.read_excel(uploaded_file, engine="openpyxl")
            st.success(f"✅ Excel file **{uploaded_file.name}** loaded successfully!")
            return df

        # --- XLS ---
        elif file_name.endswith(".xls"):
            if importlib.util.find_spec("xlrd") is None:
                st.error("⚠️ Missing dependency: `xlrd>=2.0.1` required. Run `pip install xlrd`.")
                return None
            df = pd.read_excel(uploaded_file, engine="xlrd")
            st.success(f"✅ Legacy Excel file **{uploaded_file.name}** loaded successfully!")
            return df

        # --- PDF ---
        elif file_name.endswith(".pdf"):
            if importlib.util.find_spec("pdfplumber") is None:
                st.error("⚠️ Missing dependency: `pdfplumber` required. Run `pip install pdfplumber`.")
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
                st.success(f"✅ Extracted {len(all_tables)} table(s) from **{uploaded_file.name}**.")
                return df
            else:
                st.warning("⚠️ No readable tables were found in this PDF.")
                return None

        else:
            st.error("❌ Unsupported file type. Please upload CSV, XLSX, XLS, or PDF.")
            return None

    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        return None


# ============================================================
# 🚀 Main Logic Flow
# ============================================================
if uploaded:
    df = safe_load_file(uploaded)

    if df is not None:
        st.divider()

        # --- Dataset Preview ---
        with st.expander("👁️ Preview Uploaded Dataset", expanded=True):
            st.dataframe(df.head(), use_container_width=True)

        # --- Data Cleaning ---
        st.markdown("### 🧹 Data Cleaning & Quality Check")
        with st.spinner("Running cleaning module..."):
            df_clean = auto_data_clean(df)

        if df_clean is not None:
            st.success("✅ Data cleaning completed successfully!")

            # --- Category Detection ---
            st.divider()
            st.markdown("### 🧭 Dataset Category Detection")
            sector = detect_dataset_category(df_clean)
            st.info(f"Detected dataset category: **{sector}**")

            # --- EDA ---
            st.divider()
            st.markdown("### 📈 Exploratory Data Analysis")
            st.caption("Auto-generated feature distributions, correlations, and summary metrics.")
            df_for_ai = run_eda(df_clean)

            # --- AI Summary ---
            st.divider()
            st.markdown("### 🧠 AI Dataset Summary")
            ai_summary, insights = generate_ai_summary(client, df_clean, sector)
            st.markdown(
                f"""
                <div style="background-color:#F8FAFC;padding:1rem;border-radius:10px;border:1px solid #E5E7EB;">
                <p style="color:#1E293B;">{ai_summary}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # --- Chat Module ---
            st.divider()
            st.markdown("### 💬 Basic Dataset Chat")
            launch_basic_chat(client, ai_summary, insights, df_for_ai, sector)

            # --- Next Steps ---
            st.divider()
            st.markdown("""
            ### 🚀 Next Steps
            <div style="background-color:#EFF6FF;padding:1rem;border-radius:10px;">
            For deeper <b>AI-driven analytics</b>, <b>predictive modeling</b>, or <b>automated dashboards</b>,  
            please <a href="#" target="_blank"><b>contact or hire me</b></a> to unlock advanced modules.
            </div>
            """, unsafe_allow_html=True)

    else:
        st.stop()
else:
    st.info("👆 Upload a dataset to begin your analysis.")


# ============================================================
# 🦶 Footer
# ============================================================
st.divider()
st.markdown(
    """
    <p style="text-align:center;color:gray;">
        Built with ❤️ using <b>Streamlit</b>, <b>Pandas</b>, and <b>Llama 3.1</b><br>
        <a href="https://github.com/yourusername" target="_blank">GitHub</a> |
        <a href="https://www.linkedin.com/in/yourprofile" target="_blank">LinkedIn</a>
    </p>
    """,
    unsafe_allow_html=True,
)
