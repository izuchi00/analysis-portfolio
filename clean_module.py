import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

@st.cache_data
def auto_data_clean(df):
    st.header("🧹 Data Cleaning & Quality Check")

    # -------------------------------
    # Step 1️⃣ Normalize Column Names
    # -------------------------------
    st.subheader("📛 Step 1: Column Normalization")
    st.caption("Ensures consistent and clean column naming for easy analysis.")
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9_]", "_", regex=True)
        .str.replace(r"__+", "_", regex=True)
        .str.strip("_")
    )
    st.success("✅ Column names standardized successfully.")

    df_before = df.copy()

    # Tabs for modular navigation
    tabs = st.tabs(["📋 Data Preview", "🩺 Missing Values", "🔁 Duplicates", "📈 Outlier Comparison", "📊 Summary"])

    # -------------------------------
    # Step 2️⃣ Data Preview
    # -------------------------------
    with tabs[0]:
        st.markdown("### 👀 Data Snapshot")
        st.caption("Displays the first few rows for a quick look at your dataset.")
        st.dataframe(df_before.head())
        st.info(f"Dataset shape: **{df.shape[0]} rows × {df.shape[1]} columns**")

    # -------------------------------
    # Step 3️⃣ Missing Values
    # -------------------------------
    with tabs[1]:
        st.markdown("### 🩺 Missing Values")
        st.caption("Missing data can bias analysis. We fill numeric values using mean/median and categorical with mode.")

        missing_summary = df.isna().sum()
        missing_summary = missing_summary[missing_summary > 0]

        if len(missing_summary) == 0:
            st.success("🎉 No missing values detected.")
        else:
            # Prepare table of missing value info
            fill_data = []
            for col, count in missing_summary.items():
                if df[col].dtype in [np.float64, np.int64]:
                    fill_type = "Median" if abs(df[col].skew()) > 1 else "Mean"
                    fill_value = df[col].median() if fill_type == "Median" else df[col].mean()
                    df[col].fillna(fill_value, inplace=True)
                else:
                    fill_type = "Mode"
                    fill_value = df[col].mode()[0]
                    df[col].fillna(fill_value, inplace=True)
                fill_data.append((col, int(count), fill_type, round(fill_value, 2) if isinstance(fill_value, (int, float)) else str(fill_value)))

            summary_df = pd.DataFrame(fill_data, columns=["Column", "Missing Count", "Fill Method", "Fill Value"])
            st.dataframe(summary_df)

            st.success(f"✅ Filled missing values in {len(summary_df)} columns using mean/median/mode strategy.")

    # -------------------------------
    # Step 4️⃣ Duplicates
    # -------------------------------
    with tabs[2]:
        st.markdown("### 🔁 Duplicate Detection & Removal")
        st.caption("Duplicate rows can distort results. We identify and remove them safely.")

        dup_count = df.duplicated().sum()
        if dup_count == 0:
            st.success("🎉 No duplicate rows detected.")
        else:
            st.warning(f"⚠️ Found **{dup_count}** duplicate rows.")
            dup_preview = df[df.duplicated()].head(10)
            st.dataframe(dup_preview)
            df.drop_duplicates(inplace=True)
            st.success(f"✅ Removed all duplicate rows. New shape: **{df.shape[0]} rows × {df.shape[1]} columns**")

    # -------------------------------
    # Step 5️⃣ Outliers
    # -------------------------------
    with tabs[3]:
        st.markdown("### 📈 Outlier Comparison (Before vs After)")
        st.caption("Outliers can distort your analysis. Here we cap values between 1st–99th percentiles for numeric columns.")

        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) == 0:
            st.info("No numeric columns found.")
        else:
            for col in num_cols:
                lower, upper = df[col].quantile(0.01), df[col].quantile(0.99)
                df[col] = np.clip(df[col], lower, upper)

            st.success("✅ Outliers capped successfully.")

            # Adaptive, sampled visualization
            sample_df_before = df_before.sample(min(len(df_before), 5000), random_state=42)
            sample_df_after = df.sample(min(len(df), 5000), random_state=42)

            selected_col = st.selectbox("Select a numerical column:", num_cols, key="boxplot_select")

            if selected_col in sample_df_before.columns:
                fig, ax = plt.subplots(1, 2, figsize=(8, 3))
                sns.boxplot(y=sample_df_before[selected_col], ax=ax[0], color="#f28b82")
                ax[0].set_title(f"Before Cleaning ({selected_col})")
                sns.boxplot(y=sample_df_after[selected_col], ax=ax[1], color="#81c995")
                ax[1].set_title(f"After Cleaning ({selected_col})")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("⚠️ Selected column not found.")

    # -------------------------------
    # Step 6️⃣ Summary (Before vs After)
    # -------------------------------
    with tabs[4]:
        st.markdown("### 📊 Cleaning Summary")
        st.caption("Below are key summary statistics before and after cleaning, showing how the data improved.")

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) > 0:
            st.write("#### 📉 Before Cleaning")
            st.dataframe(df_before[num_cols].describe().T.round(2))
            st.write("#### 📈 After Cleaning")
            st.dataframe(df[num_cols].describe().T.round(2))
        else:
            st.info("No numeric data available for summary statistics.")

        st.success(f"🎯 Dataset cleaned successfully! Final shape: **{df.shape[0]} rows × {df.shape[1]} columns**")

    return df
