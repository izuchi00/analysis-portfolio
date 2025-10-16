# ============================================================
# 🧹 UNIVERSAL DATA CLEANING + STRUCTURED SUMMARY (Streamlit Safe)
# ============================================================

import numpy as np
import pandas as pd
import warnings
import streamlit as st

def auto_data_clean(df, handle_outliers=True, cap_instead_of_drop=True):
    """
    Automatically clean and summarize any pandas DataFrame:
    - Normalizes column names
    - Handles missing values (reports counts and strategy)
    - Removes duplicates
    - Handles outliers (cap or drop)
    - Returns cleaned dataframe
    """
    if df is None or df.empty:
        st.warning("⚠️ No dataset found to clean.")
        return df

    st.markdown("### 🧹 Data Cleaning Log")

    df_clean = df.copy()
    summary_log = []

    # --- Normalize column names ---
    df_clean.columns = (
        df_clean.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9_]", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    summary_log.append("🪄 Normalized column names for consistent formatting.")

    # --- Handle missing values ---
    missing_report = []
    total_missing_before = df_clean.isna().sum().sum()

    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns

    for col in numeric_cols:
        n_missing = df_clean[col].isna().sum()
        if n_missing > 0:
            median_value = df_clean[col].median()
            df_clean.fillna({col: median_value}, inplace=True)
            missing_report.append(f"🧮 Filled {n_missing} missing numeric values in '{col}' with median = {median_value:.3f}")

    for col in categorical_cols:
        n_missing = df_clean[col].isna().sum()
        if n_missing > 0:
            mode_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Unknown"
            df_clean.fillna({col: mode_value}, inplace=True)
            missing_report.append(f"🔤 Filled {n_missing} missing categorical values in '{col}' with mode = '{mode_value}'")

    total_missing_after = df_clean.isna().sum().sum()
    filled_total = int(total_missing_before - total_missing_after)
    if filled_total > 0:
        summary_log.append(f"🧩 Filled {filled_total} total missing values using median/mode strategy.")
        for line in missing_report:
            summary_log.append(" • " + line)
    else:
        summary_log.append("✅ No missing values detected.")

    # --- Remove duplicates ---
    before = len(df_clean)
    df_clean.drop_duplicates(inplace=True)
    removed = before - len(df_clean)
    summary_log.append(f"🧹 Removed {removed} duplicate rows." if removed > 0 else "✅ No duplicate rows found.")

    # --- Handle outliers ---
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    if handle_outliers and len(numeric_cols) > 0:
        if cap_instead_of_drop:
            for col in numeric_cols:
                lower = df_clean[col].quantile(0.01)
                upper = df_clean[col].quantile(0.99)
                df_clean[col] = df_clean[col].clip(lower, upper)
            summary_log.append("🔧 Capped outliers between 1st–99th percentiles for numeric columns.")
        else:
            before = len(df_clean)
            for col in numeric_cols:
                Q1, Q3 = df_clean[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
            summary_log.append(f"📉 Removed {before - len(df_clean)} rows with outliers using IQR method.")

    # --- Replace infinite values ---
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_clean.fillna(0, inplace=True)
    summary_log.append("🧼 Replaced infinite values and filled residual NaNs with 0.")

    # --- Display cleaning summary ---
    with st.expander("🧾 View Cleaning Summary Log", expanded=False):
        for step in summary_log:
            st.markdown(f"- {step}")

    # --- Display sample after cleaning ---
    st.success("✅ Dataset cleaned and summarized successfully!")
    st.dataframe(df_clean.head())

    return df_clean
