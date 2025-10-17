import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def auto_data_clean(df):
    st.header("🧹 Data Cleaning & Quality Check")

    # --- 1️⃣ Column normalization ---
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9_]", "_", regex=True)
        .str.replace(r"__+", "_", regex=True)
        .str.strip("_")
    )
    st.success("✅ Normalized column names for consistent formatting.")

    df_before = df.copy()

    # --- 2️⃣ Missing Values Overview ---
    st.markdown("### 🩺 Missing Values Overview")

    missing_counts = df.isna().sum()
    if missing_counts.sum() == 0:
        st.info("🎉 No missing values detected.")
    else:
        st.dataframe(missing_counts[missing_counts > 0])

        # Heatmap visualization
        st.write("📊 Missing Value Heatmap:")
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.heatmap(df.isna(), cbar=False, cmap="Reds", yticklabels=False)
        st.pyplot(fig)

        # Fill numeric columns with median or mean
        num_cols = df.select_dtypes(include=np.number).columns
        for col in num_cols:
            if df[col].isna().any():
                fill_value = df[col].median() if df[col].skew() > 1 else df[col].mean()
                df[col].fillna(fill_value, inplace=True)

        # Fill categorical columns with mode
        cat_cols = df.select_dtypes(include="object").columns
        for col in cat_cols:
            if df[col].isna().any():
                df[col].fillna(df[col].mode()[0], inplace=True)

        st.success("✅ Filled missing values using mean/median/mode strategy.")

    # --- 3️⃣ Duplicate Check ---
    st.markdown("### 🔁 Duplicate Check")
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        st.warning(f"⚠️ Found {duplicate_count} duplicate rows.")
        st.bar_chart({"Duplicates": [duplicate_count], "Unique": [len(df) - duplicate_count]})
        df = df.drop_duplicates()
        st.success("✅ Removed duplicate rows.")
    else:
        st.info("No duplicate rows found.")

    # --- 4️⃣ Outlier Detection & Capping ---
    st.markdown("### 📉 Outlier Handling (Capping 1st–99th Percentile)")

    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) == 0:
        st.info("No numeric columns found for outlier handling.")
    else:
        for col in num_cols:
            lower, upper = df[col].quantile(0.01), df[col].quantile(0.99)
            df[col] = np.clip(df[col], lower, upper)

        st.success("✅ Outliers capped for numeric columns.")

        # --- Before vs After Boxplot Comparison ---
        st.markdown("#### 🎯 Outlier Distribution (Before vs After)")
        selected_col = st.selectbox("Select a numeric column for visualization:", num_cols)
        if selected_col not in df_before.columns or selected_col not in df.columns:
            st.warning(f"⚠️ Column '{selected_col}' not found in dataset.")
        else:
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            sns.boxplot(y=df_before[selected_col], ax=ax[0], color="salmon")
            ax[0].set_title(f"Before Cleaning ({selected_col})")
            sns.boxplot(y=df[selected_col], ax=ax[1], color="lightgreen")
            ax[1].set_title(f"After Cleaning ({selected_col})")
            st.pyplot(fig)

    # --- 5️⃣ Infinite values & final cleanup ---
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # --- 6️⃣ Final Summary ---
    st.markdown("### ✅ Data Cleaning Summary")
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.write("**Numeric Columns:**", list(num_cols))
    st.write("**Categorical Columns:**", list(df.select_dtypes(include='object').columns))
    st.success("🎯 Dataset cleaned and summarized successfully!")

    return df
