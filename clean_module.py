import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def auto_data_clean(df):
    st.header("🧹 Data Cleaning & Quality Check")

    # --- Normalize column names ---
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9_]", "_", regex=True)
        .str.replace(r"__+", "_", regex=True)
        .str.strip("_")
    )
    df_before = df.copy()

    tabs = st.tabs(["📋 Data Preview", "🔥 Missing Values", "📊 Outlier Comparison", "🔁 Duplicates", "✅ Summary"])

    # --- 1️⃣ Data Preview ---
    with tabs[0]:
        st.write("### 📋 Data Preview (Before Cleaning)")
        st.dataframe(df_before.head())

    # --- 2️⃣ Missing Values ---
    with tabs[1]:
        st.write("### 🔍 Missing Value Overview")
        missing_counts = df.isna().sum()
        if missing_counts.sum() == 0:
            st.success("🎉 No missing values detected.")
        else:
            st.dataframe(missing_counts[missing_counts > 0])
            fig, ax = plt.subplots(figsize=(8, 3))
            sns.heatmap(df.isna(), cbar=False, cmap="Reds", yticklabels=False)
            st.pyplot(fig)

            num_cols = df.select_dtypes(include=np.number).columns
            for col in num_cols:
                if df[col].isna().any():
                    fill_value = df[col].median() if df[col].skew() > 1 else df[col].mean()
                    df[col].fillna(fill_value, inplace=True)
            cat_cols = df.select_dtypes(include="object").columns
            for col in cat_cols:
                if df[col].isna().any():
                    df[col].fillna(df[col].mode()[0], inplace=True)

            st.success("✅ Missing values filled using mean/median/mode strategy.")

    # --- 3️⃣ Outlier Comparison ---
    with tabs[2]:
        st.write("### 📈 Outlier Distribution (Before vs After)")
        num_cols = df.select_dtypes(include=np.number).columns

        if len(num_cols) == 0:
            st.info("No numeric columns found for outlier handling.")
        else:
            for col in num_cols:
                lower, upper = df[col].quantile(0.01), df[col].quantile(0.99)
                df[col] = np.clip(df[col], lower, upper)

            st.success("✅ Outliers capped between 1st–99th percentiles.")

            # Safe selectbox that auto-falls back if column missing
            available_cols = [col for col in num_cols if col in df_before.columns and col in df.columns]
            if available_cols:
                selected_col = st.selectbox("Select a numerical column:", available_cols, key="outlier_col")
                if selected_col in available_cols:
                    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
                    sns.boxplot(y=df_before[selected_col], ax=ax[0], color="salmon")
                    ax[0].set_title(f"Before Cleaning ({selected_col})")
                    sns.boxplot(y=df[selected_col], ax=ax[1], color="lightgreen")
                    ax[1].set_title(f"After Cleaning ({selected_col})")
                    st.pyplot(fig)
                else:
                    st.warning("⚠️ Selected column not found after cleaning.")
            else:
                st.info("No numeric columns available for visualization.")

    # --- 4️⃣ Duplicate Check ---
    with tabs[3]:
        st.write("### 🔁 Duplicate Check")
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            st.warning(f"⚠️ Found {duplicate_count} duplicate rows.")
            st.bar_chart({"Duplicates": [duplicate_count], "Unique": [len(df) - duplicate_count]})
            df.drop_duplicates(inplace=True)
            st.success("✅ Duplicate rows removed.")
        else:
            st.success("No duplicates found.")

    # --- 5️⃣ Summary ---
    with tabs[4]:
        st.write("### ✅ Data Cleaning Summary")

        # Replace infinities and remaining NaNs
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        # Before vs After Statistics Comparison
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) > 0:
            before_stats = df_before[num_cols].describe().T[["mean", "min", "max", "std"]]
            after_stats = df[num_cols].describe().T[["mean", "min", "max", "std"]]
            comparison = before_stats.join(after_stats, lsuffix="_before", rsuffix="_after")
            st.markdown("#### 📊 Numeric Summary (Before vs After)")
            st.dataframe(comparison.round(2))
        else:
            st.info("No numeric columns to summarize.")

        st.markdown(f"**Final Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.success("🎯 Dataset cleaned and summarized successfully!")

    return df
