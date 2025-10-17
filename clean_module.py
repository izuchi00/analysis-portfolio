import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def clean_data_with_visuals(df):
    if df is None or df.empty:
        st.warning("⚠️ No dataset loaded for cleaning.")
        return None

    df_before = df.copy()

    st.markdown("## 🧹 Data Cleaning Overview")

    # --- User Control for Cleaning Strategy ---
    with st.expander("⚙️ Cleaning Settings", expanded=True):
        st.markdown("**Choose strategies for handling missing values and outliers:**")
        numeric_strategy = st.selectbox(
            "Numeric Missing Value Strategy",
            ["Mean", "Median", "Mode"],
            index=1,
            help="How to fill missing numeric values.",
        )
        clip_outliers = st.checkbox("Clip Outliers (1st–99th percentile)", True)
        remove_dupes = st.checkbox("Remove Duplicate Rows", True)

    cleaning_steps = []

    # --- Normalize column names ---
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9_]", "_", regex=True)
        .str.replace(r"__+", "_", regex=True)
        .str.strip("_")
    )
    cleaning_steps.append("Normalized column names for consistent formatting.")

    # --- Handle Missing Values ---
    missing_before = df.isna().sum().sum()
    for col in df.columns:
        if df[col].dtype in ["float64", "int64"]:
            if numeric_strategy == "Mean":
                df[col] = df[col].fillna(df[col].mean())
            elif numeric_strategy == "Median":
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
        else:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
    cleaning_steps.append(
        f"Filled {missing_before} missing values — numeric via *{numeric_strategy.lower()}*, categorical via *mode*."
    )

    # --- Handle Duplicates ---
    dup_count = df.duplicated().sum()
    if remove_dupes and dup_count > 0:
        df = df.drop_duplicates()
        cleaning_steps.append(f"Removed {dup_count} duplicate rows.")
    else:
        cleaning_steps.append("Checked for duplicates — none removed.")

    # --- Handle Outliers ---
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if clip_outliers and num_cols:
        for col in num_cols:
            lower, upper = df[col].quantile([0.01, 0.99])
            df[col] = np.clip(df[col], lower, upper)
        cleaning_steps.append("Capped outliers between 1st–99th percentiles for numeric columns.")
    else:
        cleaning_steps.append("Outlier capping skipped.")

    df_after = df.copy()

    # ============================================================
    # 📊 TABS FOR VISUALIZATION
    # ============================================================
    tabs = st.tabs([
        "📋 Data Preview",
        "🔥 Missing Values",
        "📈 Outlier Comparison",
        "🔁 Duplicates",
        "✅ Summary",
    ])

    # --- 1️⃣ Data Preview ---
    with tabs[0]:
        st.markdown("### 🧩 Before Cleaning")
        st.dataframe(df_before.head(10), use_container_width=True)
        st.markdown("### ✨ After Cleaning")
        st.dataframe(df_after.head(10), use_container_width=True)

    # --- 2️⃣ Missing Values ---
    with tabs[1]:
        st.markdown("### 🔥 Missing Values (Before vs After)")
        fig, ax = plt.subplots(1, 2, figsize=(10, 3))
        sns.heatmap(df_before.isnull(), cbar=False, ax=ax[0], cmap="Reds")
        ax[0].set_title("Before Cleaning")
        sns.heatmap(df_after.isnull(), cbar=False, ax=ax[1], cmap="Greens")
        ax[1].set_title("After Cleaning")
        st.pyplot(fig)

    # --- 3️⃣ Outlier Comparison ---
    with tabs[2]:
        st.markdown("### 📈 Outlier Distribution (Before vs After)")
        if num_cols:
            selected_col = st.selectbox("Select a numerical column:", num_cols)
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            sns.boxplot(y=df_before[selected_col], ax=ax[0], color="salmon")
            ax[0].set_title(f"Before Cleaning ({selected_col})")
            sns.boxplot(y=df_after[selected_col], ax=ax[1], color="lightgreen")
            ax[1].set_title(f"After Cleaning ({selected_col})")
            st.pyplot(fig)
        else:
            st.info("No numeric columns found for outlier visualization.")

    # --- 4️⃣ Duplicates ---
    with tabs[3]:
        st.markdown("### 🔁 Duplicate Rows Check")
        st.metric("Duplicate Rows (Before)", dup_count)
        st.metric("Duplicate Rows (After)", df_after.duplicated().sum())
        if dup_count > 0:
            st.markdown("**Sample Duplicates (Before Removal):**")
            st.dataframe(df_before[df_before.duplicated()].head(), use_container_width=True)
        else:
            st.info("No duplicate rows detected in the dataset.")

    # --- 5️⃣ Summary ---
    with tabs[4]:
        st.markdown("### 🧾 Cleaning Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows (Before)", len(df_before))
            st.metric("Missing Values (Before)", df_before.isna().sum().sum())
        with col2:
            st.metric("Rows (After)", len(df_after))
            st.metric("Missing Values (After)", df_after.isna().sum().sum())
        st.markdown("### 🧠 Steps Performed")
        for step in cleaning_steps:
            st.markdown(f"- ✅ {step}")
        st.success("✅ Dataset cleaned and summarized successfully!")

    return df_after
