import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st


# --- Core cleaning logic (cacheable) ---
@st.cache_data
def clean_data_core(df):
    df = df.copy()

    # Normalize column names
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9_]", "_", regex=True)
        .str.replace(r"__+", "_", regex=True)
        .str.strip("_")
    )

    # Fill missing values
    for col in df.columns:
        if df[col].isna().any():
            if df[col].dtype in [np.float64, np.int64]:
                fill_value = df[col].median() if abs(df[col].skew()) > 1 else df[col].mean()
                df[col].fillna(fill_value, inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)

    # Cap outliers
    for col in df.select_dtypes(include=np.number).columns:
        lower, upper = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = np.clip(df[col], lower, upper)

    # Drop duplicates and handle infinities
    df.drop_duplicates(inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df


# --- UI Wrapper ---
def auto_data_clean(df):
    st.header("🧹 Data Cleaning & Quality Check")

    # --- Save mapping between original and cleaned names ---
    original_names = df.columns.tolist()
    cleaned_names = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9_]", "_", regex=True)
        .str.replace(r"__+", "_", regex=True)
        .str.strip("_")
    )
    name_map = dict(zip(original_names, cleaned_names))  # {original -> cleaned}

    # --- Clean data ---
    df_before = df.copy()
    df.columns = cleaned_names
    df_clean = clean_data_core(df)

    # --- Tabs ---
    tabs = st.tabs(["📋 Data Preview", "🩺 Missing Values", "🔁 Duplicates", "📈 Outliers", "📊 Summary"])

    # 1️⃣ Data preview
    with tabs[0]:
        st.subheader("📋 Data Preview")
        st.dataframe(df_before.head())
        st.info(f"Rows: {df_before.shape[0]}, Columns: {df_before.shape[1]}")

    # 2️⃣ Missing values
    with tabs[1]:
        st.subheader("🩺 Missing Values Overview")
        st.caption("Missing data can bias analysis. We fill numeric values using mean/median and categorical with mode.")

        missing_summary = []
        df_filled = df_before.copy()

        for col in df_before.columns:
         missing_count = df_before[col].isna().sum()
         if missing_count > 0:
            if df_before[col].dtype in [np.float64, np.int64]:
                # Use median for skewed numeric data, mean otherwise
                fill_method = "Median" if abs(df_before[col].skew()) > 1 else "Mean"
                fill_value = df_before[col].median() if fill_method == "Median" else df_before[col].mean()
            else:
                fill_method = "Mode"
                fill_value = df_before[col].mode()[0]

            df_filled[col].fillna(fill_value, inplace=True)
            missing_summary.append({
                "Column": col,
                "Missing Count": missing_count,
                "Fill Method": fill_method,
                "Fill Value": fill_value if not isinstance(fill_value, (np.ndarray, pd.Series)) else fill_value.item()
            })

        if missing_summary:
           summary_df = pd.DataFrame(missing_summary)
           st.dataframe(summary_df)
           st.success(f"✅ Filled missing values in {len(summary_df)} columns using mean/median/mode strategy.")
        else:
           st.success("✅ No missing values detected.")


    # 3️⃣ Duplicates
    with tabs[2]:
        st.subheader("🔁 Duplicates Check")
        dup_count = df_before.duplicated().sum()
        if dup_count > 0:
            st.warning(f"⚠️ Found {dup_count} duplicate rows. Removed during cleaning.")
        else:
            st.success("✅ No duplicates found.")

    # 4️⃣ Outliers
    with tabs[3]:
        st.subheader("📈 Outlier Comparison (Before vs After)")
        st.caption("Compare numeric columns before and after cleaning (1st–99th percentile capping).")

        num_cols = df_clean.select_dtypes(include=np.number).columns
        if len(num_cols) == 0:
            st.info("No numeric columns found.")
        else:
            # Use original display names but plot with cleaned names
            display_to_cleaned = {orig: name_map[orig] for orig in original_names if name_map[orig] in num_cols}
            selected_display = st.selectbox("Select numeric column:", list(display_to_cleaned.keys()))
            selected_cleaned = display_to_cleaned[selected_display]

            # Sample for faster plotting
            sample_before = df_before.rename(columns=name_map).sample(min(len(df_before), 5000))
            sample_after = df_clean.sample(min(len(df_clean), 5000))

            if selected_cleaned in sample_before.columns:
                fig, ax = plt.subplots(1, 2, figsize=(8, 3))
                sns.boxplot(y=sample_before[selected_cleaned], ax=ax[0], color="salmon")
                sns.boxplot(y=sample_after[selected_cleaned], ax=ax[1], color="lightgreen")
                ax[0].set_title("Before Cleaning")
                ax[1].set_title("After Cleaning")
                st.pyplot(fig)
            else:
                st.warning("⚠️ Column not found after renaming. Check name normalization.")

    # 5️⃣ Summary
    with tabs[4]:
        st.subheader("📊 Summary Statistics")
        num_cols = df_clean.select_dtypes(include=np.number).columns
        if len(num_cols) > 0:
            st.write("#### Before Cleaning")
            st.dataframe(df_before.rename(columns=name_map)[num_cols].describe().T.round(2))
            st.write("#### After Cleaning")
            st.dataframe(df_clean[num_cols].describe().T.round(2))
        else:
            st.info("No numeric columns available.")
        st.success(f"🎯 Final shape: {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")

    return df_clean
