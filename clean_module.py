import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# ============================================================
# 🧹 Core Cleaning Logic (cacheable)
# ============================================================
@st.cache_data
def clean_data_core(df):
    df = df.copy()

    # --- Normalize and deduplicate column names ---
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9_]", "_", regex=True)
        .str.replace(r"__+", "_", regex=True)
        .str.strip("_")
    )

    # ✅ Ensure unique column names
    def make_unique_columns(cols):
        seen = {}
        unique_cols = []
        for col in cols:
            if col not in seen:
                seen[col] = 0
                unique_cols.append(col)
            else:
                seen[col] += 1
                unique_cols.append(f"{col}_{seen[col]}")
        return unique_cols

    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        st.warning(f"⚠️ Duplicate column names detected: {duplicates}. Renaming automatically.")
        df.columns = make_unique_columns(df.columns)

    # --- Detect and preserve datetime columns ---
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                parsed = pd.to_datetime(df[col], errors="raise", infer_datetime_format=True)
                valid_ratio = parsed.notna().mean()
                if valid_ratio > 0.7:  # convert if mostly valid dates
                    df[col] = parsed
            except Exception:
                continue

    # --- Fill missing values ---
    for col in df.columns:
        try:
            if df[col].isna().any():
                # ⏰ Skip datetime columns
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col].fillna(method="ffill", inplace=True)
                    continue

                if df[col].dtype in [np.float64, np.int64]:
                    fill_value = df[col].median() if abs(df[col].skew()) > 1 else df[col].mean()
                    df[col].fillna(fill_value, inplace=True)
                else:
                    mode_val = df[col].mode()
                    if not mode_val.empty:
                        df[col].fillna(mode_val[0], inplace=True)
        except Exception:
            continue

    # --- Cap outliers safely (numeric columns only) ---
    for col in df.select_dtypes(include=np.number).columns:
        try:
            lower, upper = df[col].quantile(0.01), df[col].quantile(0.99)
            df[col] = np.clip(df[col], lower, upper)
        except Exception as e:
            st.warning(f"⚠️ Skipped outlier capping for '{col}' ({e})")

    # --- Final cleanup ---
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    df.drop_duplicates(inplace=True)

    return df


# ============================================================
# 🎛️ Streamlit Cleaning UI
# ============================================================
def auto_data_clean(df):
    st.header("🧹 Data Cleaning & Quality Check")

    # --- Track name normalization ---
    original_names = df.columns.tolist()
    cleaned_names = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9_]", "_", regex=True)
        .str.replace(r"__+", "_", regex=True)
        .str.strip("_")
    )
    name_map = dict(zip(original_names, cleaned_names))

    # --- Clean data ---
    df_before = df.copy()
    df.columns = cleaned_names
    df_clean = clean_data_core(df)

    # --- Overview metrics ---
    st.markdown("### 📊 Dataset Overview (Before vs After)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows (Before)", df_before.shape[0])
    c2.metric("Rows (After)", df_clean.shape[0])
    c3.metric("Columns", df_clean.shape[1])

    # --- Tabs for detail views ---
    tabs = st.tabs(["🩺 Missing Values", "🔁 Duplicates", "📈 Outliers", "📊 Summary"])

    # 🩺 Missing Values
    with tabs[0]:
        st.subheader("🩺 Missing Values Overview")
        st.caption("Numeric values filled with Mean/Median • Categorical with Mode • Datetime forward-filled.")
        missing_summary = []

        for col in df_before.columns:
            missing_count = df_before[col].isna().sum()
            if missing_count > 0:
                if pd.api.types.is_datetime64_any_dtype(df_before[col]):
                    fill_method = "Forward Fill"
                    fill_value = "Previous Date"
                elif df_before[col].dtype in [np.float64, np.int64]:
                    fill_method = "Median" if abs(df_before[col].skew()) > 1 else "Mean"
                    fill_value = df_before[col].median() if fill_method == "Median" else df_before[col].mean()
                else:
                    fill_method = "Mode"
                    mode_val = df_before[col].mode()
                    fill_value = mode_val[0] if not mode_val.empty else None

                missing_summary.append({
                    "Column": col,
                    "Missing Count": missing_count,
                    "Fill Method": fill_method,
                    "Fill Value": fill_value
                })

        if missing_summary:
            st.dataframe(pd.DataFrame(missing_summary))
            st.success(f"✅ Filled missing values in {len(missing_summary)} column(s).")
        else:
            st.success("✅ No missing values detected.")

    # 🔁 Duplicates
    with tabs[1]:
        st.subheader("🔁 Duplicate Check")
        dup_count = df_before.duplicated().sum()
        if dup_count > 0:
            st.warning(f"⚠️ Found and removed {dup_count} duplicate rows.")
        else:
            st.success("✅ No duplicate rows found.")

    # 📈 Outliers
    with tabs[2]:
        st.subheader("📈 Outlier Comparison (Before vs After)")
        st.caption("Outliers capped between 1st and 99th percentile for numeric columns.")

        num_cols = df_clean.select_dtypes(include=np.number).columns

        if len(num_cols) == 0:
            st.info("No numeric columns found.")
        else:
            display_to_cleaned = {orig: name_map[orig] for orig in original_names if name_map[orig] in num_cols}
            selected_display = st.selectbox("Select numeric column:", list(display_to_cleaned.keys()))
            selected_cleaned = display_to_cleaned[selected_display]

            # --- Chart size control (same as EDA) ---
            st.markdown("### 🪄 Visualization Settings")
            size_choice = st.radio(
                "Chart size:",
                ["Tiny", "Small", "Medium", "Large"],
                index=2,
                horizontal=True
            )
            size_map = {
                "Tiny": (3, 2),
                "Small": (4, 2.5),
                "Medium": (6, 3.5),
                "Large": (8, 5)
            }
            fig_size = size_map[size_choice]

            # --- Sample for faster plotting ---
            sample_before = df_before.rename(columns=name_map).sample(min(len(df_before), 5000))
            sample_after = df_clean.sample(min(len(df_clean), 5000))

            # ✅ Validate numeric column
            if (
                selected_cleaned not in sample_before.columns
                or selected_cleaned not in sample_after.columns
            ):
                st.warning(f"⚠️ Column '{selected_cleaned}' not found in one of the datasets.")
            elif (
                sample_before[selected_cleaned].dropna().empty
                or sample_after[selected_cleaned].dropna().empty
            ):
                st.warning(f"⚠️ No valid numeric data available for '{selected_cleaned}'.")
            else:
                try:
                    # --- Plotting style same as EDA ---
                    fig, ax = plt.subplots(1, 2, figsize=fig_size)
                    sns.boxplot(y=sample_before[selected_cleaned], ax=ax[0], color="salmon")
                    sns.boxplot(y=sample_after[selected_cleaned], ax=ax[1], color="lightgreen")

                    ax[0].set_title("Before Cleaning", fontsize=10)
                    ax[1].set_title("After Cleaning", fontsize=10)
                    for a in ax:
                        a.tick_params(labelsize=8)
                        a.set_ylabel(selected_display, fontsize=9)
                    plt.subplots_adjust(wspace=0.8)  # Increase spacing; try 0.5 for more space    

                    st.pyplot(fig, use_container_width=False)

                except Exception as e:
                    st.warning(f"⚠️ Unable to plot boxplot for '{selected_cleaned}': {e}")

    # 📊 Summary
    with tabs[3]:
        st.subheader("📊 Summary Statistics")
        num_cols = df_clean.select_dtypes(include=np.number).columns

        if len(num_cols) > 0:
            st.write("#### Before Cleaning")
            safe_cols = [c for c in num_cols if c in df_before.rename(columns=name_map).columns]
            if safe_cols:
                st.dataframe(df_before.rename(columns=name_map)[safe_cols].describe().T.round(2))
            else:
                st.warning("No matching numeric columns found before cleaning.")

            st.write("#### After Cleaning")
            st.dataframe(df_clean[num_cols].describe().T.round(2))
        else:
            st.info("No numeric columns available.")

        st.success(f"🎯 Final shape: {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")

    # 👁️ Final Preview
    st.markdown("### 👁️ Sample of Cleaned Data")
    st.dataframe(df_clean.head(), use_container_width=True)

    return df_clean
