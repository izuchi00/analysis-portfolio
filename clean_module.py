# ============================================================
# 🧹 CLEAN MODULE — Enhanced UI for Edis Analytics
# ============================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st


# ============================================================
# 🧩 Core Cleaning Logic (cacheable)
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

    def make_unique_columns(cols):
        seen, unique_cols = {}, []
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
                parsed = pd.to_datetime(df[col], errors="raise")
                valid_ratio = parsed.notna().mean()
                if valid_ratio > 0.7:
                    df[col] = parsed
            except Exception:
                continue

    # --- Fill missing values ---
    for col in df.columns:
        try:
            if df[col].isna().any():
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

    # --- Cap outliers safely ---
    for col in df.select_dtypes(include=np.number).columns:
        try:
            lower, upper = df[col].quantile(0.01), df[col].quantile(0.99)
            df[col] = np.clip(df[col], lower, upper)
        except Exception:
            pass

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    df.drop_duplicates(inplace=True)

    return df


# ============================================================
# 🎛️ Streamlit Cleaning UI (Enhanced Layout)
# ============================================================
def auto_data_clean(df):
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='section-subheader'>🧹 Data Cleaning & Quality Check</h3>", unsafe_allow_html=True)

    # --- Track original vs cleaned names ---
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

    # --- Clean Data ---
    df_before = df.copy()
    df.columns = cleaned_names
    df_clean = clean_data_core(df)

    # --- Summary metrics ---
    st.markdown("<div class='metrics-container'>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows (Before)", df_before.shape[0])
    c2.metric("Rows (After)", df_clean.shape[0])
    c3.metric("Columns", df_clean.shape[1])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # --- Tabbed interface ---
    tabs = st.tabs([
        "🩺 Missing Values",
        "🔁 Duplicates",
        "📈 Outliers",
        "📊 Summary"
    ])

    # 🩺 Missing Values
    with tabs[0]:
        st.markdown("<div class='tab-card'>", unsafe_allow_html=True)
        st.subheader("🩺 Missing Values Overview")
        st.caption("Numeric → Mean/Median | Categorical → Mode | Dates → Forward Fill")
        missing_summary = []

        for col in df_before.columns:
            missing_count = df_before[col].isna().sum()
            if missing_count.sum() > 0:
                if pd.api.types.is_datetime64_any_dtype(df_before[col]):
                    fill_method, fill_value = "Forward Fill", "Previous Date"
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
            st.dataframe(pd.DataFrame(missing_summary), use_container_width=True)
            st.success(f"✅ Filled missing values in {len(missing_summary)} column(s).")
        else:
            st.success("✅ No missing values detected.")
        st.markdown("</div>", unsafe_allow_html=True)

    # 🔁 Duplicates
    with tabs[1]:
        st.markdown("<div class='tab-card'>", unsafe_allow_html=True)
        st.subheader("🔁 Duplicate Rows Check")
        dup_count = df_before.duplicated().sum()
        if dup_count > 0:
            st.warning(f"⚠️ Found and removed {dup_count} duplicate rows.")
        else:
            st.success("✅ No duplicate rows found.")
        st.markdown("</div>", unsafe_allow_html=True)

    # 📈 Outliers
    with tabs[2]:
        st.markdown("<div class='tab-card'>", unsafe_allow_html=True)
        st.subheader("📈 Outlier Comparison (Before vs After)")
        st.caption("Outliers capped between 1st and 99th percentile for numeric columns.")

        num_cols = df_clean.select_dtypes(include=np.number).columns
        if len(num_cols) == 0:
            st.info("No numeric columns found.")
        else:
            selected_col = st.selectbox("Select numeric column:", num_cols)

            # ✅ Match original column name flexibly
            before_col = next(
                (col for col in df_before.columns if col.lower().replace(" ", "_") == selected_col.lower()),
                None
            )

            try:
                if before_col and before_col in df_before.columns and selected_col in df_clean.columns:
                    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
                    sns.boxplot(y=df_before[before_col], ax=ax[0], color="salmon")
                    sns.boxplot(y=df_clean[selected_col], ax=ax[1], color="lightgreen")
                    ax[0].set_title("Before Cleaning")
                    ax[1].set_title("After Cleaning")
                    plt.subplots_adjust(wspace=0.6)
                    st.pyplot(fig, use_container_width=False)
                else:
                    st.warning(f"⚠️ Could not display outlier comparison: '{selected_col}' not found in original data.")
            except Exception as e:
                st.warning(f"⚠️ Could not display outlier comparison: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # 📊 Summary
    with tabs[3]:
        st.markdown("<div class='tab-card'>", unsafe_allow_html=True)
        st.subheader("📊 Summary Statistics")
        num_cols = df_clean.select_dtypes(include=np.number).columns

        if len(num_cols) > 0:
            st.write("#### Before Cleaning")

            # ✅ Match numeric columns across both dataframes
            before_cols = [
                col for col in df_before.columns
                if col.lower().strip().replace(" ", "_") in [c.lower().strip() for c in num_cols]
            ]

            if before_cols:
                st.dataframe(df_before[before_cols].describe().T.round(2), use_container_width=True)
            else:
                st.warning("⚠️ No matching numeric columns found before cleaning.")

            st.write("#### After Cleaning")
            st.dataframe(df_clean[num_cols].describe().T.round(2), use_container_width=True)
        else:
            st.info("No numeric columns available.")
        st.markdown("</div>", unsafe_allow_html=True)

    # 👁️ Final Preview
    st.markdown("<h4 class='section-subheader'>👁️ Sample of Cleaned Data</h4>", unsafe_allow_html=True)
    st.dataframe(df_clean.head(), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    return df_clean
