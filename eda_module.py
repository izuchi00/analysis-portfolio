# ============================================================
# 📊 UNIVERSAL FEATURE DISTRIBUTION & EDA (Streamlit Version - Responsive & Compact)
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def run_eda(df):
    """
    Responsive, Streamlit-safe EDA with chart size control.
    Returns encoded DataFrame ready for AI processing.
    """
    if df is None or df.empty:
        st.warning("⚠️ No dataset provided for EDA.")
        return None

    # --- Normalize column names ---
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9_]", "_", regex=True)
        .str.replace(r"__+", "_", regex=True)
        .str.strip("_")
    )

    # --- Detect column types ---
    cat_cols = [c for c in df.columns if df[c].dtype == 'object' or df[c].nunique() < 20]
    num_cols = [c for c in df.columns if df[c].dtype in ['int64', 'float64'] and c not in cat_cols]

    st.markdown("### 🧾 Feature Overview")
    st.write(f"**Categorical columns:** {cat_cols if cat_cols else 'None detected'}")
    st.write(f"**Numerical columns:** {num_cols if num_cols else 'None detected'}")

    # --- Auto-detect encoded categoricals ---
    auto_cats = [
        c for c in df.columns
        if (df[c].dtype in ['int64', 'float64']) and (2 <= df[c].nunique() <= 10)
    ]
    for c in auto_cats:
        if c not in cat_cols:
            cat_cols.append(c)

    # --- Create encoded copy for AI ---
    df_encoded = df.copy()
    for col in cat_cols:
        df_encoded[col] = df_encoded[col].astype('category').cat.codes

    # --- Chart size control ---
    st.markdown("### 🪄 Visualization Settings")
    size_choice = st.radio(
        "Chart size:",
        ["Tiny", "Small", "Medium", "Large"],
        index=1,
        horizontal=True
    )
    size_map = {
        "Tiny": (3, 2),
        "Small": (4, 2.5),
        "Medium": (6, 3.5),
        "Large": (8, 5)
    }
    fig_size = size_map[size_choice]

    # --- Section: Feature Distributions ---
    st.markdown("### 📊 Feature Distributions")

    cols = st.columns(2)  # side-by-side layout

    # --- Numerical distributions ---
    if num_cols:
        num_var = df[num_cols].var().sort_values(ascending=False)
        selected_num = num_var.index[:4].tolist()
        for i, col in enumerate(selected_num):
            fig, ax = plt.subplots(figsize=fig_size)
            sns.histplot(df[col], kde=True, bins=20, color='cornflowerblue', ax=ax)
            ax.set_title(f"Distribution of {col}")
            ax.tick_params(labelsize=8)
            ax.set_xlabel(col, fontsize=9)
            with cols[i % 2]:
                st.pyplot(fig, use_container_width=False)

    # --- Categorical distributions ---
    skip_cats = {"cluster", "segment", "id", "index", "target"}
    selected_cat = [c for c in cat_cols if c.lower() not in skip_cats][:4]
    for i, col in enumerate(selected_cat):
        unique_vals = df[col].nunique()
        fig, ax = plt.subplots(figsize=fig_size)
        if unique_vals > 30:
            top_values = df[col].value_counts().nlargest(15)
            sns.countplot(
                y=col,
                data=df[df[col].isin(top_values.index)],
                order=top_values.index,
                palette="pastel",
                ax=ax
            )
            ax.set_title(f"Top 15 Categories of {col}")
        else:
            sns.countplot(
                x=col,
                data=df,
                order=df[col].value_counts().index,
                palette="pastel",
                ax=ax
            )
            ax.set_title(f"Distribution of {col}")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=15, fontsize=8)
        with cols[i % 2]:
            st.pyplot(fig, use_container_width=False)

    # --- Correlation Heatmap ---
    if len(num_cols) >= 2:
        st.markdown("### 🔥 Correlation Heatmap (Numerical Features)")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", square=True, ax=ax)
        st.pyplot(fig, use_container_width=False)

    st.success("✅ EDA Complete — Encoded DataFrame ready for AI summary and chat.")
    return df_encoded
