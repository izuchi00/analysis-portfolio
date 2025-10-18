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

    # --- Section: Overview ---
    st.markdown("<h2 style='color:#2563EB;'>📊 Exploratory Data Analysis</h2>", unsafe_allow_html=True)
    st.caption("Automated EDA showing feature distributions, categorical summaries, and correlation heatmaps.")

    st.markdown("### 🧾 Feature Overview")
    with st.container():
        c1, c2 = st.columns(2)
        c1.info(f"**Categorical Columns** ({len(cat_cols)}): {cat_cols if cat_cols else 'None detected'}")
        c2.info(f"**Numerical Columns** ({len(num_cols)}): {num_cols if num_cols else 'None detected'}")

    st.divider()

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

    st.divider()
    st.markdown("### 📊 Feature Distributions")
    st.caption("Displays the most variable numerical and categorical features side by side for quick insights.")

    # --- Dual-column layout ---
    cols = st.columns(2)

    # --- Numerical distributions ---
    if num_cols:
        num_var = df[num_cols].var().sort_values(ascending=False)
        selected_num = num_var.index[:4].tolist()
        for i, col in enumerate(selected_num):
            fig, ax = plt.subplots(figsize=fig_size)
            sns.histplot(df[col], kde=True, bins=20, color='#3B82F6', ax=ax)
            ax.set_title(f"Distribution of {col}", fontsize=10, color="#1E3A8A")
            ax.tick_params(labelsize=8)
            ax.set_xlabel(col, fontsize=9)
            plt.tight_layout()
            with cols[i % 2]:
                st.pyplot(fig, use_container_width=False)

    # --- Categorical distributions ---
    skip_cats = {"cluster", "segment", "id", "index", "target"}
    selected_cat = [c for c in cat_cols if c.lower() not in skip_cats][:4]

    for i, col in enumerate(selected_cat):
        unique_vals = df[col].nunique()

        # 📏 Auto-adjust figure size for many categories
        dynamic_size = (
            (fig_size[0] + min(unique_vals / 10, 3), fig_size[1])
            if unique_vals > 10 else fig_size
        )

        fig, ax = plt.subplots(figsize=dynamic_size)

        # ✂️ Truncate long labels
        df_display = df.copy()
        if df_display[col].dtype == 'object':
            df_display[col] = df_display[col].astype(str).apply(
                lambda x: x if len(x) <= 15 else x[:12] + "..."
            )

        if unique_vals > 30:
            top_values = df_display[col].value_counts().nlargest(15)
            sns.countplot(
                y=col,
                data=df_display[df_display[col].isin(top_values.index)],
                order=top_values.index,
                palette="Blues_r",
                ax=ax
            )
            ax.set_title(f"Top 15 Categories of {col}", fontsize=10, color="#1E3A8A")
        else:
            sns.countplot(
                x=col,
                data=df_display,
                order=df_display[col].value_counts().index,
                palette="Blues_r",
                ax=ax
            )
            ax.set_title(f"Distribution of {col}", fontsize=10, color="#1E3A8A")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)

        plt.tight_layout()
        with cols[i % 2]:
            st.pyplot(fig, use_container_width=False)

    # --- Correlation Heatmap ---
    if len(num_cols) >= 2:
        st.divider()
        st.markdown("### 🔥 Correlation Heatmap (Numerical Features)")
        st.caption("Shows relationships between numerical variables to identify potential dependencies.")

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            df[num_cols].corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            square=True,
            cbar_kws={"shrink": 0.75},
            ax=ax
        )
        ax.set_title("Correlation Matrix", fontsize=10, color="#1E3A8A")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)

    st.divider()
    st.success("✅ EDA Complete — Encoded DataFrame ready for AI summary and chat.")
    return df_encoded
