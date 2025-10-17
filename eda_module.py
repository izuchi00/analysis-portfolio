# ============================================================
# 📊 UNIVERSAL FEATURE DISTRIBUTION & EDA (Streamlit Version)
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def run_eda(df):
    """
    Perform automated exploratory data analysis and encoding.
    Streamlit-safe version that renders visual outputs inline.
    Returns:
        df_encoded: cleaned, encoded DataFrame ready for AI use
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

    # --- Detect categorical & numerical columns ---
    cat_cols = [c for c in df.columns if df[c].dtype == 'object' or df[c].nunique() < 20]
    num_cols = [c for c in df.columns if df[c].dtype in ['int64', 'float64'] and c not in cat_cols]

    st.markdown("### 🧾 Feature Overview")
    st.write(f"**Categorical columns:** {cat_cols if cat_cols else 'None detected'}")
    st.write(f"**Numerical columns:** {num_cols if num_cols else 'None detected'}")

    # --- Auto-detect encoded categoricals (few unique numeric) ---
    auto_cats = [
        c for c in df.columns
        if (df[c].dtype in ['int64', 'float64']) and (2 <= df[c].nunique() <= 10)
    ]
    for c in auto_cats:
        if c not in cat_cols:
            cat_cols.append(c)

    # --- Create encoded copy for AI ---
    df_encoded = df.copy()
    category_mappings = {}
    for col in cat_cols:
        df_encoded[col] = df_encoded[col].astype('category')
        mapping = dict(enumerate(df_encoded[col].cat.categories))
        category_mappings[col] = mapping
        df_encoded[col] = df_encoded[col].cat.codes

    if category_mappings:
        with st.expander("📚 View Encoded Category Mappings", expanded=False):
            for col, mapping in category_mappings.items():
                st.markdown(f"**{col}**")
                for code, label in mapping.items():
                    st.markdown(f"- `{code}` → `{label}`")

    # --- Adaptive figure sizing ---
    def adaptive_figsize(df, n_unique=None):
        n = len(df)
        if n_unique and n_unique > 30:
            return (7, 3)
        elif n < 1000:
            return (4, 2.5)
        elif n < 10000:
            return (5, 3)
        elif n < 50000:
            return (6, 3.5)
        else:
            return (7, 4)

    st.markdown("### 📊 Feature Distributions")

    # --- Plot numerical distributions ---
    if num_cols:
        num_var = df[num_cols].var().sort_values(ascending=False)
        selected_num = num_var.index[:3].tolist()
        for col in selected_num:
            fig, ax = plt.subplots(figsize=adaptive_figsize(df))
            sns.histplot(df[col], kde=True, bins=20, color='cornflowerblue', ax=ax)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)

    # --- Plot categorical distributions ---
    skip_cats = {"cluster", "segment", "id", "index", "target"}
    selected_cat = [c for c in cat_cols if c.lower() not in skip_cats][:3]
    for col in selected_cat:
        unique_vals = df[col].nunique()
        fig, ax = plt.subplots(figsize=adaptive_figsize(df, unique_vals))
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
            ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
        st.pyplot(fig)

    # --- Correlation Heatmap ---
    if len(num_cols) >= 2:
        st.markdown("### 🔥 Correlation Heatmap (Numerical Features)")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", square=True, ax=ax)
        st.pyplot(fig)

        st.markdown("### 🎯 Initial Segmentation Hint")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(
            x=num_cols[0],
            y=num_cols[1],
            data=df,
            alpha=0.6,
            color="teal",
            edgecolor=None,
            ax=ax
        )
        st.pyplot(fig)

    st.success("✅ EDA Complete — Encoded DataFrame ready for AI summary and chat.")
    return df_encoded
