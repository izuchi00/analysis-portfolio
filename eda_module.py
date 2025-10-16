# ============================================================
# 🧠 UNIVERSAL FEATURE DISTRIBUTION & EDA + AUTO ENCODING (SMART FINAL — STREAMLIT SAFE)
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display

def run_eda(df):
    """
    Perform automated exploratory data analysis and encoding.
    Returns:
        df_encoded: cleaned, encoded DataFrame ready for AI use
    """

    if df is None or df.empty:
        display(Markdown("⚠️ **No dataset provided for EDA.**"))
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

    display(Markdown("### 🧾 Feature Overview"))
    display(Markdown(f"- **Categorical columns:** {cat_cols if cat_cols else 'None detected'}"))
    display(Markdown(f"- **Numerical columns:** {num_cols if num_cols else 'None detected'}"))

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
        display(Markdown("✅ **Categorical encodings prepared for AI/ML (`df_encoded`):**"))
        for col, mapping in category_mappings.items():
            mapping_text = "\n".join([f"• `{code}` → `{label}`" for code, label in mapping.items()])
            display(Markdown(f"**{col}**\n{mapping_text}"))

    # --- Adaptive figure sizing ---
    def adaptive_figsize(df, n_unique=None):
        n = len(df)
        if n_unique and n_unique > 30:
            return (10, 4)
        elif n < 1000:
            return (5, 3)
        elif n < 10000:
            return (6, 4)
        elif n < 50000:
            return (7, 5)
        else:
            return (8, 6)

    display(Markdown("### 📊 Feature Distributions"))

    # --- Select representative columns ---
    if num_cols:
        num_var = df[num_cols].var().sort_values(ascending=False)
        selected_num = num_var.index[:3].tolist()
    else:
        selected_num = []

    skip_cats = {"cluster", "segment", "id", "index", "target"}
    selected_cat = [c for c in cat_cols if c.lower() not in skip_cats][:3]

    # --- Plot numerical distributions ---
    for col in selected_num:
        plt.figure(figsize=adaptive_figsize(df))
        sns.histplot(df[col], kde=True, bins=20, color='cornflowerblue', alpha=0.7)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

    # --- Plot categorical distributions (smart detection for year-like columns) ---
    for col in selected_cat:
        unique_vals = df[col].nunique()

        # ✅ Detect and handle year-like columns properly
        if "year" in col.lower() or "date" in col.lower() or col.lower().endswith("_span"):
            try:
                years = pd.to_numeric(df[col], errors="coerce").dropna()
                if not years.empty:
                    year_counts = years.value_counts().sort_index()
                    plt.figure(figsize=(8, 4))
                    sns.barplot(
                        x=year_counts.index.astype(int),
                        y=year_counts.values,
                        palette="pastel"
                    )
                    plt.title(f"Trend over Time: {col}")
                    plt.xlabel("Year")
                    plt.ylabel("Count")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.show()
                    continue
            except Exception as e:
                print(f"⚠️ Could not parse {col} as numeric year:", e)

        # ✅ Standard categorical visualization
        plt.figure(figsize=adaptive_figsize(df, unique_vals))
        if unique_vals > 30:
            top_values = df[col].value_counts().nlargest(15)
            sns.countplot(
                y=col,
                data=df[df[col].isin(top_values.index)],
                order=top_values.index,
                palette="pastel"
            )
            plt.title(f"Top 15 Categories of {col}")
            plt.xlabel("Count")
            plt.ylabel(col)
        else:
            sns.countplot(
                x=col,
                data=df,
                order=df[col].value_counts().index,
                palette="pastel"
            )
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.xticks(rotation=15)
        plt.tight_layout()
        plt.show()

    # --- Optional: Correlation Heatmap & Segmentation Scatter ---
    if len(num_cols) >= 2:
        display(Markdown("### 🔥 Correlation Heatmap (Numerical Features)"))
        plt.figure(figsize=(6, 5))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", square=True)
        plt.title("Correlation Heatmap of Numerical Features")
        plt.tight_layout()
        plt.show()

        display(Markdown("### 🎯 Initial Segmentation Hint"))
        plt.figure(figsize=(6, 5))
        sns.scatterplot(
            x=num_cols[0],
            y=num_cols[1],
            data=df,
            alpha=0.6,
            color="teal",
            edgecolor=None
        )
        plt.title(f"Segmentation Hint: {num_cols[0]} vs {num_cols[1]}")
        plt.tight_layout()
        plt.show()

    display(Markdown("✅ **EDA Complete** — Encoded DataFrame ready for AI summary and chat."))

    return df_encoded
