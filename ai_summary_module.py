# ============================================================
# 🤖 CONTEXT-AWARE AI DATASET SUMMARY + SMART INSIGHTS
# ============================================================

import numpy as np
import pandas as pd
import streamlit as st

def generate_ai_summary(client, df, sector="Auto-Detected"):
    """
    Generates a contextual dataset description (based on column names & sample values)
    plus concise insights.
    """

    if df is None or df.empty:
        st.warning("⚠️ No dataset provided for AI summary.")
        return "No data available", []

    rows, cols = df.shape

    # --- Detect column types ---
    cat_cols = [c for c in df.columns if df[c].dtype == 'object' or df[c].nunique() < 20]
    num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    datetime_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]

    # --- Summaries for prompt ---
    missing_pct = round(df.isna().sum().sum() / (rows * cols) * 100, 2)
    dup_count = df.duplicated().sum()

    # --- Column preview (names + examples) ---
    preview = df.head(3).to_dict(orient="records")
    preview_text = "\n".join([str(r) for r in preview])

    # --- Build prompt ---
    prompt = f"""
    You are a professional data analyst. Analyze the dataset sample below and
    generate a short (2–4 sentences) executive summary describing what this dataset
    likely represents and what it could be used for. Be concrete and contextual
    based on the column names and values — not generic.

    Then, provide exactly 3 concise insights or analysis ideas as bullet points.

    Dataset info:
    - Rows: {rows:,}
    - Columns: {cols}
    - Sector: {sector}
    - Missing Values: {missing_pct}%
    - Duplicate Rows: {dup_count}
    - Columns: {', '.join(df.columns[:15])}{'...' if len(df.columns) > 15 else ''}

    Sample rows:
    {preview_text}
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system",
                 "content": "You summarize datasets naturally and briefly for humans — using data context and meaning, not lists."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=450,
        )

        ai_text = getattr(response.choices[0].message, "content", "").strip()

        # --- Parse output ---
        lines = ai_text.split("\n")
        summary, insights = "", []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith(("•", "-", "1.", "2.", "3.")):
                insights.append(line.strip("•- ").strip())
            elif not summary:
                summary = line

        # --- Display in Streamlit ---
        st.markdown("### 🧠 Executive Summary")
        st.markdown(summary or "_No AI summary generated._")

        if insights:
            st.markdown("### 🔍 Key Insights")
            for i, insight in enumerate(insights, 1):
                st.markdown(f"**{i}.** {insight}")

        return summary, insights

    except Exception as e:
        st.warning(f"⚠️ AI summary generation failed ({e}).")
        fallback_summary = (
            f"This dataset contains **{rows:,} rows** and **{cols} columns** "
            f"with both categorical and numerical data, suitable for analysis "
            f"in the {sector} domain."
        )
        fallback_insights = [
            "Explore feature correlations to identify drivers of key metrics.",
            "Assess trends across time or categories.",
            "Clean missing or duplicate data before modeling."
        ]

        st.markdown(fallback_summary)
        return fallback_summary, fallback_insights
