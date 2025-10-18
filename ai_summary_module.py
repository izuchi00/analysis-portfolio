# ============================================================
# 🤖 AI SUMMARY + SMART INSIGHTS (Streamlit-Compatible)
# ============================================================

from IPython.display import Markdown, display
import numpy as np

def generate_ai_summary(client, df, sector="General / Unspecified"):
    """
    Generate AI-powered dataset description and insights using Groq API.
    Produces a brief narrative summary instead of a technical column list.
    """

    if df is None or df.empty:
        display(Markdown("⚠️ **No dataset provided for AI summary.**"))
        return "No data available", []

    display(Markdown("### 🤖 Generating AI Summary & Insights..."))

    # --- Prepare dataset info ---
    df_summary = df.copy()
    rows, cols = df_summary.shape

    drop_cols = {"cluster", "segment", "target"}
    df_summary = df_summary[[c for c in df_summary.columns if c.lower() not in drop_cols]]

    skip_cols = {"id", "index", "cluster", "segment", "target"}
    cat_cols = [
        c for c in df_summary.columns
        if (df_summary[c].dtype.name == "category" or df_summary[c].nunique() < 20)
        and c.lower() not in skip_cols
    ]
    num_cols = [
        c for c in df_summary.columns
        if (df_summary[c].dtype in ["int64", "float64"])
        and c.lower() not in skip_cols
        and c not in cat_cols
    ]

    # --- Prepare context for AI ---
    column_context = ", ".join(df_summary.columns[:15]) + ("..." if len(df_summary.columns) > 15 else "")
    data_context = f"""
    - Rows: {rows:,}
    - Columns: {cols}
    - Sector: {sector}
    - Example columns: {column_context}
    """

    # --- Ask AI for a short human-style summary ---
    try:
        summary_prompt = f"""
        You are a data analyst. Write a brief, natural-language paragraph (3–4 sentences)
        that describes the dataset below. Mention what type of data it appears to contain,
        its likely purpose, and potential use — *based on the context provided*.
        Avoid listing columns or technical terms like "categorical" or "numerical".

        Dataset Info:
        {data_context}
        """

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You write short, professional dataset summaries in plain English."},
                {"role": "user", "content": summary_prompt},
            ],
            temperature=0.4,
            max_tokens=250,
        )

        ai_summary_text = response.choices[0].message.content.strip()

    except Exception as e:
        ai_summary_text = (
            f"This dataset contains **{rows:,} records** and **{cols} columns**, "
            f"likely representing data related to the **{sector}** domain."
        )
        display(Markdown(f"⚠️ AI summary generation failed ({e}). Using fallback."))

    display(Markdown(f"### 🧠 AI Dataset Summary\n\n{ai_summary_text}"))

    # --- Generate AI insights ---
    try:
        insight_prompt = f"""
        You are a professional data analyst. Based on the dataset summary below,
        provide 4 concise insights or analytical ideas (no code). Avoid generic phrasing.

        Dataset Sector: {sector}
        Dataset Summary:
        {ai_summary_text}
        """

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You generate focused analytical insights that follow from dataset summaries."
                },
                {"role": "user", "content": insight_prompt},
            ],
            temperature=0.4,
            max_tokens=300,
        )

        ai_text = response.choices[0].message.content.strip()
        insights = [line.strip("•- ").strip() for line in ai_text.split("\n") if line.strip()]

        display(Markdown("### 🔍 Key Insights (AI-Generated):"))
        for i, insight in enumerate(insights, 1):
            display(Markdown(f"{i}. {insight}"))

    except Exception as e:
        display(Markdown(f"⚠️ *AI insight generation failed ({e}). Using defaults.*"))
        insights = [
            "Explore key trends, averages, and distributions.",
            "Investigate feature relationships and correlations.",
            "Identify segments, anomalies, or emerging patterns.",
            "Analyze time-based or category-based variations."
        ]

    display(Markdown("✅ **AI Summary and Insights generated successfully.**"))
    return ai_summary_text, insights
