# ============================================================
# 🤖 AI SUMMARY + SMART INSIGHTS (Streamlit-Compatible)
# ============================================================

from IPython.display import Markdown, display
from groq import Groq
import os
import numpy as np

def generate_ai_summary(client, df, sector="General / Unspecified"):
    """
    Generate AI-powered dataset summary and insights using Groq API.

    Args:
        client (Groq): Initialized Groq client
        df (pd.DataFrame): Dataset to summarize
        sector (str): Detected dataset category (optional)

    Returns:
        tuple: (ai_summary, insights)
    """
    if df is None or df.empty:
        display(Markdown("⚠️ **No dataset provided for AI summary.**"))
        return "No data available", []

    display(Markdown("### 🤖 Generating AI Summary & Insights..."))

    # --- Use encoded DataFrame if available ---
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

    ai_summary = f"""
    Dataset contains **{rows:,} rows** and **{cols:,} columns**.

    **Categorical features:** {', '.join(cat_cols) if cat_cols else 'None detected'}  
    **Numerical features:** {', '.join(num_cols) if num_cols else 'None detected'}  

    This dataset is suitable for exploratory analysis and insight discovery in the **{sector}** domain.
    """

    display(Markdown(f"### 🧠 AI Summary\n{ai_summary}"))

    # --- Generate AI insights via Groq ---
    try:
        insight_prompt = f"""
        You are a professional data analyst. Based on the dataset summary below,
        provide 4 short insights or directions for analysis (no code).

        Dataset Sector: {sector}
        Dataset Summary:
        {ai_summary}
        """

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You generate concise, relevant analytical insights for data summaries."
                },
                {"role": "user", "content": insight_prompt},
            ],
            temperature=0,
            max_tokens=300,
        )

        ai_text = response.choices[0].message.content.strip()
        insights = [line.strip("•- ").strip() for line in ai_text.split("\n") if line.strip()]

        display(Markdown("### 🔍 Key Insights (AI-Generated):"))
        for i, insight in enumerate(insights, 1):
            display(Markdown(f"{i}. {insight}"))  # keep formatted Markdown display

    except Exception as e:
        display(Markdown(f"⚠️ *AI insight generation failed ({e}). Using defaults.*"))
        insights = [
            "Explore feature distributions and categorical balance.",
            "Check correlations among numerical variables.",
            "Identify segmentation or clustering opportunities."
        ]

    display(Markdown("✅ **AI Summary and Insights generated successfully.**"))
    return ai_summary, insights
