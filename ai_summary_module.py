# ============================================================
# 🤖 AI SUMMARY + SMART INSIGHTS (Groq + detect_category integrated)
# ============================================================

from IPython.display import Markdown, display
import numpy as np
import pandas as pd

def generate_ai_summary(client, df, sector="Auto-Detected"):
    """
    Generate an AI-powered dataset summary and insights using Groq API.

    Args:
        client (Groq): Initialized Groq client.
        df (pd.DataFrame): Dataset to summarize.
        sector (str): Category detected by detect_category module.

    Returns:
        tuple: (structured_summary, insights)
    """

    if df is None or df.empty:
        display(Markdown("⚠️ **No dataset provided for AI summary.**"))
        return "No data available", []

    display(Markdown("### 🤖 Generating AI Summary & Insights..."))

    # --- Prepare DataFrame ---
    df_summary = df.copy()
    rows, cols = df_summary.shape

    skip_cols = {"id", "index", "cluster", "segment", "target"}
    cat_cols = [
        c for c in df_summary.columns
        if (df_summary[c].dtype.name == "category" or df_summary[c].nunique() < 20)
        and c.lower() not in skip_cols
    ]
    num_cols = [
        c for c in df_summary.columns
        if np.issubdtype(df_summary[c].dtype, np.number)
        and c.lower() not in skip_cols
        and c not in cat_cols
    ]
    datetime_cols = [
        c for c in df_summary.columns if np.issubdtype(df_summary[c].dtype, np.datetime64)
    ]

    # --- Quality Metrics ---
    missing_pct = df_summary.isna().sum().sum() / (rows * cols) * 100
    dup_count = df_summary.duplicated().sum()
    avg_corr = (
        df_summary[num_cols].corr().abs().mean().mean()
        if len(num_cols) > 1 else 0
    )

    # --- Top Values ---
    cat_summary = []
    for c in cat_cols[:3]:
        try:
            mode_val = df_summary[c].mode().iloc[0]
            cat_summary.append(f"{c}: '{mode_val}'")
        except Exception:
            pass

    num_summary = []
    for c in num_cols[:3]:
        try:
            num_summary.append(f"{c}: avg {df_summary[c].mean():,.2f}")
        except Exception:
            pass

    # --- Date Range ---
    date_range = ""
    if datetime_cols:
        try:
            col = datetime_cols[0]
            date_range = f"{df_summary[col].min().date()} → {df_summary[col].max().date()}"
        except Exception:
            pass
    elif any("date" in c.lower() for c in df_summary.columns):
        try:
            col = [c for c in df_summary.columns if "date" in c.lower()][0]
            temp = pd.to_datetime(df_summary[col], errors="coerce")
            date_range = f"{temp.min().date()} → {temp.max().date()}"
        except Exception:
            pass

    # --- Structured Summary ---
    structured_summary = f"""
    Dataset has **{rows:,} rows** and **{cols:,} columns**.

    **Column Overview**
    - Categorical features: {', '.join(cat_cols) if cat_cols else 'None detected'}
    - Numerical features: {', '.join(num_cols) if num_cols else 'None detected'}
    - Datetime features: {', '.join(datetime_cols) if datetime_cols else 'None detected'}

    **Data Quality**
    - Missing values: {missing_pct:.1f}% of total
    - Duplicate rows: {dup_count:,}
    - Avg numeric correlation: {avg_corr:.2f}

    **Key Patterns**
    - Top categorical modes → {', '.join(cat_summary) if cat_summary else 'N/A'}
    - Key numeric averages → {', '.join(num_summary) if num_summary else 'N/A'}
    - Date range → {date_range if date_range else 'N/A'}

    **Detected Sector:** {sector}
    """

    display(Markdown(f"### 🧠 AI Summary\n{structured_summary}"))

    # --- AI Commentary via Groq ---
    try:
        ai_prompt = f"""
        You are a professional data analyst. Given the structured dataset summary below,
        write a concise yet insightful overview and 4 actionable analytical recommendations.

        Dataset Sector: {sector}
        Summary:
        {structured_summary}
        """

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You analyze data and produce meaningful, structured insights."},
                {"role": "user", "content": ai_prompt},
            ],
            temperature=0.3,
            max_tokens=400,
        )

        ai_text = response.choices[0].message.content.strip()

        # --- Parse AI output ---
        lines = [l.strip("•- ").strip() for l in ai_text.split("\n") if l.strip()]
        insights = [l for l in lines if len(l) < 250][:5]

        display(Markdown("### 🔍 Key Insights (AI-Generated):"))
        for i, insight in enumerate(insights, 1):
            display(Markdown(f"{i}. {insight}"))

        display(Markdown("✅ **AI Summary and Insights generated successfully.**"))

    except Exception as e:
        display(Markdown(f"⚠️ *AI insight generation failed ({e}). Using defaults.*"))
        insights = [
            "Examine the correlation matrix to find strong relationships.",
            "Explore time-based trends for potential seasonality.",
            "Compare distributions across categorical groups.",
            "Identify outliers that might affect modeling accuracy.",
        ]

    return structured_summary, insights
