# ============================================================
# 🤖 AI SUMMARY + SMART INSIGHTS
# ============================================================

from IPython.display import Markdown, display
import numpy as np
from groq import Groq
import os

display(Markdown("### 🤖 Generating AI Summary & Insights..."))

api_key = os.getenv("GROQ_API_KEY", "").strip()
if not api_key or not api_key.startswith("gsk_"):
    raise ValueError("❌ Missing or invalid API key. Please re-run setup.")
client = Groq(api_key=api_key)

# --- Use encoded DataFrame if available ---
if "df_for_ai" in globals():
    df_summary = df_for_ai.copy()
else:
    df_summary = df.copy()

rows, cols = df_summary.shape
drop_cols = {"cluster", "segment", "target"}
df_summary = df_summary[[c for c in df_summary.columns if c.lower() not in drop_cols]]

skip_cols = {"id", "index", "cluster", "segment", "target"}
cat_cols = [c for c in df_summary.columns if (df_summary[c].dtype.name == "category" or df_summary[c].nunique() < 20) and c.lower() not in skip_cols]
num_cols = [c for c in df_summary.columns if (df_summary[c].dtype in ["int64", "float64"]) and c.lower() not in skip_cols and c not in cat_cols]

if "sector" not in globals() or not sector:
    sector = "General / Unspecified"

ai_summary = f"""
Dataset contains **{rows:,} rows** and **{cols:,} columns**.

**Categorical features:** {', '.join(cat_cols) if cat_cols else 'None detected'}
**Numerical features:** {', '.join(num_cols) if num_cols else 'None detected'}

This dataset is suitable for exploratory analysis and insight discovery in the **{sector}** domain.
"""
display(Markdown(f"### 🧠 AI Summary\n{ai_summary}"))

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
            {"role": "system", "content": "You generate concise, relevant analytical insights for data summaries."},
            {"role": "user", "content": insight_prompt},
        ],
        temperature=0,
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
        "Explore feature distributions and categorical balance",
        "Check correlations among numerical variables",
        "Identify segmentation or clustering opportunities"
    ]

globals()["ai_summary"] = ai_summary
globals()["insights"] = insights

display(Markdown("✅ **AI Summary and Insights generated successfully.**"))
