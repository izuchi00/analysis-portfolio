# ============================================================
# 💬 BASIC GUIDED DATASET CHAT (Button-Only Mode, Stable Output)
# ============================================================

import ipywidgets as widgets
from IPython.display import Markdown, display, clear_output
from groq import Groq
import os, traceback

# --- Ensure Groq API key ---
api_key = os.getenv("GROQ_API_KEY", "").strip()
if not api_key or not api_key.startswith("gsk_"):
    raise ValueError("❌ Missing or invalid API key. Please re-run setup.")

client = Groq(api_key=api_key)
chat_history = []

# ============================================================
# 🧠 LIMITED CHAT ENGINE (Predefined Prompts Only)
# ============================================================
def groq_guided_chat(question, summary, insights, df, sector):
    """Handles short AI responses for all buttons except 'Next steps'."""
    context = f"""
You are a polite, friendly data assistant.
You are chatting about a dataset that has already undergone EDA.

Dataset Sector: {sector}
Dataset Summary:
{summary}

Key Insights:
{', '.join(insights[:5])}

The user clicked a predefined button labeled: "{question}".

Rules:
- Give short, clear, conversational answers (max 4 sentences).
- No Python code or complex stats.
"""
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a concise, helpful data analysis assistant."},
                {"role": "user", "content": context},
            ],
            temperature=0,  # ✅ deterministic output (consistent)
            max_tokens=350,
        )
        answer = response.choices[0].message.content.strip()
        chat_history.append({"user": question, "ai": answer})
        render_chat()
    except Exception as e:
        traceback.print_exc()
        display(Markdown(f"❌ **Error during AI response:** {e}"))


# ============================================================
# 💬 RENDER CHAT (Display History)
# ============================================================
def render_chat():
    clear_output(wait=True)
    display(Markdown("## 💬 Guided Dataset Chat"))
    display(Markdown("🟢 Select a question below to explore your dataset."))

    for msg in chat_history[-5:]:
        display(Markdown(f"🧑 **You:** {msg['user']}"))
        display(Markdown(f"🤖 **AI:** {msg['ai']}"))
        print("-" * 50)

    display_suggestions()


# ============================================================
# 🚀 CONSISTENT “NEXT STEPS” MESSAGE
# ============================================================
def show_next_steps(sector):
    """Displays a consistent, predefined CTA based on dataset sector."""
    clear_output(wait=True)
    display(Markdown("## 💬 Guided Dataset Chat"))
    display(Markdown("🟢 Chat active — select a question below to explore your dataset."))

    # --- Sector-specific advanced analysis suggestions ---
    sector_recommendations = {
        "marketing": [
            "Customer segmentation & targeting",
            "Campaign performance prediction",
            "Churn and retention analysis",
            "Ad spend optimization"
        ],
        "finance": [
            "Revenue forecasting & risk modeling",
            "Portfolio performance optimization",
            "Expense anomaly detection",
            "Profitability and KPI tracking"
        ],
        "retail": [
            "Product demand forecasting",
            "Dynamic pricing optimization",
            "Inventory trend prediction",
            "Sales region clustering"
        ],
        "healthcare": [
            "Patient outcome prediction",
            "Treatment effectiveness analysis",
            "Operational efficiency optimization",
            "Cost-benefit modeling"
        ],
        "general": [
            "Predictive modeling & forecasting",
            "Clustering and segmentation analysis",
            "Automated dashboard reporting",
            "KPI correlation and trend detection"
        ]
    }

    # Pick sector or default to general
    key = sector.lower() if sector and sector.lower() in sector_recommendations else "general"
    recs = sector_recommendations[key]

    # --- Display consistent CTA ---
    display(Markdown(f"""
### 🚀 Next Steps — Advanced Analysis for the **{sector.title()}** Domain
To perform deeper **AI-driven analytics**, consider exploring the following advanced methods:
"""))
    for r in recs:
        display(Markdown(f"- {r}"))
    display(Markdown("""
---
For custom **predictive modeling, automation, or decision intelligence solutions**,
please **[contact or hire the analyst](#)** to access the advanced modules.
_(These advanced features are available as part of premium or enterprise-level projects.)_
---
"""))

    display_suggestions()


# ============================================================
# 💡 FIXED PROMPT BUTTONS
# ============================================================
def display_suggestions():
    options = [
        ("Summarize dataset", "Give a brief summary of this dataset."),
        ("Explain correlations", "What correlations exist in this dataset?"),
        ("Describe patterns", "Describe any patterns or trends you observed."),
        ("Segmentation hints", "What do the segment or group differences show?"),
        ("Next steps", "Show advanced analysis recommendations."),
    ]

    display(Markdown("### 💡 Choose an option to continue:"))
    box = widgets.HBox()
    btn_list = []

    for label, query in options:
        btn = widgets.Button(description=label, button_style='info', layout=widgets.Layout(margin='2px'))

        def make_callback(q, label):
            def on_click(b):
                if "Next steps" in label:
                    show_next_steps(sector)
                else:
                    groq_guided_chat(q, ai_summary, insights, df, sector)
            return on_click

        btn.on_click(make_callback(query, label))
        btn_list.append(btn)

    box.children = btn_list
    display(box)


# ============================================================
# 🎯 INITIALIZE GUIDED CHAT UI
# ============================================================
display(Markdown("✅ **AI Summary detected — launching guided dataset chat...**"))
display(Markdown("🧠 This simplified chat lets you explore quick insights using predefined options below."))
display_suggestions()
