# ============================================================
# 💬 BASIC GUIDED DATASET CHAT (Streamlit-Compatible, Button-Driven)
# ============================================================

import streamlit as st
from groq import Groq
import traceback

# ============================================================
# 💬 MAIN CHAT FUNCTION
# ============================================================
def launch_basic_chat(client, ai_summary, insights, df, sector="General"):
    """
    Launch a guided chat in Streamlit with pre-defined questions.
    Args:
        client (Groq): Initialized Groq client
        ai_summary (str): AI-generated dataset summary
        insights (list): Key AI insights
        df (pd.DataFrame): The dataset
        sector (str): Detected dataset category
    """

    st.markdown("### 💬 Guided Dataset Chat")
    st.markdown("🧠 Choose a question below to explore dataset insights:")

    # --- Predefined options ---
    options = [
        "Summarize dataset",
        "Explain correlations",
        "Describe patterns",
        "Segmentation hints",
        "Next steps"
    ]

    choice = st.selectbox("Choose a guided question:", options)

    if st.button("Ask"):
        if choice == "Next steps":
            show_next_steps(sector)
        else:
            with st.spinner("🤖 Thinking..."):
                response = groq_guided_chat(client, choice, ai_summary, insights, df, sector)
                if response:
                    st.success(response)


# ============================================================
# 🧠 CHAT ENGINE (Short, Friendly Responses)
# ============================================================
def groq_guided_chat(client, question, summary, insights, df, sector):
    """Generate concise answers for predefined questions."""
    context = f"""
You are a friendly data assistant discussing a dataset that has undergone EDA.

Dataset Sector: {sector}
Dataset Summary:
{summary}

Key Insights:
{', '.join(insights[:5])}

The user clicked a predefined button labeled: "{question}".

Rules:
- Be short, clear, and conversational (max 4 sentences).
- No Python code or technical jargon.
- Focus on interpretive insights and reasoning.
"""
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a concise, helpful data analysis assistant."},
                {"role": "user", "content": context},
            ],
            temperature=0,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        traceback.print_exc()
        return f"⚠️ Error generating AI response: {e}"


# ============================================================
# 🚀 NEXT STEPS RECOMMENDATIONS
# ============================================================
def show_next_steps(sector):
    """Display sector-specific advanced recommendations."""
    st.markdown("### 🚀 Next Steps — Advanced Analysis Recommendations")

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

    key = sector.lower() if sector and sector.lower() in sector_recommendations else "general"
    recs = sector_recommendations[key]

    for r in recs:
        st.markdown(f"- {r}")

    st.markdown("""
---
For deeper **AI-driven analytics**, predictive modeling, or automated reporting,  
please **[contact or hire me](#)** to unlock advanced features.
---
""")
