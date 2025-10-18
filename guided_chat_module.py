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

    # --- Section Header ---
    st.markdown("<h2 style='color:#2563EB;'>💬 Guided Dataset Chat</h2>", unsafe_allow_html=True)
    st.caption("Ask high-level interpretive questions to explore insights from your dataset.")

    with st.container():
        st.markdown(
            """
            <div style="padding:10px; border-radius:8px; background-color:#F3F4F6; border-left:4px solid #2563EB;">
                🧠 <strong>Tip:</strong> These guided prompts help you interpret data patterns without code.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Choose a guided question:")
    options = [
        "Summarize dataset",
        "Explain correlations",
        "Describe patterns",
        "Segmentation hints",
        "Next steps"
    ]

    choice = st.selectbox("", options, index=0, label_visibility="collapsed")

    # --- Interaction Button ---
    col1, col2 = st.columns([0.2, 0.8])
    with col1:
        ask_btn = st.button("💬 Ask", use_container_width=True)
    with col2:
        st.caption("Select a question and click *Ask* to get AI-powered insights.")

    # --- Handle Interaction ---
    if ask_btn:
        st.markdown("---")
        if choice == "Next steps":
            show_next_steps(sector)
        else:
            with st.spinner("🤖 Thinking... generating insights..."):
                response = groq_guided_chat(client, choice, ai_summary, insights, df, sector)
                if response:
                    st.success(response, icon="💡")


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
    st.markdown("<h3 style='color:#2563EB;'>🚀 Next Steps — Advanced Analysis Recommendations</h3>", unsafe_allow_html=True)

    st.caption("Suggested deeper analyses and projects tailored to your dataset category.")

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

    st.markdown("<ul style='margin-top: 10px;'>", unsafe_allow_html=True)
    for r in recs:
        st.markdown(f"<li>📈 {r}</li>", unsafe_allow_html=True)
    st.markdown("</ul>", unsafe_allow_html=True)

    st.markdown(
        """
        <hr style='margin-top:20px; margin-bottom:10px;'>
        <div style="padding:10px; background-color:#F9FAFB; border-radius:8px; border-left:4px solid #2563EB;">
            For deeper <strong>AI-driven analytics</strong>, predictive modeling, or automated reporting,<br>
            please <a href="#" style="color:#2563EB; text-decoration:none; font-weight:500;">contact or hire me</a>
            to unlock advanced features.
        </div>
        """,
        unsafe_allow_html=True,
    )
