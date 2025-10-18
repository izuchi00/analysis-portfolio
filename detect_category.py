# ============================================================
# 🧭 AUTOMATIC DATASET CATEGORY DETECTION (Streamlit-Safe)
# ============================================================

import re
import streamlit as st
from IPython.display import Markdown, display


def detect_dataset_category(df):
    """
    Automatically detect dataset type (e.g., customer, finance, sales, health, etc.)
    based on column names and data features.
    Returns a string label for the detected dataset category.
    """

    if df is None:
        st.warning("⚠️ No dataset provided for category detection.")
        return "Unknown"

    # Convert column names to lowercase for consistent matching
    cols = [str(c).lower() for c in df.columns]

    # --- Keyword groups for category detection ---
    category_keywords = {
        "Customer / People": [
            "customer", "client", "name", "gender", "age", "income",
            "education", "segment", "marital", "occupation"
        ],
        "Finance / Banking": [
            "balance", "loan", "credit", "debit", "account", "transaction",
            "bank", "payment", "interest", "amount", "salary"
        ],
        "Sales / Retail": [
            "product", "sale", "price", "discount", "revenue", "profit",
            "category", "store", "region", "quantity", "brand"
        ],
        "Healthcare / Medical": [
            "patient", "disease", "diagnosis", "treatment", "doctor",
            "hospital", "medical", "symptom", "test", "result", "lab"
        ],
        "Technology / Usage": [
            "device", "app", "usage", "session", "click", "login",
            "duration", "user_id", "platform", "os", "browser"
        ],
        "Education": [
            "student", "grade", "school", "exam", "teacher", "course",
            "subject", "marks", "attendance"
        ],
        "Operations / Logistics": [
            "shipment", "order", "supply", "warehouse", "inventory",
            "logistic", "vehicle", "route", "delivery"
        ],
    }

    # --- Keyword matching ---
    match_scores = {category: 0 for category in category_keywords}

    for col in cols:
        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if re.search(rf"\b{keyword}\b", col):
                    match_scores[category] += 1

    # --- Determine best match ---
    best_category = max(match_scores, key=match_scores.get)
    if match_scores[best_category] == 0:
        best_category = "General / Other"

    # --- Streamlit-friendly UI output ---
    st.markdown("---")
    st.markdown(
        f"""
        <div style="padding: 10px; border-radius: 8px; background-color: #F3F4F6; border-left: 4px solid #2563EB;">
            <strong>🧭 Detected Dataset Category:</strong> <span style="color:#2563EB;">{best_category}</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")

    return best_category


# ============================================================
# Notes:
# ✅ No logic changes — only visual improvements for Streamlit display.
# ✅ Keeps compatibility with app.py integration.
# ============================================================
