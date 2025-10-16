# ============================================================
# 🧭 4️⃣ AUTOMATIC DATASET CATEGORY DETECTION
# ============================================================

import re
from IPython.display import Markdown, display

def detect_dataset_category(df):
    """
    Automatically detect dataset type (e.g., customer, finance, sales, health, etc.)
    based on column name patterns and data features.
    """

    # Convert column names to lowercase for consistent matching
    cols = " ".join(df.columns.astype(str).str.lower())

    # Define simple keyword groups
    categories = {
        "Customer / Marketing": ["gender", "age", "income", "spending", "customer", "segment", "region"],
        "Finance / Banking": ["balance", "loan", "credit", "account", "transaction", "payment", "interest"],
        "Healthcare / Medical": ["patient", "disease", "symptom", "diagnosis", "hospital", "treatment"],
        "Sales / Retail": ["product", "sales", "revenue", "profit", "store", "quantity", "price"],
        "Human Resources": ["employee", "salary", "department", "hired", "position", "performance"],
        "Education / Academics": ["student", "grade", "exam", "score", "subject", "school"],
        "Technology / Usage": ["user", "device", "click", "app", "session", "usage"],
    }

    matched_category = "General / Other"
    max_matches = 0

    for category, keywords in categories.items():
        matches = sum(1 for kw in keywords if re.search(rf"\b{kw}\b", cols))
        if matches > max_matches:
            max_matches = matches
            matched_category = category

    display(Markdown(f"### 🧭 Detected Dataset Category: **{matched_category}**"))
    return matched_category


# --- Run detection ---
if df is not None:
    sector = detect_dataset_category(df)
else:
    sector = "Unknown"
