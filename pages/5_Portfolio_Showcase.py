import streamlit as st

st.set_page_config(page_title="🌐 Portfolio | Edis Analytics", layout="wide")

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Custom sidebar header ---
st.sidebar.image("assets/logo.png", width=60)
st.sidebar.markdown("<div class='sidebar-title'>Edis Analytics</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-subtitle'>Data Intelligence Suite</div>", unsafe_allow_html=True)
st.sidebar.markdown("---")


st.markdown("""
<h2 class='section-header'>🌐 Portfolio Showcase</h2>
<p>Explore selected analytics projects built by <strong>Edis Analytics</strong>.</p>
""", unsafe_allow_html=True)

st.image("assets/logo.png", width=120)
