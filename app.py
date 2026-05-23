import streamlit as st

st.set_page_config(
    page_title="AI Assistant Comparison",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.auth import require_auth
from src.db import init_db

init_db()

if not require_auth():
    st.stop()

from src.ui import compare_chat, memory_panel, eval_dashboard, settings

tab_chat, tab_memory, tab_evals, tab_costs = st.tabs([
    "Compare Chat", "Memory", "Evaluations", "Cost + Latency"
])

with tab_chat:
    compare_chat.render()

with tab_memory:
    memory_panel.render()

with tab_evals:
    eval_dashboard.render()

with tab_costs:
    settings.render()
