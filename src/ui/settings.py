import streamlit as st
from src.config import cfg


def render():
    st.markdown("## Cost + Latency")
    st.caption("An overview of model configuration, estimated costs, and observed performance.")

    if st.button("Clear model cache (reload clients)", help="Use this after changing API keys or model names in .env"):
        st.cache_resource.clear()
        st.success("Model cache cleared. Clients will reload on next message.")


    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### OSS Model")
        st.markdown(f"**Model:** `{cfg.OSS_MODEL_ID}`")
        st.markdown(f"**Device:** `{cfg.OSS_DEVICE}`")
        st.markdown(f"**Max tokens:** `{cfg.OSS_MAX_NEW_TOKENS}`")
        st.markdown(f"**Temperature:** `{cfg.OSS_TEMPERATURE}`")
        st.divider()
        st.markdown("**Deployment cost (Hugging Face Spaces)**")
        st.markdown("""
- Free CPU tier: ~0 USD/month  
- 0.5B model: runs on CPU with ~2-8s latency  
- GPU tier (T4): ~$0.60/hr on HF Spaces
        """)

    with col2:
        st.markdown("### Frontier Model")
        provider = cfg.FRONTIER_PROVIDER.upper()
        st.markdown(f"**Provider:** `{provider}`")
        if provider == "OPENAI":
            st.markdown(f"**Model:** `{cfg.OPENAI_MODEL}`")
        else:
            st.markdown(f"**Model:** `{cfg.ANTHROPIC_MODEL}`")
        st.markdown(f"**Max tokens:** `{cfg.FRONTIER_MAX_TOKENS}`")
        st.markdown(f"**Temperature:** `{cfg.FRONTIER_TEMPERATURE}`")
        st.divider()
        st.markdown("**Estimated API cost**")
        st.markdown("""
- GPT-5-nano: ~$0.05/1M input tokens, ~$0.40/1M output tokens  
- Claude 4.5 Haiku: ~$1.00/1M input, ~$5.00/1M output  
- Typical chat turn: ~300-800 tokens total
        """)

    st.divider()
    st.markdown("### Observed Latency from Evaluations")

    from src.evals.runner import get_eval_results
    oss_results = get_eval_results("oss")
    frontier_results = get_eval_results("frontier")

    if oss_results or frontier_results:
        def avg_ms(results):
            valid = [r["latency_ms"] for r in results if r.get("latency_ms")]
            return round(sum(valid) / len(valid)) if valid else "N/A"

        col3, col4 = st.columns(2)
        col3.metric("OSS Avg Latency", f"{avg_ms(oss_results)} ms")
        col4.metric("Frontier Avg Latency", f"{avg_ms(frontier_results)} ms")
    else:
        st.info("Run evaluations to see latency data here.")
