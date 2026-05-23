import streamlit as st
from src.config import cfg
from src.models.base import ModelClient


@st.cache_resource(show_spinner=False)
def get_oss_client() -> ModelClient:
    from src.models.oss_hf import OSSModelClient
    return OSSModelClient()


@st.cache_resource(show_spinner=False)
def get_frontier_client() -> ModelClient:
    provider = cfg.FRONTIER_PROVIDER.lower()
    if provider == "anthropic":
        from src.models.anthropic_model import AnthropicModelClient
        return AnthropicModelClient()
    from src.models.openai_model import OpenAIModelClient
    return OpenAIModelClient()
