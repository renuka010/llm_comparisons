from functools import partial

import streamlit as st
from src.chat.conversation import (
    create_conversation,
    list_conversations,
    save_message,
    get_messages,
    rename_conversation,
    delete_conversation,
)
from src.chat.context_builder import build_messages_for_model
from src.chat.memory_extractor import extract_facts, extract_facts_llm, extract_keywords
from src.chat.memory_store import save_memory_fact, add_fact_to_index, search_memory
from src.chat.summarizer import maybe_summarize
from src.models.factory import get_oss_client, get_frontier_client


def _init_session():
    if "conv_id" not in st.session_state:
        st.session_state["conv_id"] = None
    if "frontier_tool_log" not in st.session_state:
        # {conv_id: [list of tool_calls_log per turn, newest last]}
        st.session_state["frontier_tool_log"] = {}


def _sidebar():
    with st.sidebar:
        st.markdown("### Conversations")

        if st.button("+ New Chat", use_container_width=True):
            conv_id = create_conversation()
            st.session_state["conv_id"] = conv_id
            st.rerun()

        conversations = list_conversations()
        for conv in conversations:
            cols = st.columns([4, 1])
            label = conv["title"][:28] + "…" if len(conv["title"]) > 28 else conv["title"]
            active = st.session_state.get("conv_id") == conv["id"]
            if cols[0].button(
                f"{'▶ ' if active else ''}{label}",
                key=f"conv_{conv['id']}",
                use_container_width=True,
            ):
                st.session_state["conv_id"] = conv["id"]
                st.rerun()
            if cols[1].button("🗑", key=f"del_{conv['id']}"):
                delete_conversation(conv["id"])
                if st.session_state.get("conv_id") == conv["id"]:
                    st.session_state["conv_id"] = None
                st.rerun()


def _render_tool_calls(tool_calls_log: list[dict]) -> None:
    """Render a compact tool-use badge + expandable detail for one assistant turn."""
    for entry in tool_calls_log:
        with st.expander(f"🔍 Searched memory — *\"{entry['query']}\"*", expanded=False):
            result = entry["result_preview"]
            if result and result != "No relevant memories found.":
                st.markdown(result)
            else:
                st.caption("No relevant memories found.")


def _render_messages(messages: list[dict], model_type: str, conv_id: int | None = None):
    # Per-conv tool call logs aligned to frontier assistant turn index
    tool_logs: list[list[dict]] = []
    if model_type == "frontier" and conv_id is not None:
        tool_logs = st.session_state.get("frontier_tool_log", {}).get(conv_id, [])

    frontier_turn_idx = 0

    for msg in messages:
        if msg["model_type"] == "user" and msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        elif msg["model_type"] == model_type and msg["role"] == "assistant":
            # Show tool calls (if any) before the assistant bubble for this turn
            if model_type == "frontier" and frontier_turn_idx < len(tool_logs):
                _render_tool_calls(tool_logs[frontier_turn_idx])
            frontier_turn_idx += 1

            with st.chat_message("assistant"):
                if msg.get("error"):
                    st.error(f"Error: {msg['error']}")
                else:
                    st.markdown(msg["content"])
                    meta = []
                    if msg.get("latency_ms"):
                        meta.append(f"{msg['latency_ms']}ms")
                    if msg.get("token_estimate"):
                        meta.append(f"~{msg['token_estimate']} tokens")
                    if meta:
                        st.caption(" · ".join(meta))


def _auto_title(conv_id: int, user_text: str) -> None:
    """Set conversation title to first ~6 words of the opening message."""
    existing_user_msgs = [m for m in get_messages(conv_id) if m["role"] == "user"]
    if len(existing_user_msgs) == 0:
        words = user_text.split()
        title = " ".join(words[:6]) + ("…" if len(words) > 6 else "")
        rename_conversation(conv_id, title)


def _save_facts(conv_id: int, facts: list[str]) -> None:
    for fact in facts:
        keywords = extract_keywords(fact)
        fact_id = save_memory_fact(conv_id, fact, keywords)
        add_fact_to_index(fact_id, fact, conv_id)


def _send_to_models(conv_id: int, user_text: str) -> None:
    _auto_title(conv_id, user_text)
    save_message(conv_id, "user", "user", user_text)

    # --- OSS: rule-based extraction → inject into context via RAG ---
    oss_facts = extract_facts(user_text)
    _save_facts(conv_id, oss_facts)

    oss_messages = build_messages_for_model(conv_id, user_text, "oss")

    with st.spinner("OSS model thinking…"):
        oss_client = get_oss_client()
        oss_resp = oss_client.generate(oss_messages)
        save_message(
            conv_id, "oss", "assistant",
            oss_resp.text or "",
            latency_ms=oss_resp.latency_ms,
            token_estimate=oss_resp.token_estimate,
            error=oss_resp.error,
        )

    # --- Frontier: no memory in prompt; model can call search_memory as a tool ---
    frontier_messages = build_messages_for_model(conv_id, user_text, "frontier")
    tool_map = {"search_memory": partial(search_memory, conv_id=conv_id)}

    with st.spinner("Frontier model thinking…"):
        frontier_client = get_frontier_client()
        frontier_resp = frontier_client.generate_with_tools(frontier_messages, tool_map)
        save_message(
            conv_id, "frontier", "assistant",
            frontier_resp.text or "",
            latency_ms=frontier_resp.latency_ms,
            token_estimate=frontier_resp.token_estimate,
            error=frontier_resp.error,
        )

    # Always append (even empty list) so turn index stays aligned with tool_logs
    logs = st.session_state["frontier_tool_log"].setdefault(conv_id, [])
    logs.append(frontier_resp.tool_calls_log)

    # --- Frontier: LLM-based fact extraction after response ---
    frontier_facts = extract_facts_llm(user_text)
    _save_facts(conv_id, frontier_facts)

    maybe_summarize(conv_id, "oss")
    maybe_summarize(conv_id, "frontier")


def render():
    _init_session()
    _sidebar()

    st.markdown("## Side-by-Side Chat")

    conv_id = st.session_state.get("conv_id")
    if conv_id is None:
        st.info("Start a new chat using the sidebar.")
        return

    messages = get_messages(conv_id)

    from src.config import cfg
    col_oss, col_frontier = st.columns(2)

    with col_oss:
        st.markdown(f"**OSS — {cfg.OSS_MODEL_ID.split('/')[-1]}**")
        with st.container(height=520, border=False):
            _render_messages(messages, "oss", conv_id)

    with col_frontier:
        provider = cfg.FRONTIER_PROVIDER.upper()
        st.markdown(f"**Frontier — {provider}**")
        with st.container(height=520, border=False):
            _render_messages(messages, "frontier", conv_id)

    st.divider()

    user_input = st.chat_input("Type your message…")
    if user_input:
        _send_to_models(conv_id, user_input)
        st.rerun()
