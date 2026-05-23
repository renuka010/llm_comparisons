import streamlit as st
from src.chat.conversation import list_conversations
from src.chat.memory_store import (
    get_memory_facts,
    save_memory_fact,
    delete_memory_fact,
    rebuild_faiss_index,
    has_synthetic_data,
    add_fact_to_index,
)
from src.chat.memory_extractor import extract_keywords
from src.data.synthetic_facts import SYNTHETIC_FACTS, SYNTHETIC_SEED_KEYWORD


def render():
    st.markdown("## Memory")
    st.caption("Facts the assistant has learned from your conversations.")

    conversations = list_conversations()
    if not conversations:
        st.info("No conversations yet. Start chatting to build memory.")
        return

    conv_options = {f"[{c['id']}] {c['title']}": c["id"] for c in conversations}
    selected_label = st.selectbox("Select conversation", list(conv_options.keys()))
    conv_id = conv_options[selected_label]

    facts = get_memory_facts(conv_id)

    st.markdown(f"**{len(facts)} fact(s) stored for this conversation**")

    for fact in facts:
        cols = st.columns([8, 1])
        cols[0].markdown(f"- {fact['fact']}")
        if cols[1].button("✕", key=f"del_fact_{fact['id']}"):
            delete_memory_fact(fact["id"])
            st.rerun()

    st.divider()
    st.markdown("#### Add a fact manually")
    with st.form("add_fact_form"):
        new_fact = st.text_area("Fact", placeholder="e.g. User prefers concise summaries.")
        submitted = st.form_submit_button("Save fact")

    if submitted and new_fact.strip():
        keywords = extract_keywords(new_fact.strip())
        fact_id = save_memory_fact(conv_id, new_fact.strip(), keywords)
        add_fact_to_index(fact_id, new_fact.strip(), conv_id)
        st.success("Fact saved.")
        st.rerun()

    st.divider()
    if st.button("Rebuild FAISS index from all facts"):
        count = rebuild_faiss_index(conv_id)
        st.success(f"Index rebuilt with {count} facts.")

    st.divider()
    st.markdown("#### Ingest Synthetic Data")
    st.caption(
        "Load a pre-built set of 25 realistic personal facts into this conversation's memory. "
        "Use this to verify that memory retrieval and tool-use are working correctly."
    )

    already_seeded = has_synthetic_data(conv_id)

    if already_seeded:
        st.info("Synthetic data has already been ingested for this conversation.", icon="✅")
        with st.expander("View synthetic facts", expanded=False):
            for fact in SYNTHETIC_FACTS:
                st.markdown(f"- {fact}")
    else:
        if st.button("Ingest Synthetic Data", type="primary"):
            ingested = []
            for fact in SYNTHETIC_FACTS:
                fact_id = save_memory_fact(conv_id, fact, SYNTHETIC_SEED_KEYWORD)
                add_fact_to_index(fact_id, fact, conv_id)
                ingested.append(fact)
            st.success(f"Ingested {len(ingested)} synthetic facts into memory.")
            with st.expander("Facts added", expanded=True):
                for fact in ingested:
                    st.markdown(f"- {fact}")
            st.rerun()
