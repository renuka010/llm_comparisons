import json
from pathlib import Path
import streamlit as st
from src.config import cfg
from src.db import get_db


def _conv_index_path(conv_id: int) -> Path:
    return Path(cfg.FAISS_DIR) / f"conv_{conv_id}.index"


def _conv_map_path(conv_id: int) -> Path:
    return Path(cfg.FAISS_DIR) / f"conv_{conv_id}_map.json"


def save_memory_fact(conv_id: int, fact: str, keywords: str = "", importance: int = 1) -> int:
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO memory_facts (conversation_id, fact, keywords, importance) VALUES (?, ?, ?, ?)",
            (conv_id, fact, keywords, importance),
        )
        return cur.lastrowid


def get_memory_facts(conv_id: int) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM memory_facts WHERE conversation_id = ? ORDER BY created_at ASC",
            (conv_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_memory_fact(fact_id: int) -> None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT conversation_id FROM memory_facts WHERE id = ?", (fact_id,)
        ).fetchone()
        conv_id = row["conversation_id"] if row else None
        conn.execute("DELETE FROM memory_facts WHERE id = ?", (fact_id,))
    if conv_id is not None:
        _rebuild_conv_index(conv_id)


def delete_conversation_index(conv_id: int) -> None:
    """Remove the per-conversation FAISS index files when a conversation is deleted."""
    index_path = _conv_index_path(conv_id)
    map_path = _conv_map_path(conv_id)
    if index_path.exists():
        index_path.unlink()
    if map_path.exists():
        map_path.unlink()


def _get_conv_facts_for_index(conv_id: int) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, fact FROM memory_facts WHERE conversation_id = ?", (conv_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def _rebuild_conv_index(conv_id: int) -> int:
    import faiss

    facts = _get_conv_facts_for_index(conv_id)
    index_path = _conv_index_path(conv_id)
    map_path = _conv_map_path(conv_id)

    if not facts:
        if index_path.exists():
            index_path.unlink()
        if map_path.exists():
            map_path.unlink()
        return 0

    embedder = _get_embedder(cfg.EMBEDDING_MODEL)
    texts = [f["fact"] for f in facts]
    vectors = embedder.encode(texts, convert_to_numpy=True).astype("float32")

    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with open(map_path, "w") as f:
        json.dump([{"id": f["id"], "fact": f["fact"]} for f in facts], f)

    return len(facts)


@st.cache_resource(show_spinner="Loading embedding model…")
def _get_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def _get_conv_index_and_map(conv_id: int):
    import faiss

    index_path = _conv_index_path(conv_id)
    map_path = _conv_map_path(conv_id)

    if index_path.exists() and map_path.exists():
        index = faiss.read_index(str(index_path))
        with open(map_path) as f:
            memory_map = json.load(f)
        return index, memory_map

    return None, []


def rebuild_faiss_index(conv_id: int) -> int:
    """Rebuild the FAISS index for a specific conversation from SQLite."""
    return _rebuild_conv_index(conv_id)


def add_fact_to_index(fact_id: int, fact_text: str, conv_id: int) -> None:
    import faiss

    embedder = _get_embedder(cfg.EMBEDDING_MODEL)
    vector = embedder.encode([fact_text], convert_to_numpy=True).astype("float32")

    index_path = _conv_index_path(conv_id)
    map_path = _conv_map_path(conv_id)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    if index_path.exists() and map_path.exists():
        index = faiss.read_index(str(index_path))
        with open(map_path) as f:
            memory_map = json.load(f)
    else:
        dim = vector.shape[1]
        index = faiss.IndexFlatL2(dim)
        memory_map = []

    index.add(vector)
    memory_map.append({"id": fact_id, "fact": fact_text})

    faiss.write_index(index, str(index_path))
    with open(map_path, "w") as f:
        json.dump(memory_map, f)


def has_synthetic_data(conv_id: int) -> bool:
    """Return True if synthetic seed facts have already been ingested for this conversation."""
    from src.data.synthetic_facts import SYNTHETIC_SEED_KEYWORD
    with get_db() as conn:
        row = conn.execute(
            "SELECT 1 FROM memory_facts WHERE conversation_id = ? AND keywords = ? LIMIT 1",
            (conv_id, SYNTHETIC_SEED_KEYWORD),
        ).fetchone()
    return row is not None


def search_memory(query: str, conv_id: int, top_k: int | None = None) -> list[str]:
    k = top_k or cfg.MEMORY_TOP_K
    index, memory_map = _get_conv_index_and_map(conv_id)

    if index is None or not memory_map:
        return []

    embedder = _get_embedder(cfg.EMBEDDING_MODEL)
    vector = embedder.encode([query], convert_to_numpy=True).astype("float32")

    k = min(k, len(memory_map))
    _, indices = index.search(vector, k)

    results = []
    for idx in indices[0]:
        if 0 <= idx < len(memory_map):
            results.append(memory_map[idx]["fact"])
    return results
