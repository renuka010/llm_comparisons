from datetime import datetime
from src.db import get_db


def create_conversation(title: str = "New Chat") -> int:
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO conversations (title) VALUES (?)", (title,)
        )
        return cur.lastrowid


def list_conversations() -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, title, updated_at FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def rename_conversation(conv_id: int, title: str) -> None:
    with get_db() as conn:
        conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, datetime.utcnow(), conv_id),
        )


def delete_conversation(conv_id: int) -> None:
    from src.chat.memory_store import delete_conversation_index
    with get_db() as conn:
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
        conn.execute("DELETE FROM memory_facts WHERE conversation_id = ?", (conv_id,))
        conn.execute(
            "DELETE FROM conversation_summaries WHERE conversation_id = ?", (conv_id,)
        )
        conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    delete_conversation_index(conv_id)


def save_message(
    conv_id: int,
    model_type: str,
    role: str,
    content: str,
    latency_ms: int | None = None,
    token_estimate: int | None = None,
    error: str | None = None,
) -> int:
    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO messages
               (conversation_id, model_type, role, content, latency_ms, token_estimate, error)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (conv_id, model_type, role, content, latency_ms, token_estimate, error),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (datetime.utcnow(), conv_id),
        )
        return cur.lastrowid


def get_messages(conv_id: int) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            """SELECT * FROM messages WHERE conversation_id = ?
               ORDER BY created_at ASC""",
            (conv_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_conversation_summary(conv_id: int, model_type: str) -> str | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT summary FROM conversation_summaries WHERE conversation_id = ? AND model_type = ?",
            (conv_id, model_type),
        ).fetchone()
    return row["summary"] if row else None


def upsert_conversation_summary(conv_id: int, model_type: str, summary: str) -> None:
    with get_db() as conn:
        conn.execute(
            """INSERT INTO conversation_summaries (conversation_id, model_type, summary, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(conversation_id, model_type) DO UPDATE SET
               summary = excluded.summary, updated_at = excluded.updated_at""",
            (conv_id, model_type, summary, datetime.utcnow()),
        )


def get_recent_turn_pairs(conv_id: int, model_type: str, n: int) -> list[dict]:
    """
    Efficiently fetch only the last N user/assistant pairs for a given model
    directly from DB — no full history load.
    """
    with get_db() as conn:
        user_rows = conn.execute(
            """SELECT content FROM messages
               WHERE conversation_id = ? AND role = 'user' AND model_type = 'user'
               ORDER BY created_at DESC LIMIT ?""",
            (conv_id, n),
        ).fetchall()

        asst_rows = conn.execute(
            """SELECT content FROM messages
               WHERE conversation_id = ? AND role = 'assistant' AND model_type = ?
               ORDER BY created_at DESC LIMIT ?""",
            (conv_id, model_type, n),
        ).fetchall()

    # Both lists are newest-first; zip stops at the shorter one, then reverse
    # to restore chronological order for the prompt.
    pairs: list[dict] = []
    for um, am in zip(reversed(user_rows), reversed(asst_rows)):
        pairs.append({"role": "user", "content": um["content"]})
        pairs.append({"role": "assistant", "content": am["content"]})
    return pairs
