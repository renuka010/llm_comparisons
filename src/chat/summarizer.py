from src.config import cfg
from src.chat.conversation import upsert_conversation_summary
from src.db import get_db


def _user_turn_count(conv_id: int) -> int:
    with get_db() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as c FROM messages WHERE conversation_id = ? AND role = 'user' AND model_type = 'user'",
            (conv_id,),
        ).fetchone()
    return row["c"]


def _needs_summarization(conv_id: int) -> bool:
    """Trigger every RECENT_TURN_PAIRS user turns so the summary stays in sync
    with the context window size."""
    count = _user_turn_count(conv_id)
    return count > 0 and count % cfg.RECENT_TURN_PAIRS == 0


def _build_transcript(conv_id: int, model_type: str) -> str:
    """Build a per-model transcript: all user turns interleaved with that
    model's assistant replies only."""
    with get_db() as conn:
        user_rows = conn.execute(
            """SELECT content, created_at FROM messages
               WHERE conversation_id = ? AND role = 'user' AND model_type = 'user'
               ORDER BY created_at ASC""",
            (conv_id,),
        ).fetchall()

        asst_rows = conn.execute(
            """SELECT content, created_at FROM messages
               WHERE conversation_id = ? AND role = 'assistant' AND model_type = ?
               ORDER BY created_at ASC""",
            (conv_id, model_type),
        ).fetchall()

    label = "OSS" if model_type == "oss" else "Frontier"
    all_lines: list[tuple] = []
    for r in user_rows:
        all_lines.append((r["created_at"], f"User: {r['content']}"))
    for r in asst_rows:
        all_lines.append((r["created_at"], f"{label}: {r['content'][:400]}"))

    all_lines.sort(key=lambda x: x[0])
    return "\n".join(line for _, line in all_lines[-40:])


def generate_and_store_summary(conv_id: int, model_type: str, client) -> str | None:
    """
    Summarize the per-model conversation history and persist it.
    OSS history is summarized by the OSS model; frontier by the frontier model.
    """
    transcript = _build_transcript(conv_id, model_type)
    if not transcript.strip():
        return None

    summarize_prompt = (
        "Summarize the following conversation in 3-5 concise sentences. "
        "Capture key decisions, stated preferences, and important context "
        "that would help the assistant pick up the thread later.\n\n"
        f"<conversation>:\n{transcript}\n </conversation>"
    )

    response = client.generate([{"role": "user", "content": summarize_prompt}])
    if response.error or not response.text:
        return None

    summary = response.text.strip()
    upsert_conversation_summary(conv_id, model_type, summary)
    return summary


def maybe_summarize(conv_id: int, model_type: str) -> bool:
    """
    Summarize if RECENT_TURN_PAIRS threshold is hit.
    Uses the OSS client for OSS history and frontier client for frontier history.
    Returns True if a new summary was written.
    """
    if not _needs_summarization(conv_id):
        return False

    try:
        if model_type == "oss":
            from src.models.factory import get_oss_client
            client = get_oss_client()
        else:
            from src.models.factory import get_frontier_client
            client = get_frontier_client()
    except Exception:
        return False

    result = generate_and_store_summary(conv_id, model_type, client)
    return result is not None
