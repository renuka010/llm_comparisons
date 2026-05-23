from src.config import cfg
from src.chat.conversation import get_conversation_summary, get_recent_turn_pairs
from src.chat.memory_store import search_memory
from src.prompts import build_context_block, context_to_messages


def build_messages_for_model(conv_id: int, user_message: str, model_type: str) -> list[dict]:
    """
    Assemble the prompt context for a specific model (oss or frontier).

    OSS  — injects top-K FAISS memory facts directly into the context block
           so the small model has the information without needing tool calls.
    Frontier — omits memory injection; the model receives a search_memory
               tool it can invoke on its own via the API tool-use mechanism.
    """
    recent_turns = get_recent_turn_pairs(conv_id, model_type, cfg.RECENT_TURN_PAIRS)
    summary = get_conversation_summary(conv_id, model_type)

    if model_type == "oss":
        memory_facts = search_memory(user_message, conv_id, top_k=cfg.MEMORY_TOP_K)
    else:
        memory_facts = []

    context_block = build_context_block(
        memory_facts=memory_facts,
        recent_turns=recent_turns,
        summary=summary,
        user_message=user_message,
    )

    return context_to_messages(context_block, model_type=model_type)
