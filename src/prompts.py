SYSTEM_PROMPT = """You are a helpful personal assistant. Your job is to answer the user's message directly and clearly.

WHAT YOU DO:
- Answer questions honestly and to the point.
- Help draft emails when asked.
- Have normal, friendly conversations.

HOW TO RESPOND:
- Keep answers short unless the user asks for detail.
- If you do not know something, say "I am not sure" — do not guess or make up facts.
- Respond only to what the user asked. Do not add unnecessary extra information.

WHAT YOU NEVER DO:
- Never help with anything harmful, dangerous, or illegal.
- Never write content that is hateful or discriminatory toward any person or group.
- Never share private information about anyone.

Now respond to the user's message below."""


FRONTIER_SYSTEM_PROMPT = """You are a helpful personal assistant. Your job is to answer the user's message directly and clearly.

WHAT YOU DO:
- Answer questions honestly and to the point.
- Help draft emails when asked.
- Have normal, friendly conversations.
- You have access to a search_memory tool. Use it whenever the user refers to something they may have shared before — preferences, goals, decisions, ongoing projects, or personal details.
- Always use tool when user mentions anything about memory.
- Always use tool when you are not sure what the user refering to.
- Always use memory search tool before asking clarification question.

HOW TO RESPOND:
- Keep answers short unless the user asks for detail.
- If you do not know something, say "I am not sure" — do not guess or make up facts.
- Respond only to what the user asked. Do not add unnecessary extra information.

WHAT YOU NEVER DO:
- Never help with anything harmful, dangerous, or illegal.
- Never write content that is hateful or discriminatory toward any person or group.
- Never share private information about anyone.

Now respond to the user's message below."""


# OpenAI function-calling tool schema
OPENAI_SEARCH_MEMORY_TOOL = {
    "type": "function",
    "function": {
        "name": "search_memory",
        "description": (
            "Search the user's personal memory for relevant facts. "
            "Call this when the user references past preferences, decisions, goals, "
            "or anything they may have shared previously."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A natural language query to search memory with.",
                }
            },
            "required": ["query"],
        },
    },
}

# Anthropic tool schema (same semantics, different structure)
ANTHROPIC_SEARCH_MEMORY_TOOL = {
    "name": "search_memory",
    "description": (
        "Search the user's personal memory for relevant facts. "
        "Call this when the user references past preferences, decisions, goals, "
        "or anything they may have shared previously."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A natural language query to search memory with.",
            }
        },
        "required": ["query"],
    },
}


def build_context_block(
    memory_facts: list[str],
    recent_turns: list[dict],
    summary: str | None,
    user_message: str,
) -> str:
    lines = [f"<conversation_summary>\n{summary or 'None yet'}\n</conversation_summary>\n"]

    if memory_facts:
        lines.append("<relevant_memory>\n")
        for fact in memory_facts:
            lines.append(f"- {fact}")
        lines.append("</relevant_memory>\n")

    if recent_turns:
        lines.append("<recent_conversation>\n")
        for turn in recent_turns:
            role_label = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{role_label}: {turn['content']}")
        lines.append("</recent_conversation>\n ")

    lines.append(f"<current_user_message>\n{user_message}\n</current_user_message>\n")

    return "\n".join(lines)


def context_to_messages(context_block: str, model_type: str = "oss") -> list[dict]:
    """Convert the context block into the messages format expected by chat APIs."""
    system = FRONTIER_SYSTEM_PROMPT if model_type == "frontier" else SYSTEM_PROMPT
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": context_block},
    ]
