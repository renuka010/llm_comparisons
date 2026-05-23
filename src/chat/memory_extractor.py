import json
import re

TRIGGER_PHRASES = [
    r"\bI prefer\b",
    r"\bRemember\b",
    r"\bI chose\b",
    r"\bI decided\b",
    r"\bMy goal is\b",
    r"\bI want to\b",
    r"\bWe are going to\b",
    r"\bI am building\b",
    r"\bI need to\b",
    r"\bI'm working on\b",
]

_PATTERNS = [re.compile(p, re.IGNORECASE) for p in TRIGGER_PHRASES]

_EXTRACTION_SYSTEM_PROMPT = """You are a memory extraction assistant. \
Your job is to identify personal facts worth remembering from a user's message.

Think step-by-step about whether the message reveals anything memorable about the user: \
preferences, goals, decisions, personal information, ongoing projects, dietary habits, \
life circumstances, etc.

Respond ONLY with a valid JSON object in this exact format:
{
  "reasoning": "<your step-by-step thinking>",
  "facts": ["<fact 1>", "<fact 2>"]
}

Rules:
- "facts" must be a JSON array of strings (can be empty []).
- Each fact should be a short, self-contained sentence.
- Do NOT include trivial or question-only messages as facts.
- Do NOT invent facts not implied by the message."""


def extract_facts(user_message: str) -> list[str]:
    """
    Rule-based extraction used for the OSS model.
    Returns the message (or matching sentences) when a trigger phrase is found.
    """
    if any(p.search(user_message) for p in _PATTERNS):
        cleaned = user_message.strip()
        if len(cleaned) < 300:
            return [cleaned]
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        facts = [s.strip() for s in sentences if any(p.search(s) for p in _PATTERNS)]
        return facts[:3]
    return []


def extract_facts_llm(user_message: str) -> list[str]:
    """
    LLM-based extraction used for the frontier model.
    Makes a separate call to JUDGE_MODEL with a chain-of-thought reasoning step
    before emitting structured facts. Returns an empty list on any failure.
    """
    from openai import OpenAI
    from src.config import cfg

    if not cfg.OPENAI_API_KEY:
        return []

    client = OpenAI(api_key=cfg.OPENAI_API_KEY)

    try:
        response = client.chat.completions.create(
            model=cfg.JUDGE_MODEL,
            messages=[
                {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": f"User message:\n{user_message}"},
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=512,
            temperature=0,
        )

        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        facts = data.get("facts", [])
        return [f for f in facts if isinstance(f, str) and f.strip()]

    except Exception:
        return []


def extract_keywords(fact: str) -> str:
    stop_words = {"i", "a", "the", "is", "to", "and", "of", "in", "it", "that", "this"}
    words = re.findall(r"\b\w+\b", fact.lower())
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    return ", ".join(list(dict.fromkeys(keywords))[:8])
