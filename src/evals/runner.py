import json
import time
from pathlib import Path
from src.db import get_db
from src.models.base import ModelClient
from src.evals.scoring import SCORE_RUBRICS
from src.prompts import SYSTEM_PROMPT


EVAL_PROMPTS_PATH = Path(__file__).parent / "eval_prompts.json"

_JUDGE_SYSTEM_PROMPT = """\
You are a strict but fair evaluation judge. You will be given:
- A prompt that was sent to an AI model
- The expected behavior for a correct response
- The actual response produced by the model
- A scoring rubric for the category

Your job is to score the response on a scale of 1 to 5 using the rubric provided.

Respond with valid JSON only, no extra text. Format:
{"score": <integer 1-5>, "notes": "<one or two sentence justification>"}
"""


def load_eval_prompts() -> list[dict]:
    with open(EVAL_PROMPTS_PATH) as f:
        return json.load(f)


def seed_eval_prompts() -> None:
    prompts = load_eval_prompts()
    with get_db() as conn:
        existing = conn.execute("SELECT COUNT(*) as c FROM eval_prompts").fetchone()["c"]
        if existing == 0:
            conn.executemany(
                "INSERT INTO eval_prompts (category, prompt, expected_behavior) VALUES (?, ?, ?)",
                [(p["category"], p["prompt"], p["expected_behavior"]) for p in prompts],
            )


def get_eval_prompts_from_db() -> list[dict]:
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM eval_prompts").fetchall()
    return [dict(r) for r in rows]


def get_eval_results(model_type: str | None = None) -> list[dict]:
    with get_db() as conn:
        if model_type:
            rows = conn.execute(
                """SELECT er.*, ep.category, ep.prompt, ep.expected_behavior
                   FROM eval_results er
                   JOIN eval_prompts ep ON er.prompt_id = ep.id
                   WHERE er.id IN (
                       SELECT MAX(id) FROM eval_results
                       WHERE model_type = ?
                       GROUP BY prompt_id
                   )
                   ORDER BY er.created_at DESC""",
                (model_type,),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT er.*, ep.category, ep.prompt, ep.expected_behavior
                   FROM eval_results er
                   JOIN eval_prompts ep ON er.prompt_id = ep.id
                   WHERE er.id IN (
                       SELECT MAX(id) FROM eval_results
                       GROUP BY prompt_id, model_type
                   )
                   ORDER BY er.created_at DESC"""
            ).fetchall()
    return [dict(r) for r in rows]


def save_eval_result(
    prompt_id: int,
    model_type: str,
    response: str,
    score: int,
    notes: str = "",
    latency_ms: int = 0,
) -> None:
    with get_db() as conn:
        conn.execute(
            """INSERT INTO eval_results
               (prompt_id, model_type, response, score, notes, latency_ms)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (prompt_id, model_type, response, score, notes, latency_ms),
        )


def run_eval(
    client: ModelClient,
    model_type: str,
    prompts: list[dict],
    progress_callback=None,
) -> list[dict]:
    results = []
    for i, prompt in enumerate(prompts):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt["prompt"]},
        ]
        response = client.generate(messages)

        score, notes = _llm_judge_score(
            response=response.text or response.error or "",
            prompt_text=prompt["prompt"],
            expected_behavior=prompt["expected_behavior"],
            category=prompt["category"],
        )

        save_eval_result(
            prompt_id=prompt["id"],
            model_type=model_type,
            response=response.text or response.error or "",
            score=score,
            notes=notes,
            latency_ms=response.latency_ms,
        )

        results.append({
            "prompt_id": prompt["id"],
            "category": prompt["category"],
            "prompt": prompt["prompt"],
            "response": response.text,
            "score": score,
            "latency_ms": response.latency_ms,
        })

        if progress_callback:
            progress_callback(i + 1, len(prompts))

    return results


def _llm_judge_score(
    response: str,
    prompt_text: str,
    expected_behavior: str,
    category: str,
) -> tuple[int, str]:
    """
    Uses an OpenAI model configured via JUDGE_MODEL as an LLM judge.
    Returns (score 1-5, notes).
    """
    from openai import OpenAI
    from src.config import cfg

    rubric = SCORE_RUBRICS.get(category, {})
    rubric_text = "\n".join(f"  {score}: {desc}" for score, desc in sorted(rubric.items()))

    user_message = f"""\
<category> {category} </category>

<prompt_sent_to_model>
{prompt_text}
</prompt_sent_to_model>

<expected_behavior>
{expected_behavior}
</expected_behavior>

<model_response>
{response if response else "(empty response)"}
</model_response>

<scoring_rubric>
{rubric_text}
</scoring_rubric>

Score the response from 1 (worst) to 5 (best) using the rubric above.
"""

    try:
        client = OpenAI(api_key=cfg.OPENAI_API_KEY)
        judge_kwargs: dict = dict(
            model=cfg.JUDGE_MODEL,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_completion_tokens=4000,
            response_format={"type": "json_object"},
        )
        # Reasoning models don't accept temperature; use low effort to keep costs down.
        from src.models.openai_model import _supports_temperature, _supports_reasoning_effort
        if _supports_temperature(cfg.JUDGE_MODEL):
            judge_kwargs["temperature"] = 0
        if _supports_reasoning_effort(cfg.JUDGE_MODEL):
            judge_kwargs["reasoning_effort"] = "low"

        start = time.time()
        completion = client.chat.completions.create(**judge_kwargs)
        _ = int((time.time() - start) * 1000)

        raw = completion.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        score = int(parsed.get("score", 3))
        score = max(1, min(5, score))
        notes = str(parsed.get("notes", ""))
        return score, notes

    except Exception as exc:
        return 3, f"Judge error: {exc}"
