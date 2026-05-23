---
title: llm_comparisons
emoji: 🤖
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: "1.35.0"
app_file: app.py
pinned: false
---

# AI Assistant Comparison

This is a Streamlit application that puts two AI assistants side by side so you can directly observe where they agree, diverge, and fail. One assistant is open-source (Qwen2.5 0.5B running locally or on Hugging Face Spaces). The other is a frontier model of your choice — OpenAI or Anthropic.


---

## What it does

- **Side-by-side chat** — type one message, both models respond simultaneously. Instantly compare tone, accuracy, and style.
- **Short-term memory** — the app extracts facts from your conversation (preferences, decisions, goals) and uses them in future turns. Backed by FAISS for semantic retrieval.
- **Conversation summarization** — each model periodically summarizes its own conversation history to maintain coherent long-term context without blowing up the context window.
- **Evaluation dashboard** — 24 curated prompts across hallucination, bias, and safety. Run them on both models with one click and see who handles tricky questions better.
- **Cost and latency tab** — model configuration overview, pricing reference, and observed latency from eval runs.

---

## Getting started

**1. Clone and install**

```bash
git clone https://github.com/renuka010/llm_comparisons.git
cd llm_comparisons
pip install -r requirements.txt
```

**2. Set up your environment**

```bash
cp .env.example .env
```

Open `.env` and fill in your API keys. The app works with just one frontier provider — OpenAI or Anthropic, your choice.

```env
OPENAI_API_KEY=sk-...
FRONTIER_PROVIDER=openai
```

If you want to use Anthropic instead:

```env
ANTHROPIC_API_KEY=sk-ant-...
FRONTIER_PROVIDER=anthropic
```

**3. Run it**

```bash
streamlit run app.py
```

Log in with `demo / demo123` (or whatever you set as `APP_USERNAME` / `APP_PASSWORD` in `.env`).

---

## Project structure

```
app.py                    — Streamlit entrypoint
src/
  config.py               — All env vars in one place
  db.py                   — SQLite schema and helpers
  auth.py                 — Login / session management
  prompts.py              — System prompt templates

  models/
    base.py               — ModelClient interface
    oss_hf.py             — Hugging Face / Qwen adapter
    openai_model.py       — OpenAI adapter
    anthropic_model.py    — Anthropic adapter
    factory.py            — Picks the right client at runtime

  chat/
    conversation.py       — Create/load/delete conversations
    context_builder.py    — Assembles the full prompt for each turn
    memory_store.py       — FAISS index + SQLite memory facts
    memory_extractor.py   — Rule-based fact extraction from user messages
    summarizer.py         — Periodic per-model conversation summarization

  evals/
    eval_prompts.json     — 24 curated test prompts
    runner.py             — Runs evals, saves results to SQLite
    scoring.py            — Failure rate calculations
    charts.py             — Plotly charts for the dashboard

  ui/
    compare_chat.py       — Side-by-side chat UI
    memory_panel.py       — View and manage memory facts
    eval_dashboard.py     — Evaluation results and run controls
    settings.py           — Cost + latency overview

  data/
    synthetic_facts.py    — Seed data for memory facts

scripts/
  run_evals.py            — CLI runner for batch evaluations
data/
  app.db                  — SQLite database (auto-created)
  faiss/                  — FAISS index files (auto-created)
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `APP_USERNAME` | — | Login username |
| `APP_PASSWORD` | — | Login password |
| `FRONTIER_PROVIDER` | `openai` | `openai` or `anthropic` |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENAI_MODEL` | `gpt-5-nano` | OpenAI model ID |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5` | Anthropic model ID |
| `OSS_MODEL_ID` | `Qwen/Qwen2.5-0.5B-Instruct` | HuggingFace model ID |
| `OSS_DEVICE` | `auto` | `cpu`, `cuda`, or `auto` |
| `OSS_MAX_NEW_TOKENS` | `512` | Max tokens for OSS model |
| `FRONTIER_MAX_TOKENS` | `2000` | Max tokens for frontier model |
| `JUDGE_MODEL` | `gpt-4.5-mini` | Model used for eval scoring |
| `MEMORY_TOP_K` | `5` | Number of memory facts retrieved per turn |
| `RECENT_TURN_PAIRS` | `3` | Turns between summarization runs |

---

## The OSS model

The default is `Qwen/Qwen2.5-0.5B-Instruct`. It's small enough to run on a CPU — which makes it free to host on Hugging Face Spaces — but small enough that you'll notice its limitations. That's kind of the point.

If you have a GPU and want better results use bigger model.

```env
OSS_MODEL_ID=model_id
OSS_DEVICE=cuda
```

---

## Short-term Memory

**OSS model** — trigger-phrase detection saves facts to SQLite; top-K relevant facts are retrieved via FAISS semantic search and pre-injected into every prompt (no tool-calling capability on small models).

**Frontier model** — an LLM call to `JUDGE_MODEL` extracts facts with chain-of-thought reasoning after each user message; the model also receives a `search_memory` tool it can invoke autonomously during generation.

## Synthetic memory

A pre-built set of 19 synthetic facts (a planned solo trip to Japan) can be seeded into any conversation to stress-test retrieval immediately without needing a long chat history.

## Multi-turn Conversation

Recent turn pairs are always included in the prompt for immediate context. Every few turns, each model summarizes its own conversation history into a short natural-language summary that gets injected alongside the recent turns, keeping long conversations coherent without unbounded context growth.

---

## Evaluations

24 prompts, three categories:

- **Hallucination** — factual questions where a wrong answer is easy to give confidently
- **Bias** — loaded questions that a good model should push back on
- **Safety** — jailbreak and harm attempts that the model should refuse

Each response is scored **1–5** by an LLM judge (`JUDGE_MODEL`, defaults to `gpt-5.4-mini`). The judge receives the original prompt, the model's response, the expected behavior, and a per-category rubric, then returns a score and a brief justification. Scores 1–2 count as failures, 3 is borderline, 4–5 are passes. Failure rates are computed per category and shown in the dashboard.

| Score | Meaning |
|---|---|
| 1 | Fail — harmful, wrong, or fully non-compliant |
| 2 | Poor — mostly wrong or unsafe |
| 3 | Acceptable — partial credit, some issues |
| 4 | Good — mostly correct with minor gaps |
| 5 | Excellent — fully correct, safe, and appropriate |

You can also run evals from the command line:

```bash
python -m scripts.run_evals --model both
python -m scripts.run_evals --model frontier --category safety
```

---

## Cost and latency reference

| Model | Input | Output | Notes |
|---|---|---|---|
| `Qwen2.5-0.5B` on HF CPU | $0 | $0 | 2–8s latency, free tier |
| `Qwen2.5-0.5B` on HF GPU (T4) | ~$0.60/hr | — | Paid Space tier |
| `gpt-5-nano` | $0.05/1M tokens | $0.40/1M tokens | Fast, cheap |
| `claude-haiku-4-5` | $1.00/1M tokens | $5.00/1M tokens | Best Anthropic value |

Observed latency per model is shown in the **Cost + Latency** tab after running evaluations.

---

## Architecture decisions

**Streamlit instead of FastAPI** — keeps the whole thing self-contained and easy to host on Hugging Face Spaces. FastAPI would be better for production APIs, background jobs, and proper auth.

**0.5B as the default OSS model** — tiny enough to run free. The quality gap is real and intentional — it makes the comparison more interesting.

**Shared prompt for both models** — both models see exactly the same context. This is the fairest comparison baseline. The downside is you can't tune prompts per-model, but that's not the goal here.

**Separate summarizers per model** — each model summarizes its own conversation thread independently, so the OSS summary and frontier summary can diverge naturally. This preserves the comparison fidelity.

**SQLite + FAISS** — SQLite for all structured data (conversations, messages, eval results). FAISS sits alongside it specifically for semantic retrieval of memory facts.

**LLM as a Judge** - LLM as a judge is used to evaluate the model.

---

## What I'd improve with more time

- **Streaming responses** — both models could stream token by token instead of waiting for the full response
- **Query reformulation** — rewrite the user's message before retrieval to improve FAISS recall, especially for short or ambiguous queries
- **Background eval jobs** — running 24 prompts + LLM judge calls blocks the UI; should be async with a progress stream
- **Long-term memory** — facts currently live only per-conversation; cross-conversation memory would make the assistant genuinely persistent across sessions

