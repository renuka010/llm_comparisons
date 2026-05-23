"""
CLI script to run evaluations outside of the Streamlit UI.

Usage:
    python -m scripts.run_evals --model oss
    python -m scripts.run_evals --model frontier
    python -m scripts.run_evals --model both
    python -m scripts.run_evals --model both --category safety
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db import init_db
from src.evals.runner import seed_eval_prompts, get_eval_prompts_from_db, run_eval
from src.evals.scoring import compute_failure_rates
from src.models.factory import get_oss_client, get_frontier_client


def print_summary(results: list[dict], model_type: str) -> None:
    stats = compute_failure_rates(results)
    print(f"\n{'=' * 50}")
    print(f"Results for: {model_type.upper()}")
    print(f"{'=' * 50}")
    for cat, s in stats.items():
        label = cat.capitalize()
        print(f"  {label}: failure={s['failure_rate'] * 100:.1f}%  severe={s['severe_failure_rate'] * 100:.1f}%  (n={s['total']})")
    print()


def main():
    parser = argparse.ArgumentParser(description="Run evaluations from CLI")
    parser.add_argument("--model", choices=["oss", "frontier", "both"], default="both")
    parser.add_argument("--category", choices=["hallucination", "bias", "safety", "all"], default="all")
    args = parser.parse_args()

    init_db()
    seed_eval_prompts()

    prompts = get_eval_prompts_from_db()
    if args.category != "all":
        prompts = [p for p in prompts if p["category"] == args.category]

    print(f"Running {len(prompts)} prompts | model={args.model} | judge=llm")

    def progress(done, total):
        bar = "#" * int(done / total * 30)
        print(f"\r  [{bar:<30}] {done}/{total}", end="", flush=True)

    if args.model in ("oss", "both"):
        print("\nLoading OSS model…")
        client = get_oss_client()
        results = run_eval(client, "oss", prompts, progress_callback=progress)
        print()
        print_summary(results, "oss")

    if args.model in ("frontier", "both"):
        print("\nQuerying frontier model…")
        client = get_frontier_client()
        results = run_eval(client, "frontier", prompts, progress_callback=progress)
        print()
        print_summary(results, "frontier")

    print("Done. Results saved to SQLite.")


if __name__ == "__main__":
    main()
