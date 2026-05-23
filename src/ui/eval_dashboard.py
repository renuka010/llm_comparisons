import streamlit as st
from src.evals.runner import (
    seed_eval_prompts,
    get_eval_prompts_from_db,
    get_eval_results,
    run_eval,
    save_eval_result,
)
from src.evals.scoring import compute_failure_rates, SCORE_LABELS, SCORE_RUBRICS
from src.evals.charts import failure_rate_bar, latency_comparison_bar, results_to_dataframe
from src.models.factory import get_oss_client, get_frontier_client
from src.config import cfg


def render():
    st.markdown("## Evaluations")
    st.caption("Curated prompts testing hallucination, bias, and safety across both models.")

    seed_eval_prompts()

    oss_results = get_eval_results("oss")
    frontier_results = get_eval_results("frontier")

    tab_results, tab_run, tab_rubric = st.tabs(["Results", "Run Eval", "Scoring Rubric"])

    with tab_results:
        if not oss_results and not frontier_results:
            st.info("No evaluation results yet. Go to 'Run Eval' to generate them.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(failure_rate_bar(oss_results, "oss"), use_container_width=True)
            with col2:
                st.plotly_chart(failure_rate_bar(frontier_results, "frontier"), use_container_width=True)

            st.plotly_chart(latency_comparison_bar(oss_results, frontier_results), use_container_width=True)

            st.markdown("#### OSS Results")
            df_oss = results_to_dataframe(oss_results)
            if not df_oss.empty:
                st.dataframe(df_oss, use_container_width=True, hide_index=True)

            st.markdown("#### Frontier Results")
            df_frontier = results_to_dataframe(frontier_results)
            if not df_frontier.empty:
                st.dataframe(df_frontier, use_container_width=True, hide_index=True)

    with tab_run:
        prompts = get_eval_prompts_from_db()
        st.markdown(f"**{len(prompts)} prompts loaded** across hallucination, bias, and safety.")
        st.caption(f"Responses are scored automatically by the LLM judge (`{cfg.JUDGE_MODEL}`) on a 1–5 scale.")

        categories = list(set(p["category"] for p in prompts))
        selected_cats = st.multiselect("Filter categories", categories, default=categories)
        filtered = [p for p in prompts if p["category"] in selected_cats]

        col_oss, col_frontier = st.columns(2)

        with col_oss:
            st.markdown("**Run on OSS model**")
            if st.button("Run OSS Eval", use_container_width=True):
                progress = st.progress(0)
                client = get_oss_client()

                def update_oss(done, total):
                    progress.progress(done / total)

                run_eval(client, "oss", filtered, progress_callback=update_oss)
                st.success("OSS evaluation complete.")
                st.rerun()

        with col_frontier:
            st.markdown("**Run on Frontier model**")
            if st.button("Run Frontier Eval", use_container_width=True):
                progress = st.progress(0)
                client = get_frontier_client()

                def update_frontier(done, total):
                    progress.progress(done / total)

                run_eval(client, "frontier", filtered, progress_callback=update_frontier)
                st.success("Frontier evaluation complete.")
                st.rerun()

        st.divider()
        st.markdown("#### Manual Score Override")
        st.caption("Select a result row from the Results tab and override the score here.")
        prompt_labels = {f"[{p['id']}] {p['prompt'][:60]}": p for p in prompts}
        selected_prompt = st.selectbox("Prompt", list(prompt_labels.keys()))
        model_type = st.selectbox("Model", ["oss", "frontier"])
        score = st.selectbox("Score", [1, 2, 3, 4, 5], format_func=lambda s: f"{s} — {SCORE_LABELS[s]}")
        notes = st.text_input("Notes")
        if st.button("Save manual score"):
            prompt = prompt_labels[selected_prompt]
            save_eval_result(prompt["id"], model_type, "manual override", score, notes)
            st.success("Score saved.")

    with tab_rubric:
        st.caption(f"Scores are assigned by the LLM judge (`{cfg.JUDGE_MODEL}`) on a **1–5 scale** — 1 is worst, 5 is best.")
        for category, rubric in SCORE_RUBRICS.items():
            st.markdown(f"**{category.capitalize()}**")
            for score, desc in sorted(rubric.items()):
                st.markdown(f"- `{score}` — {desc}")
            st.markdown("")
