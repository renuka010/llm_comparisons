import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from src.evals.scoring import compute_failure_rates


def failure_rate_bar(results: list[dict], model_type: str) -> go.Figure:
    if not results:
        fig = go.Figure()
        fig.update_layout(title=f"No results yet for {model_type}")
        return fig

    stats = compute_failure_rates(results)
    categories = [k for k in stats if k != "overall"]

    fig = go.Figure(data=[
        go.Bar(
            name="Failure rate",
            x=categories,
            y=[stats[c]["failure_rate"] * 100 for c in categories],
            marker_color=["#e74c3c", "#f39c12", "#9b59b6"],
        ),
        go.Bar(
            name="Severe failure rate",
            x=categories,
            y=[stats[c]["severe_failure_rate"] * 100 for c in categories],
            marker_color=["#c0392b", "#d68910", "#76448a"],
        ),
    ])
    fig.update_layout(
        barmode="group",
        title=f"Failure Rates — {model_type.upper()}",
        yaxis_title="Rate (%)",
        xaxis_title="Category",
        legend=dict(orientation="h"),
        height=350,
    )
    return fig


def latency_comparison_bar(oss_results: list[dict], frontier_results: list[dict]) -> go.Figure:
    def avg_latency(results):
        if not results:
            return 0
        return sum(r.get("latency_ms", 0) for r in results) / len(results)

    fig = go.Figure(data=[
        go.Bar(name="OSS", x=["Avg Latency (ms)"], y=[avg_latency(oss_results)], marker_color="#3498db"),
        go.Bar(name="Frontier", x=["Avg Latency (ms)"], y=[avg_latency(frontier_results)], marker_color="#2ecc71"),
    ])
    fig.update_layout(
        barmode="group",
        title="Average Latency Comparison",
        yaxis_title="Milliseconds",
        height=300,
    )
    return fig


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    cols = ["category", "prompt", "response", "score", "notes", "latency_ms"]
    return df[[c for c in cols if c in df.columns]]
