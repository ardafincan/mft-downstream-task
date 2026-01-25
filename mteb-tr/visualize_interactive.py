#!/usr/bin/env python3
"""
Interactive visualizer for MTEB-TR results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd

try:
    import gradio as gr
    import plotly.express as px
except ImportError as exc:
    raise SystemExit(
        "Missing dependencies. Install with: pip install -e '.[leaderboard]'"
    ) from exc


DEFAULT_RESULTS_DIR = "results"
NON_TASK_COLUMNS = {"Model", "Overall"}


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include="number").columns.tolist()


def _task_columns(df: pd.DataFrame) -> List[str]:
    numeric_cols = _numeric_columns(df)
    return [col for col in numeric_cols if col not in NON_TASK_COLUMNS]


def _safe_read_results(path: str) -> Tuple[pd.DataFrame, str]:
    resolved = Path(path).expanduser()
    if not resolved.exists():
        return pd.DataFrame(), f"Results directory not found: {resolved}"
    if not resolved.is_dir():
        return pd.DataFrame(), f"Results path is not a directory: {resolved}"

    results = {}
    for model_dir in resolved.iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name.replace("__", "/")
        for revision_dir in model_dir.iterdir():
            if not revision_dir.is_dir():
                continue
            revision = revision_dir.name
            display_model = f"{model_name} [{revision}]"
            for json_file in revision_dir.glob("*.json"):
                if json_file.name == "model_meta.json":
                    continue
                try:
                    with json_file.open("r", encoding="utf-8") as handle:
                        task_result = json.load(handle)
                except Exception:
                    continue
                if "scores" not in task_result or "test" not in task_result["scores"]:
                    continue
                test_scores = task_result["scores"]["test"]
                if not test_scores:
                    continue
                main_score = test_scores[0].get("main_score")
                if main_score is None:
                    continue
                if display_model not in results:
                    results[display_model] = {}
                results[display_model][json_file.stem] = main_score

    if not results:
        return pd.DataFrame(), f"No results found in {resolved}"

    df = pd.DataFrame(results).T
    df.index.name = "Model"
    df = df.reset_index()
    return df, f"Loaded {len(df)} runs from {resolved}"


def load_results(path: str):
    df, status = _safe_read_results(path)
    if df.empty:
        return (
            df,
            status,
            gr.Dropdown(choices=[], value=[]),
            gr.Dropdown(choices=[], value=[]),
            gr.Dropdown(choices=["None", "Overall (desc)", "Overall (asc)"], value="None"),
            pd.DataFrame(),
            None,
            None,
        )

    model_choices = sorted(df["Model"].dropna().unique().tolist())
    task_choices = _task_columns(df)
    default_tasks = task_choices[: min(8, len(task_choices))]
    status += f" | {len(model_choices)} models, {len(task_choices)} task columns."

    return (
        df,
        status,
        gr.Dropdown(choices=model_choices, value=model_choices),
        gr.Dropdown(choices=task_choices, value=default_tasks),
        gr.Dropdown(choices=["None", "Overall (desc)", "Overall (asc)"], value="Overall (desc)"),
        df,
        None,
        None,
    )


def _filter_df(
    df: pd.DataFrame,
    models: Optional[Iterable[str]],
    tasks: Optional[Iterable[str]],
) -> pd.DataFrame:
    filtered = df.copy()
    if models:
        filtered = filtered[filtered["Model"].isin(models)]
    if tasks:
        keep_cols = ["Model"] + [t for t in tasks if t in filtered.columns]
        filtered = filtered[keep_cols]
    return filtered


def _aggregate_scores(df: pd.DataFrame, tasks: List[str], agg: str) -> pd.Series:
    if not tasks:
        return pd.Series(dtype="float64")
    data = df[tasks]
    if agg == "Median":
        return data.median(axis=1)
    if agg == "Max":
        return data.max(axis=1)
    return data.mean(axis=1)


def _overall_scores(df: pd.DataFrame, tasks: List[str]) -> pd.Series:
    if not tasks:
        return pd.Series(dtype="float64")
    return df[tasks].mean(axis=1)


def update_plots(
    df: pd.DataFrame,
    models: List[str],
    tasks: List[str],
    agg: str,
    sort_mode: str,
):
    if df is None or df.empty:
        return pd.DataFrame(), None, None

    models = models or []
    tasks = tasks or []
    filtered = _filter_df(df, models, tasks)
    if filtered.empty:
        return pd.DataFrame(), None, None

    task_cols = [col for col in tasks if col in filtered.columns]
    if not task_cols:
        task_cols = _task_columns(filtered)

    overall = _overall_scores(filtered, task_cols).rename("Overall")
    table = filtered.copy()
    table.insert(1, "Overall", overall)

    if sort_mode == "Overall (desc)":
        table = table.sort_values("Overall", ascending=False)
    elif sort_mode == "Overall (asc)":
        table = table.sort_values("Overall", ascending=True)

    if task_cols:
        heatmap_df = table.set_index("Model")[task_cols]
        heatmap_fig = px.imshow(
            heatmap_df,
            aspect="auto",
            color_continuous_scale="RdBu",
            labels=dict(x="Task", y="Model", color="Score"),
        )
        heatmap_fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        agg_scores = _aggregate_scores(table, task_cols, agg)
        bar_df = pd.DataFrame({"Model": table["Model"], "Score": agg_scores})
        bar_fig = px.bar(
            bar_df.sort_values("Score", ascending=False),
            x="Score",
            y="Model",
            orientation="h",
            title=f"{agg} score across selected tasks",
        )
        bar_fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    else:
        heatmap_fig = px.imshow([[0]], aspect="auto", labels=dict(color="Score"))
        bar_fig = px.bar(pd.DataFrame({"Model": [], "Score": []}), x="Score", y="Model")

    return table, heatmap_fig, bar_fig


def build_app():
    with gr.Blocks(title="MTEB-TR Results Visualizer") as demo:
        gr.Markdown(
            "# MTEB-TR Results Visualizer\n"
            "Load local results and explore interactive tables and plots."
        )

        df_state = gr.State(pd.DataFrame())

        with gr.Row():
            results_path = gr.Textbox(label="Results Directory", value=DEFAULT_RESULTS_DIR)
            load_btn = gr.Button("Load")

        status = gr.Markdown("Ready.")

        with gr.Row():
            model_filter = gr.Dropdown(
                label="Models",
                multiselect=True,
                choices=[],
                value=[],
            )
            task_filter = gr.Dropdown(
                label="Tasks (numeric columns)",
                multiselect=True,
                choices=[],
                value=[],
            )
            agg = gr.Dropdown(
                label="Aggregate",
                choices=["Mean", "Median", "Max"],
                value="Mean",
            )
            sort_mode = gr.Dropdown(
                label="Sort table by Overall",
                choices=["None", "Overall (desc)", "Overall (asc)"],
                value="Overall (desc)",
            )

        table = gr.Dataframe(label="Filtered Table", interactive=False)
        heatmap = gr.Plot(label="Task Heatmap")
        bars = gr.Plot(label="Aggregate Comparison")

        load_btn.click(
            load_results,
            inputs=results_path,
            outputs=[
                df_state,
                status,
                model_filter,
                task_filter,
                sort_mode,
                table,
                heatmap,
                bars,
            ],
        )

        for widget in (model_filter, task_filter, agg, sort_mode):
            widget.change(
                update_plots,
                inputs=[df_state, model_filter, task_filter, agg, sort_mode],
                outputs=[table, heatmap, bars],
            )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
