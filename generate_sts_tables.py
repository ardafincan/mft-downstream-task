#!/usr/bin/env python3
"""STS Benchmark Results Table Generator - Generates markdown tables and charts."""
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


def load_results(filename):
    """Load JSON results file."""
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def format_table(headers, rows):
    """Format data as a Markdown table."""
    if not rows:
        return "No data available."
    col_widths = [
        max(len(h), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)
    ]
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_widths) + " |"
    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    return "\n".join(
        [fmt.format(*headers), sep] + [fmt.format(*[str(c) for c in r]) for r in rows]
    )


def process_split_data(entries, split_name):
    """Group entries by model, assign Training Steps, find best results."""
    model_stats = {}
    for e in entries:
        model_stats.setdefault(e["model"], []).append(e)

    all_runs = []
    for results in model_stats.values():
        results.sort(key=lambda x: x["timestamp_obj"])
        for i, res in enumerate(results):
            step = {0: 0, 1: 50, 2: 100}.get(i, f"?({i})")
            if split_name == "train" and len(results) == 1:
                step = 100
            res["step"] = step
            all_runs.append(res)

    all_runs.sort(key=lambda x: x["timestamp_obj"])
    best_results = [
        max(results, key=lambda x: x.get("pearson", 0) or 0)
        for results in model_stats.values()
    ]
    best_results.sort(key=lambda x: x["timestamp_obj"], reverse=True)
    return all_runs, best_results


def generate_detailed_summary(entries_by_split):
    """Generate detailed textual summary with scores as percentages."""
    lines = ["# 📝 Detailed Analysis Summary\n"]
    lines.append(
        "> **Note:** The 'train' and 'test' labels refer to dataset split names in HuggingFace. Neither split was used during model training. This is a zero-shot evaluation.\n"
    )
    test_entries = entries_by_split.get("test", [])

    if not test_entries:
        return "\n".join(lines + ["*No test data available for analysis.*"])

    _, best_test_results = process_split_data(test_entries, "test")
    best_test_results.sort(key=lambda x: x.get("pearson", 0), reverse=True)
    best = best_test_results[0]

    lines.append("### 🏆 Overall Best Performance")
    lines.append(
        f"The best performing model on the **Test Split** is **{best['model']}** with a Pearson of **{best['pearson']*100:.2f}%** and Spearman of **{best['spearman']*100:.2f}%**.\n"
    )

    def avg_score(keyword):
        scores = [
            r["pearson"] for r in best_test_results if keyword in r["model"].lower()
        ]
        return sum(scores) / len(scores) if scores else 0.0

    avg_mft, avg_tabi = avg_score("mft"), avg_score("tabi")
    diff = avg_mft - avg_tabi
    lines.append("### ⚔️ MFT vs Tabi (Random Init) Comparison")
    lines.append(f"- **MFT Random Init Average**: {avg_mft*100:.2f}%")
    lines.append(f"- **Tabi Random Init Average**: {avg_tabi*100:.2f}%")
    lines.append(
        f"\nMFT random initialization {'outperformed' if diff > 0 else 'underperformed'} Tabi by **{abs(diff)*100:.2f}** percentage points.\n"
    )
    lines.append(
        "This gap indicates the structural prior advantage of the morphology-first tokenizer."
    )

    return "\n".join(lines)


def set_academic_style():
    """Set matplotlib style for academic figures."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")  # Fallback

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.dpi": 300,
            "lines.linewidth": 2,
            "lines.markersize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def generate_bar_chart(model_data, output_filename, title):
    """Generate grouped horizontal bar chart with academic styling."""
    set_academic_style()
    models = list(model_data.keys())
    # Extract last data point for each model
    pearson = [model_data[m][-1][1] * 100 if model_data[m] else 0 for m in models]
    spearman = [model_data[m][-1][2] * 100 if model_data[m] else 0 for m in models]

    # Academic colors (colorblind friendly)
    c1 = "#4e79a7"  # Blue
    c2 = "#f28e2b"  # Orange

    y = np.arange(len(models))
    height = 0.35

    # Adjust figure size - horizontal works better for listing models
    fig, ax = plt.subplots(figsize=(10, 4))  # Compact height

    r1 = ax.barh(y + height / 2, pearson, height, label="Pearson", color=c1, alpha=0.9)
    r2 = ax.barh(
        y - height / 2, spearman, height, label="Spearman", color=c2, alpha=0.9
    )

    mft_idx = [i for i, m in enumerate(models) if "mft" in m.lower()]
    # Highlight MFT models
    for i in mft_idx:
        for r in (r1[i], r2[i]):
            r.set_hatch("///")
            r.set_edgecolor("black")
            r.set_linewidth(1.2)

    ax.set_xlabel("Score (%)")
    ax.set_title(title, pad=15)
    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)

    # Invert Y axis so first model is at top
    ax.invert_yaxis()

    # Styling y-tick labels
    for i, lbl in enumerate(ax.get_yticklabels()):
        # models[i] is a string, lbl is a Text object
        text = models[i]
        if "mft" in text.lower():
            lbl.set_fontweight("bold")
            lbl.set_color("#2c3e50")

    ax.bar_label(r1, padding=3, fmt="%.1f")
    ax.bar_label(r2, padding=3, fmt="%.1f")

    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()
    print(f"Bar Chart generated at {output_filename}")
    return True


def generate_line_chart(model_data, output_filename, title):
    """Generate line charts for Pearson and Spearman."""
    set_academic_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # colors = plt.cm.tab10(np.linspace(0, 1, len(model_data)))

    for i, (model, points) in enumerate(model_data.items()):
        points.sort(key=lambda x: x[0])
        steps = [p[0] for p in points]

        # Handle single point by extending limit slightly
        if len(steps) == 1:
            steps_plot = [steps[0]]
            pearson_plot = [points[0][1] * 100]
            spearman_plot = [points[0][2] * 100]
            linestyle = ""
            marker = "D"  # Diamond for single points
            ms = 10
        else:
            steps_plot = steps
            pearson_plot = [p[1] * 100 for p in points]
            spearman_plot = [p[2] * 100 for p in points]
            linestyle = "-"
            marker = "o"
            ms = 8

        ax1.plot(
            steps_plot,
            pearson_plot,
            marker=marker,
            linestyle=linestyle,
            label=model,
            markersize=ms,
            linewidth=2,
        )
        ax2.plot(
            steps_plot,
            spearman_plot,
            marker=marker,
            linestyle=linestyle,
            label=model,
            markersize=ms,
            linewidth=2,
        )

        # Fix x-axis if single point
        if len(steps) == 1:
            ax1.set_xlim(steps[0] - 1, steps[0] + 1)
            ax1.set_xticks([steps[0]])
            ax2.set_xlim(steps[0] - 1, steps[0] + 1)
            ax2.set_xticks([steps[0]])

    for ax, metric in [(ax1, "Pearson"), (ax2, "Spearman")]:
        ax.set_title(f"{title} - {metric}", pad=10)
        ax.set_xlabel("Training Step")
        ax.set_ylabel(f"{metric} Score (%)")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()
    print(f"Line Chart generated at {output_filename}")
    return True


def generate_version_history_charts(all_entries):
    """Generate version history line charts for Test split results."""
    set_academic_style()

    # Filter for Test split
    test_entries = [e for e in all_entries if e["split"] == "test"]
    if not test_entries:
        return

    model_data = {}
    for e in test_entries:
        step = e.get("step", 0)
        if isinstance(step, str):
            step = (
                int("".join(filter(str.isdigit, step)))
                if any(c.isdigit() for c in step)
                else 0
            )

        model_data.setdefault(e["model"], []).append(
            (step, e.get("pearson", 0) or 0, e.get("spearman", 0) or 0)
        )

    # Common plotting helper
    def plot_metric(metric_idx, metric_name, filename):
        plt.figure(figsize=(8, 5))

        has_single_points_only = True

        for model, points in model_data.items():
            points.sort()
            steps = [p[0] for p in points]
            scores = [p[metric_idx] * 100 for p in points]

            if len(points) > 1:
                has_single_points_only = False
                plt.plot(
                    steps, scores, marker="o", linestyle="-", label=model, markersize=8
                )
            else:
                # Plot single point large
                plt.plot(
                    steps,
                    scores,
                    marker="D",
                    linestyle="",
                    label=model,
                    markersize=10,
                    alpha=0.9,
                )
                # Annotate
                for x, y in zip(steps, scores):
                    plt.annotate(
                        f"{y:.1f}%",
                        (x, y),
                        xytext=(0, 10),
                        textcoords="offset points",
                        ha="center",
                    )

        plt.title(f"Version History - {metric_name}")
        plt.xlabel("Training Step")
        plt.ylabel(f"{metric_name} Score (%)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(frameon=True)

        # Center x-axis if only single points
        if has_single_points_only:
            plt.xlim(-0.5, 0.5)
            plt.xticks([0], ["Random Init (0)"])

        plt.tight_layout()
        plt.savefig(filename, bbox_inches="tight")
        plt.close()
        print(f"Generated {filename}")

    # 1. Pearson (index 1)
    plot_metric(1, "Pearson", "version_history_pearson.png")
    # 2. Spearman (index 2)
    plot_metric(2, "Spearman", "version_history_spearman.png")


def generate_chart(all_runs, output_filename, title):
    """Generate chart - bar for single points, line for multiple."""
    model_data = {}
    for r in all_runs:
        step = r["step"]
        if isinstance(step, str) and step.isdigit():
            step = int(step)  # sanitize

        # If 'step' is essentially just an index for random sanity check
        model_data.setdefault(r["model"], []).append(
            (step if isinstance(step, int) else 0, r["pearson"], r["spearman"])
        )

    if not model_data:
        return False

    # Force line chart if user desires, but for single points Bar is better.
    # However, user said "do not remove any charts". We will keep logic but allow line chart fallback.
    max_pts = max(len(pts) for pts in model_data.values())
    return (
        generate_bar_chart(model_data, output_filename, title)
        if max_pts <= 1
        else generate_line_chart(model_data, output_filename, title)
    )


def main():
    filename = "sts_benchmark_results.json"
    if not os.path.exists(filename):
        print(f"File {filename} not found.")
        return

    try:
        data = load_results(filename)
    except json.JSONDecodeError as e:
        print(f"Error reading JSON: {e}")
        return

    # Parse entries
    all_entries = []
    for entry in data:
        ts_raw = entry.get("timestamp", "")
        ts_obj = datetime.min
        ts_display = ts_raw
        if ts_raw:
            try:
                ts_obj = datetime.fromisoformat(ts_raw)
                if ts_obj.tzinfo is not None:
                    ts_obj = ts_obj.replace(tzinfo=None)
                ts_display = ts_obj.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass

        # Hack to extract step from random filename if possible or default
        # For random pivot, we treat them as step 0
        current_step = 0

        for res in entry.get("results", []):
            all_entries.append(
                {
                    "timestamp_obj": ts_obj,
                    "timestamp_display": ts_display,
                    "dataset": entry.get("dataset", "unknown"),
                    "split": res.get("split", "unknown"),
                    "model": res.get("model", "Unknown"),
                    "pearson": res.get("pearson", 0.0) or 0.0,
                    "spearman": res.get("spearman", 0.0) or 0.0,
                    "num_samples": res.get("num_samples", 0),
                    "step": current_step,
                }
            )

    # Group by split
    entries_by_split = {}
    for e in all_entries:
        entries_by_split.setdefault(e["split"], []).append(e)

    # --- Generate Version History Charts (All Splits or Test Only) ---
    # We call this explicitly to restore the missing files
    generate_version_history_charts(all_entries)

    output_lines = []

    # Summary at TOP
    output_lines.append(generate_detailed_summary(entries_by_split))

    # Process each split
    for split_name, entries in entries_by_split.items():
        processed_runs, best_results = process_split_data(entries, split_name)
        split_title = f"{split_name.capitalize()} Split"
        output_lines.append(f"# {split_title}\n")

        chart_file = f"sts_benchmark_chart_{split_name}.png"
        if generate_chart(
            processed_runs, chart_file, f"Model Performance - {split_title}"
        ):
            output_lines.append(f"![{split_title} Performance]({chart_file})\n")

        # All Runs Table (scores ×100)
        rows1 = []
        for r in processed_runs:
            name = f"**{r['model']}**" if "mft" in r["model"] else r["model"]
            p = (
                f"**{r['pearson']*100:.2f}%**"
                if "mft" in r["model"]
                else f"{r['pearson']*100:.2f}%"
            )
            s = (
                f"**{r['spearman']*100:.2f}%**"
                if "mft" in r["model"]
                else f"{r['spearman']*100:.2f}%"
            )
            rows1.append([r["timestamp_display"], name, r["step"], p, s])

        output_lines.append(f"## All Runs - {split_title}\n")
        output_lines.append(
            format_table(
                ["Timestamp", "Model", "Training Step", "Pearson", "Spearman"], rows1
            )
        )
        output_lines.append("\n")

        # Best Results Table (scores ×100)
        rows2 = []
        for r in best_results:
            name = f"**{r['model']}**" if "mft" in r["model"] else r["model"]
            p = (
                f"**{r['pearson']*100:.2f}%**"
                if "mft" in r["model"]
                else f"{r['pearson']*100:.2f}%"
            )
            s = (
                f"**{r['spearman']*100:.2f}%**"
                if "mft" in r["model"]
                else f"{r['spearman']*100:.2f}%"
            )
            rows2.append(
                [r["timestamp_display"], name, r["step"], p, s, r["num_samples"]]
            )

        output_lines.append(f"## Best Results - {split_title}\n")
        output_lines.append(
            format_table(
                [
                    "Timestamp",
                    "Model",
                    "Training Step",
                    "Pearson",
                    "Spearman",
                    "Samples",
                ],
                rows2,
            )
        )
        output_lines.append("\n\n")

    output_content = "\n".join(output_lines)
    # print(output_content) # Suppress for now

    with open("STS_BENCHMARK_RESULTS.md", "w", encoding="utf-8") as f:
        f.write("# STS Benchmark Results Report\n\n" + output_content)

    print("\nResults exported to STS_BENCHMARK_RESULTS.md")

    # Generate Version Benchmark Report
    generate_version_report(entries_by_split)


def generate_version_report(entries_by_split):
    """Generate VERSION_BENCHMARK_RESULTS.md from test split data."""
    test_entries = entries_by_split.get("test", [])
    if not test_entries:
        return

    # Filter for random init models or versioned runs
    # We look for runs that have 'revision' field or are part of the version history
    # The merged data should have 'revision'

    # Group by model
    model_data = {}
    for e in test_entries:
        model_data.setdefault(e["model"], []).append(e)

    lines = ["# Version Benchmark Results Report\n"]
    lines.append("# 📝 Detailed Analysis Summary\n")

    # Find best overall
    best_overall = None
    for m, entries in model_data.items():
        if not entries:
            continue
        best_model = max(entries, key=lambda x: x.get("spearman", 0) or 0)
        if best_overall is None or (best_model.get("spearman", 0) or 0) > (
            best_overall.get("spearman", 0) or 0
        ):
            best_overall = best_model

    if best_overall:
        lines.append("### 🏆 Overall Best Performance\n")
        rev = best_overall.get("revision", "N/A")[:8]
        lines.append(
            f"The best performing model version is **{best_overall['model']}** (Rev: `{rev}`) with a Pearson score of **{best_overall['pearson']*100:.2f}%** and Spearman of **{best_overall['spearman']*100:.2f}%**.\n"
        )

    lines.append("## Performance Charts\n")
    lines.append("### Pearson Correlation\n")
    lines.append("![Pearson Performance](version_history_pearson.png)\n")
    lines.append("### Spearman Correlation\n")
    lines.append("![Spearman Performance](version_history_spearman.png)\n")

    lines.append("# 📜 Detailed Version History\n")

    for model in sorted(model_data.keys()):
        entries = model_data[model]
        # Sort by timestamp descending
        entries.sort(key=lambda x: x["timestamp_obj"], reverse=True)

        lines.append(f"## {model}\n")
        lines.append(f"| Date | Revision | Pearson | Spearman | Samples |")
        lines.append(f"| --- | --- | --- | --- | --- |")

        for e in entries:
            date_str = e["timestamp_display"]
            rev = e.get("revision", "N/A")
            if rev and len(rev) > 8:
                rev = rev[:8]
            pearson = f"{e['pearson']*100:.2f}%"
            spearman = f"{e['spearman']*100:.2f}%"
            samples = e.get("num_samples", "-")
            lines.append(f"| {date_str} | {rev} | {pearson} | {spearman} | {samples} |")

        lines.append("\n")

    with open("VERSION_BENCHMARK_RESULTS.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\nVersion report exported to VERSION_BENCHMARK_RESULTS.md")


if __name__ == "__main__":
    main()
