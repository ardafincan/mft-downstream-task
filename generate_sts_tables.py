#!/usr/bin/env python3
import json
import os
from datetime import datetime


def load_results(filename):
    with open(filename, "r") as f:
        return json.load(f)


def format_table(headers, rows):
    """Formats data as a Markdown table."""
    if not rows:
        return "No data available."

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    # Create format string
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_widths) + " |"

    # Create separator
    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"

    lines = []
    lines.append(fmt.format(*headers))
    lines.append(sep)
    for row in rows:
        lines.append(fmt.format(*[str(c) for c in row]))

    return "\n".join(lines)


def main():
    filename = "sts_benchmark_results.json"
    if not os.path.exists(filename):
        print(f"File {filename} not found.")
        return

    try:
        data = load_results(filename)
    except json.JSONDecodeError as e:
        print(f"Error reading JSON file: {e}")
        return

    all_entries = []

    # Flatten all results first
    for entry in data:
        timestamp_raw = entry.get("timestamp", "")
        ts_display = timestamp_raw
        ts_obj = datetime.min
        if timestamp_raw:
            try:
                ts_obj = datetime.fromisoformat(timestamp_raw)
                ts_display = ts_obj.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass

        dataset = entry.get("dataset", "unknown")

        for res in entry.get("results", []):
            split = res.get("split", "unknown")
            model = res.get("model", "Unknown")
            pearson = res.get("pearson", 0.0) or 0.0
            spearman = res.get("spearman", 0.0) or 0.0
            proc_time = res.get("processing_time_seconds", 0.0) or 0.0

            res_entry = {
                "timestamp_obj": ts_obj,
                "timestamp_display": ts_display,
                "dataset": dataset,
                "split": split,
                "model": model,
                "pearson": pearson,
                "spearman": spearman,
                "time": proc_time,
                "num_samples": res.get("num_samples", 0),
            }
            all_entries.append(res_entry)

    # Group by split
    entries_by_split = {}
    for e in all_entries:
        s = e["split"]
        if s not in entries_by_split:
            entries_by_split[s] = []
        entries_by_split[s].append(e)

    output_lines = []

    # Process each split
    for split_name, entries in entries_by_split.items():
        processed_runs, best_results = process_split_data(entries, split_name)

        split_title = f"{split_name.capitalize()} Split"
        output_lines.append(f"# {split_title}\n")

        # Generate Chart
        chart_filename = f"sts_benchmark_chart_{split_name}.png"
        generated_chart = generate_chart(
            processed_runs, chart_filename, f"Model Performance - {split_title}"
        )

        if generated_chart:
            output_lines.append(f"![{split_title} Performance]( {chart_filename} )\n\n")

        # Table 1: All Runs
        headers1 = ["Timestamp", "Model", "Checkpoint", "Pearson", "Spearman"]
        rows1 = []
        for r in processed_runs:
            model_name = r["model"]
            pearson_str = f"{r['pearson']:.4f}"
            spearman_str = f"{r['spearman']:.4f}"

            if "mft" in model_name:
                model_name = f"**{model_name}**"
                pearson_str = f"**{pearson_str}**"
                spearman_str = f"**{spearman_str}**"

            rows1.append(
                [
                    r["timestamp_display"],
                    model_name,
                    r["step"],
                    pearson_str,
                    spearman_str,
                ]
            )

        output_lines.append(f"## All Runs - {split_title}\n")
        output_lines.append(format_table(headers1, rows1))
        output_lines.append("\n")

        # Table 2: Best Results
        headers2 = [
            "Timestamp",
            "Model",
            "Checkpoint",
            "Best Pearson",
            "Spearman",
            "Samples",
        ]
        rows2 = []
        for r in best_results:
            model_name = r["model"]
            pearson_str = f"{r['pearson']:.4f}"
            spearman_str = f"{r['spearman']:.4f}"

            if "mft" in model_name:
                model_name = f"**{model_name}**"
                pearson_str = f"**{pearson_str}**"
                spearman_str = f"**{spearman_str}**"

            rows2.append(
                [
                    r["timestamp_display"],
                    model_name,
                    r["step"],
                    pearson_str,
                    spearman_str,
                    r["num_samples"],
                ]
            )

        output_lines.append(f"## Best Results - {split_title} (Sorted by Timestamp)\n")
        output_lines.append(format_table(headers2, rows2))
        output_lines.append("\n\n")

    output_content = "\n".join(output_lines)

    # Print to stdout
    print(output_content)

    # Write to file
    output_filename = "STS_BENCHMARK_RESULTS.md"
    with open(output_filename, "w") as f:
        f.write("# STS Benchmark Results Report\n\n")
        f.write(output_content)

    print(f"\nResults exported to {output_filename}")


def process_split_data(entries, split_name):
    """
    Groups entries by model, assigns checkpoints, and finds best results.
    Return (all_runs_sorted, best_results_sorted)
    """
    model_stats = {}
    for e in entries:
        m = e["model"]
        if m not in model_stats:
            model_stats[m] = []
        model_stats[m].append(e)

    all_runs = []

    # Assign steps
    for model, results in model_stats.items():
        # Sort by timestamp ascending
        results.sort(key=lambda x: x["timestamp_obj"])

        for i, res in enumerate(results):
            # Existing logic for test split or general sequence
            # If train split likely has only 1 run (latest), we might want to special case it?
            # User said "checkpoint colon too, each change of model is 0, 50 and 100 step"
            # If there is only 1 run, assigning '0' might be confusing if it's the 100th step.
            # But without explicit step info in JSON, we can only imply from order or count.
            # For now, sticking to the order logic but let's be aware.

            step = "?"
            if i == 0:
                step = 0
            elif i == 1:
                step = 50
            elif i == 2:
                step = 100
            else:
                step = f"? ({i})"

            # Heuristic override for single run on 'train' split if likely 'latest'
            # But "random models" are excluded.
            # If we see only 1 run and it's 'train', maybe let's just label it 100?
            # Risk: What if they run step 0 on train?
            # Let's check user intent: "run ... with the latest checkpoint ... draw another chart ... no random models"
            # It implies a single point per model. Labeling it '0' (start) is definitely wrong.
            # Let's label it 'Latest' or '100'?
            if split_name == "train" and len(results) == 1:
                step = 100  # Assumption based on user prompt "latest checkpoint"

            res["step"] = step
            all_runs.append(res)

    # Sort all runs by timestamp
    all_runs.sort(key=lambda x: x["timestamp_obj"], reverse=False)

    # Best results (Latest timestamp usually, or max pearson)
    best_results = []
    for model, results in model_stats.items():
        best = max(results, key=lambda x: x.get("pearson", 0) or 0)
        best_results.append(best)

    best_results.sort(key=lambda x: x["timestamp_obj"], reverse=True)

    return all_runs, best_results


def generate_chart(all_runs, output_filename, title):
    """Generates a chart for the given runs. Automatically switches between Line and Bar chart."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False

    model_data = {}
    valid_points_count = 0
    max_points_per_model = 0

    for r in all_runs:
        model = r["model"]
        step = r["step"]
        pearson = r["pearson"]
        spearman = r["spearman"]

        # Accept integer steps or if it's a single point which might have been assigned an int
        if isinstance(step, int):
            if model not in model_data:
                model_data[model] = []
            model_data[model].append((step, pearson, spearman))
            valid_points_count += 1
            max_points_per_model = max(max_points_per_model, len(model_data[model]))

    if valid_points_count == 0:
        print(
            f"No valid data points with integer steps for chart generation for {title}."
        )
        return False

    # Decide Chart Type
    # If max points per model is 1, use Bar Chart
    if max_points_per_model <= 1:
        return generate_bar_chart(model_data, output_filename, title, plt)
    else:
        return generate_line_chart(model_data, output_filename, title, plt)


def generate_bar_chart(model_data, output_filename, title, plt):
    """Generates a grouped vertical bar chart for Pearson and Spearman scores."""
    import numpy as np

    models = list(model_data.keys())
    pearson_scores = []
    spearman_scores = []

    for m in models:
        # data is list of (step, pearson, spearman), take the last/only one
        if model_data[m]:
            # Sort by step just in case, distinct points
            model_data[m].sort(key=lambda x: x[0])
            pearson_scores.append(model_data[m][-1][1])
            spearman_scores.append(model_data[m][-1][2])
        else:
            pearson_scores.append(0)
            spearman_scores.append(0)

    # Set up the bar chart
    x = np.arange(len(models))  # label locations
    width = 0.35  # width of the bars

    # Dynamic figsize
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define hatches for MFT highlighting
    # We need to pass a list of hatches same length as bars, OR plot individually
    # Easier to plot individually to control hatches?
    # Or just iterate?
    # Matplotlib bar only accepts single hatch or None usually, unless repeated?
    # Actually it can take a list in newer versions, checking... let's assume simple approach

    # Determine highlight indices
    mft_indices = [i for i, m in enumerate(models) if "mft" in m.lower()]

    # Plot Pearson
    rects1 = ax.bar(
        x - width / 2, pearson_scores, width, label="Pearson", color="skyblue"
    )
    # Plot Spearman
    rects2 = ax.bar(
        x + width / 2, spearman_scores, width, label="Spearman", color="orange"
    )

    # Apply highlighting (Hatching) to MFT bars
    # "highlight mft models" -> Hatching makes them distinct
    for i in mft_indices:
        rects1[i].set_hatch("///")
        rects2[i].set_hatch("///")

        # Also maybe a slightly darker edge?
        rects1[i].set_edgecolor("black")
        rects2[i].set_edgecolor("black")

    # Add text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    # Highlight specific tick labels
    for i, label in enumerate(ax.get_xticklabels()):
        if "mft" in models[i].lower():
            label.set_fontweight("bold")
            label.set_color("#2c3e50")  # Mild dark blue/grey for emphasis

    # Add values on top of bars
    ax.bar_label(rects1, padding=3, fmt="%.4f", rotation=90)
    ax.bar_label(rects2, padding=3, fmt="%.4f", rotation=90)

    # Add extra margin for top labels
    plt.margins(y=0.2)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Grouped Vertical Bar Chart generated at {output_filename}")
    return True


def generate_line_chart(model_data, output_filename, title, plt):
    """Generates line charts for Pearson and Spearman."""
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Plot Pearson
    for model, points in model_data.items():
        points.sort(key=lambda x: x[0])
        steps = [p[0] for p in points]
        p_scores = [p[1] for p in points]
        ax1.plot(steps, p_scores, marker="o", label=model)

    ax1.set_title(f"{title} - Pearson")
    ax1.set_xlabel("Checkpoint")
    ax1.set_ylabel("Pearson Score")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True)

    # Plot Spearman
    for model, points in model_data.items():
        points.sort(key=lambda x: x[0])
        steps = [p[0] for p in points]
        s_scores = [p[2] for p in points]
        ax2.plot(steps, s_scores, marker="o", label=model)

    ax2.set_title(f"{title} - Spearman")
    ax2.set_xlabel("Checkpoint")
    ax2.set_ylabel("Spearman Score")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Line Chart generated at {output_filename}")
    return True


if __name__ == "__main__":
    main()
