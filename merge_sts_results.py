import json
import os
from datetime import datetime


def merge_results():
    sts_file = "sts_benchmark_results.json"
    version_file = "version_eval_alibayram_cosmosGPT2_random_init.json"

    if not os.path.exists(version_file):
        print(f"Version file {version_file} not found. Skipping merge.")
        return

    with open(version_file, "r") as f:
        v_data = json.load(f)

    runs_to_add = []
    model_name = v_data.get("model", "alibayram/cosmosGPT2-random-init")

    # We filter out runs that are already in sts_benchmark_results to avoid duplicates?
    # Or just append?
    # Let's read existing first
    existing_data = []
    if os.path.exists(sts_file):
        with open(sts_file, "r") as f:
            existing_data = json.load(f)

    # Check for duplicates based on timestamp maybe?
    existing_timestamps = {e.get("timestamp") for e in existing_data}

    count = 0
    for res in v_data.get("results", []):
        ts = res.get("commit_date")
        if not ts:
            ts = datetime.now().isoformat()

        # We can't easily dedup by timestamp if it was generated now, but commit_date is stable
        # Let's check if this specific result is already logged?
        # Actually, let's just create the entry

        entry = {
            "timestamp": ts,
            "dataset": "figenfikri/stsb_tr",
            "results": [
                {
                    "model": model_name,
                    "split": res.get("split", "test"),
                    "pearson": res.get("pearson"),
                    "spearman": res.get("spearman"),
                    "num_samples": res.get("num_samples"),
                    "processing_time_seconds": res.get("processing_time_seconds"),
                    "revision": res.get("revision"),  # Store revision for reference
                }
            ],
        }

        # Simple dedup: check if same timestamp and model and spearman score exists
        is_dup = False
        for ex in existing_data:
            if ex.get("timestamp") == ts:
                for r in ex.get("results", []):
                    if r.get("model") == model_name and r.get("spearman") == res.get(
                        "spearman"
                    ):
                        is_dup = True
                        break
            if is_dup:
                break

        if not is_dup:
            existing_data.append(entry)
            count += 1

    print(f"Merging {count} new entries to {sts_file}")

    with open(sts_file, "w") as f:
        json.dump(existing_data, f, indent=2)


if __name__ == "__main__":
    merge_results()
