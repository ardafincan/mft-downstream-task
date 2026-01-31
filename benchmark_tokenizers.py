import time
import turk_mft
from transformers import AutoTokenizer
from datasets import load_dataset
import sys


def benchmark():
    print("Loading dataset 'alibayram/cosmos-corpus-00-5'...")
    try:
        ds = load_dataset("alibayram/cosmos-corpus-00-5", split="train", streaming=True)
        it = iter(ds)
        first_item = next(it)
        text_column = "text" if "text" in first_item else list(first_item.keys())[0]

        rows = [first_item[text_column]]
        # Increase to 5000 rows for significant load
        print("Fetching 5000 rows...")
        for _ in range(4999):
            try:
                item = next(it)
                rows.append(item[text_column])
            except StopIteration:
                break
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Use 500 rows for long text (approx 3-4MB)
    long_text = "".join(rows[:500])
    short_texts = rows  # All 5000

    print(
        f"Prepared Data: Long Text ({len(long_text)} chars), {len(short_texts)} Short Texts."
    )

    # Suppress warnings
    import transformers

    transformers.logging.set_verbosity_error()

    print("Initializing tokenizers...")
    try:
        models = {
            "MFT (Rust)": turk_mft.TurkishTokenizer(),
            "Mursit": AutoTokenizer.from_pretrained(
                "alibayram/newmindaiMursit-random-init", use_fast=True
            ),
            "Cosmos": AutoTokenizer.from_pretrained(
                "alibayram/cosmosGPT2-random-init", use_fast=True
            ),
            "Tabi": AutoTokenizer.from_pretrained(
                "alibayram/tabi-random-init", use_fast=True
            ),
        }
    except Exception as e:
        print(f"Error initializing tokenizers: {e}")
        return

    # Warmup
    print("Warming up...")
    for name, tok in models.items():
        try:
            tok.encode("merhaba dünya")
        except:
            pass

    # Test 1: Long Text
    print("\nTest 1: Long Text (Concatenated 50 rows)")
    print("-" * 80)
    print(
        f"{'Model':<15} | {'Time (ms)':<10} | {'Speedup (vs MFT)':<20} | {'Tokens':<10}"
    )
    print("-" * 80)

    mft_time = 0

    # Run MFT first
    start = time.perf_counter()
    mft_tokens = models["MFT (Rust)"].encode(long_text)
    mft_time = (time.perf_counter() - start) * 1000
    print(
        f"{'MFT (Rust)':<15} | {mft_time:>10.2f} | {'Baseline':>20} | {len(mft_tokens):<10}"
    )

    for name, tok in models.items():
        if name == "MFT (Rust)":
            continue

        start = time.perf_counter()
        tokens = tok.encode(long_text, add_special_tokens=False)
        duration = (time.perf_counter() - start) * 1000

        speedup = duration / mft_time if mft_time > 0 else 0
        speedup_str = (
            f"{speedup:.2f}x slower" if speedup > 1 else f"{1/speedup:.2f}x faster"
        )
        print(f"{name:<15} | {duration:>10.2f} | {speedup_str:>20} | {len(tokens):<10}")

    # Test 2: 500 Lines Loop
    print("\nTest 2: 500 Lines (Sequential Loop)")
    print("-" * 80)
    print(
        f"{'Model':<15} | {'Time (ms)':<10} | {'Speedup (vs MFT)':<20} | {'Total Tokens':<12}"
    )
    print("-" * 80)

    mft_time_loop = 0
    start = time.perf_counter()
    mft_total = 0
    for text in short_texts:
        mft_total += len(models["MFT (Rust)"].encode(text))
    mft_time_loop = (time.perf_counter() - start) * 1000
    print(
        f"{'MFT (Rust)':<15} | {mft_time_loop:>10.2f} | {'Baseline':>20} | {mft_total:<12}"
    )

    for name, tok in models.items():
        if name == "MFT (Rust)":
            continue

        start = time.perf_counter()
        total = 0
        for text in short_texts:
            total += len(tok.encode(text, add_special_tokens=False))
        duration = (time.perf_counter() - start) * 1000

        speedup = duration / mft_time_loop if mft_time_loop > 0 else 0
        speedup_str = (
            f"{speedup:.2f}x slower" if speedup > 1 else f"{1/speedup:.2f}x faster"
        )
        print(f"{name:<15} | {duration:>10.2f} | {speedup_str:>20} | {total:<12}")


if __name__ == "__main__":
    benchmark()
