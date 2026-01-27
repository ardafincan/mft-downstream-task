import os
import csv
import torch
import logging
import argparse
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import sys

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_sentences(filepath: str) -> List[Tuple[str, str]]:
    sentence_pairs = []
    # Use utf-8-sig to handle potential BOM
    with open(filepath, "r", encoding="utf-8-sig") as file:
        reader = csv.reader(file, delimiter=";")
        try:
            next(reader)  # Skip header
        except StopIteration:
            logger.warning(f"Empty file: {filepath}")
            return []

        for row in reader:
            if len(row) >= 2:
                good = row[0].strip()
                bad = row[1].strip()
                if good and bad:
                    sentence_pairs.append((good, bad))

    logger.info(f"Loaded {len(sentence_pairs)} pairs from {os.path.basename(filepath)}")
    return sentence_pairs


def evaluate_model(model_path: str, data_dir: str, output_dir: str, is_mft: bool):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading SentenceTransformer from {model_path} on {device}...")

    try:
        model = SentenceTransformer(model_path, trust_remote_code=True)
        model.to(device)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Inject custom tokenizer if needed
    if is_mft:
        logger.info("Attempting to inject custom TurkishTokenizer...")
        try:
            from turkish_tokenizer import TurkishTokenizer

            custom_tokenizer = TurkishTokenizer()

            # Inject into the first module (Transformer)
            if len(model) > 0 and hasattr(model[0], "tokenizer"):
                model[0].tokenizer = custom_tokenizer
                # Some custom implementations check this flag
                model[0]._is_custom_tokenizer = True
                logger.info("✓ Custom TurkishTokenizer injected into model[0]")
            else:
                logger.warning(
                    "Could not find .tokenizer on model[0]. Injection might have failed."
                )

        except ImportError:
            logger.error(
                "Could not import TurkishTokenizer. Make sure turkish_tokenizer.py is in the path."
            )
        except Exception as e:
            logger.error(f"Error injecting tokenizer: {e}")

    # List CSV files
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        return

    file_names = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])
    if not file_names:
        logger.warning(f"No CSV files found in {data_dir}")
        return

    results_summary = []

    print("\n" + "=" * 80)
    print(f"{'Category':<50} | {'Avg Cosine Sim':<15} | {'Pairs':<10}")
    print("-" * 80)

    for file_name in file_names:
        file_path = os.path.join(data_dir, file_name)
        pairs = load_sentences(file_path)

        if not pairs:
            continue

        sentences1 = [p[0] for p in pairs]  # Good
        sentences2 = [p[1] for p in pairs]  # Bad

        # Compute embeddings
        with torch.no_grad():
            embeddings1 = model.encode(
                sentences1,
                batch_size=32,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            embeddings2 = model.encode(
                sentences2,
                batch_size=32,
                convert_to_tensor=True,
                show_progress_bar=False,
            )

        # Compute Cosine Similarity
        cosine_scores = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)
        avg_sim = torch.mean(cosine_scores).item()

        category_name = file_name.replace("augmented_", "").replace(".csv", "")
        print(f"{category_name:<50} | {avg_sim:.4f}          | {len(pairs):<10}")

        results_summary.append(
            {
                "category": category_name,
                "avg_similarity": avg_sim,
                "num_pairs": len(pairs),
            }
        )

    print("=" * 80 + "\n")

    # Save summary
    model_slug = model_path.split("/")[-1]
    summary_filename = f"{model_slug}_turblimp_sensitivity.csv"
    summary_path = os.path.join(output_dir, summary_filename)

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["category", "avg_similarity", "num_pairs"]
        )
        writer.writeheader()
        writer.writerows(results_summary)

    logger.info(f"Saved results to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="TurBLiMP/data/base")
    parser.add_argument("--output_dir", type=str, default="turblimp_results")
    parser.add_argument("--is_mft", action="store_true")
    args = parser.parse_args()

    evaluate_model(args.model_path, args.data_dir, args.output_dir, args.is_mft)
