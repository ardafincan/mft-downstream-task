"""
Run MTEB Turkish evaluation with MFT (Turkish) tokenizer.

This script uses the modified sentence_transformers package that supports
custom_tokenizer parameter to use the MFT tokenizer with SentenceTransformer models.
"""

import os
import sys

# Add parent directory to path for importing turkish_tokenizer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mteb
from dotenv import load_dotenv
from mteb import MTEB

import turkish_tokenizer as tt
from sentence_transformers import SentenceTransformer

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")


def evaluate_model_with_mft(model_name: str, output_folder: str = "results"):
    """
    Evaluate a model using MFT Turkish tokenizer.

    Args:
        model_name: The model name/path to evaluate
        output_folder: Folder to save results
    """
    # Initialize MFT Turkish tokenizer
    tokenizer = tt.TurkishTokenizer()
    print(f"MFT Tokenizer vocab size: {tokenizer.vocab_size}")

    # Load SentenceTransformer with MFT tokenizer
    model = SentenceTransformer(
        model_name,
        trust_remote_code=True,
        custom_tokenizer=tokenizer,
        token=HF_TOKEN,
    )

    # Get Turkish benchmark tasks
    mteb_tr = mteb.get_benchmark("MTEB(Turkish)")

    # Initialize MTEB evaluation
    evaluation = MTEB(tasks=mteb_tr)

    # Run evaluation
    results = evaluation.run(model, output_folder=output_folder)

    return results


def evaluate_model_standard(model_name: str, output_folder: str = "results"):
    """
    Evaluate a model using its default tokenizer (for comparison).

    Args:
        model_name: The model name/path to evaluate
        output_folder: Folder to save results
    """
    # Load SentenceTransformer with default tokenizer
    model = SentenceTransformer(
        model_name,
        trust_remote_code=True,
        token=HF_TOKEN,
    )

    # Get Turkish benchmark tasks
    mteb_tr = mteb.get_benchmark("MTEB(Turkish)")

    # Initialize MTEB evaluation
    evaluation = MTEB(tasks=mteb_tr)

    # Run evaluation
    results = evaluation.run(model, output_folder=output_folder)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MTEB Turkish evaluation")
    parser.add_argument(
        "model_name",
        type=str,
        help="Model name or path to evaluate",
    )
    parser.add_argument(
        "--use-mft",
        action="store_true",
        help="Use MFT Turkish tokenizer instead of model's default tokenizer",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="results",
        help="Folder to save results (default: results)",
    )

    args = parser.parse_args()

    print(f"Evaluating model: {args.model_name}")
    print(f"Using MFT tokenizer: {args.use_mft}")
    print(f"Output folder: {args.output_folder}")
    print("-" * 50)

    if args.use_mft:
        evaluate_model_with_mft(args.model_name, args.output_folder)
    else:
        evaluate_model_standard(args.model_name, args.output_folder)

    print("Done!")
