#!/usr/bin/env python3

import argparse
import os
import sys

from sentence_transformers import SentenceTransformer

import mteb
from mteb import MTEB


def evaluate_model(model_name, output_folder, batch_size, device):
    """
    Evaluate a model using MTEB-TR benchmark

    Args:
        model_name (str): Name or path of the model to evaluate
        output_folder (str): Path to save the evaluation results
        batch_size (int): Batch size for evaluation
        device (str): Device to use for computation (e.g., 'cpu', 'cuda', 'mps')
    """
    try:
        # Get MTEB-TR benchmark
        mteb_tr = mteb.get_benchmark("MTEB(Turkish)")

        # Initialize model
        print(f"Loading model: {model_name} on {device}")
        model = SentenceTransformer(model_name, trust_remote_code=True, device=device)

        # Initialize MTEB evaluation
        evaluation = MTEB(tasks=mteb_tr)

        # Create output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Run evaluation
        print(
            f"Starting evaluation with batch size {batch_size}. Results will be saved to: {output_folder}"
        )
        results = evaluation.run(
            model, output_folder=output_folder, batch_size=batch_size
        )

        print("Evaluation completed successfully!")
        return True

    except Exception as e:
        print(f"Error during evaluation: {str(e)}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run MTEB-TR benchmark evaluation for a given model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "model_name",
        help="Name or path of the model to evaluate (e.g., 'sentence-transformers/LaBSE' or path to local model)",
    )

    parser.add_argument(
        "--output-folder",
        "-o",
        default="results",
        help="Path to save the evaluation results",
    )

    parser.add_argument(
        "--batch-size", "-b", type=int, default=32, help="Batch size for evaluation"
    )

    parser.add_argument(
        "--device",
        "-d",
        default="cpu",
        help="Device to use for computation (e.g., 'cpu', 'cuda', 'mps')",
    )

    args = parser.parse_args()

    success = evaluate_model(
        args.model_name, args.output_folder, args.batch_size, args.device
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
