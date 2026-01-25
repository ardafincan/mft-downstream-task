#!/usr/bin/env python3
"""
Simple script to push a local SentenceTransformer model to HuggingFace Hub.
Usage:
    python push_model_to_hub.py /path/to/model magibu/embeddingmagibu-152m-ft
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()


def push_model(model_path: str, repo_id: str, commit_message: str = "Update model"):
    """Push a model to HuggingFace Hub."""
    
    # Verify HF_TOKEN exists
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not found in environment. Please set it in .env file.")
    
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    print(f"Loading model from: {model_path}")
    model = SentenceTransformer(str(model_path))
    
    print(f"Pushing to Hub: {repo_id}")
    print(f"Commit message: {commit_message}")
    
    try:
        model.push_to_hub(
            repo_id=repo_id,
            token=token,
            commit_message=commit_message,
            exist_ok=True,  # Don't error if repo already exists
            private=False,
        )
        print(f"✅ Successfully pushed to {repo_id}")
    except Exception as e:
        print(f"❌ Error pushing to hub: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Push a SentenceTransformer model to HuggingFace Hub"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the local model directory",
    )
    parser.add_argument(
        "repo_id",
        type=str,
        help="HuggingFace Hub repository ID (e.g., username/model-name)",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Update model",
        help="Commit message for the push",
    )
    
    args = parser.parse_args()
    
    push_model(args.model_path, args.repo_id, args.commit_message)


if __name__ == "__main__":
    main()
