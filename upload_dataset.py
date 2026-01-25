"""Upload encoded dataset to HuggingFace Hub using Xet storage."""

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = "alibayram/cosmos-corpus-encoded"
LOCAL_PATH = "./encoded_dataset"


def main():
    print("=" * 60)
    print("Upload Dataset to HuggingFace Hub (Xet)")
    print("=" * 60)

    api = HfApi(token=HF_TOKEN)

    # Create repo if it doesn't exist
    print(f"\n1. Creating repo: {REPO_ID}")
    try:
        api.create_repo(REPO_ID, repo_type="dataset", exist_ok=True)
        print("   ✓ Repo ready")
    except Exception as e:
        print(f"   Repo already exists or error: {e}")

    # Upload the entire folder
    print(f"\n2. Uploading from: {LOCAL_PATH}")
    print("   This uses Xet storage for faster uploads...")

    api.upload_folder(
        folder_path=LOCAL_PATH,
        repo_id=REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN,
        commit_message="Upload encoded dataset with MFT and TabiBERT input_ids",
    )

    print("\n   ✓ Upload complete!")
    print(f"\n3. Dataset available at: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
