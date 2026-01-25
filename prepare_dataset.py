"""Dataset preparation script for embedding distillation training.

This script prepares a training dataset by:
1. Loading the source dataset with teacher embeddings
2. Encoding texts with both MFT and TabiBERT tokenizers
3. Filtering texts that exceed max sequence length (2048 tokens)
4. Pushing the prepared dataset to HuggingFace Hub

The resulting dataset has columns:
- text: Original text
- mft_input_ids: Token IDs from MFT tokenizer
- tabi_input_ids: Token IDs from TabiBERT tokenizer
- teacher_embedding_final: Teacher model embeddings

Usage:
    python prepare_dataset.py
"""

import os

from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoTokenizer

from turkish_tokenizer import TurkishTokenizer

load_dotenv()

# Configuration
SOURCE_DATASET = "alibayram/cosmos-corpus-0-05-with-embeddings"
OUTPUT_DATASET = "alibayram/cosmos-corpus-encoded"
MAX_SEQ_LENGTH = 2048
TEXT_COLUMN = "text"
TEACHER_EMBEDDING_COLUMN = "teacher_embedding_final"

HF_TOKEN = os.environ.get("HF_TOKEN")


def main():
    print("=" * 60)
    print("Dataset Preparation for Embedding Distillation")
    print("=" * 60)

    # Load source dataset
    print(f"\n1. Loading source dataset: {SOURCE_DATASET}")
    dataset = load_dataset(SOURCE_DATASET, split="train")
    print(f"   Original size: {len(dataset):,} examples")

    # Initialize tokenizers
    print("\n2. Initializing tokenizers...")

    # MFT Tokenizer
    mft_tokenizer = TurkishTokenizer()
    print(f"   MFT tokenizer vocab size: {mft_tokenizer.vocab_size}")

    # TabiBERT Tokenizer
    tabi_tokenizer = AutoTokenizer.from_pretrained(
        "alibayram/TabiBERT-tokenizer-32k",
        token=HF_TOKEN,
        use_fast=True,
    )
    print(f"   TabiBERT tokenizer vocab size: {tabi_tokenizer.vocab_size}")

    # Encode with both tokenizers and filter
    print(f"\n3. Encoding and filtering (max_seq_length={MAX_SEQ_LENGTH})...")

    def encode_and_filter(example):
        """Encode text with both tokenizers and check length."""
        text = example[TEXT_COLUMN]

        # MFT encoding
        mft_input_ids = mft_tokenizer.encode(text)

        # TabiBERT encoding
        tabi_input_ids = tabi_tokenizer.encode(text, add_special_tokens=True)

        # Check if both are within max length
        keep = (
            len(mft_input_ids) <= MAX_SEQ_LENGTH
            and len(tabi_input_ids) <= MAX_SEQ_LENGTH
        )

        return {
            "mft_input_ids": mft_input_ids,
            "tabi_input_ids": tabi_input_ids,
            "_keep": keep,
        }

    # Process dataset
    dataset = dataset.map(
        encode_and_filter,
        desc="Encoding with MFT & TabiBERT",
        num_proc=4,  # Use multiple processes for speed
    )

    # Filter by length
    original_size = len(dataset)
    dataset = dataset.filter(
        lambda x: x["_keep"],
        desc="Filtering by max length",
    )
    dataset = dataset.remove_columns(["_keep"])
    filtered_size = len(dataset)

    print(
        f"   Kept {filtered_size:,} / {original_size:,} examples ({100*filtered_size/original_size:.1f}%)"
    )

    # Select final columns
    final_columns = [
        TEXT_COLUMN,
        "mft_input_ids",
        "tabi_input_ids",
        TEACHER_EMBEDDING_COLUMN,
    ]

    # Check if all columns exist
    missing = [col for col in final_columns if col not in dataset.column_names]
    if missing:
        print(f"   Warning: Missing columns: {missing}")
        final_columns = [col for col in final_columns if col in dataset.column_names]

    print(f"\n4. Final dataset columns: {final_columns}")

    # Save locally first (so we don't lose work if push fails)
    local_path = "./encoded_dataset"
    print(f"\n5. Saving locally to: {local_path}")
    dataset.save_to_disk(local_path)
    print("   ✓ Saved locally!")

    # Push to HuggingFace Hub
    print(f"\n6. Pushing to HuggingFace Hub: {OUTPUT_DATASET}")
    dataset.push_to_hub(
        OUTPUT_DATASET,
        token=HF_TOKEN,
        private=False,
    )
    print("   ✓ Upload complete!")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Source dataset: {SOURCE_DATASET}")
    print(f"Output dataset: {OUTPUT_DATASET}")
    print(f"Local path:     {local_path}")
    print(f"Original size:  {original_size:,} examples")
    print(f"Final size:     {filtered_size:,} examples")
    print(f"Max seq length: {MAX_SEQ_LENGTH}")
    print(f"Columns:        {', '.join(final_columns)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
