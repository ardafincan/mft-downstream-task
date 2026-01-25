"""
Random Initialization Script for Both MFT and TabiBERT Tokenizers

Creates a single SentenceTransformer model with random initialization
and pushes it to both MFT and TabiBERT repositories. Since both use
identical random weights (seed=42), we only need to create once.

The only difference between repos:
- MFT repo: No tokenizer files (uses custom TurkishTokenizer at inference)
- TabiBERT repo: Includes TabiBERT tokenizer files
"""

import os
import shutil
import torch
import torch.nn as nn
import random
import numpy as np

from dotenv import load_dotenv
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import turkish_tokenizer as tt

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

# Fixed seed for reproducibility
SEED = 42
VOCAB_SIZE = 32768  # Both tokenizers have same vocab size
org_model_id = "google/embeddinggemma-300m"
clone_dir = "random_init_cloned"


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_weights(module):
    """Initialize all weights randomly with Xavier/Glorot initialization."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.xavier_uniform_(module.weight)
    elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.ones_(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'weight') and module.weight is not None:
        if module.weight.dim() >= 2:
            nn.init.xavier_uniform_(module.weight)
        else:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)


if __name__ == "__main__":
    print(f"Random Initialization with SEED={SEED}")
    print("Creating single model, pushing to both repos")
    print("=" * 60)
    
    # Set seed
    set_seed(SEED)
    
    # Load SentenceTransformer structure
    print("Loading SentenceTransformer structure...")
    model = SentenceTransformer(org_model_id, token=HF_TOKEN)
    
    # Resize embeddings to 32K
    model[0].auto_model.resize_token_embeddings(VOCAB_SIZE)
    
    # Apply random initialization
    set_seed(SEED)
    print(f"Applying random initialization with seed={SEED}...")
    model.apply(init_weights)
    model = model.to(torch.bfloat16)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Save model
    model.save_pretrained(clone_dir)
    
    # === Push to MFT repo (without tokenizer) ===
    print("\n--- MFT Repository ---")
    
    # Remove tokenizer files for MFT
    for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model"]:
        path = os.path.join(clone_dir, f)
        if os.path.exists(path):
            os.remove(path)
    
    # Load with custom tokenizer and push
    mft_tokenizer = tt.TurkishTokenizer()
    print(f"MFT tokenizer vocab size: {mft_tokenizer.vocab_size}")
    
    mft_model = SentenceTransformer(clone_dir, custom_tokenizer=mft_tokenizer)
    mft_model = mft_model.to(torch.bfloat16)
    
    print("Uploading to alibayram/mft-random-init...")
    mft_model.push_to_hub("alibayram/mft-random-init", token=HF_TOKEN, exist_ok=True)
    print("✓ Uploaded alibayram/mft-random-init")
    
    del mft_model
    
    # === Push to TabiBERT repo (with tokenizer) ===
    print("\n--- TabiBERT Repository ---")
    
    # Add TabiBERT tokenizer
    tabi_tokenizer = AutoTokenizer.from_pretrained(
        "alibayram/TabiBERT-tokenizer-32k",
        token=HF_TOKEN,
        use_fast=False
    )
    print(f"TabiBERT tokenizer vocab size: {tabi_tokenizer.vocab_size}")
    tabi_tokenizer.save_pretrained(clone_dir)
    
    # Reload and push
    tabi_model = SentenceTransformer(clone_dir)
    tabi_model = tabi_model.to(torch.bfloat16)
    
    print("Uploading to alibayram/tabi-random-init...")
    tabi_model.push_to_hub("alibayram/tabi-random-init", token=HF_TOKEN, exist_ok=True)
    print("✓ Uploaded alibayram/tabi-random-init")
    
    # Cleanup
    shutil.rmtree(clone_dir)
    
    print("\n" + "=" * 60)
    print("✓ Both repos updated with identical random weights!")
    print(f"✓ Seed: {SEED}")
    print(f"✓ Vocab size: {VOCAB_SIZE}")
    print("=" * 60)
