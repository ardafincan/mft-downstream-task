"""
Test script for custom tokenizer integration with SentenceTransformer.
Uses TurkishTokenizer with a SentenceTransformer model.
"""

import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import turkish_tokenizer as tt

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

# Initialize TurkishTokenizer
tokenizer = tt.TurkishTokenizer()
print(f"TurkishTokenizer vocab size: {tokenizer.vocab_size}")

# Load SentenceTransformer model with custom tokenizer
# Using a base model for testing (you can replace with "gemma3_cloned" if available)
model = SentenceTransformer(
    "gemma3_cloned", 
    custom_tokenizer=tokenizer,
    token=HF_TOKEN
)

# Test sentences (mix of English and Turkish)
sentences = [
    "That is a happy person",
    "That is a happy dog",
    "That is a very happy person",
    "Today is a sunny day",
    "Bu mutlu bir insan",
    "Bu mutlu bir köpek",
    "Bugün güneşli bir gün",
]

print("\nEncoding sentences...")
embeddings = model.encode(sentences)
print(f"Embeddings shape: {embeddings.shape}")

print("\nCalculating similarities...")
similarities = model.similarity(embeddings, embeddings)
print("\nSimilarity matrix:")
print(similarities)

# Print pairwise similarities for readability
print("\n\nPairwise similarities:")
for i, sent1 in enumerate(sentences):
    for j, sent2 in enumerate(sentences):
        if i < j:
            print(f"  '{sent1[:30]}...' <-> '{sent2[:30]}...': {similarities[i][j]:.4f}")
