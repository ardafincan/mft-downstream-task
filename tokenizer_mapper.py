import os
import time
import json
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
import turkish_tokenizer as tt

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

org_tokenizer = AutoTokenizer.from_pretrained("google/embeddinggemma-300m", token=HF_TOKEN, use_fast=False)
org_model = AutoModelForCausalLM.from_pretrained("google/embeddinggemma-300m", token=HF_TOKEN)
org_model = org_model.to(torch.bfloat16)

target_tokenizer = tt.TurkishTokenizer()

source_vocab = org_tokenizer.get_vocab()
target_vocab = target_tokenizer.get_vocab()

token_id_map = {}
direct_matches = 0
tokenized_matches = 0

for token_str, target_id in target_vocab.items():
    token_str = token_str.replace(" ", "▁")
    if token_str in source_vocab:
        # Direct match - use source token ID directly
        token_id_map[target_id] = [source_vocab[token_str]]
        direct_matches += 1
    else:
        # Token not in source vocab - need to tokenize
        encoded = org_tokenizer.encode(token_str, add_special_tokens=False)
        if encoded:
            token_id_map[target_id] = encoded
            tokenized_matches += 1

print(f"   Direct matches: {direct_matches}")
print(f"   Tokenized matches: {tokenized_matches}")
print(f"   Total mapped: {len(token_id_map)}")


source_embeddings = org_model.model.embed_tokens.weight.clone().to(org_model.device)
org_model.resize_token_embeddings(target_tokenizer.vocab_size)

errors = []
for i in range(target_tokenizer.vocab_size):
    if i not in token_id_map or not token_id_map[i]:
        errors.append(i)
        # Initialize with zeros for unmapped tokens
        continue
    
    source_ids = token_id_map[i]
    # MEAN token strategy: average the source token embeddings
    # remove <bos> if present
    if source_ids[0] == org_tokenizer.bos_token_id:
        source_ids = source_ids[1:]
    embeddings_to_average = source_embeddings[source_ids]
    with torch.no_grad():
        org_model.model.embed_tokens.weight[i] = embeddings_to_average.mean(dim=0)
    if (i + 1) % 10000 == 0:
        print(f"   Mapped {i + 1}/{target_tokenizer.vocab_size} embeddings", flush=True)

if errors:
    print(f"   ⚠️ {len(errors)} tokens could not be mapped (initialized with mean)")

org_model.save_pretrained("gemma3_cloned")
