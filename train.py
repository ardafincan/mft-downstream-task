"""Training script for embedding distillation using EmbeddingDistillationTrainer.

This module configures and runs the embedding distillation process,
training a student model to match teacher embeddings from a pre-encoded dataset.

Prerequisites:
    1. Run prepare_dataset.py first to create the encoded dataset
    2. Create .env file with: WANDB_API_KEY=xxx and HF_TOKEN=xxx
"""

import os

from dotenv import load_dotenv

from embedding_trainer import EmbeddingDistillationTrainer, EmbeddingTrainerConfig

load_dotenv()

WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")

if WANDB_API_KEY:
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY

# Pre-encoded dataset from prepare_dataset.py
DATASET_ID = "alibayram/cosmos-corpus-encoded"

# Choose model and corresponding input_ids column
# MFT models use: mft_input_ids
# TabiBERT models use: tabi_input_ids

MODEL_ID = "alibayram/mft-downstream-task-embeddinggemma"
INPUT_IDS_COLUMN = "mft_input_ids"

# Uncomment for other models:
# MODEL_ID = "alibayram/mft-downstream-task-embeddingmagibu"
# INPUT_IDS_COLUMN = "mft_input_ids"

# MODEL_ID = "alibayram/mft-random-init"
# INPUT_IDS_COLUMN = "mft_input_ids"

# MODEL_ID = "alibayram/tabi-downstream-task-embeddinggemma"
# INPUT_IDS_COLUMN = "tabi_input_ids"

# MODEL_ID = "alibayram/tabi-downstream-task-embeddingmagibu"
# INPUT_IDS_COLUMN = "tabi_input_ids"

# MODEL_ID = "alibayram/tabi-random-init"
# INPUT_IDS_COLUMN = "tabi_input_ids"


config = EmbeddingTrainerConfig(
    student_model=MODEL_ID,
    # Training hyperparameters
    num_epochs=1,
    batch_size=256,
    learning_rate=5e-5,
    warmup_ratio=0.01,
    weight_decay=0.01,
    max_grad_norm=1.0,
    # Loss function
    loss_type="cosine",
    # Pre-encoded dataset column
    input_ids_column=INPUT_IDS_COLUMN,
    embedding_column="teacher_embedding_final",
    # Optimization
    use_bf16=True,
    gradient_checkpointing=True,
    compile_model=True,
    # Output
    output_dir="./trained_model",
    save_steps=100,
    logging_steps=20,
    # WandB
    use_wandb=True,
    wandb_project="distillation",
    wandb_run_name=f"{MODEL_ID.split('/')[-1]}-distillation",
    # Push to Hub
    push_to_hub=True,
    hub_model_id=MODEL_ID,
    hub_token=HF_TOKEN,
)

trainer = EmbeddingDistillationTrainer(config)
metrics = trainer.train(DATASET_ID)

print(f"Final loss: {metrics['train_loss']:.4f}")
