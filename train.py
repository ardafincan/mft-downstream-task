"""Training script for embedding distillation using EmbeddingDistillationTrainer.

This module configures and runs the embedding distillation process,
training a student model to match teacher embeddings from a pre-encoded dataset.

Prerequisites:
    1. Run prepare_dataset.py first to create the encoded dataset
    2. Create .env file with: WANDB_API_KEY=xxx and HF_TOKEN=xxx
"""

import logging
import os
import sys
import traceback

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv

from embedding_trainer import EmbeddingDistillationTrainer, EmbeddingTrainerConfig

load_dotenv()

WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    logger.warning("HF_TOKEN not found in environment variables or .env file!")
if not WANDB_API_KEY:
    logger.warning("WANDB_API_KEY not found in environment variables or .env file!")

if WANDB_API_KEY:
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY

# Pre-encoded dataset from prepare_dataset.py
DATASET_ID = "alibayram/cosmos-corpus-encoded"

# Models configuration
MODELS = [
    # MFT Models (Costomized Tokenizer)
    {
        "name": "mft-embeddinggemma",
        "model_id": "alibayram/mft-downstream-task-embeddinggemma",
        "input_ids_column": "mft_input_ids",
    },
    {
        "name": "mft-embeddingmagibu",
        "model_id": "alibayram/mft-downstream-task-embeddingmagibu",
        "input_ids_column": "mft_input_ids",
    },
    {
        "name": "mft-random-init",
        "model_id": "alibayram/mft-random-init",
        "input_ids_column": "mft_input_ids",
    },
    # TabiBERT Models (BERT Tokenizer)
    {
        "name": "tabi-embeddinggemma",
        "model_id": "alibayram/tabi-downstream-task-embeddinggemma",
        "input_ids_column": "tabi_input_ids",
    },
    {
        "name": "tabi-embeddingmagibu",
        "model_id": "alibayram/tabi-downstream-task-embeddingmagibu",
        "input_ids_column": "tabi_input_ids",
    },
    {
        "name": "tabi-random-init",
        "model_id": "alibayram/tabi-random-init",
        "input_ids_column": "tabi_input_ids",
    },
]

logger.info(f"Found {len(MODELS)} models to train.")

for i, model_cfg in enumerate(MODELS):
    model_name = model_cfg["name"]
    model_id = model_cfg["model_id"]
    input_column = model_cfg["input_ids_column"]

    logger.info(f"\n[{i+1}/{len(MODELS)}] Starting training for: {model_name}")
    logger.info(f"Model ID: {model_id}")
    logger.info(f"Input Column: {input_column}")

    # Unique run name for WandB
    run_name = f"{model_name}-distillation"

    config = EmbeddingTrainerConfig(
        student_model=model_id,
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
        input_ids_column=input_column,
        embedding_column="teacher_embedding_final",
        # Optimization
        use_bf16=True,
        gradient_checkpointing=True,
        compile_model=True,
        # Output
        output_dir=f"./trained_models/{model_name}",
        save_steps=200,
        logging_steps=20,
        # WandB
        use_wandb=True,
        wandb_project="mft-downstream-distillation",
        wandb_run_name=run_name,
        # Push to Hub
        push_to_hub=True,
        hub_model_id=model_id,  # Overwrite the student model repo
        hub_token=HF_TOKEN,
    )

    trainer = None
    try:
        trainer = EmbeddingDistillationTrainer(config)
        metrics = trainer.train(DATASET_ID)
        logger.info(f"✓ Finished {model_name}. Loss: {metrics['train_loss']:.4f}")
    except Exception:
        logger.error(f"✗ Failed training {model_name}")
        traceback.print_exc()

    # Cleanup memory to avoid OOM
    if trainer:
        del trainer
    if config:
        del config
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Cleared CUDA cache for next model.\n")
