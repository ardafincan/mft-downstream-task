#!/usr/bin/env python3
"""
Gradual Fine-tuning Script for Sentence Transformers

This script fine-tunes sentence transformer models using a curriculum learning approach,
starting with a small portion of the data and progressively increasing the dataset size.

Usage:
    # Basic usage (MPS/local testing)
    python gradual_finetune.py

    # With custom phases
    python gradual_finetune.py --phases 0.1 0.25 0.5 1.0

    # Dry run to verify configuration
    python gradual_finetune.py --dry-run

    # Full training with WandB and Hub push
    python gradual_finetune.py --use-wandb --push-to-hub
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import (
    BinaryClassificationEvaluator,
    EmbeddingSimilarityEvaluator,
    SequentialEvaluator,
)
from sentence_transformers.losses import (
    CoSENTLoss,
    MultipleNegativesRankingLoss,
    SoftmaxLoss,
)
from sentence_transformers.training_args import BatchSamplers

# Load environment variables from .env file
load_dotenv()


@dataclass
class GradualFinetuneConfig:
    """Configuration for gradual fine-tuning."""

    # Model configuration
    base_model: str = "magibu/embeddingmagibu-152m"
    output_dir: str = "./gradual_finetune_output"
    hub_model_id: str = "magibu/embeddingmagibu-152m-ft"

    # Gradual training phases (percentage of dataset)
    training_phases: list[float] = field(
        default_factory=lambda: [0.1, 0.25, 0.5, 1.0]
    )

    # Training hyperparameters
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 64
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    num_epochs_per_phase: int = 1
    fp16: bool = False  # Will be set based on device
    bf16: bool = False  # Will be set based on device
    gradient_accumulation_steps: int = 1

    # Logging and saving
    use_wandb: bool = False
    wandb_project: str = "gradual-finetune-turkish"
    push_to_hub: bool = False
    push_after_each_phase: bool = True  # Push to hub after each phase
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100

    # MTEB Evaluation
    run_mteb_eval: bool = True  # Run MTEB-TR evaluation after each phase
    mteb_results_dir: str = "./mteb_results"
    
    # Early stopping
    early_stopping: bool = True  # Stop if performance degrades
    early_stopping_metric: str = "avg_score"  # Metric to monitor
    early_stopping_patience: int = 1  # Stop if no improvement for N phases

    # Device configuration
    device: str = "auto"  # auto, cuda, mps, cpu

    def __post_init__(self):
        """Set device-specific configurations."""
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                self.fp16 = True
            elif torch.backends.mps.is_available():
                self.device = "mps"
                self.fp16 = False  # MPS doesn't support fp16 well
            else:
                self.device = "cpu"
                self.fp16 = False


# Dataset configurations
DATASET_CONFIGS = {
    "msmarco-tr": {
        "name": "selmanbaysan/msmarco-tr_fine_tuning_dataset",
        "columns": {"anchor": "anchor", "positive": "positive"},
        "loss_type": "mnrl",
        "train_split": "train",
        "eval_split": "test",
    },
    "fiqa-tr": {
        "name": "selmanbaysan/fiqa-tr_fine_tuning_dataset",
        "columns": {"anchor": "anchor", "positive": "positive"},
        "loss_type": "mnrl",
        "train_split": "train",
        "eval_split": "test",
    },
    "scifact-tr": {
        "name": "selmanbaysan/scifact-tr_fine_tuning_dataset",
        "columns": {"anchor": "anchor", "positive": "positive"},
        "loss_type": "mnrl",
        "train_split": "train",
        "eval_split": None,  # No eval split
    },
    "nfcorpus-tr": {
        "name": "selmanbaysan/nfcorpus-tr_fine_tuning_dataset",
        "columns": {"anchor": "anchor", "positive": "positive"},
        "loss_type": "mnrl",
        "train_split": "train",
        "eval_split": "test",
    },
    "quora-tr": {
        "name": "selmanbaysan/quora-tr_fine_tuning_dataset",
        "columns": {"anchor": "anchor", "positive": "positive"},
        "loss_type": "mnrl",
        "train_split": "dev",  # This dataset uses dev as training
        "eval_split": "test",
    },
    "wmt16": {
        "name": "selmanbaysan/wmt16_en_tr_fine_tuning_dataset",
        "columns": {"anchor": "anchor", "positive": "positive"},
        "loss_type": "mnrl",
        "train_split": "train",
        "eval_split": "test",
    },
    "multinli-tr": {
        "name": "selmanbaysan/multinli_tr_fine_tuning_dataset",
        "columns": {"premise": "premise", "hypothesis": "hypothesis", "label": "label"},
        "loss_type": "softmax",
        "train_split": "train",
        "eval_split": None,  # No eval split
    },
    "snli-tr": {
        "name": "selmanbaysan/snli_tr_fine_tuning_dataset",
        "columns": {"premise": "premise", "hypothesis": "hypothesis", "label": "label"},
        "loss_type": "softmax",
        "train_split": "train",
        "eval_split": "test",
    },
    "xnli-tr": {
        "name": "selmanbaysan/xnli_tr_fine_tuning_dataset",
        "columns": {"premise": "premise", "hypothesis": "hypothesis", "label": "label"},
        "loss_type": "softmax",
        "train_split": "validation",  # This dataset uses validation as training
        "eval_split": "test",
    },
    "stsb-tr": {
        "name": "selmanbaysan/stsb-tr",
        "columns": {"sentence1": "sentence1", "sentence2": "sentence2", "score": "score"},
        "loss_type": "cosent",
        "train_split": "train",
        "eval_split": "test",
    },
}


def run_mteb_evaluation(
    model_path: str,
    output_dir: str,
    device: str = "auto",
    batch_size: int = 64,
) -> dict[str, Any]:
    """
    Run MTEB-TR benchmark evaluation on a model.
    
    Args:
        model_path: Path to the model to evaluate
        output_dir: Directory to save results
        device: Device to use for evaluation
        batch_size: Batch size for encoding
        
    Returns:
        Dictionary with evaluation results summary
    """
    import subprocess
    
    print(f"\n{'='*60}")
    print("Running MTEB-TR Benchmark Evaluation")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    
    # Run mteb_tr_cli.py
    cmd = [
        sys.executable,
        "mteb_tr_cli.py",
        model_path,
        "--output-folder", output_dir,
        "--batch-size", str(batch_size),
    ]
    if device != "auto":
        cmd.extend(["--device", device])
    
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout
        )
        
        if result.returncode != 0:
            print(f"MTEB evaluation failed: {result.stderr}")
            return {"error": result.stderr, "success": False}
        
        print(result.stdout)
        
    except subprocess.TimeoutExpired:
        print("MTEB evaluation timed out")
        return {"error": "timeout", "success": False}
    except Exception as e:
        print(f"Error running MTEB evaluation: {e}")
        return {"error": str(e), "success": False}
    
    # Parse results
    return parse_mteb_results(output_dir, model_path)


def parse_mteb_results(results_dir: str, model_name: str) -> dict[str, Any]:
    """
    Parse MTEB results from JSON files.
    
    Returns:
        Dictionary with task scores and average score
    """
    results_path = Path(results_dir)
    
    # Find the model results directory
    model_dir_name = model_name.replace("/", "__")
    if "/" in model_name:
        model_dir = results_path / model_dir_name
    else:
        model_dir = results_path / model_name
    
    if not model_dir.exists():
        # Try to find any subdirectory
        subdirs = [d for d in results_path.iterdir() if d.is_dir()]
        if subdirs:
            model_dir = subdirs[0]
        else:
            return {"error": "No results found", "success": False}
    
    # Find the revision directory (usually a hash)
    revision_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
    if not revision_dirs:
        return {"error": "No revision directory found", "success": False}
    
    revision_dir = revision_dirs[0]
    
    # Parse all JSON result files
    task_scores = {}
    for json_file in revision_dir.glob("*.json"):
        if json_file.name == "model_meta.json":
            continue
        
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            task_name = json_file.stem
            
            # Extract main score based on task type
            if "scores" in data:
                scores = data["scores"]
                if "test" in scores:
                    test_scores = scores["test"][0] if scores["test"] else {}
                elif "validation" in scores:
                    test_scores = scores["validation"][0] if scores["validation"] else {}
                else:
                    test_scores = {}
                
                # Get the main metric
                main_score = test_scores.get("main_score", 0)
                task_scores[task_name] = {
                    "main_score": main_score,
                    "all_scores": test_scores,
                }
        except Exception as e:
            print(f"  Warning: Could not parse {json_file.name}: {e}")
    
    if not task_scores:
        return {"error": "No task scores found", "success": False}
    
    # Calculate average score
    avg_score = sum(t["main_score"] for t in task_scores.values()) / len(task_scores)
    
    return {
        "success": True,
        "avg_score": avg_score,
        "task_scores": task_scores,
        "num_tasks": len(task_scores),
    }


def create_commit_message(phase: int, percentage: float, results: dict[str, Any]) -> str:
    """Create a commit message summarizing the phase results."""
    lines = [
        f"Phase {phase} ({percentage*100:.0f}% data) training complete",
        "",
        f"Average Score: {results.get('avg_score', 0):.4f}",
        f"Tasks Evaluated: {results.get('num_tasks', 0)}",
        "",
        "Task Scores:",
    ]
    
    task_scores = results.get("task_scores", {})
    for task_name, scores in sorted(task_scores.items()):
        main_score = scores.get("main_score", 0)
        lines.append(f"  - {task_name}: {main_score:.4f}")
    
    return "\n".join(lines)


def check_early_stopping(
    current_score: float,
    best_score: float,
    patience_counter: int,
    patience: int,
) -> tuple[bool, float, int]:
    """
    Check if training should stop early.
    
    Returns:
        (should_stop, new_best_score, new_patience_counter)
    """
    if current_score > best_score:
        # Improvement - reset counter
        return False, current_score, 0
    else:
        # No improvement
        patience_counter += 1
        if patience_counter >= patience:
            return True, best_score, patience_counter
        return False, best_score, patience_counter


def get_device_info() -> dict[str, Any]:
    """Get information about available devices."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    return info


def load_dataset_with_percentage(
    dataset_name: str,
    percentage: float,
    split: str = "train",
    seed: int = 42,
) -> Dataset:
    """Load a percentage of the dataset for gradual training."""
    print(f"  Loading {dataset_name} ({split} split, {percentage*100:.0f}%)...")
    
    try:
        dataset = load_dataset(dataset_name, split=split)
    except Exception as e:
        print(f"  Warning: Could not load {dataset_name}: {e}")
        return None

    if percentage < 1.0:
        n_samples = max(1, int(len(dataset) * percentage))
        dataset = dataset.shuffle(seed=seed).select(range(n_samples))
    
    print(f"    Loaded {len(dataset)} samples")
    return dataset


def prepare_dataset_for_training(
    dataset: Dataset,
    config: dict[str, Any],
    loss_type: str,
) -> Dataset:
    """Prepare dataset columns for sentence transformers training."""
    columns = config["columns"]
    
    if loss_type == "mnrl":
        # For MultipleNegativesRankingLoss: needs anchor, positive
        dataset = dataset.rename_columns({
            columns["anchor"]: "anchor",
            columns["positive"]: "positive",
        })
        # Filter out samples with None values
        dataset = dataset.filter(
            lambda x: x["anchor"] is not None and x["positive"] is not None
        )
    elif loss_type == "softmax":
        # For SoftmaxLoss: needs premise, hypothesis, label
        dataset = dataset.rename_columns({
            columns["premise"]: "premise",
            columns["hypothesis"]: "hypothesis",
            columns["label"]: "label",
        })
        # Filter out samples with None values
        dataset = dataset.filter(
            lambda x: x["premise"] is not None and x["hypothesis"] is not None and x["label"] is not None
        )
    elif loss_type == "cosent":
        # For CoSENTLoss: needs sentence1, sentence2, score
        dataset = dataset.rename_columns({
            columns["sentence1"]: "sentence1",
            columns["sentence2"]: "sentence2",
            columns["score"]: "score",
        })
        # Filter out samples with None values
        dataset = dataset.filter(
            lambda x: x["sentence1"] is not None and x["sentence2"] is not None and x["score"] is not None
        )
    
    return dataset


def get_loss_function(
    loss_type: str,
    model: SentenceTransformer,
    num_labels: int = 3,
) -> Any:
    """Get the appropriate loss function based on dataset type."""
    if loss_type == "mnrl":
        return MultipleNegativesRankingLoss(model=model)
    elif loss_type == "softmax":
        return SoftmaxLoss(
            model=model,
            sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
            num_labels=num_labels,
        )
    elif loss_type == "cosent":
        return CoSENTLoss(model=model)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def create_evaluators(model: SentenceTransformer) -> SequentialEvaluator:
    """Create evaluators for multiple evaluation datasets."""
    evaluators = []
    
    # STSb-TR evaluator (Semantic Similarity)
    print("Loading STSb-TR evaluation dataset...")
    try:
        stsb_eval = load_dataset("selmanbaysan/stsb-tr", split="test")
        stsb_evaluator = EmbeddingSimilarityEvaluator(
            sentences1=stsb_eval["sentence1"],
            sentences2=stsb_eval["sentence2"],
            scores=[s / 5.0 for s in stsb_eval["score"]],  # Normalize to 0-1
            name="stsb-tr",
        )
        evaluators.append(stsb_evaluator)
    except Exception as e:
        print(f"  Warning: Could not load STSb-TR: {e}")
    
    # SNLI-TR evaluator (Binary Classification)
    print("Loading SNLI-TR evaluation dataset...")
    try:
        snli_eval = load_dataset("selmanbaysan/snli_tr_fine_tuning_dataset", split="test")
        # Filter to only entailment (1) vs non-entailment (0)
        snli_eval = snli_eval.filter(lambda x: x["label"] in [0, 1])
        snli_evaluator = BinaryClassificationEvaluator(
            sentences1=snli_eval["premise"],
            sentences2=snli_eval["hypothesis"],
            labels=snli_eval["label"],
            name="snli-tr",
        )
        evaluators.append(snli_evaluator)
    except Exception as e:
        print(f"  Warning: Could not load SNLI-TR: {e}")
    
    # XNLI-TR evaluator (Binary Classification)
    print("Loading XNLI-TR evaluation dataset...")
    try:
        xnli_eval = load_dataset("selmanbaysan/xnli_tr_fine_tuning_dataset", split="test")
        xnli_eval = xnli_eval.filter(lambda x: x["label"] in [0, 1])
        xnli_evaluator = BinaryClassificationEvaluator(
            sentences1=xnli_eval["premise"],
            sentences2=xnli_eval["hypothesis"],
            labels=xnli_eval["label"],
            name="xnli-tr",
        )
        evaluators.append(xnli_evaluator)
    except Exception as e:
        print(f"  Warning: Could not load XNLI-TR: {e}")
    
    if not evaluators:
        print("Warning: No evaluators could be created!")
        return None
    
    return SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[0])


def load_all_datasets_for_phase(
    percentage: float,
    selected_datasets: list[str] | None = None,
) -> dict[str, Dataset]:
    """Load all datasets with the specified percentage."""
    datasets = {}
    dataset_keys = selected_datasets or list(DATASET_CONFIGS.keys())
    
    for key in dataset_keys:
        config = DATASET_CONFIGS[key]
        train_split = config.get("train_split", "train")
        dataset = load_dataset_with_percentage(
            config["name"],
            percentage,
            split=train_split,
        )
        if dataset is not None:
            dataset = prepare_dataset_for_training(
                dataset, config, config["loss_type"]
            )
            datasets[key] = dataset
    
    return datasets



def train_phase(
    model: SentenceTransformer,
    config: GradualFinetuneConfig,
    phase_idx: int,
    percentage: float,
    datasets: dict[str, Dataset],
    evaluator: SequentialEvaluator | None,
) -> dict[str, Any]:
    """Train a single phase of the gradual fine-tuning."""
    phase_output_dir = Path(config.output_dir) / f"phase_{phase_idx + 1}_{int(percentage * 100)}pct"
    phase_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare loss functions for each dataset type
    losses = {}
    for key, dataset in datasets.items():
        loss_type = DATASET_CONFIGS[key]["loss_type"]
        loss = get_loss_function(loss_type, model)
        losses[key] = loss
    
    # Create training arguments
    run_name = f"phase_{phase_idx + 1}_{int(percentage * 100)}pct"
    
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(phase_output_dir),
        num_train_epochs=config.num_epochs_per_phase,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        fp16=config.fp16,
        bf16=config.bf16,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps" if evaluator else "no",
        eval_steps=config.eval_steps if evaluator else None,
        save_strategy="steps",
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        run_name=run_name,
        report_to=["wandb"] if config.use_wandb else [],
        load_best_model_at_end=True if evaluator else False,
        metric_for_best_model="stsb-tr_spearman_cosine" if evaluator else None,
    )
    
    # Create trainer with multi-dataset support
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets,
        eval_dataset=None,  # We use evaluators instead
        evaluator=evaluator,
        loss=losses,
    )
    
    # Train
    print(f"\n{'='*60}")
    print(f"Starting training for Phase {phase_idx + 1} ({percentage*100:.0f}% data)")
    print(f"{'='*60}")
    
    train_result = trainer.train()
    
    # Save the model
    model.save(str(phase_output_dir / "final_model"))
    
    # Evaluate
    eval_results = {}
    if evaluator:
        print("\nRunning evaluation...")
        eval_results = evaluator(model)
        print(f"Evaluation results: {eval_results}")
    
    return {
        "phase": phase_idx + 1,
        "percentage": percentage,
        "train_loss": train_result.training_loss,
        "eval_results": eval_results,
        "output_dir": str(phase_output_dir),
    }


def gradual_finetune(config: GradualFinetuneConfig, dry_run: bool = False) -> list[dict]:
    """Main function to run gradual fine-tuning."""
    # Setup
    print("\n" + "="*60)
    print("Gradual Fine-tuning for Sentence Transformers")
    print("="*60)
    
    # Check device
    device_info = get_device_info()
    print(f"\nDevice information: {json.dumps(device_info, indent=2)}")
    print(f"Using device: {config.device}")
    
    # Setup WandB if enabled
    if config.use_wandb:
        try:
            import wandb
            wandb.login(key=os.getenv("WANDB_API_KEY"))
            wandb.init(
                project=config.wandb_project,
                config={
                    "base_model": config.base_model,
                    "training_phases": config.training_phases,
                    "learning_rate": config.learning_rate,
                    "batch_size": config.per_device_train_batch_size,
                },
            )
        except Exception as e:
            print(f"Warning: Could not initialize WandB: {e}")
            config.use_wandb = False
    
    # Setup HuggingFace Hub token
    if config.push_to_hub:
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)
        else:
            print("Warning: HF_TOKEN not found in .env, push to hub may fail")
    
    # Load model
    print(f"\nLoading base model: {config.base_model}")
    model = SentenceTransformer(config.base_model, device=config.device)
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    
    # Create output directory
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create evaluators
    print("\nCreating evaluators...")
    evaluator = create_evaluators(model)
    
    if dry_run:
        print("\n" + "="*60)
        print("DRY RUN - Configuration Summary")
        print("="*60)
        print(f"Base model: {config.base_model}")
        print(f"Training phases: {config.training_phases}")
        print(f"Datasets: {list(DATASET_CONFIGS.keys())}")
        print(f"Output directory: {config.output_dir}")
        print(f"Device: {config.device}")
        print(f"WandB: {config.use_wandb}")
        print(f"Push to Hub: {config.push_to_hub} -> {config.hub_model_id}")
        
        # Load a small sample to verify datasets
        print("\nVerifying datasets (loading 1% samples)...")
        test_datasets = load_all_datasets_for_phase(0.01)
        for name, ds in test_datasets.items():
            print(f"  {name}: {len(ds)} samples, columns: {ds.column_names}")
        
        print("\nDry run completed successfully!")
        return []
    
    # Training loop
    all_results = []
    best_score = -float("inf")
    patience_counter = 0
    
    for phase_idx, percentage in enumerate(config.training_phases):
        print(f"\n{'#'*60}")
        print(f"# PHASE {phase_idx + 1} of {len(config.training_phases)}: Training with {percentage*100:.0f}% data")
        print(f"{'#'*60}")
        
        # Load datasets for this phase
        print(f"\nLoading datasets ({percentage*100:.0f}% of each)...")
        datasets = load_all_datasets_for_phase(percentage)
        
        total_samples = sum(len(ds) for ds in datasets.values())
        print(f"Total training samples for this phase: {total_samples:,}")
        
        # Train this phase
        phase_results = train_phase(
            model=model,
            config=config,
            phase_idx=phase_idx,
            percentage=percentage,
            datasets=datasets,
            evaluator=evaluator,
        )
        
        # Save phase model
        phase_model_path = output_path / f"phase_{phase_idx + 1}_{int(percentage * 100)}pct" / "final_model"
        
        # Run MTEB evaluation if enabled
        mteb_results = {}
        if config.run_mteb_eval:
            mteb_output_dir = Path(config.mteb_results_dir) / f"phase_{phase_idx + 1}"
            mteb_results = run_mteb_evaluation(
                model_path=str(phase_model_path),
                output_dir=str(mteb_output_dir),
                device=config.device,
                batch_size=config.per_device_eval_batch_size,
            )
            phase_results["mteb_results"] = mteb_results
            
            if mteb_results.get("success"):
                print(f"\nMTEB Average Score: {mteb_results['avg_score']:.4f}")
                print(f"Tasks Evaluated: {mteb_results['num_tasks']}")
        
        all_results.append(phase_results)
        
        # Save results
        results_file = output_path / "training_results.json"
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\nPhase {phase_idx + 1} completed!")
        print(f"Results saved to: {results_file}")
        
        # Push to Hub after each phase if enabled
        if config.push_to_hub and config.push_after_each_phase:
            phase_hub_id = f"{config.hub_model_id}-phase{phase_idx + 1}"
            commit_message = create_commit_message(phase_idx + 1, percentage, mteb_results)
            
            print(f"\nPushing to Hub: {config.hub_model_id}")
            try:
                model.push_to_hub(
                    config.hub_model_id,
                    commit_message=commit_message,
                )
                print("Model pushed successfully!")
                print(f"Commit message:\n{commit_message}")
            except Exception as e:
                print(f"Error pushing to hub: {e}")
        
        # Check early stopping
        if config.early_stopping and mteb_results.get("success"):
            current_score = mteb_results.get("avg_score", 0)
            
            should_stop, best_score, patience_counter = check_early_stopping(
                current_score=current_score,
                best_score=best_score,
                patience_counter=patience_counter,
                patience=config.early_stopping_patience,
            )
            
            if should_stop:
                print(f"\n{'!'*60}")
                print(f"! EARLY STOPPING: Performance degraded for {patience_counter} phase(s)")
                print(f"! Best score: {best_score:.4f}, Current score: {current_score:.4f}")
                print(f"! Stopping training at phase {phase_idx + 1}")
                print(f"{'!'*60}")
                break
            else:
                if current_score > best_score - 0.0001:  # Allow small tolerance
                    print(f"\nPerformance improved: {best_score:.4f} -> {current_score:.4f}")
                else:
                    print(f"\nPerformance dropped: {best_score:.4f} -> {current_score:.4f}")
                    print(f"Patience: {patience_counter}/{config.early_stopping_patience}")
    
    # Final save and push to hub
    final_model_path = output_path / "final_model"
    model.save(str(final_model_path))
    print(f"\nFinal model saved to: {final_model_path}")
    
    if config.push_to_hub:
        print(f"\nPushing final model to Hub: {config.hub_model_id}")
        try:
            # Create final commit message with all phase results
            final_commit_lines = ["Final model after gradual fine-tuning", ""]
            for result in all_results:
                phase = result["phase"]
                pct = result["percentage"] * 100
                if "mteb_results" in result and result["mteb_results"].get("success"):
                    avg = result["mteb_results"]["avg_score"]
                    final_commit_lines.append(f"Phase {phase} ({pct:.0f}%): {avg:.4f}")
            
            model.push_to_hub(
                config.hub_model_id,
                commit_message="\n".join(final_commit_lines),
            )
            print("Final model pushed successfully!")
        except Exception as e:
            print(f"Error pushing to hub: {e}")
    
    # Final summary
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print("\nPhase Results Summary:")
    for result in all_results:
        print(f"  Phase {result['phase']} ({result['percentage']*100:.0f}%): "
              f"Loss={result['train_loss']:.4f}")
        if "mteb_results" in result and result["mteb_results"].get("success"):
            print(f"    MTEB Avg Score: {result['mteb_results']['avg_score']:.4f}")
        if result.get('eval_results'):
            for metric, value in result['eval_results'].items():
                print(f"    {metric}: {value:.4f}")
    
    return all_results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Gradual Fine-tuning Script for Sentence Transformers"
    )
    
    parser.add_argument(
        "--base-model",
        type=str,
        default="magibu/embeddingmagibu-152m",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./gradual_finetune_output",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        default="magibu/embeddingmagibu-152m-ft",
        help="Model ID for pushing to Hub",
    )
    parser.add_argument(
        "--phases",
        type=float,
        nargs="+",
        default=[0.1, 0.25, 0.5, 1.0],
        help="Training phases (percentages of dataset)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size per device",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--epochs-per-phase",
        type=int,
        default=1,
        help="Number of epochs per training phase",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable WandB logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="gradual-finetune-turkish",
        help="WandB project name",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push final model to HuggingFace Hub",
    )
    parser.add_argument(
        "--no-push-after-phase",
        action="store_true",
        help="Don't push to Hub after each phase (only push final model)",
    )
    parser.add_argument(
        "--no-mteb-eval",
        action="store_true",
        help="Skip MTEB-TR benchmark evaluation after each phase",
    )
    parser.add_argument(
        "--mteb-results-dir",
        type=str,
        default="./mteb_results",
        help="Directory to save MTEB evaluation results",
    )
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Disable early stopping even if performance degrades",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=1,
        help="Stop if no improvement for N phases",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without training to verify configuration",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use for training",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    config = GradualFinetuneConfig(
        base_model=args.base_model,
        output_dir=args.output_dir,
        hub_model_id=args.hub_model_id,
        training_phases=args.phases,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_epochs_per_phase=args.epochs_per_phase,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        push_to_hub=args.push_to_hub,
        push_after_each_phase=not args.no_push_after_phase,
        run_mteb_eval=not args.no_mteb_eval,
        mteb_results_dir=args.mteb_results_dir,
        early_stopping=not args.no_early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        device=args.device,
    )
    
    gradual_finetune(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
