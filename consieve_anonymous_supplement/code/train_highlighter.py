#!/usr/bin/env python3
"""Train a Provence-style semantic highlighter model using open_provence framework.

This script fine-tunes a semantic highlighting model on your
temporal QA data with hard negatives from dimension perturbation.

Two training modes:
1. Local only: Fine-tune on your annotated temporal QA data
2. Joint: Fine-tune on combined local + Zilliz data

Usage:
    # Train using YAML config (recommended)
    python scripts/train_highlighter.py --config configs/ts-highlighter-v1.yaml
    python scripts/train_highlighter.py --config configs/ts-highlighter-v1.yaml --wandb

    # Preview data stats
    python scripts/train_highlighter.py --stats

    # Train on local data only (Zilliz base model)
    python scripts/train_highlighter.py --local-only --output-dir models/highlighter-local

    # Train with ModernBERT base model
    python scripts/train_highlighter.py --local-only --base-model modernbert

    # Train on joint data (local + Zilliz)
    python scripts/train_highlighter.py --with-zilliz --output-dir models/highlighter-joint

    # Train from HuggingFace dataset (works anywhere!)
    python scripts/train_highlighter.py --hf-dataset anonymous/ts-highlighter-data --wandb

    # Dry run (prepare data but don't train)
    python scripts/train_highlighter.py --config configs/ts-highlighter-v1.yaml --dry-run

Architecture (OpenProvence):
    Base model: zilliz/semantic-highlight-bilingual-v1 (BGE-M3 Reranker, 0.6B params)
             or Alibaba-NLP/gte-reranker-modernbert-base (ModernBERT)
    Ranking head: BOS token -> scalar score (for knowledge distillation)
    Pruning head: Per-token binary classification (keep/prune)
    Loss: ranking_weight * MSE + pruning_weight * CrossEntropy

References:
    - Provence paper: https://arxiv.org/abs/2501.16214
    - Zilliz model: https://huggingface.co/zilliz/semantic-highlight-bilingual-v1
    - Open Provence: https://github.com/hotchpotch/open_provence
"""
import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import yaml

from evaluation.query_formatter import format_and_validate_query

# Add project root and open_provence to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "open_provence"))

logger = logging.getLogger(__name__)

# Rich imports for modern display (optional)
try:
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TaskProgressColumn
    from rich.live import Live
    from rich import box
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

# =============================================================================
# RICH DISPLAY HELPERS
# =============================================================================


def display_training_config(config, world_size: int = 1, is_distributed: bool = False):
    """Display training configuration in a modern, rich format."""
    if not RICH_AVAILABLE:
        # Fallback to plain text
        print(f"\n{'='*60}")
        print("SEMANTIC HIGHLIGHTER TRAINING (OpenProvence)")
        print(f"{'='*60}")
        print(f"Base model: {config.base_model}")
        print(f"Output dir: {config.output_dir}")
        print(f"Max length: {config.max_length}")
        effective_batch = config.batch_size * config.gradient_accumulation_steps * world_size
        if is_distributed:
            print(f"Distributed training: {world_size} GPUs")
        print(f"Effective batch size: {effective_batch}")
        print(f"Learning rate: {config.learning_rate}")
        print(f"Epochs: {config.num_epochs}")
        print(f"Ranking weight: {config.ranking_weight}")
        print(f"Pruning weight: {config.pruning_weight}")
        return

    # Rich display
    effective_batch = config.batch_size * config.gradient_accumulation_steps * world_size

    # Create config table
    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    # Model info
    table.add_row("Base Model", config.base_model.split("/")[-1])
    table.add_row("Output Dir", str(config.output_dir))
    table.add_row("Max Length", str(config.max_length))
    table.add_row("", "")  # Spacer

    # Training params
    if is_distributed:
        batch_str = f"{config.batch_size} × {config.gradient_accumulation_steps} × {world_size} = [bold]{effective_batch}[/bold]"
        table.add_row("GPUs", f"[bold yellow]{world_size}[/bold yellow]")
    else:
        batch_str = f"{config.batch_size} × {config.gradient_accumulation_steps} = [bold]{effective_batch}[/bold]"
    table.add_row("Batch Size", batch_str)
    table.add_row("Learning Rate", f"{config.learning_rate:.1e}")
    table.add_row("Epochs", str(config.num_epochs))
    table.add_row("Warmup Ratio", str(config.warmup_ratio))
    table.add_row("", "")  # Spacer

    # Loss weights
    table.add_row("Ranking Weight", f"{config.ranking_weight:.1f}")
    table.add_row("Pruning Weight", f"{config.pruning_weight:.1f}")

    # Source weights if configured
    if config.source_weights:
        weights_str = ", ".join(f"{k}:{v}x" for k, v in config.source_weights.items())
        table.add_row("Source Weights", weights_str)

    # Create panel
    panel = Panel(
        table,
        title="[bold blue]🚀 Semantic Highlighter Training[/bold blue]",
        subtitle="[dim]OpenProvence Framework[/dim]",
        border_style="blue",
    )
    console.print(panel)


def display_dataset_info(train_size: int, valid_size: int | None, eval_size: int | None,
                         hard_neg_count: int, hard_neg_pct: float, source_counts: dict | None = None):
    """Display dataset information in a rich format."""
    if not RICH_AVAILABLE:
        print(f"\nDataset loaded:")
        print(f"  Train: {train_size} samples")
        if valid_size:
            print(f"  Valid: {valid_size} samples")
        if eval_size:
            print(f"  Eval: {eval_size} samples")
        print(f"  Hard negatives: {hard_neg_count} ({hard_neg_pct:.1f}%)")
        if source_counts:
            print(f"  Sources: {source_counts}")
        return

    table = Table(title="📊 Dataset", box=box.ROUNDED, show_lines=False)
    table.add_column("Split", style="cyan")
    table.add_column("Samples", style="green", justify="right")
    table.add_column("Details", style="dim")

    table.add_row("Train", f"{train_size:,}", f"{hard_neg_count:,} hard negatives ({hard_neg_pct:.1f}%)")
    if valid_size:
        table.add_row("Valid", f"{valid_size:,}", "")
    if eval_size:
        table.add_row("Eval", f"{eval_size:,}", "")

    console.print(table)

    # Show source distribution if available
    if source_counts:
        src_table = Table(title="📁 Source Distribution", box=box.SIMPLE, show_header=False)
        src_table.add_column("Source", style="cyan")
        src_table.add_column("Count", style="green", justify="right")
        for source, count in sorted(source_counts.items()):
            src_table.add_row(source, f"{count:,}")
        console.print(src_table)


def display_model_info(model_name: str, hidden_size: int, num_params: float, flash_attn: bool):
    """Display model information."""
    if not RICH_AVAILABLE:
        print(f"\nModel loaded: {model_name}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Parameters: {num_params:.1f}M")
        print(f"  FlashAttention: {'Yes' if flash_attn else 'No'}")
        return

    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    table.add_column("", style="cyan")
    table.add_column("", style="green")
    table.add_row("Model", model_name.split("/")[-1])
    table.add_row("Hidden Size", str(hidden_size))
    table.add_row("Parameters", f"{num_params:.1f}M")
    table.add_row("FlashAttention", "[green]✓[/green]" if flash_attn else "[yellow]✗[/yellow]")

    panel = Panel(table, title="[bold]🤖 Model[/bold]", border_style="green")
    console.print(panel)


def display_training_complete(output_path: str, best_metric: float | None = None):
    """Display training completion message."""
    if not RICH_AVAILABLE:
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Model saved to: {output_path}")
        if best_metric is not None:
            print(f"Best eval loss: {best_metric:.4f}")
        return

    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    table.add_column("", style="cyan")
    table.add_column("", style="green")
    table.add_row("Model Path", str(output_path))
    if best_metric is not None:
        table.add_row("Best Eval Loss", f"{best_metric:.4f}")

    panel = Panel(
        table,
        title="[bold green]✅ Training Complete[/bold green]",
        border_style="green",
    )
    console.print(panel)


# =============================================================================
# RICH PROGRESS CALLBACK
# =============================================================================


class RichProgressCallback:
    """Custom callback for rich progress display during training.

    Shows a live progress bar with metrics instead of dict spam.
    """

    def __init__(self, total_steps: int, console: Console):
        self.total_steps = total_steps
        self.console = console
        self.progress = None
        self.task_id = None
        self.live = None
        self.current_metrics = {}
        self.current_step = 0
        self.current_epoch = 0

    def _create_progress(self):
        """Create the progress display."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=self.console,
            transient=False,
        )

    def _format_metrics_table(self) -> Table:
        """Format current metrics as a compact table."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        # Key metrics to show
        if "loss" in self.current_metrics:
            table.add_row("Loss", f"{self.current_metrics['loss']:.4f}")
        if "pruning_loss" in self.current_metrics:
            table.add_row("Pruning", f"{self.current_metrics['pruning_loss']:.4f}")
        if "eval_loss" in self.current_metrics:
            table.add_row("Eval Loss", f"{self.current_metrics['eval_loss']:.4f}")
        if "learning_rate" in self.current_metrics:
            lr = self.current_metrics["learning_rate"]
            table.add_row("LR", f"{lr:.2e}")

        return table

    def on_train_begin(self):
        """Called at the start of training."""
        self.progress = self._create_progress()
        self.task_id = self.progress.add_task(
            f"Epoch {self.current_epoch + 1}",
            total=self.total_steps,
        )
        self.live = Live(self.progress, console=self.console, refresh_per_second=4)
        self.live.start()

    def on_train_end(self):
        """Called at the end of training."""
        if self.live:
            self.live.stop()

    def on_step_end(self, step: int, logs: dict | None = None):
        """Called at the end of each step."""
        self.current_step = step
        if logs:
            self.current_metrics.update(logs)

        if self.progress and self.task_id is not None:
            # Update progress
            self.progress.update(self.task_id, completed=step)

            # Update description with key metrics
            desc_parts = [f"Epoch {self.current_epoch + 1}"]
            if "loss" in self.current_metrics:
                desc_parts.append(f"loss={self.current_metrics['loss']:.4f}")
            if "eval_loss" in self.current_metrics:
                desc_parts.append(f"eval={self.current_metrics['eval_loss']:.4f}")

            self.progress.update(self.task_id, description=" • ".join(desc_parts))

    def on_epoch_end(self, epoch: int):
        """Called at the end of each epoch."""
        self.current_epoch = epoch


def create_rich_callback_class():
    """Create a TrainerCallback class that uses RichProgressCallback.

    This is done dynamically to avoid importing transformers at module load time.
    """
    from transformers import TrainerCallback, TrainerState, TrainerControl
    from transformers.trainer_callback import TrainerCallback

    class RichTrainerCallback(TrainerCallback):
        """HuggingFace Trainer callback that uses rich for progress display."""

        def __init__(self):
            self.rich_progress = None
            self._initialized = False

        def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
            """Initialize rich progress at training start."""
            if RICH_AVAILABLE and console:
                self.rich_progress = RichProgressCallback(
                    total_steps=state.max_steps,
                    console=console,
                )
                self.rich_progress.on_train_begin()
                self._initialized = True

        def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
            """Clean up rich progress at training end."""
            if self.rich_progress:
                self.rich_progress.on_train_end()

        def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
            """Update progress bar on every step for smooth animation."""
            if self.rich_progress:
                self.rich_progress.on_step_end(state.global_step)

        def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
            """Update metrics on log (but don't print - just update internal state)."""
            if self.rich_progress and logs:
                # Filter out non-metric keys
                metrics = {k: v for k, v in logs.items()
                          if isinstance(v, (int, float)) and not k.startswith("_")}
                self.rich_progress.current_metrics.update(metrics)

        def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
            """Update epoch counter."""
            if self.rich_progress:
                self.rich_progress.on_epoch_end(int(state.epoch))

    return RichTrainerCallback


# =============================================================================
# CONFIGURATION
# =============================================================================

# Model presets
MODEL_PRESETS = {
    # Zilliz model (BGE-M3 based, 0.6B params, bilingual EN/ZH)
    "zilliz": "zilliz/semantic-highlight-bilingual-v1",
    # Open Provence ModernBERT (310M params, bilingual EN/JA, MIT license)
    "open-provence": "hotchpotch/open-provence-reranker-large-v1",
    "modernbert": "hotchpotch/open-provence-reranker-large-v1",  # Alias
    # GTE ModernBERT base (for experimentation)
    "gte-modernbert": "Alibaba-NLP/gte-reranker-modernbert-base",
    # Japanese small model
    "japanese-xsmall": "hotchpotch/japanese-reranker-xsmall-v2",
}


def load_config_from_yaml(yaml_path: Path) -> dict:
    """Load training config from YAML file."""
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    # Flatten nested config into HighlighterConfig fields
    flat = {}

    # Model args
    if "model_args" in config:
        model_args = config["model_args"]
        if "model_name_or_path" in model_args:
            flat["base_model"] = model_args["model_name_or_path"]
        if "revision" in model_args:
            flat["base_model_revision"] = model_args["revision"]

    # Data args
    if "data_args" in config:
        data_args = config["data_args"]
        if "hf_dataset" in data_args:
            flat["hf_dataset"] = data_args["hf_dataset"]
        if "train_file" in data_args:
            flat["train_file"] = data_args["train_file"]
        if "valid_file" in data_args:
            flat["valid_file"] = data_args["valid_file"]
        if "eval_file" in data_args:
            flat["eval_file"] = data_args["eval_file"]
        if "source_weights" in data_args:
            flat["source_weights"] = data_args["source_weights"]
        # Data filtering options (for ablation studies)
        if "exclude_perturbed" in data_args:
            flat["exclude_perturbed"] = data_args["exclude_perturbed"]
        if "exclude_empty" in data_args:
            flat["exclude_empty"] = data_args["exclude_empty"]
        if "exclude_sources" in data_args:
            flat["exclude_sources"] = data_args["exclude_sources"]

    # Training args
    if "training_args" in config:
        ta = config["training_args"]
        if "output_dir" in ta:
            flat["output_dir"] = ta["output_dir"]
        if "learning_rate" in ta:
            flat["learning_rate"] = ta["learning_rate"]
        if "per_device_train_batch_size" in ta:
            flat["batch_size"] = ta["per_device_train_batch_size"]
        if "gradient_accumulation_steps" in ta:
            flat["gradient_accumulation_steps"] = ta["gradient_accumulation_steps"]
        if "num_train_epochs" in ta:
            flat["num_epochs"] = ta["num_train_epochs"]
        if "warmup_ratio" in ta:
            flat["warmup_ratio"] = ta["warmup_ratio"]
        if "weight_decay" in ta:
            flat["weight_decay"] = ta["weight_decay"]
        if "logging_steps" in ta:
            flat["logging_steps"] = ta["logging_steps"]
        if "eval_steps" in ta:
            flat["eval_steps"] = ta["eval_steps"]
        if "save_steps" in ta:
            flat["save_steps"] = ta["save_steps"]

    # Loss args
    if "loss_args" in config:
        la = config["loss_args"]
        if "ranking_weight" in la:
            flat["ranking_weight"] = la["ranking_weight"]
        if "pruning_weight" in la:
            flat["pruning_weight"] = la["pruning_weight"]

    # Wandb
    if "wandb" in config:
        wb = config["wandb"]
        if wb.get("enabled"):
            flat["use_wandb"] = True
        if "project" in wb:
            flat["wandb_project"] = wb["project"]
        if "run_name" in wb:
            flat["wandb_run_name"] = wb["run_name"]

    # Top-level
    if "seed" in config:
        flat["seed"] = config["seed"]
    if "max_length" in config:
        flat["max_length"] = config["max_length"]

    return flat


def create_config_from_yaml_and_args(yaml_config: dict, args: argparse.Namespace) -> "HighlighterConfig":
    """Create HighlighterConfig from YAML config, with CLI args as overrides."""
    # Start with YAML values
    config_dict = dict(yaml_config)

    # CLI overrides (only if explicitly provided)
    if args.base_model != "zilliz":  # Default value
        config_dict["base_model"] = MODEL_PRESETS.get(args.base_model.lower(), args.base_model)
    if args.max_length != 2048:
        config_dict["max_length"] = args.max_length
    if args.batch_size != 4:
        config_dict["batch_size"] = args.batch_size
    if args.gradient_accumulation != 16:
        config_dict["gradient_accumulation_steps"] = args.gradient_accumulation
    if args.learning_rate != 5e-5:
        config_dict["learning_rate"] = args.learning_rate
    if args.epochs != 1:
        config_dict["num_epochs"] = args.epochs
    if args.ranking_weight != 0.0:
        config_dict["ranking_weight"] = args.ranking_weight
    if args.pruning_weight != 1.0:
        config_dict["pruning_weight"] = args.pruning_weight
    if args.seed != 42:
        config_dict["seed"] = args.seed
    if args.output_dir:
        config_dict["output_dir"] = args.output_dir
    if args.wandb:
        config_dict["use_wandb"] = True
    if args.wandb_project != "ts-highlighter":
        config_dict["wandb_project"] = args.wandb_project
    if args.wandb_run_name:
        config_dict["wandb_run_name"] = args.wandb_run_name

    # Resolve base model preset
    if "base_model" in config_dict:
        config_dict["base_model"] = MODEL_PRESETS.get(
            config_dict["base_model"].lower() if isinstance(config_dict["base_model"], str) else config_dict["base_model"],
            config_dict["base_model"]
        )

    # Filter to valid HighlighterConfig fields
    valid_fields = {f.name for f in fields(HighlighterConfig)}
    filtered = {k: v for k, v in config_dict.items() if k in valid_fields}

    return HighlighterConfig(**filtered)


@dataclass
class HighlighterConfig:
    """Configuration for highlighter training using OpenProvence framework."""

    # Model
    base_model: str = "zilliz/semantic-highlight-bilingual-v1"
    base_model_revision: str | None = None
    max_length: int = 2048  # Max tokens (model supports 8192, but shorter is faster)
    num_labels: int = 2  # 2-class for ranking head (Provence default)

    # Training (OpenProvence defaults)
    learning_rate: float = 5e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 16  # Effective batch = 64
    num_epochs: int = 1  # OpenProvence uses 1 epoch
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # Loss weights (pruning-only by default for highlighter)
    ranking_weight: float = 0.0  # Set to 0.05 for joint ranking+pruning
    pruning_weight: float = 1.0

    # Data
    train_file: str = "datasets/training_local/train.jsonl"
    valid_file: str = "datasets/training_local/valid.jsonl"
    eval_file: str = "datasets/training_local/eval.jsonl"
    hf_dataset: str | None = None  # HuggingFace dataset name (e.g., "anonymous/ts-highlighter-data")

    # Source weighting (upsampling)
    # Keys are source names, values are integer multipliers (1 = no change, 2 = duplicate)
    source_weights: dict[str, int] | None = None

    # Data filtering options (for ablation studies)
    # exclude_perturbed: bool - exclude samples where source contains "perturbed"
    # exclude_empty: bool - exclude samples with no relevant chunks (all labels=0)
    # exclude_sources: list[str] - exclude samples from specific sources (e.g., ["timeqa"])
    exclude_perturbed: bool = False
    exclude_empty: bool = False
    exclude_sources: list[str] | None = None

    # Hard negative handling
    hard_negative_weight: float = 1.0  # Weight for hard negative samples

    # Output
    output_dir: str = "models/highlighter"
    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 200

    # Wandb
    use_wandb: bool = False
    wandb_project: str = "ts-highlighter"
    wandb_run_name: str | None = None

    # Seed
    seed: int = 42


# =============================================================================
# DATA LOADING & CONVERSION
# =============================================================================


def load_training_data(file_path: Path) -> list[dict]:
    """Load training data from JSONL file."""
    samples = []
    with open(file_path) as f:
        for line in f:
            sample = json.loads(line)
            # Skip failed annotations
            if not sample.get("annotation_success", True):
                continue
            # Detect hard negatives from source field
            source = sample.get("source", "")
            if "hard_negative" in source or "perturbed" in source:
                sample["is_hard_negative"] = True
            samples.append(sample)
    return samples


def convert_to_open_provence_format(samples: list[dict]) -> list[dict]:
    """Convert our sentence-label format to OpenProvence format.

    Our format:
        - query: str
        - texts: list[str] (sentences)
        - labels: list[int] (0/1 for each sentence)

    OpenProvence format:
        - query: str
        - texts: list[str] (documents - we use one document per sample)
        - chunks_pos: list[list[tuple[int, int]]] (character positions per doc)
        - relevant_chunks: list[list[int]] (indices of relevant chunks per doc)
        - labels: list[int] (document-level labels for ranking)
    """
    converted = []

    for sample in samples:
        query = format_and_validate_query(sample)
        sentences = sample.get("texts", sample.get("sentences", []))
        sentence_labels = sample.get("labels", sample.get("sentence_labels", []))
        is_hard_negative = sample.get("is_hard_negative", False)

        # Skip empty samples
        if not sentences:
            continue

        # For hard negatives, all sentences are irrelevant
        if is_hard_negative:
            sentence_labels = [0] * len(sentences)

        # Ensure labels match sentences
        if len(sentence_labels) != len(sentences):
            sentence_labels = sentence_labels[: len(sentences)]
            sentence_labels += [0] * (len(sentences) - len(sentence_labels))

        # Concatenate sentences to form the document
        document = " ".join(sentences)

        # Calculate character positions for each sentence (chunk)
        chunks_pos = []
        pos = 0
        for sent in sentences:
            start = pos
            end = pos + len(sent)
            chunks_pos.append([start, end])
            pos = end + 1  # +1 for space

        # Find indices of relevant chunks (sentences with label=1)
        relevant_chunks = [i for i, label in enumerate(sentence_labels) if label == 1]

        # Document-level label: 1 if any sentence is relevant, else 0
        doc_label = 1 if any(sentence_labels) else 0

        # Teacher score (simulated - use doc_label as score)
        # In real training with teacher model, this would be the reranker score
        teacher_score = float(doc_label)

        converted.append(
            {
                "query": query,
                "texts": [document],  # Single document
                "chunks_pos": [chunks_pos],  # Chunk positions for the document
                "relevant_chunks": [relevant_chunks],  # Relevant chunk indices
                "labels": [doc_label],  # Document label for ranking
                "teacher_score": [teacher_score],  # Teacher scores for distillation
                "is_hard_negative": is_hard_negative,
                # Preserve original data for debugging
                "_original_sentences": sentences,
                "_original_labels": sentence_labels,
            }
        )

    return converted


def create_hf_dataset(samples: list[dict]):
    """Create HuggingFace Dataset from converted samples."""
    from datasets import Dataset

    # Remove internal fields for HF dataset
    clean_samples = []
    for s in samples:
        clean = {k: v for k, v in s.items() if not k.startswith("_")}
        clean_samples.append(clean)

    return Dataset.from_list(clean_samples)


def upsample_dataset(dataset, source_weights: dict[str, int], is_main_process: bool = True):
    """Upsample dataset based on source weights.

    Args:
        dataset: HuggingFace Dataset with 'source' column
        source_weights: Dict mapping source names to integer multipliers
                       e.g., {"timeqa": 2, "streamingqa": 1}
                       Sources not in dict default to 1 (no change)
        is_main_process: Whether to print progress info

    Returns:
        Upsampled HuggingFace Dataset
    """
    from collections import Counter
    from datasets import Dataset

    if not source_weights:
        return dataset

    # Count original distribution
    original_counts = Counter(dataset["source"])

    # Build upsampled indices
    upsampled_indices = []
    for idx, source in enumerate(dataset["source"]):
        weight = source_weights.get(source, 1)
        # Add this sample 'weight' times
        upsampled_indices.extend([idx] * weight)

    # Shuffle the indices
    random.shuffle(upsampled_indices)

    # Create new dataset from selected indices
    upsampled_data = dataset.select(upsampled_indices)

    if is_main_process:
        # Count new distribution
        new_counts = Counter(upsampled_data["source"])
        if RICH_AVAILABLE:
            console.print(f"\n  [bold]Upsampling applied[/bold]")
            table = Table(show_header=True, box=box.SIMPLE, padding=(0, 1))
            table.add_column("Source", style="cyan")
            table.add_column("Original", justify="right")
            table.add_column("Weight", justify="center", style="yellow")
            table.add_column("Upsampled", justify="right", style="green")
            for source in sorted(set(original_counts.keys()) | set(new_counts.keys())):
                orig = original_counts.get(source, 0)
                new = new_counts.get(source, 0)
                weight = source_weights.get(source, 1)
                table.add_row(source, f"{orig:,}", f"{weight}x", f"{new:,}")
            console.print(table)
            console.print(f"  [dim]Total: {len(dataset):,} → {len(upsampled_data):,} samples[/dim]")
        else:
            print(f"\n  Upsampling applied (source_weights: {source_weights}):")
            print(f"    Original: {len(dataset)} samples")
            print(f"    Upsampled: {len(upsampled_data)} samples")
            print(f"    Distribution change:")
            for source in sorted(set(original_counts.keys()) | set(new_counts.keys())):
                orig = original_counts.get(source, 0)
                new = new_counts.get(source, 0)
                weight = source_weights.get(source, 1)
                print(f"      {source}: {orig} → {new} (weight={weight})")

    return upsampled_data


def filter_dataset(dataset, exclude_perturbed: bool = False, exclude_empty: bool = False,
                   exclude_sources: list[str] | None = None, is_main_process: bool = True):
    """Filter dataset based on criteria (for ablation studies).

    Args:
        dataset: HuggingFace Dataset with 'source' and 'relevant_chunks' columns
        exclude_perturbed: If True, exclude samples where source contains "perturbed"
        exclude_empty: If True, exclude samples where relevant_chunks is empty (all labels=0)
        exclude_sources: List of source prefixes to exclude (e.g., ["timeqa"] excludes "timeqa" and "timeqa_perturbed")
        is_main_process: Whether to print progress info

    Returns:
        Filtered HuggingFace Dataset
    """
    if not exclude_perturbed and not exclude_empty and not exclude_sources:
        return dataset

    original_size = len(dataset)

    def should_keep(sample):
        source = sample.get("source", "")

        # Check exclude_sources filter (matches prefix)
        if exclude_sources:
            for excluded in exclude_sources:
                if source.startswith(excluded):
                    return False

        # Check perturbed filter
        if exclude_perturbed:
            if "perturbed" in source.lower():
                return False

        # Check empty filter (no relevant chunks)
        if exclude_empty:
            relevant_chunks = sample.get("relevant_chunks", [[]])
            # For our format, relevant_chunks is a list of lists (one per document)
            # A sample is "empty" if all documents have no relevant chunks
            is_empty = all(len(chunks) == 0 for chunks in relevant_chunks)
            if is_empty:
                return False

        return True

    # Filter the dataset
    filtered_dataset = dataset.filter(should_keep)
    filtered_size = len(filtered_dataset)

    if is_main_process:
        removed = original_size - filtered_size
        filters_applied = []
        if exclude_perturbed:
            filters_applied.append("exclude_perturbed")
        if exclude_empty:
            filters_applied.append("exclude_empty")
        if exclude_sources:
            filters_applied.append(f"exclude_sources={exclude_sources}")

        if RICH_AVAILABLE:
            console.print(f"\n  [bold]Data filtering applied[/bold] ({', '.join(filters_applied)})")
            console.print(f"  [dim]Original: {original_size:,} → Filtered: {filtered_size:,} (removed {removed:,})[/dim]")
        else:
            print(f"\n  Data filtering applied ({', '.join(filters_applied)}):")
            print(f"    Original: {original_size} samples")
            print(f"    Filtered: {filtered_size} samples (removed {removed})")

    return filtered_dataset


# =============================================================================
# TRAINING
# =============================================================================


def train(config: HighlighterConfig, dry_run: bool = False, simple_output: bool = False):
    """Train the highlighter model using OpenProvence framework."""
    import os

    import torch
    from transformers import TrainingArguments

    # Import from open_provence
    from open_provence import (
        OpenProvenceDataCollator,
        OpenProvenceEncoder,
        OpenProvenceLoss,
        OpenProvenceTrainer,
    )

    # Detect distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = local_rank != -1
    is_main_process = local_rank in (-1, 0)

    # Display training config on main process
    if is_main_process:
        display_training_config(config, world_size, is_distributed)

    # Initialize wandb only on main process
    if config.use_wandb and is_main_process:
        try:
            import wandb

            # Generate run name if not provided
            run_name = config.wandb_run_name
            if not run_name:
                from datetime import datetime
                model_short = config.base_model.split("/")[-1][:20]
                run_name = f"{model_short}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            wandb.init(
                project=config.wandb_project,
                name=run_name,
                config={
                    "base_model": config.base_model,
                    "max_length": config.max_length,
                    "learning_rate": config.learning_rate,
                    "batch_size": config.batch_size,
                    "gradient_accumulation_steps": config.gradient_accumulation_steps,
                    "effective_batch_size": config.batch_size * config.gradient_accumulation_steps * world_size,
                    "num_epochs": config.num_epochs,
                    "ranking_weight": config.ranking_weight,
                    "pruning_weight": config.pruning_weight,
                    "warmup_ratio": config.warmup_ratio,
                    "weight_decay": config.weight_decay,
                    "seed": config.seed,
                    "world_size": world_size,
                    "source_weights": config.source_weights,
                },
            )
            if RICH_AVAILABLE:
                console.print(f"  [dim]W&B:[/dim] [cyan]{config.wandb_project}[/cyan] / [green]{run_name}[/green]")
            else:
                print(f"  W&B: {config.wandb_project} / {run_name}")
        except ImportError:
            if RICH_AVAILABLE:
                console.print("[yellow]⚠ wandb not installed, disabling logging[/yellow]")
            else:
                print("Warning: wandb not installed, disabling wandb logging")
            config.use_wandb = False
    elif config.use_wandb and not is_main_process:
        # Disable wandb on non-main processes
        config.use_wandb = False

    # Check FlashAttention availability
    flash_attn_available = False
    try:
        import flash_attn  # noqa: F401
        flash_attn_available = True
    except ImportError:
        pass

    # Set seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Load and convert training data
    if is_main_process:
        if RICH_AVAILABLE:
            console.print("\n[bold cyan]Loading training data...[/bold cyan]")
        else:
            print(f"\nLoading training data...")

    if config.hf_dataset:
        # Load from HuggingFace Hub
        from datasets import load_dataset
        from collections import Counter
        if is_main_process:
            if RICH_AVAILABLE:
                console.print(f"  [dim]Source:[/dim] HuggingFace Hub ([green]{config.hf_dataset}[/green])")
            else:
                print(f"  Source: HuggingFace Hub ({config.hf_dataset})")
        hf_data = load_dataset(config.hf_dataset)

        train_dataset = hf_data["train"]
        valid_dataset = hf_data.get("valid")
        test_dataset = hf_data.get("eval")

        # Count hard negatives and sources
        train_hn = sum(1 for s in train_dataset if s.get("is_hard_negative"))
        train_hn_pct = 100 * train_hn / len(train_dataset) if len(train_dataset) > 0 else 0

        # Get source distribution
        source_counts = None
        if "source" in train_dataset.column_names:
            source_counts = dict(Counter(train_dataset["source"]))

        if is_main_process:
            display_dataset_info(
                train_size=len(train_dataset),
                valid_size=len(valid_dataset) if valid_dataset else None,
                eval_size=len(test_dataset) if test_dataset else None,
                hard_neg_count=train_hn,
                hard_neg_pct=train_hn_pct,
                source_counts=source_counts,
            )

        # Apply data filtering if configured (for ablation studies)
        if config.exclude_perturbed or config.exclude_empty or config.exclude_sources:
            train_dataset = filter_dataset(
                train_dataset,
                exclude_perturbed=config.exclude_perturbed,
                exclude_empty=config.exclude_empty,
                exclude_sources=config.exclude_sources,
                is_main_process=is_main_process,
            )
            if valid_dataset:
                valid_dataset = filter_dataset(
                    valid_dataset,
                    exclude_perturbed=config.exclude_perturbed,
                    exclude_empty=config.exclude_empty,
                    exclude_sources=config.exclude_sources,
                    is_main_process=False,  # Don't print twice
                )
            if test_dataset:
                test_dataset = filter_dataset(
                    test_dataset,
                    exclude_perturbed=config.exclude_perturbed,
                    exclude_empty=config.exclude_empty,
                    exclude_sources=config.exclude_sources,
                    is_main_process=False,
                )

        # Apply source-based upsampling if configured
        if config.source_weights:
            train_dataset = upsample_dataset(train_dataset, config.source_weights, is_main_process)

        if dry_run:
            if is_main_process:
                if RICH_AVAILABLE:
                    console.print("\n[yellow]⚠ DRY RUN[/yellow] - Would train on this data. Exiting.")
                else:
                    print("\n[DRY RUN] Would train on this data. Exiting.")
                sample = train_dataset[0]
                if RICH_AVAILABLE:
                    console.print("\n[bold]Sample from HuggingFace dataset:[/bold]")
                    console.print(f"  [cyan]Query:[/cyan] {sample['query'][:80]}...")
                    console.print(f"  [cyan]Document length:[/cyan] {len(sample['texts'][0])} chars")
                    console.print(f"  [cyan]Num chunks:[/cyan] {len(sample['chunks_pos'][0])}")
                    console.print(f"  [cyan]Relevant chunks:[/cyan] {sample['relevant_chunks'][0]}")
                    console.print(f"  [cyan]Source:[/cyan] {sample.get('source', 'unknown')}")
                else:
                    print("\nSample from HuggingFace dataset:")
                    print(f"  Query: {sample['query'][:80]}...")
                    print(f"  Document length: {len(sample['texts'][0])} chars")
                    print(f"  Num chunks: {len(sample['chunks_pos'][0])}")
                    print(f"  Relevant chunks: {sample['relevant_chunks'][0]}")
                    print(f"  Source: {sample.get('source', 'unknown')}")
            return
    else:
        # Load from local files
        if is_main_process:
            if RICH_AVAILABLE:
                console.print(f"  [dim]Source:[/dim] Local files")
            else:
                print(f"  Source: Local files")

        train_samples = load_training_data(Path(config.train_file))
        train_converted = convert_to_open_provence_format(train_samples)

        valid_samples = []
        valid_converted = []
        if Path(config.valid_file).exists():
            valid_samples = load_training_data(Path(config.valid_file))
            valid_converted = convert_to_open_provence_format(valid_samples)

        eval_samples = []
        eval_converted = []
        if Path(config.eval_file).exists():
            eval_samples = load_training_data(Path(config.eval_file))
            eval_converted = convert_to_open_provence_format(eval_samples)

        # Count hard negatives
        train_hn = sum(1 for s in train_converted if s.get("is_hard_negative"))
        train_hn_pct = 100 * train_hn / len(train_converted) if len(train_converted) > 0 else 0

        if is_main_process:
            display_dataset_info(
                train_size=len(train_converted),
                valid_size=len(valid_converted) if valid_converted else None,
                eval_size=len(eval_converted) if eval_converted else None,
                hard_neg_count=train_hn,
                hard_neg_pct=train_hn_pct,
            )

        if dry_run:
            if is_main_process:
                if RICH_AVAILABLE:
                    console.print("\n[yellow]⚠ DRY RUN[/yellow] - Would train on this data. Exiting.")
                else:
                    print("\n[DRY RUN] Would train on this data. Exiting.")
                # Show sample conversion
                if train_converted:
                    sample = train_converted[0]
                    if RICH_AVAILABLE:
                        console.print("\n[bold]Sample conversion:[/bold]")
                        console.print(f"  [cyan]Query:[/cyan] {sample['query'][:80]}...")
                        console.print(f"  [cyan]Document length:[/cyan] {len(sample['texts'][0])} chars")
                        console.print(f"  [cyan]Num chunks:[/cyan] {len(sample['chunks_pos'][0])}")
                        console.print(f"  [cyan]Relevant chunks:[/cyan] {sample['relevant_chunks'][0]}")
                        console.print(f"  [cyan]Doc label:[/cyan] {sample['labels'][0]}")
                    else:
                        print("\nSample conversion:")
                        print(f"  Query: {sample['query'][:80]}...")
                        print(f"  Document length: {len(sample['texts'][0])} chars")
                        print(f"  Num chunks: {len(sample['chunks_pos'][0])}")
                        print(f"  Relevant chunks: {sample['relevant_chunks'][0]}")
                        print(f"  Doc label: {sample['labels'][0]}")
            return

        # Create HuggingFace datasets
        train_dataset = create_hf_dataset(train_converted)
        valid_dataset = create_hf_dataset(valid_converted) if valid_converted else None
        test_dataset = create_hf_dataset(eval_converted) if eval_converted else None

    # Initialize OpenProvence model
    if is_main_process:
        if RICH_AVAILABLE:
            console.print(f"\n[bold cyan]Loading model...[/bold cyan]")
        else:
            print(f"\nLoading model: {config.base_model}...")

    # Determine attn_implementation based on FlashAttention availability
    # Note: Not all models support flash_attention_2 (e.g., ModernBERT-based models don't)
    # Only enable for models known to support it (Zilliz/BGE-M3 based)
    model_args = {"trust_remote_code": True}
    if config.base_model_revision:
        model_args["revision"] = config.base_model_revision
    use_flash_attn = flash_attn_available and "zilliz" in config.base_model.lower()
    if use_flash_attn:
        model_args["attn_implementation"] = "flash_attention_2"

    # Check if model is already an OpenProvence model (like Zilliz)
    from transformers import AutoConfig, AutoModel
    model_config = AutoConfig.from_pretrained(config.base_model, **model_args)

    if getattr(model_config, "model_type", None) == "open_provence":
        # Load existing OpenProvence model directly
        try:
            loaded_model = AutoModel.from_pretrained(config.base_model, **model_args)
        except ValueError as e:
            # Flash attention not supported by this model, retry without it
            if "Flash Attention" in str(e) and "attn_implementation" in model_args:
                if is_main_process:
                    if RICH_AVAILABLE:
                        console.print("[yellow]Flash Attention not supported by this model, falling back to default[/yellow]")
                    else:
                        print("Flash Attention not supported by this model, falling back to default")
                del model_args["attn_implementation"]
                use_flash_attn = False
                loaded_model = AutoModel.from_pretrained(config.base_model, **model_args)
            else:
                raise
        # Wrap in OpenProvenceEncoder for compatibility with our training loop
        # The model already has pruning head, so we just need to wrap it
        import torch.nn as nn
        encoder = OpenProvenceEncoder.__new__(OpenProvenceEncoder)
        nn.Module.__init__(encoder)
        encoder.model_name_or_path = config.base_model
        encoder.mode = "reranking_pruning"
        encoder.max_length = config.max_length
        encoder.num_labels = getattr(model_config, "num_labels", 1)
        encoder.device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder.config = model_config
        encoder.tokenizer = loaded_model.tokenizer
        encoder.ranking_model = loaded_model.ranking_model
        encoder.pruning_head = loaded_model.pruning_head
        encoder._keys_to_ignore_on_save = []
        model = encoder
    else:
        # Create new OpenProvenceEncoder for base models
        model = OpenProvenceEncoder(
            model_name_or_path=config.base_model,
            num_labels=config.num_labels,
            max_length=config.max_length,
            model_args=model_args,
            pruning_config={
                "dropout": 0.1,
                "sentence_pooling": "mean",
                "use_weighted_pooling": False,
            },
        )

    if is_main_process:
        hidden_size = model.ranking_model.config.hidden_size
        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        display_model_info(
            model_name=config.base_model,
            hidden_size=hidden_size,
            num_params=num_params,
            flash_attn=use_flash_attn,
        )

    # Create data collator
    # Only warn about missing scores column if ranking_weight > 0 (when it matters)
    data_collator = OpenProvenceDataCollator(
        tokenizer=model.tokenizer,
        max_length=config.max_length,
        query_column="query",
        texts_column="texts",
        labels_column="labels",
        scores_column="teacher_score",
        chunks_pos_column="chunks_pos",
        relevant_chunks_column="relevant_chunks",
        warn_missing_scores=(config.ranking_weight > 0),
    )

    # Create loss function
    loss_fn = OpenProvenceLoss(
        model=model,
        ranking_weight=config.ranking_weight,
        pruning_weight=config.pruning_weight,
        is_regression=True,  # Use MSE for ranking
    )

    # Training arguments
    # When using rich (and not --simple), disable tqdm and use our custom progress callback
    use_rich_progress = RICH_AVAILABLE and is_main_process and not simple_output

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        eval_strategy="steps" if valid_dataset else "no",
        eval_steps=config.eval_steps if valid_dataset else None,
        save_steps=config.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True if valid_dataset else False,
        metric_for_best_model="eval_loss" if valid_dataset else None,
        greater_is_better=False if valid_dataset else None,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        dataloader_num_workers=4,
        seed=config.seed,
        report_to=["wandb"] if config.use_wandb else "none",
        remove_unused_columns=False,  # Keep all columns for data collator
        optim="adafactor",  # OpenProvence default
        lr_scheduler_type="cosine",
        # Disable default progress/logging when using rich
        disable_tqdm=use_rich_progress,
        log_level="warning" if use_rich_progress else "info",
    )

    # Create callbacks list
    callbacks = []
    if use_rich_progress:
        RichTrainerCallback = create_rich_callback_class()
        callbacks.append(RichTrainerCallback())

    # Create trainer
    trainer = OpenProvenceTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        loss_fn=loss_fn,
        callbacks=callbacks if callbacks else None,
    )

    # Remove default PrinterCallback when using rich (it causes dict spam)
    if use_rich_progress:
        from transformers.trainer_callback import PrinterCallback
        trainer.remove_callback(PrinterCallback)

    # Train
    if is_main_process:
        if RICH_AVAILABLE:
            console.print(f"\n[bold cyan]Starting training...[/bold cyan]")
        else:
            print(f"\nStarting training...")
    trainer.train()

    # Save final model (only on main process)
    if is_main_process:
        final_model_path = Path(config.output_dir) / "final_model"
        if RICH_AVAILABLE:
            console.print(f"\n[dim]Saving model to {final_model_path}...[/dim]")
        else:
            print(f"\nSaving model to {final_model_path}...")
        model.save_pretrained(str(final_model_path))

        # Save training config
        config_path = final_model_path / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(
                {
                    "base_model": config.base_model,
                    "base_model_revision": config.base_model_revision,
                    "max_length": config.max_length,
                    "num_labels": config.num_labels,
                    "ranking_weight": config.ranking_weight,
                    "pruning_weight": config.pruning_weight,
                    "learning_rate": config.learning_rate,
                    "batch_size": config.batch_size,
                    "num_epochs": config.num_epochs,
                    "world_size": world_size,
                },
                f,
                indent=2,
            )

        # Finish wandb run
        if config.use_wandb:
            import wandb
            wandb.finish()

        # Get best metric if available
        best_metric = None
        if trainer.state.best_metric is not None:
            best_metric = trainer.state.best_metric

        display_training_complete(str(final_model_path), best_metric)


# =============================================================================
# STATISTICS
# =============================================================================


def show_stats(train_file: Path, valid_file: Path):
    """Show statistics about training data."""
    if RICH_AVAILABLE:
        console.print(Panel("[bold]Training Data Statistics[/bold]", border_style="blue"))
    else:
        print(f"\n{'='*60}")
        print("TRAINING DATA STATISTICS")
        print(f"{'='*60}")

    for name, file_path in [("Train", train_file), ("Valid", valid_file)]:
        if not file_path.exists():
            if RICH_AVAILABLE:
                console.print(f"\n[yellow]{name}:[/yellow] File not found ({file_path})")
            else:
                print(f"\n{name}: File not found ({file_path})")
            continue

        samples = load_training_data(file_path)

        # Count by source
        sources: dict[str, int] = {}
        hn_count = 0
        total_sentences = 0
        total_relevant = 0

        for s in samples:
            src = s.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
            if s.get("is_hard_negative"):
                hn_count += 1

            labels = s.get("labels", s.get("sentence_labels", []))
            total_sentences += len(labels)
            total_relevant += sum(labels)

        hn_pct = 100 * hn_count / len(samples) if len(samples) > 0 else 0
        rel_pct = 100 * total_relevant / total_sentences if total_sentences > 0 else 0

        if RICH_AVAILABLE:
            # Main stats table
            table = Table(title=f"[bold]{name.upper()}[/bold] ({len(samples):,} samples)", box=box.ROUNDED)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green", justify="right")
            table.add_row("Hard negatives", f"{hn_count:,} ({hn_pct:.1f}%)")
            table.add_row("Total sentences", f"{total_sentences:,}")
            if total_sentences > 0:
                table.add_row("Relevant sentences", f"{total_relevant:,} ({rel_pct:.1f}%)")
            console.print(table)

            # Source distribution table
            src_table = Table(title="By Source", box=box.SIMPLE, show_header=False)
            src_table.add_column("Source", style="cyan")
            src_table.add_column("Count", style="green", justify="right")
            for src, count in sorted(sources.items()):
                src_table.add_row(src, f"{count:,}")
            console.print(src_table)
        else:
            print(f"\n{name.upper()}: {len(samples)} samples")
            print(f"  Hard negatives: {hn_count} ({hn_pct:.1f}%)")
            print(f"  Total sentences: {total_sentences}")
            if total_sentences > 0:
                print(f"  Relevant sentences: {total_relevant} ({rel_pct:.1f}%)")
            print(f"\n  By source:")
            for src, count in sorted(sources.items()):
                print(f"    {src}: {count}")


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train Provence-style semantic highlighter using OpenProvence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show data stats
  python scripts/train_highlighter.py --stats

  # Train on local data with Zilliz model
  python scripts/train_highlighter.py --local-only

  # Train with ModernBERT
  python scripts/train_highlighter.py --local-only --base-model modernbert

  # Train on joint data
  python scripts/train_highlighter.py --with-zilliz
""",
    )

    # Config file (highest priority mode)
    parser.add_argument(
        "--config", type=str, metavar="YAML",
        help="Load config from YAML file (e.g., 'configs/ts-highlighter-v1.yaml')"
    )

    # Mode (required if --config not provided)
    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument(
        "--local-only", action="store_true", help="Train on local annotated data only"
    )
    mode_group.add_argument(
        "--with-zilliz", action="store_true", help="Train on local + Zilliz combined data"
    )
    mode_group.add_argument(
        "--hf-dataset", type=str, metavar="REPO",
        help="Train from HuggingFace dataset (e.g., 'anonymous/ts-highlighter-data')"
    )
    mode_group.add_argument("--stats", action="store_true", help="Show training data statistics")

    # Paths
    parser.add_argument(
        "--train-file", type=str, default=None, help="Training file (default: auto based on mode)"
    )
    parser.add_argument("--valid-file", type=str, default=None, help="Validation file")
    parser.add_argument("--eval-file", type=str, default=None, help="Evaluation file (for final eval)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for model")

    # Model params
    parser.add_argument(
        "--base-model",
        type=str,
        default="zilliz",
        help="Base model: 'zilliz', 'modernbert', or full HF path (default: zilliz)",
    )
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum sequence length")

    # Training params (defaults match OpenProvence)
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per device (default: 4)")
    parser.add_argument(
        "--gradient-accumulation", type=int, default=16, help="Gradient accumulation steps (default: 16, effective batch=64)"
    )
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate (default: 5e-5)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs (default: 1, OpenProvence style)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Loss weights
    parser.add_argument(
        "--ranking-weight", type=float, default=0.0, help="Weight for ranking loss (default: 0.0 for pruning-only)"
    )
    parser.add_argument(
        "--pruning-weight", type=float, default=1.0, help="Weight for pruning loss (default: 1.0)"
    )

    # Wandb
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument(
        "--wandb-project", type=str, default="ts-highlighter", help="Wandb project name"
    )
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Wandb run name")

    # Options
    parser.add_argument("--dry-run", action="store_true", help="Load data but don't train")
    parser.add_argument("--simple", action="store_true", help="Use simple text output instead of rich progress bars")

    args = parser.parse_args()

    # Handle --config mode (YAML config file)
    if args.config:
        yaml_path = Path(args.config)
        if not yaml_path.exists():
            print(f"Error: Config file not found: {yaml_path}")
            sys.exit(1)

        if RICH_AVAILABLE:
            console.print(f"[dim]Loading config from:[/dim] [cyan]{yaml_path}[/cyan]")
        else:
            print(f"Loading config from: {yaml_path}")
        yaml_config = load_config_from_yaml(yaml_path)
        config = create_config_from_yaml_and_args(yaml_config, args)

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
        )

        # Train
        train(config, dry_run=args.dry_run, simple_output=args.simple)
        return

    # Require a mode if --config not provided
    if not any([args.local_only, args.with_zilliz, args.hf_dataset, args.stats]):
        print("Error: Must specify one of: --config, --local-only, --with-zilliz, --hf-dataset, --stats")
        sys.exit(1)

    # Handle --stats
    if args.stats:
        train_file = Path("datasets/training_local/train.jsonl")
        valid_file = Path("datasets/training_local/valid.jsonl")
        show_stats(train_file, valid_file)

        train_file = Path("datasets/training_joint/train.jsonl")
        valid_file = Path("datasets/training_joint/valid.jsonl")
        if train_file.exists():
            show_stats(train_file, valid_file)
        return

    # Resolve base model preset
    base_model = MODEL_PRESETS.get(args.base_model.lower(), args.base_model)

    # Determine paths based on mode
    hf_dataset = None
    if args.local_only:
        train_file = Path(args.train_file or "datasets/training_local/train.jsonl")
        valid_file = Path(args.valid_file or "datasets/training_local/valid.jsonl")
        eval_file = Path(args.eval_file or "datasets/training_local/eval.jsonl")
        model_suffix = args.base_model.lower() if args.base_model.lower() in MODEL_PRESETS else "custom"
        output_dir = args.output_dir or f"models/highlighter-local-{model_suffix}"
    elif args.with_zilliz:
        train_file = Path(args.train_file or "datasets/training_joint/train.jsonl")
        valid_file = Path(args.valid_file or "datasets/training_joint/valid.jsonl")
        eval_file = Path(args.eval_file or "datasets/training_joint/eval.jsonl")
        model_suffix = args.base_model.lower() if args.base_model.lower() in MODEL_PRESETS else "custom"
        output_dir = args.output_dir or f"models/highlighter-joint-{model_suffix}"
    elif args.hf_dataset:
        hf_dataset = args.hf_dataset
        train_file = Path("unused")
        valid_file = Path("unused")
        eval_file = Path("unused")
        model_suffix = args.base_model.lower() if args.base_model.lower() in MODEL_PRESETS else "custom"
        output_dir = args.output_dir or f"models/highlighter-hf-{model_suffix}"

    # Check files exist (skip for HuggingFace dataset)
    if not hf_dataset and not train_file.exists():
        print(f"Error: Training file not found: {train_file}")
        print("\nRun prepare_joint_training.py first:")
        print(f"  python scripts/prepare_joint_training.py --local-only")
        sys.exit(1)

    # Create config
    config = HighlighterConfig(
        base_model=base_model,
        max_length=args.max_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        ranking_weight=args.ranking_weight,
        pruning_weight=args.pruning_weight,
        train_file=str(train_file),
        valid_file=str(valid_file),
        eval_file=str(eval_file),
        hf_dataset=hf_dataset,
        output_dir=output_dir,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        seed=args.seed,
    )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    # Train
    train(config, dry_run=args.dry_run, simple_output=args.simple)


if __name__ == "__main__":
    main()
