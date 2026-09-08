"""
Main evaluation pipeline orchestrator.

Implements the efficient batch processing flow:
1. Load all queries
2. Retrieve for all queries (batch) - with caching
3. Apply highlighter compression (optional) - with caching
4. Generate answers for all queries (batch)
5. Compute metrics

Optimized execution order:
- Complete all BM25 retrieval first (results cached)
- Group highlighter methods by model to minimize loading/unloading
- Generate LLM answers in batches
"""

import os
import json
import gzip
import datetime
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

from .llm_client import create_llm_client, BatchGenerator, BaseLLMClient
from .retriever import create_retriever, format_context, HybridRetriever, RetrievedDoc
from .metrics import (
    evaluate_predictions,
    evaluate_predictions_grouped,
    format_metrics_report,
    format_grouped_metrics_report,
    AggregatedMetrics,
    GroupedMetrics,
)
from .llm_judge import LLMJudge, aggregate_cached_judgments
from .query_formatter import format_and_validate_query
from .token_accounting import create_context_token_counter

logger = logging.getLogger(__name__)


@dataclass
class EvalEntry:
    """A single evaluation entry."""
    id: str
    question: str
    question_ts: str
    answers: List[str]
    additional_answers: List[str]

    # Perturbation status
    is_hard_negative: bool = False

    # Dataset source (streamingqa, timeqa, streamingqa_perturbed, timeqa_perturbed)
    source: str = ""

    # Ground truth context from dataset (for gt_llm/gt_highlighter methods)
    gt_context: str = ""
    gt_document_id: str = ""
    evidence_sentences: List[str] = None
    evidence_labels: List[int] = None

    # Filled during pipeline
    context: str = ""
    retrieved_docs: List[str] = None  # Raw retrieved documents
    retrieved_doc_ids: List[str] = None
    highlighted_docs: List[str] = None  # After highlighter compression
    model_answer: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    context_tokens: int = 0

    # Method metadata
    highlighter_model: str = ""
    compression_ratio: float = 0.0
    gt_retrieved: bool = False  # Whether GT doc was found in retrieved docs
    noise_answer_overlap_doc_ids: List[str] = None
    evidence_precision: Optional[float] = None
    evidence_recall: Optional[float] = None
    evidence_tp: Optional[int] = None
    evidence_fp: Optional[int] = None
    evidence_fn: Optional[int] = None
    compression_latency_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "question_ts": self.question_ts,
            "answers": self.answers,
            "additional_answers": self.additional_answers,
            "is_hard_negative": self.is_hard_negative,
            "source": self.source,
            "context": self.context,
            "retrieved_doc_ids": self.retrieved_doc_ids or [],
            "gt_document_id": self.gt_document_id,
            "noise_answer_overlap_doc_ids": self.noise_answer_overlap_doc_ids or [],
            "evidence_precision": self.evidence_precision,
            "evidence_recall": self.evidence_recall,
            "evidence_tp": self.evidence_tp,
            "evidence_fp": self.evidence_fp,
            "evidence_fn": self.evidence_fn,
            "evidence_labeled": self.evidence_labels is not None,
            "compression_latency_seconds": self.compression_latency_seconds,
            "highlighter_model": self.highlighter_model,
            "compression_ratio": self.compression_ratio,
            "gt_retrieved": self.gt_retrieved,
            "model_answer": self.model_answer,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "context_tokens": self.context_tokens,
        }


def load_qa_dataset(
    path: str,
    field_mappings: Dict[str, str] = None,
) -> List[EvalEntry]:
    """
    Load QA dataset from JSON or JSONL file.
    """
    if field_mappings is None:
        field_mappings = {
            "question": "question",
            "answers": "answers",
            "additional_answers": "additional_answers",
            "question_ts": "question_timestamp",
            "id": "id",
            "is_hard_negative": "is_hard_negative",
        }

    path = Path(path)
    entries = []

    # Load data
    if path.suffix == ".gz":
        open_fn = lambda p: gzip.open(p, "rt", encoding="utf-8")
    else:
        open_fn = lambda p: open(p, "r", encoding="utf-8")

    with open_fn(path) as f:
        if path.suffix in [".jsonl", ".gz"] or (path.suffix == ".gz" and ".jsonl" in path.name):
            data = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
            if not isinstance(data, list):
                data = [data]

    for item in data:
        # Extract fields with mappings
        question = item.get(field_mappings.get("question", "question"), "")
        answers = item.get(field_mappings.get("answers", "answers"), [])
        additional = item.get(field_mappings.get("additional_answers", "additional_answers"), [])
        question_ts = item.get(field_mappings.get("question_ts", "question_timestamp"), "")
        entry_id = item.get(field_mappings.get("id", "id"), "")
        is_hard_negative = item.get(field_mappings.get("is_hard_negative", "is_hard_negative"), False)
        gt_context = item.get(field_mappings.get("gt_context", "context"), "")
        source = item.get(field_mappings.get("source", "source"), "")

        # Ensure lists
        if isinstance(answers, str):
            answers = [answers]
        if isinstance(additional, str):
            additional = [additional]

        # Format timestamp
        if question_ts:
            question_ts = format_timestamp(question_ts)

        entries.append(EvalEntry(
            id=entry_id,
            question=question,
            question_ts=question_ts,
            answers=answers,
            additional_answers=additional,
            is_hard_negative=bool(is_hard_negative),
            source=source,
            gt_context=gt_context,
        ))

    return entries


def attach_gold_document_ids(
    entries: List[EvalEntry], dataset_config: Dict[str, Any]
) -> None:
    """Attach stable corpus IDs from the immutable Clean v2 sidecar mapping."""
    mapping_path = dataset_config.get("gold_document_map_path")
    if not mapping_path:
        return
    split = dataset_config.get("gold_document_map_split", "eval")
    with open(mapping_path, encoding="utf-8") as handle:
        mapping = json.load(handle)
    rows = mapping.get("splits", {}).get(split)
    if rows is None:
        raise ValueError(f"Gold document mapping has no split {split!r}: {mapping_path}")
    by_id = {str(row["id"]): str(row["gold_document_id"]) for row in rows}
    missing = []
    for entry in entries:
        entry.gt_document_id = by_id.get(str(entry.id), "")
        if not entry.gt_document_id:
            missing.append(str(entry.id))
    if missing:
        raise ValueError(
            f"Gold document mapping is missing {len(missing)} selected entries; "
            f"first IDs: {missing[:5]}"
        )


def attach_evidence_labels(
    entries: List[EvalEntry], dataset_config: Dict[str, Any]
) -> None:
    """Attach optional sentence evidence labels without changing QA selection."""
    path = dataset_config.get("evidence_label_path")
    if not path:
        return
    by_id = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                by_id[str(row["id"])] = row
    for entry in entries:
        row = by_id.get(str(entry.id))
        if row is None:
            continue
        sentences = list(row["texts"])
        labels = [int(value) for value in row["labels"]]
        if len(sentences) != len(labels):
            raise ValueError(f"Evidence sentence/label mismatch for {entry.id}")
        entry.evidence_sentences = sentences
        entry.evidence_labels = labels


def compute_entry_evidence_metrics(entry: EvalEntry) -> None:
    """Compute conservative sentence evidence metrics for one selected context.

    Labeled sentences come only from the gold document. Selected sentences from
    any other retrieved document count as false positives; possible alternate
    sufficient evidence is separately exposed by noise diagnostics.
    """
    if entry.evidence_labels is None or entry.evidence_sentences is None:
        return
    selected_docs = entry.highlighted_docs if entry.highlighted_docs is not None else entry.retrieved_docs
    selected_docs = selected_docs or []
    selected_ids = entry.retrieved_doc_ids or []
    gold_selected_text = ""
    non_gold_docs = []
    for doc, doc_id in zip(selected_docs, selected_ids):
        if entry.gt_document_id and doc_id == entry.gt_document_id:
            gold_selected_text += " " + doc
        else:
            non_gold_docs.append(doc)
    predictions = []
    for sentence in entry.evidence_sentences:
        stripped = sentence.strip()
        predictions.append(bool(stripped) and stripped in gold_selected_text)
    tp = sum(pred and label == 1 for pred, label in zip(predictions, entry.evidence_labels))
    fp = sum(pred and label == 0 for pred, label in zip(predictions, entry.evidence_labels))
    fn = sum((not pred) and label == 1 for pred, label in zip(predictions, entry.evidence_labels))
    if non_gold_docs:
        from src.processors.sentence_splitter import SentenceSplitter

        splitter = SentenceSplitter()
        fp += sum(len(splitter.split(doc)) for doc in non_gold_docs if doc)
    entry.evidence_tp = int(tp)
    entry.evidence_fp = int(fp)
    entry.evidence_fn = int(fn)
    entry.evidence_precision = tp / (tp + fp) if tp + fp else 0.0
    entry.evidence_recall = tp / (tp + fn) if tp + fn else 0.0


def format_timestamp(ts_value) -> str:
    """Format timestamp to 'Month Day, Year' format."""
    if isinstance(ts_value, str):
        # Try parsing ISO format
        try:
            dt = datetime.datetime.fromisoformat(ts_value.replace("Z", "+00:00"))
            return f"{dt.strftime('%b')} {dt.day}, {dt.year}"
        except ValueError:
            pass
        # Return as-is if already formatted
        return ts_value

    if isinstance(ts_value, (int, float)):
        dt = datetime.datetime.utcfromtimestamp(int(ts_value))
        return f"{dt.strftime('%b')} {dt.day}, {dt.year}"

    return str(ts_value)


def build_messages(
    entry: EvalEntry,
    system_prompt: str,
    user_template: str,
    context: str = "",
) -> List[Dict[str, str]]:
    """Build chat messages for an entry."""
    messages = []

    # System message with context
    system_content = system_prompt
    if context:
        system_content = f"{system_prompt}\n\n{context}"

    messages.append({"role": "system", "content": system_content})

    # User message
    user_content = user_template.format(
        question=format_and_validate_query(entry),
        question_ts=entry.question_ts,
        context=context,
    )
    messages.append({"role": "user", "content": user_content})

    return messages


def expand_methods(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expand method configurations.

    Methods with 'highlighter: true' are expanded to create one method
    per highlighter model in the config.

    Returns methods sorted for optimal execution:
    1. Non-highlighter methods first, sorted by top_k descending (so larger
       retrievals are cached first and can be reused by smaller ones)
    2. Highlighter methods grouped by highlighter model, also sorted by top_k descending
    """
    methods = config.get("methods", [])
    highlighter_cfg = config.get("highlighter", {})
    highlighter_models = highlighter_cfg.get("models", [])
    model_aliases = highlighter_cfg.get("model_aliases", {})

    def get_top_k(method):
        """Get top_k for sorting (larger first, 0 for no retrieval)."""
        retrieval = method.get("retrieval")
        if not retrieval:
            return 0
        return retrieval.get("top_k", 0)

    expanded = []

    # First pass: non-highlighter methods (sorted by top_k descending)
    non_hl_methods = [m.copy() for m in methods if not m.get("highlighter")]
    non_hl_methods.sort(key=get_top_k, reverse=True)
    expanded.extend(non_hl_methods)

    # Second pass: highlighter methods (grouped by model for efficiency)
    highlighter_methods = [m for m in methods if m.get("highlighter")]
    # Sort by top_k descending within highlighter methods
    highlighter_methods.sort(key=get_top_k, reverse=True)

    if highlighter_methods and highlighter_models:
        # Group by highlighter model first (to minimize model loading)
        for model_id in highlighter_models:
            # Local checkpoints conventionally end in ``final_model``.  An
            # explicit alias prevents distinct local runs from collapsing onto
            # the same expanded method name while preserving legacy naming.
            model_short = model_aliases.get(
                model_id,
                model_id.split("/")[-1] if "/" in model_id else model_id,
            )

            for method in highlighter_methods:
                expanded_method = method.copy()
                expanded_method["name"] = f"{method['name']}_{model_short}"
                expanded_method["highlighter_model"] = model_id
                expanded_method["description"] = f"{method.get('description', '')} (model: {model_short})"
                expanded.append(expanded_method)

    return expanded


def get_retrieval_key(method_config: Dict[str, Any]) -> str:
    """Get a cache key for retrieval configuration."""
    retrieval = method_config.get("retrieval")
    if not retrieval:
        return "none"

    parts = [
        retrieval.get("type", "bm25"),
        str(retrieval.get("top_k", 0)),
    ]
    if retrieval.get("rerank_top_k"):
        parts.append(f"rerank_{retrieval['rerank_top_k']}")

    return "_".join(parts)


class EvaluationPipeline:
    """
    Main evaluation pipeline with batch processing and caching.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        retriever: Optional[HybridRetriever] = None,
        highlighter_manager=None,
        config: Dict[str, Any] = None,
    ):
        self.llm_client = llm_client
        self.retriever = retriever
        self.highlighter_manager = highlighter_manager
        self.config = config or {}

        # Extract config values
        prompts_cfg = self.config.get("prompts", {})
        self.system_prompt = prompts_cfg.get("system", "You are a helpful assistant.")
        self.system_prompt_bare = prompts_cfg.get("system_bare", self.system_prompt)
        self.user_template = prompts_cfg.get("user", "{question}")
        self.context_format = prompts_cfg.get("context_format", "Retrieved Document {rank}: {content}")
        self.context_separator = prompts_cfg.get("context_separator", "\n\n")

        llm_cfg = self.config.get("llm", {})
        self.sampling_params = llm_cfg.get("sampling", {})
        # Get batch_size from mode-specific config
        llm_mode = llm_cfg.get("mode", "server")
        if llm_mode == "server":
            self.batch_size = llm_cfg.get("server", {}).get("batch_size", 64)
        else:
            self.batch_size = llm_cfg.get("local", {}).get("batch_size", 8)

        highlighter_cfg = self.config.get("highlighter", {})
        self.highlighter_threshold = highlighter_cfg.get("threshold", 0.5)
        self.highlighter_thresholds = highlighter_cfg.get("thresholds", {})
        self.context_token_counter = create_context_token_counter(self.config)

        # Cache for retrieval results: (retrieval_key, entry_id) -> (docs, doc_ids)
        self._retrieval_cache: Dict[str, Dict[str, Tuple[List[str], List[str]]]] = {}

        # Track maximum top_k retrieved per query for reuse
        # Key: entry_id -> {type: (max_top_k, all_docs, all_doc_ids)}
        self._max_retrieval_cache: Dict[str, Tuple[int, List[str], List[str]]] = {}

    def _get_cached_retrieval(
        self,
        entries: List[EvalEntry],
        retrieval_cfg: Dict[str, Any],
    ) -> Dict[str, Tuple[List[str], List[str]]]:
        """
        Get retrieval results, using cache if available.

        Optimizations:
        - Reuses larger top_k results (e.g., top100 can provide top5)
        - Caches results across methods

        Returns: {entry_id: (doc_contents, doc_ids)}
        """
        retrieval_key = get_retrieval_key({"retrieval": retrieval_cfg})
        retrieval_type = retrieval_cfg.get("type", "bm25")
        top_k = retrieval_cfg.get("top_k", 10)
        rerank_top_k = retrieval_cfg.get("rerank_top_k") if "rerank" in retrieval_type else None

        # For non-rerank BM25, check if we can reuse a larger cached retrieval
        if "rerank" not in retrieval_type:
            # Check if exact match exists
            if retrieval_key in self._retrieval_cache:
                cached = self._retrieval_cache[retrieval_key]
                if all(e.id in cached for e in entries):
                    return cached

            # Check if a larger top_k exists that we can slice from
            for cached_key, cached_data in self._retrieval_cache.items():
                # Parse the cached key to get its top_k
                # Key format: "bm25_100" or "bm25_5"
                if cached_key.startswith("bm25_") and "_rerank" not in cached_key:
                    try:
                        cached_top_k = int(cached_key.split("_")[1])
                        if cached_top_k >= top_k and all(e.id in cached_data for e in entries):
                            # Reuse by slicing
                            print(f"  Reusing cached BM25 top_{cached_top_k} (slicing to top_{top_k})")
                            sliced_data = {
                                entry_id: (docs[:top_k], ids[:top_k])
                                for entry_id, (docs, ids) in cached_data.items()
                                if entry_id in {e.id for e in entries}
                            }
                            # Also cache this for future exact matches
                            self._retrieval_cache[retrieval_key] = sliced_data
                            return sliced_data
                    except (IndexError, ValueError):
                        continue

        # Check exact cache match (including rerank methods)
        if retrieval_key in self._retrieval_cache:
            cached = self._retrieval_cache[retrieval_key]
            if all(e.id in cached for e in entries):
                return cached

        # Need to run retrieval
        if not self.retriever:
            return {}

        # One canonical logical query is shared by retrieval, highlighting, and generation.
        queries = [format_and_validate_query(e) for e in entries]
        query_ids = [e.id for e in entries]

        # Batch retrieve
        print(f"  Running BM25 retrieval (top_k={top_k})...")
        results = self.retriever.batch_retrieve(
            queries=queries,
            query_ids=query_ids,
            top_k=top_k,
            rerank_top_k=rerank_top_k,
        )

        # Convert to cache format
        cache_data = {}
        for entry in entries:
            docs = results.get(entry.id, [])
            doc_contents = [d.content for d in docs]
            doc_ids = [d.doc_id for d in docs]
            cache_data[entry.id] = (doc_contents, doc_ids)

        # Store in cache
        if retrieval_key not in self._retrieval_cache:
            self._retrieval_cache[retrieval_key] = {}
        self._retrieval_cache[retrieval_key].update(cache_data)

        return cache_data

    def _apply_highlighter(
        self,
        entries: List[EvalEntry],
        highlighter_model: str,
    ):
        """Apply highlighter compression to retrieved documents."""
        if not self.highlighter_manager:
            logger.warning("Highlighter requested but no manager available")
            return

        # Free GPU memory from reranker before loading highlighter
        if self.retriever and hasattr(self.retriever, 'reranker') and self.retriever.reranker:
            try:
                import torch
                reranker = self.retriever.reranker
                if hasattr(reranker, 'model') and hasattr(reranker.model, 'model'):
                    # Move CrossEncoder model to CPU to free GPU memory
                    reranker.model.model.cpu()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.info("Moved reranker to CPU to free GPU memory for highlighter")
            except Exception as e:
                logger.debug(f"Could not move reranker to CPU: {e}")

        print(f"  Applying highlighter: {highlighter_model.split('/')[-1]}...")
        threshold = self.highlighter_thresholds.get(highlighter_model, self.highlighter_threshold)
        if threshold is None:
            raise ValueError(
                f"No validation-frozen threshold configured for highlighter model: {highlighter_model}"
            )

        for entry in tqdm(entries, desc="Highlighting"):
            if not entry.retrieved_docs:
                entry.highlighted_docs = []
                continue

            query = format_and_validate_query(entry)

            # Prune documents
            pruned = self.highlighter_manager.prune_documents(
                model_id=highlighter_model,
                query=query,
                documents=entry.retrieved_docs,
                threshold=threshold,
            )

            entry.highlighted_docs = pruned
            entry.highlighter_model = highlighter_model

            # Calculate compression ratio
            orig_len = sum(len(d) for d in entry.retrieved_docs)
            pruned_len = sum(len(d) for d in pruned)
            entry.compression_ratio = pruned_len / orig_len if orig_len > 0 else 0.0

    def _format_context(
        self,
        docs: List[str],
    ) -> str:
        """Format documents into context string."""
        if not docs:
            return ""

        formatted = []
        for rank, content in enumerate(docs, start=1):
            if content.strip():  # Skip empty documents
                text = self.context_format.format(rank=rank, content=content)
                formatted.append(text)

        return self.context_separator.join(formatted)

    def run(
        self,
        entries: List[EvalEntry],
        method_config: Dict[str, Any],
    ) -> List[EvalEntry]:
        """
        Run the evaluation pipeline for a specific method.

        Args:
            entries: List of evaluation entries
            method_config: Method-specific configuration

        Returns:
            List of entries with model answers filled in
        """
        retrieval_cfg = method_config.get("retrieval")
        method_name = method_config.get("name", "unknown")
        highlighter_model = method_config.get("highlighter_model")

        print(f"\n{'='*60}")
        print(f"Running method: {method_name}")
        print(f"{'='*60}")

        # Step 1: Retrieval (if enabled)
        if retrieval_cfg and self.retriever:
            print("\nStep 1: Retrieval...")
            retrieval_data = self._get_cached_retrieval(entries, retrieval_cfg)

            for entry in entries:
                docs, doc_ids = retrieval_data.get(entry.id, ([], []))
                entry.retrieved_docs = docs
                entry.retrieved_doc_ids = doc_ids
        else:
            print("\nStep 1: No retrieval (bare LLM)")
            for entry in entries:
                entry.retrieved_docs = []
                entry.retrieved_doc_ids = []

        # Step 2: Highlighter (if enabled)
        if highlighter_model and self.highlighter_manager:
            print("\nStep 2: Highlighter compression...")
            self._apply_highlighter(entries, highlighter_model)
        else:
            print("\nStep 2: No highlighter")
            for entry in entries:
                entry.highlighted_docs = entry.retrieved_docs

        # Step 3: Build all messages
        print("\nStep 3: Building prompts...")
        messages_list = []
        for entry in tqdm(entries, desc="Building prompts"):
            # Use highlighted docs if available, otherwise raw retrieved docs
            docs = entry.highlighted_docs if entry.highlighted_docs is not None else entry.retrieved_docs
            context = self._format_context(docs or [])
            entry.context = context
            if self.context_token_counter:
                entry.context_tokens = self.context_token_counter.count(context)

            # Choose system prompt based on whether we have retrieval
            system = self.system_prompt if retrieval_cfg else self.system_prompt_bare

            messages = build_messages(
                entry=entry,
                system_prompt=system,
                user_template=self.user_template,
                context=context,
            )
            messages_list.append(messages)

        # Step 4: Batch LLM generation
        print("\nStep 4: Batch LLM generation...")
        generator = BatchGenerator(
            client=self.llm_client,
            batch_size=self.batch_size,
            sampling_params=self.sampling_params,
        )

        responses = generator.generate_all(messages_list, desc="Generating answers")

        # Step 5: Collect results
        print("\nStep 5: Collecting results...")
        for entry, response in zip(entries, responses):
            entry.model_answer = response.text
            entry.input_tokens = response.input_tokens
            entry.output_tokens = response.output_tokens

        return entries


def run_evaluation(
    config: Dict[str, Any],
    methods: List[str] = None,
) -> Dict[str, GroupedMetrics]:
    """
    Run full evaluation pipeline with optimized execution order.

    Args:
        config: Full configuration dictionary
        methods: List of method names to run (None = all)

    Returns:
        Dictionary mapping method name to GroupedMetrics (all, non-perturbed, perturbed)
    """
    # Load dataset
    dataset_cfg = config.get("dataset", {})
    qa_path = dataset_cfg.get("qa_path")
    field_mappings = dataset_cfg.get("fields", {})

    print(f"Loading dataset from {qa_path}...")
    entries = load_qa_dataset(qa_path, field_mappings)
    print(f"Loaded {len(entries)} entries")

    # Count perturbation status
    perturbed_count = sum(1 for e in entries if e.is_hard_negative)
    non_perturbed_count = len(entries) - perturbed_count
    print(f"  Non-perturbed: {non_perturbed_count}, Perturbed: {perturbed_count}")

    # Create LLM client
    print("\nInitializing LLM client...")
    llm_client = create_llm_client(config.get("llm", {}))

    # Create retriever
    print("Initializing retriever...")
    retriever = create_retriever(config.get("retrieval", {}))

    # Create highlighter manager (if enabled)
    highlighter_manager = None
    highlighter_cfg = config.get("highlighter", {})
    if highlighter_cfg.get("enabled", False) and highlighter_cfg.get("models"):
        print("Initializing highlighter manager...")
        from .highlighter import create_highlighter_manager
        highlighter_manager = create_highlighter_manager(config)

    # Create LLM judge (if enabled)
    llm_judge = None
    judge_cfg = config.get("judge", {})
    if judge_cfg.get("run_during_eval", False):
        print("Initializing LLM judge...")
        try:
            judge_client = create_llm_client(judge_cfg)
            llm_judge = LLMJudge(
                llm_client=judge_client,
                batch_size=judge_cfg.get("batch_size", 128),
                sampling_params=judge_cfg.get("sampling", {}),
                cache_path=judge_cfg.get("cache_path"),
                model_alias=judge_cfg.get(judge_cfg.get("mode", "qwen"), {}).get(
                    "model_name", "unknown"
                ),
                model_version=judge_cfg.get("model_version"),
                prompt_version=judge_cfg.get("prompt_version", "clean_v2_yes_no_v1"),
            )
            print("  LLM judge ready")
        except Exception as e:
            print(f"  Warning: Could not initialize judge: {e}")

    # Create pipeline
    pipeline = EvaluationPipeline(
        llm_client=llm_client,
        retriever=retriever,
        highlighter_manager=highlighter_manager,
        config=config,
    )

    # Expand methods (highlighter methods become multiple methods)
    all_methods = expand_methods(config)

    # Filter by requested methods
    if methods:
        # Match by base name or full name
        filtered = []
        for m in all_methods:
            name = m.get("name", "")
            base_name = name.split("_")[0] if "_" in name else name
            # Match if name starts with any requested method or exact match
            if name in methods or any(name.startswith(req) for req in methods):
                filtered.append(m)
            # Also match base highlighter methods
            elif any(m.get("highlighter") and req == base_name for req in methods):
                filtered.append(m)
        all_methods = filtered

    print(f"\nMethods to run ({len(all_methods)}):")
    for m in all_methods:
        hl = f" [HL: {m.get('highlighter_model', '').split('/')[-1]}]" if m.get("highlighter_model") else ""
        print(f"  - {m['name']}{hl}")

    # Output directory
    output_cfg = config.get("output", {})
    output_dir = Path(output_cfg.get("dir", "./evaluation/results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run each method
    all_metrics = {}

    for method_cfg in all_methods:
        method_name = method_cfg.get("name", "unknown")

        # Make a fresh copy of entries for each method
        method_entries = [
            EvalEntry(
                id=e.id,
                question=e.question,
                question_ts=e.question_ts,
                answers=e.answers.copy(),
                additional_answers=e.additional_answers.copy(),
                is_hard_negative=e.is_hard_negative,
                source=e.source,
                gt_context=e.gt_context,
            )
            for e in entries
        ]

        # Run pipeline
        results = pipeline.run(method_entries, method_cfg)

        # Convert to dicts for metrics
        results_dicts = [e.to_dict() for e in results]

        # Compute grouped metrics (all, non-perturbed, perturbed)
        grouped_metrics = evaluate_predictions_grouped(results_dicts)

        # Run LLM judge if enabled
        if llm_judge:
            print("\nStep 6: Running LLM judge...")
            judge_results = llm_judge.evaluate(results_dicts)

            # Update metrics with judge accuracy
            grouped_metrics.all.judge_correct = judge_results.correct
            grouped_metrics.all.judge_accuracy = judge_results.accuracy
            grouped_metrics.all.n_judge = judge_results.entries_with_refs

            # Also compute judge accuracy for subgroups
            non_perturbed_dicts = [r for r in results_dicts if not r.get("is_hard_negative", False)]
            perturbed_dicts = [r for r in results_dicts if r.get("is_hard_negative", False)]

            if non_perturbed_dicts:
                np_judge = aggregate_cached_judgments(non_perturbed_dicts, judge_results.judgments)
                grouped_metrics.non_perturbed.judge_correct = np_judge.correct
                grouped_metrics.non_perturbed.judge_accuracy = np_judge.accuracy
                grouped_metrics.non_perturbed.n_judge = np_judge.entries_with_refs

            if perturbed_dicts:
                p_judge = aggregate_cached_judgments(perturbed_dicts, judge_results.judgments)
                grouped_metrics.perturbed.judge_correct = p_judge.correct
                grouped_metrics.perturbed.judge_accuracy = p_judge.accuracy
                grouped_metrics.perturbed.n_judge = p_judge.entries_with_refs

            print(f"  Judge accuracy: {judge_results.accuracy:.4f} ({judge_results.correct}/{judge_results.entries_with_refs})")

        all_metrics[method_name] = grouped_metrics

        # Print grouped metrics
        print(format_grouped_metrics_report(grouped_metrics, method_name))

        # Save results
        if output_cfg.get("save_predictions", True):
            output_path = output_dir / f"{method_name}_{timestamp}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results_dicts, f, ensure_ascii=False, indent=2)
            print(f"Predictions saved to: {output_path}")

    # Unload highlighter to free memory
    if highlighter_manager:
        highlighter_manager.unload()

    # Save aggregated metrics
    if output_cfg.get("save_metrics", True):
        def metrics_to_dict(m: AggregatedMetrics) -> Dict[str, Any]:
            result = {
                "n_total": m.n_total,
                "n_qa": m.n_qa,
                "n_token": m.n_token,
                "n_judge": m.n_judge,
                "entries_with_refs": m.entries_with_refs,
                "f1_micro": m.f1_micro,
                "em_micro": m.em_micro,
                "total_input_tokens": m.total_input_tokens,
                "total_output_tokens": m.total_output_tokens,
                "total_context_tokens": m.total_context_tokens,
                "avg_input_tokens": m.avg_input_tokens,
                "avg_output_tokens": m.avg_output_tokens,
                "avg_context_tokens": m.avg_context_tokens,
            }
            # Include judge metrics if available
            if m.judge_accuracy > 0 or m.judge_correct > 0:
                result["judge_correct"] = m.judge_correct
                result["judge_accuracy"] = m.judge_accuracy
            return result

        metrics_summary = {
            name: {
                "all": metrics_to_dict(gm.all),
                "non_perturbed": metrics_to_dict(gm.non_perturbed),
                "perturbed": metrics_to_dict(gm.perturbed),
            }
            for name, gm in all_metrics.items()
        }
        metrics_path = output_dir / f"metrics_{timestamp}.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_summary, f, ensure_ascii=False, indent=2)
        print(f"\nMetrics summary saved to: {metrics_path}")

    return all_metrics
