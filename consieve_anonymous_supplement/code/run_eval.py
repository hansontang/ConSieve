#!/usr/bin/env python3
"""
Evaluation pipeline CLI entry point.

Usage:
    # Run all methods
    python -m evaluation.run_eval --config evaluation/config.yaml

    # Run specific methods
    python -m evaluation.run_eval --config evaluation/config.yaml --methods bare_llm bm25_top5

    # Dry run (show what would be done)
    python -m evaluation.run_eval --dry-run

    # Test run (5 examples per method)
    python -m evaluation.run_eval --test-run
    python -m evaluation.run_eval --test-run --test-samples 10

    # Build highlighter cache only (no LLM/judge required)
    python -m evaluation.run_eval --highlighter-only
    python -m evaluation.run_eval --highlighter-only --methods bm25_top5_highlighter

    # Run LLM judge on existing predictions
    python -m evaluation.run_eval --judge evaluation/results/bm25_top5_20240119.json

    # Override config values
    python -m evaluation.run_eval --config evaluation/config.yaml --llm-mode server --top-k 10

    # Qwen Batch API (submit and come back later):
    python -m evaluation.run_eval --batch-submit --methods bm25_top5_highlighter
    python -m evaluation.run_eval --batch-status
    python -m evaluation.run_eval --batch-retrieve <batch_id>
"""

# Set CUDA allocator config BEFORE importing PyTorch to help with memory fragmentation
import os
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import gc
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    from .config_utils import resolve_environment

    with open(config_path, "r", encoding="utf-8") as f:
        return resolve_environment(yaml.safe_load(f), strict=True)


def merge_config_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Merge command-line overrides into config."""
    # LLM mode override
    if args.llm_mode:
        config.setdefault("llm", {})["mode"] = args.llm_mode

    # Judge mode override
    if args.judge_mode:
        config.setdefault("judge", {})["mode"] = args.judge_mode

    # Server URL override
    if args.server_url:
        config.setdefault("llm", {}).setdefault("server", {})["base_url"] = args.server_url

    # Model name override
    if args.model:
        config.setdefault("llm", {}).setdefault("server", {})["model_name"] = args.model
        config.setdefault("llm", {}).setdefault("local", {})["model_name"] = args.model

    # Index path override
    if args.index_path:
        config.setdefault("retrieval", {})["index_path"] = args.index_path

    # QA path override
    if args.qa_path:
        config.setdefault("dataset", {})["qa_path"] = args.qa_path

    # Output dir override
    if args.output_dir:
        config.setdefault("output", {})["dir"] = args.output_dir

    return config


def run_dry_run(config: Dict[str, Any], methods: List[str] = None, highlighter_only: bool = False):
    """Show what would be done without actually running."""
    from .pipeline import expand_methods

    print("\n" + "=" * 70)
    print("DRY RUN - Configuration Summary" + (" (highlighter-only mode)" if highlighter_only else ""))
    print("=" * 70)

    # LLM Config
    llm_cfg = config.get("llm", {})
    llm_mode = llm_cfg.get("mode", "server")
    print("\n[LLM Configuration]")
    print(f"  Mode: {llm_mode}")
    if llm_mode == "server":
        server_cfg = llm_cfg.get("server", {})
        print(f"  Server URL: {server_cfg.get('base_url', 'N/A')}")
        print(f"  Model: {server_cfg.get('model_name', 'N/A')}")
        print(f"  Max Concurrency: {server_cfg.get('max_concurrency', 'N/A')}")
    else:
        local_cfg = llm_cfg.get("local", {})
        print(f"  Model: {local_cfg.get('model_name', 'N/A')}")
        print(f"  Tensor Parallel: {local_cfg.get('tensor_parallel_size', 1)}")

    sampling = llm_cfg.get("sampling", {})
    print(f"  Sampling: max_tokens={sampling.get('max_tokens')}, temp={sampling.get('temperature')}")

    # Judge Config
    judge_cfg = config.get("judge", {})
    judge_mode = judge_cfg.get("mode", "server")
    print("\n[Judge Configuration]")
    print(f"  Mode: {judge_mode}")
    if judge_mode == "server":
        server_cfg = judge_cfg.get("server", {})
        print(f"  Server URL: {server_cfg.get('base_url', 'N/A')}")
        print(f"  Model: {server_cfg.get('model_name', 'N/A')}")

    # Retrieval Config
    retrieval_cfg = config.get("retrieval", {})
    print("\n[Retrieval Configuration]")
    print(f"  Index Path: {retrieval_cfg.get('index_path', 'N/A')}")
    index_path = Path(retrieval_cfg.get("index_path", ""))
    if index_path.exists():
        print(f"  Index Status: EXISTS")
    else:
        print(f"  Index Status: NOT FOUND (retrieval methods will fail)")

    bm25_cfg = retrieval_cfg.get("bm25", {})
    print(f"  BM25 Params: k1={bm25_cfg.get('k1')}, b={bm25_cfg.get('b')}")
    print(f"  BM25 Cache: {retrieval_cfg.get('bm25_cache_path', 'N/A')}")

    reranker_cfg = retrieval_cfg.get("reranker", {})
    if reranker_cfg:
        print(f"  Reranker: {reranker_cfg.get('model_name', 'N/A')}")

    # Highlighter Config
    highlighter_cfg = config.get("highlighter", {})
    print("\n[Highlighter Configuration]")
    print(f"  Enabled: {highlighter_cfg.get('enabled', False)}")
    if highlighter_cfg.get("enabled"):
        print(f"  Threshold: {highlighter_cfg.get('threshold', 0.5)}")
        print(f"  Cache Path: {highlighter_cfg.get('cache_path', 'N/A')}")
        print(f"  Models ({len(highlighter_cfg.get('models', []))}):")
        for model in highlighter_cfg.get("models", []):
            model_short = model.split("/")[-1] if "/" in model else model
            print(f"    - {model_short}")

    # Dataset Config
    dataset_cfg = config.get("dataset", {})
    qa_path = dataset_cfg.get("qa_path", "")
    print("\n[Dataset Configuration]")
    print(f"  QA Path: {qa_path}")

    if Path(qa_path).exists():
        # Count entries
        try:
            with open(qa_path, "r") as f:
                if qa_path.endswith(".jsonl"):
                    count = sum(1 for line in f if line.strip())
                else:
                    data = json.load(f)
                    count = len(data) if isinstance(data, list) else 1
            print(f"  Dataset Status: EXISTS ({count:,} entries)")
        except Exception as e:
            print(f"  Dataset Status: ERROR reading ({e})")
    else:
        print(f"  Dataset Status: NOT FOUND")

    # Expand methods (highlighter methods become multiple)
    all_methods = expand_methods(config)
    if methods:
        # Filter by requested methods
        filtered = []
        for m in all_methods:
            name = m.get("name", "")
            if name in methods or any(name.startswith(req) for req in methods):
                filtered.append(m)
        all_methods = filtered

    # In highlighter-only mode, filter to only highlighter methods
    if highlighter_only:
        hl_only = [m for m in all_methods if m.get("highlighter_model")]
        skipped = len(all_methods) - len(hl_only)
        if skipped > 0:
            print(f"\n  [Highlighter-only mode: {skipped} non-highlighter methods will be skipped]")
        all_methods = hl_only

    # Count method types
    non_hl_methods = [m for m in all_methods if not m.get("highlighter_model")]
    hl_methods = [m for m in all_methods if m.get("highlighter_model")]

    print(f"\n[Methods to Run] ({len(all_methods)} total)")
    if not highlighter_only:
        print(f"  Non-highlighter methods: {len(non_hl_methods)}")
    print(f"  Highlighter methods: {len(hl_methods)}")

    # Show non-highlighter methods
    if non_hl_methods:
        print("\n  Non-Highlighter Methods:")
        for i, method in enumerate(non_hl_methods, 1):
            name = method.get("name", "unknown")
            retrieval = method.get("retrieval")
            ret_str = f"{retrieval.get('type')} (top_k={retrieval.get('top_k')})" if retrieval else "None"
            print(f"    {i}. {name} - Retrieval: {ret_str}")

    # Show highlighter methods (grouped by base method)
    if hl_methods:
        print("\n  Highlighter Methods:")
        # Group by base method name
        base_methods = {}
        for m in hl_methods:
            # Extract base name (before model suffix)
            name = m.get("name", "")
            parts = name.rsplit("_", 1)
            if len(parts) > 1 and any(hm.split("/")[-1] == parts[-1] for hm in highlighter_cfg.get("models", [])):
                base = "_".join(name.split("_")[:-1])
            else:
                base = name
            if base not in base_methods:
                base_methods[base] = []
            base_methods[base].append(m)

        for base_name, group in base_methods.items():
            retrieval = group[0].get("retrieval")
            ret_str = f"{retrieval.get('type')} (top_k={retrieval.get('top_k')})" if retrieval else "None"
            models = [m.get("highlighter_model", "").split("/")[-1] for m in group]
            print(f"    {base_name} - Retrieval: {ret_str}")
            print(f"      → {len(models)} variants: {', '.join(models[:3])}{'...' if len(models) > 3 else ''}")

    # Output Config
    output_cfg = config.get("output", {})
    print("\n[Output Configuration]")
    print(f"  Output Dir: {output_cfg.get('dir', 'N/A')}")
    print(f"  Save Predictions: {output_cfg.get('save_predictions', True)}")
    print(f"  Save Metrics: {output_cfg.get('save_metrics', True)}")

    print("\n" + "=" * 70)
    print("To run evaluation, remove --dry-run flag")
    print("=" * 70)


def run_test_run(config: Dict[str, Any], methods: List[str] = None, num_samples: int = 5, notify=None, notify_error=None, force_download: bool = False, use_cache: bool = False):
    """Run evaluation on a small number of samples for testing.

    GPU memory optimization: Run in order that avoids reloading vLLM:
    1. Retrieval + Reranking for ALL methods
    2. Unload reranker
    3. Highlighting for highlighter methods
    4. Unload highlighter
    5. Load vLLM once
    6. LLM generation for ALL methods

    Args:
        force_download: Force re-download highlighter models from HuggingFace
        use_cache: Enable caching (by default, test runs disable cache)
    """
    import gc
    import time
    import torch
    from tqdm import tqdm
    from .pipeline import load_qa_dataset, expand_methods, EvalEntry
    from .llm_client import create_llm_client, BatchGenerator
    from .retriever import create_retriever
    from .metrics import evaluate_predictions_grouped

    start_time = time.time()

    print("\n" + "=" * 70)
    print(f"TEST RUN - Running {num_samples} samples per method")
    print("=" * 70)

    # Load dataset
    dataset_cfg = config.get("dataset", {})
    qa_path = dataset_cfg.get("qa_path")
    field_mappings = dataset_cfg.get("fields", {})

    print(f"\nLoading dataset from {qa_path}...")
    all_entries = load_qa_dataset(qa_path, field_mappings)
    print(f"Total entries: {len(all_entries)}")

    test_entries = all_entries[:num_samples]
    print(f"Using first {len(test_entries)} entries for test run")

    # Show sample questions
    print("\n[Sample Questions]")
    for i, entry in enumerate(test_entries[:3], 1):
        q = entry.question[:80] + "..." if len(entry.question) > 80 else entry.question
        print(f"  {i}. {q}")

    # Expand and filter methods
    all_methods = expand_methods(config)
    if methods:
        filtered = []
        for m in all_methods:
            name = m.get("name", "")
            if name in methods or any(name.startswith(req) for req in methods):
                filtered.append(m)
        all_methods = filtered

    non_hl_methods = [m for m in all_methods if not m.get("highlighter_model")]
    hl_methods = [m for m in all_methods if m.get("highlighter_model")]

    print(f"\n[Methods to Run] ({len(all_methods)} total)")
    print(f"  Non-highlighter: {len(non_hl_methods)}")
    print(f"  Highlighter: {len(hl_methods)}")

    # Get config values
    highlighter_cfg = config.get("highlighter", {})
    highlighter_enabled = highlighter_cfg.get("enabled", False) and highlighter_cfg.get("models")
    judge_cfg = config.get("judge", {})
    llm_cfg = config.get("llm", {})
    llm_mode = llm_cfg.get("mode", "server")

    # Prompt configuration
    prompts_cfg = config.get("prompts", {})
    system_prompt = prompts_cfg.get("system", "You are a helpful assistant.")
    system_prompt_bare = prompts_cfg.get("system_bare", system_prompt)
    user_template = prompts_cfg.get("user", "{question}")
    context_format = prompts_cfg.get("context_format", "Retrieved Document {rank}: {content}")
    context_separator = prompts_cfg.get("context_separator", "\n\n")

    def format_context(docs):
        if not docs:
            return ""
        parts = []
        for i, doc in enumerate(docs, 1):
            parts.append(context_format.format(rank=i, content=doc))
        return context_separator.join(parts)

    # Store all method entries: method_name -> (entries, method_cfg)
    method_entries_map = {}

    # =========================================================================
    # PHASE 1: Retrieval + Reranking for ALL methods
    # =========================================================================
    print("\n" + "-" * 70)
    print("PHASE 1: Retrieval + Reranking")
    print("-" * 70)

    print("  - Loading Retriever...", end=" ", flush=True)
    if not use_cache:
        print("BM25 cache disabled (test mode)")
    try:
        retriever = create_retriever(config.get("retrieval", {}), skip_cache=not use_cache)
        if retriever:
            print("OK")
        else:
            print("SKIPPED (no index configured)")
    except Exception as e:
        print(f"FAILED: {e}")
        retriever = None

    for method_cfg in all_methods:
        method_name = method_cfg.get("name", "unknown")
        retrieval_cfg = method_cfg.get("retrieval")

        # Make a fresh copy of entries
        method_entries = [
            EvalEntry(
                id=e.id,
                question=e.question,
                question_ts=e.question_ts,
                answers=e.answers.copy(),
                additional_answers=e.additional_answers.copy(),
                is_hard_negative=e.is_hard_negative,
                source=e.source,
            )
            for e in test_entries
        ]

        if retrieval_cfg and retriever:
            top_k = retrieval_cfg.get("top_k", 5)
            use_rerank = retrieval_cfg.get("rerank", False)
            rerank_top_k = retrieval_cfg.get("rerank_top_k", top_k) if use_rerank else None
            rerank_info = f", rerank_top_k={rerank_top_k}" if use_rerank else ""

            print(f"  [{method_name}] Retrieval (top_k={top_k}{rerank_info})")
            try:
                for entry in tqdm(method_entries, desc=f"    {method_name}", leave=True):
                    # HybridRetriever.retrieve() returns List[RetrievedDoc]
                    # If rerank_top_k is set, it will rerank the results
                    retrieved_docs = retriever.retrieve(
                        entry.question,
                        top_k=top_k,
                        rerank_top_k=rerank_top_k,
                    )
                    # Extract content strings from RetrievedDoc objects
                    entry.retrieved_docs = [doc.content for doc in retrieved_docs]

                print(f"    Done. Retrieved {len(method_entries)} queries")
            except Exception as e:
                print(f"FAILED: {e}")
                if notify_error:
                    notify_error(context="Retrieval", error=str(e), method_name=method_name)
                continue
        elif retrieval_cfg and not retriever:
            print(f"  [{method_name}] SKIPPED (retriever not available)")
            continue
        else:
            # bare_llm - no retrieval needed
            print(f"  [{method_name}] No retrieval needed")
            for entry in method_entries:
                entry.retrieved_docs = []

        method_entries_map[method_name] = (method_entries, method_cfg)

    # =========================================================================
    # PHASE 2: Unload Reranker
    # =========================================================================
    if retriever and hasattr(retriever, 'reranker') and retriever.reranker:
        print("\n" + "-" * 70)
        print("PHASE 2: Unloading Reranker")
        print("-" * 70)
        try:
            reranker = retriever.reranker
            if hasattr(reranker, 'model'):
                if hasattr(reranker.model, 'cpu'):
                    reranker.model.cpu()
                del reranker.model
            retriever.reranker = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("  Reranker unloaded")
        except Exception as e:
            print(f"  Warning: Could not unload reranker: {e}")

    # =========================================================================
    # PHASE 3: Highlighting (for highlighter methods only)
    # =========================================================================
    if hl_methods and highlighter_enabled:
        print("\n" + "-" * 70)
        print("PHASE 3: Highlighting")
        print("-" * 70)

        print("  - Loading Highlighter Manager...", end=" ", flush=True)
        try:
            from .highlighter import create_highlighter_manager
            highlighter_manager = create_highlighter_manager(config, skip_cache=not use_cache, force_download=force_download)
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")
            highlighter_manager = None

        if highlighter_manager:
            for method_cfg in hl_methods:
                method_name = method_cfg.get("name", "unknown")
                highlighter_model = method_cfg.get("highlighter_model")

                if method_name not in method_entries_map:
                    continue

                method_entries, _ = method_entries_map[method_name]
                threshold = highlighter_cfg.get("threshold", 0.5)
                model_short = highlighter_model.split("/")[-1] if "/" in highlighter_model else highlighter_model

                print(f"  [{method_name}] Highlighting {len(method_entries)} entries (threshold={threshold})")
                try:
                    compression_ratios = []
                    highlighter_manager.reset_cache_stats()

                    pbar = tqdm(method_entries, desc=f"    {model_short}", leave=True)
                    for entry in pbar:
                        if not entry.retrieved_docs:
                            entry.highlighted_docs = []
                            entry.compression_ratio = 0.0
                            entry.highlighter_model = highlighter_model
                            continue

                        from .query_formatter import format_and_validate_query
                        query = format_and_validate_query(entry)

                        pruned = highlighter_manager.prune_documents(
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
                        compression_ratios.append(entry.compression_ratio)

                        # Update progress bar with cache stats
                        hits, total = highlighter_manager.get_cache_stats()
                        pbar.set_postfix(hit=hits, new=total-hits, refresh=True)

                    # Show summary
                    hits, total = highlighter_manager.get_cache_stats()
                    if compression_ratios:
                        avg_ratio = sum(compression_ratios) / len(compression_ratios)
                        print(f"    Done. Avg compression: {avg_ratio:.1%}, Cache: {hits}/{total} hits")
                except Exception as e:
                    print(f"FAILED: {e}")
                    del method_entries_map[method_name]
                    if notify_error:
                        notify_error(context="Highlighting", error=str(e), method_name=method_name)

            # Flush cache and unload highlighter
            print("\n  Flushing highlighter cache...", end=" ", flush=True)
            highlighter_manager.flush_cache()
            print("OK")
            print("  Unloading highlighter...", end=" ", flush=True)
            highlighter_manager.unload()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            print("OK")

    # =========================================================================
    # PHASE 4: Load vLLM (once) and run ALL LLM generation
    # =========================================================================
    print("\n" + "-" * 70)
    print("PHASE 4: LLM Generation (all methods)")
    print("-" * 70)

    print("  - Loading LLM Client...", end=" ", flush=True)
    try:
        llm_client = create_llm_client(llm_cfg)
        print("OK")
    except Exception as e:
        print(f"FAILED: {e}")
        return

    # Create LLM judge if enabled
    llm_judge = None
    if judge_cfg.get("run_during_eval", False):
        print("  - LLM Judge...", end=" ", flush=True)
        try:
            from .llm_judge import LLMJudge
            judge_mode = judge_cfg.get("mode", llm_mode)
            if llm_mode == "local" and judge_mode == "local":
                print("(sharing LLM client)...", end=" ", flush=True)
                judge_client = llm_client
            else:
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
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")

    # Get batch size
    if llm_mode == "local":
        batch_size = llm_cfg.get("local", {}).get("batch_size", 8)
    else:
        batch_size = llm_cfg.get("server", {}).get("batch_size", 64)
    sampling_params = llm_cfg.get("sampling", {})

    all_results = {}

    # Run LLM generation for all methods
    for method_name, (method_entries, method_cfg) in method_entries_map.items():
        retrieval_cfg = method_cfg.get("retrieval")

        print(f"\n  [{method_name}] Generating answers...")
        try:
            # Build prompts
            messages_list = []
            for entry in method_entries:
                docs = entry.highlighted_docs if entry.highlighted_docs is not None else entry.retrieved_docs
                context = format_context(docs or [])
                entry.context = context

                sys_prompt = system_prompt if retrieval_cfg else system_prompt_bare
                system_content = sys_prompt
                if context:
                    system_content = f"{sys_prompt}\n\n{context}"

                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_template.format(
                        question=entry.question,
                        question_ts=entry.question_ts,
                        context=context,
                    )}
                ]
                messages_list.append(messages)

            # Run LLM generation
            generator = BatchGenerator(
                client=llm_client,
                batch_size=batch_size,
                sampling_params=sampling_params,
            )
            responses = generator.generate_all(messages_list, desc="Generating")

            # Collect results
            for entry, response in zip(method_entries, responses):
                entry.model_answer = response.text
                entry.input_tokens = response.input_tokens
                entry.output_tokens = response.output_tokens

            results_dicts = [e.to_dict() for e in method_entries]
            grouped_metrics = evaluate_predictions_grouped(results_dicts)

            # Run LLM judge if enabled
            if llm_judge:
                print("    Running LLM judge...")
                judge_results = llm_judge.evaluate(results_dicts)
                grouped_metrics.all.judge_correct = judge_results.correct
                grouped_metrics.all.judge_accuracy = judge_results.accuracy
                grouped_metrics.all.n_judge = judge_results.entries_with_refs

                non_perturbed_dicts = [r for r in results_dicts if not r.get("is_hard_negative", False)]
                perturbed_dicts = [r for r in results_dicts if r.get("is_hard_negative", False)]

                if non_perturbed_dicts:
                    from .llm_judge import aggregate_cached_judgments
                    np_judge = aggregate_cached_judgments(non_perturbed_dicts, judge_results.judgments)
                    grouped_metrics.non_perturbed.judge_correct = np_judge.correct
                    grouped_metrics.non_perturbed.judge_accuracy = np_judge.accuracy
                    grouped_metrics.non_perturbed.n_judge = np_judge.entries_with_refs

                if perturbed_dicts:
                    from .llm_judge import aggregate_cached_judgments
                    p_judge = aggregate_cached_judgments(perturbed_dicts, judge_results.judgments)
                    grouped_metrics.perturbed.judge_correct = p_judge.correct
                    grouped_metrics.perturbed.judge_accuracy = p_judge.accuracy
                    grouped_metrics.perturbed.n_judge = p_judge.entries_with_refs

                print(f"    Judge accuracy: {judge_results.accuracy:.4f}")

            # Show sample results
            print(f"\n    Sample Results:")
            for i, entry in enumerate(method_entries[:3], 1):
                ans = entry.model_answer[:50] + "..." if len(entry.model_answer) > 50 else entry.model_answer
                ref = entry.answers[0][:30] + "..." if entry.answers and len(entry.answers[0]) > 30 else (entry.answers[0] if entry.answers else "N/A")
                print(f"      {i}. Pred: {ans}")
                print(f"         Ref:  {ref}")
                print(f"         Tokens: in={entry.input_tokens}, out={entry.output_tokens}")
                if entry.compression_ratio > 0:
                    print(f"         Compression: {entry.compression_ratio:.1%}")

            all_results[method_name] = grouped_metrics

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            if notify_error:
                notify_error(context="LLM Generation", error=str(e), method_name=method_name)

    # Print summary
    if all_results:
        has_judge = any(gm.all.judge_accuracy > 0 for gm in all_results.values())
        has_gt_ret = any(gm.all.gt_total_count > 0 for gm in all_results.values())
        width = 110
        if has_judge:
            width += 10
        if has_gt_ret:
            width += 10

        print("\n" + "=" * width)
        print("TEST RUN SUMMARY")
        print("=" * width)
        header = f"{'Method':<35} {'Category':<28} {'Count':>6} {'F1':>8} {'EM':>8}"
        if has_judge:
            header += f" {'Judge':>8}"
        if has_gt_ret:
            header += f" {'GT Ret':>8}"
        header += f" {'Avg In':>10} {'Avg Out':>10}"
        print(header)
        print("-" * width)

        # All 9 categories
        category_list = [
            ("All", "all"),
            ("Non-Perturbed", "non_perturbed"),
            ("Perturbed", "perturbed"),
            ("StreamingQA", "streamingqa"),
            ("StreamingQA Non-Perturbed", "streamingqa_non_perturbed"),
            ("StreamingQA Perturbed", "streamingqa_perturbed"),
            ("TimeQA", "timeqa"),
            ("TimeQA Non-Perturbed", "timeqa_non_perturbed"),
            ("TimeQA Perturbed", "timeqa_perturbed"),
        ]

        for name, grouped in all_results.items():
            display_name = name[:32] + "..." if len(name) > 35 else name
            for cat_name, cat_attr in category_list:
                m = getattr(grouped, cat_attr)
                if m.entries_with_refs > 0:
                    row = (
                        f"{display_name:<35} "
                        f"{cat_name:<28} "
                        f"{m.entries_with_refs:>6} "
                        f"{m.f1_micro:>8.4f} "
                        f"{m.em_micro:>8.4f}"
                    )
                    if has_judge:
                        row += f" {m.judge_accuracy:>8.4f}"
                    if has_gt_ret:
                        if m.gt_total_count > 0:
                            row += f" {m.gt_retrieval_rate:>8.4f}"
                        else:
                            row += f" {'N/A':>8}"
                    row += f" {m.avg_input_tokens:>10.1f} {m.avg_output_tokens:>10.1f}"
                    print(row)
                else:
                    row = f"{display_name:<35} {cat_name:<28} {'N/A':>6} {'N/A':>8} {'N/A':>8}"
                    if has_judge:
                        row += f" {'N/A':>8}"
                    if has_gt_ret:
                        row += f" {'N/A':>8}"
                    row += f" {'N/A':>10} {'N/A':>10}"
                    print(row)
            print("-" * width)
        print("=" * width)
        print(f"\nTest run complete! Tested {num_samples} samples per method.")
        print("To run full evaluation, remove --test-run flag")

        if notify and all_results:
            elapsed = time.time() - start_time
            notify(all_results, total_time=elapsed)


def run_pipeline(config: Dict[str, Any], methods: list = None, notify=None, notify_error=None, notify_step=None, stepwise=True, highlighter_only=False, force_download: bool = False):
    """Run the main evaluation pipeline.

    Args:
        force_download: Force re-download highlighter models from HuggingFace
    """
    import time

    start_time = time.time()

    if config.get("protocol_version") == "clean_v2":
        from .clean_protocol import assert_clean_v2_config
        from .experiment_manifest import write_experiment_manifest

        assert_clean_v2_config(config)
        write_experiment_manifest(config, config.get("output", {}).get("dir", "./evaluation/results"))

    try:
        if stepwise:
            # Use step-wise execution for better resource efficiency
            from .pipeline_stepwise import run_evaluation_stepwise
            results = run_evaluation_stepwise(
                config,
                methods=methods,
                notify_step=notify_step,
                notify_error=notify_error,
                highlighter_only=highlighter_only,
                force_download=force_download,
            )
        else:
            # Use original method-by-method execution
            from .pipeline import run_evaluation
            results = run_evaluation(config, methods=methods)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        if notify_error:
            notify_error(context="Full pipeline", error=str(e))
        return

    # Skip summary if highlighter-only mode (returns empty dict)
    if not results:
        return

    # Print summary with grouped metrics
    # Check if any method has judge metrics
    has_judge = any(gm.all.judge_accuracy > 0 for gm in results.values())
    has_gt_ret = any(gm.all.gt_total_count > 0 for gm in results.values())
    width = 110
    if has_judge:
        width += 10
    if has_gt_ret:
        width += 10

    print("\n" + "=" * width)
    print("EVALUATION SUMMARY")
    print("=" * width)

    header = f"{'Method':<40} {'Category':<28} {'Count':>6} {'F1':>8} {'EM':>8}"
    if has_judge:
        header += f" {'Judge':>8}"
    if has_gt_ret:
        header += f" {'GT Ret':>8}"
    header += f" {'In Tok':>12} {'Out Tok':>12}"
    print(header)
    print("-" * width)

    # All 9 categories
    category_list = [
        ("All", "all"),
        ("Non-Perturbed", "non_perturbed"),
        ("Perturbed", "perturbed"),
        ("StreamingQA", "streamingqa"),
        ("StreamingQA Non-Perturbed", "streamingqa_non_perturbed"),
        ("StreamingQA Perturbed", "streamingqa_perturbed"),
        ("TimeQA", "timeqa"),
        ("TimeQA Non-Perturbed", "timeqa_non_perturbed"),
        ("TimeQA Perturbed", "timeqa_perturbed"),
    ]

    for name, grouped in results.items():
        # Truncate long names
        display_name = name[:37] + "..." if len(name) > 40 else name
        for cat_name, cat_attr in category_list:
            m = getattr(grouped, cat_attr)
            if m.entries_with_refs > 0:
                row = (
                    f"{display_name:<40} "
                    f"{cat_name:<28} "
                    f"{m.entries_with_refs:>6} "
                    f"{m.f1_micro:>8.4f} "
                    f"{m.em_micro:>8.4f}"
                )
                if has_judge:
                    row += f" {m.judge_accuracy:>8.4f}"
                if has_gt_ret:
                    if m.gt_total_count > 0:
                        row += f" {m.gt_retrieval_rate:>8.4f}"
                    else:
                        row += f" {'N/A':>8}"
                row += f" {m.total_input_tokens:>12,} {m.total_output_tokens:>12,}"
                print(row)
            else:
                row = f"{display_name:<40} {cat_name:<28} {'N/A':>6} {'N/A':>8} {'N/A':>8}"
                if has_judge:
                    row += f" {'N/A':>8}"
                if has_gt_ret:
                    row += f" {'N/A':>8}"
                row += f" {'N/A':>12} {'N/A':>12}"
                print(row)
        print("-" * width)
    print("=" * width)

    # Send notification
    if notify and results:
        elapsed = time.time() - start_time
        notify(results, total_time=elapsed)


def run_judge(predictions_path: str, config: Dict[str, Any], output_path: str = None):
    """Run LLM judge on predictions."""
    from .llm_judge import run_llm_judge

    if output_path is None:
        pred_path = Path(predictions_path)
        output_path = str(pred_path.parent / f"{pred_path.stem}_judge.json")

    run_llm_judge(predictions_path, config, output_path)


def run_metrics_only(predictions_path: str):
    """Compute metrics on existing predictions without LLM."""
    from .metrics import evaluate_predictions, format_metrics_report

    with open(predictions_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    metrics = evaluate_predictions(predictions)
    print(format_metrics_report(metrics, predictions_path))


# =============================================================================
# Batch API Operations
# =============================================================================

def run_batch_submit(config: Dict[str, Any], methods: List[str] = None, test_mode: bool = False):
    """
    Submit a batch job to Qwen API.

    This prepares all prompts and submits them as a batch, then exits.
    You can come back later to retrieve results with --batch-retrieve.

    Args:
        config: Configuration dictionary
        methods: List of method names to process (default: all)
        test_mode: If True, use batch-test-model for testing (max 1MB, 100 rows)
    """
    from .pipeline import load_qa_dataset, expand_methods, EvalEntry, build_messages
    from .pipeline_stepwise import run_evaluation_stepwise
    from .retriever import create_retriever
    from .highlighter import LLMCache, create_highlighter_manager
    from .llm_client import QwenBatchClient

    print("\n" + "=" * 70)
    print("BATCH SUBMIT MODE" + (" [TEST]" if test_mode else ""))
    print("=" * 70)

    if test_mode:
        print("\n*** TEST MODE: Using batch-test-model ***")
        print("  - Max file size: 1MB")
        print("  - Max rows per file: 100")
        print("  - Max concurrent tasks: 2")
        print("  - Endpoint: /v1/chat/ds-test\n")

    # Get batch API config
    llm_cfg = config.get("llm", {})
    batch_cfg = llm_cfg.get("batch", {})

    # For test mode, we don't require full batch config
    api_key = batch_cfg.get("api_key") or os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("Error: DASHSCOPE_API_KEY not set")
        print("Set it via environment variable or in config.yaml under llm.batch.api_key")
        return

    # Initialize batch client
    print("Initializing Qwen batch client...")
    try:
        batch_client = QwenBatchClient(
            api_key=api_key,
            model_name=batch_cfg.get("model_name", "qwen-turbo"),
            base_url=batch_cfg.get("base_url"),
            test_mode=test_mode,
        )
        print(f"  Model: {batch_client.model_name}")
        print(f"  Endpoint: {batch_client.endpoint}")
    except Exception as e:
        print(f"Error: Could not initialize batch client: {e}")
        return

    # Initialize LLM cache for batch job tracking
    llm_cache_path = llm_cfg.get("cache_path", "./evaluation/cache/llm_cache.db")
    llm_cache = LLMCache(llm_cache_path)
    print(f"  Cache: {llm_cache_path}")

    # Load dataset
    dataset_cfg = config.get("dataset", {})
    qa_path = dataset_cfg.get("qa_path")
    field_mappings = dataset_cfg.get("fields", {})

    print(f"\nLoading dataset from {qa_path}...")
    base_entries = load_qa_dataset(qa_path, field_mappings)
    print(f"Loaded {len(base_entries)} entries")

    # Apply eval_samples limit from config
    eval_samples = dataset_cfg.get("eval_samples")
    if eval_samples and len(base_entries) > eval_samples:
        print(f"  Limiting to first {eval_samples} entries (eval_samples config)")
        base_entries = base_entries[:eval_samples]

    # In test mode, limit to 50 entries (batch-test-model max is 100 rows)
    if test_mode:
        max_test_entries = 50
        if len(base_entries) > max_test_entries:
            print(f"  [TEST MODE] Limiting to first {max_test_entries} entries")
            base_entries = base_entries[:max_test_entries]

    # Expand and filter methods
    all_methods = expand_methods(config)
    if methods:
        filtered = []
        for m in all_methods:
            name = m.get("name", "")
            if name in methods or any(name.startswith(req) for req in methods):
                filtered.append(m)
        all_methods = filtered

    print(f"\nMethods to process: {len(all_methods)}")
    for m in all_methods:
        print(f"  - {m['name']}")

    # Create retriever for BM25
    print("\nInitializing retriever...")
    retriever = create_retriever(config.get("retrieval", {}))

    # Initialize highlighter if needed
    highlighter_cfg = config.get("highlighter", {})
    highlighter_manager = None
    if highlighter_cfg.get("enabled", False):
        from .highlighter import create_highlighter_manager
        highlighter_manager = create_highlighter_manager(config)

    # Prompt templates
    prompts_cfg = config.get("prompts", {})
    system_prompt = prompts_cfg.get("system", "You are a helpful assistant.")
    system_prompt_bare = prompts_cfg.get("system_bare", system_prompt)
    # Reason from context prompts
    system_prompt_rfc = prompts_cfg.get("system_reason_from_context", system_prompt)
    system_prompt_rfc_bare = prompts_cfg.get("system_reason_from_context_bare", system_prompt_bare)
    user_template = prompts_cfg.get("user", "{question}")
    context_format = prompts_cfg.get("context_format", "Retrieved Document {rank}: {content}")
    context_separator = prompts_cfg.get("context_separator", "\n\n")
    sampling_params = llm_cfg.get("sampling", {})

    # Load reason_from_context dataset if needed
    has_rfc_methods = any(m.get("reason_from_context") for m in all_methods)
    rfc_entries = None
    if has_rfc_methods:
        from .pipeline_stepwise import load_reason_from_context_dataset
        print("\nLoading reason_from_context dataset...")
        rfc_entries = load_reason_from_context_dataset(config, field_mappings)

    def format_context(docs):
        if not docs:
            return ""
        parts = []
        for rank, content in enumerate(docs, start=1):
            if content.strip():
                parts.append(context_format.format(rank=rank, content=content))
        return context_separator.join(parts)

    # Process each method
    for method_cfg in all_methods:
        method_name = method_cfg.get("name", "unknown")
        retrieval_cfg = method_cfg.get("retrieval")
        highlighter_model = method_cfg.get("highlighter_model")
        is_rfc = method_cfg.get("reason_from_context", False)

        # Select appropriate system prompt
        if is_rfc:
            system = system_prompt_rfc if retrieval_cfg else system_prompt_rfc_bare
        else:
            system = system_prompt if retrieval_cfg else system_prompt_bare

        print(f"\n{'='*70}")
        print(f"Processing: {method_name}" + (" [RFC]" if is_rfc else ""))
        print("="*70)

        # Use appropriate dataset
        source_entries = rfc_entries if is_rfc and rfc_entries else base_entries

        # Create entries for this method
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
            for e in source_entries
        ]

        # Run retrieval if needed
        if retrieval_cfg:
            ret_type = retrieval_cfg.get("type", "bm25")

            if ret_type == "ground_truth":
                # Use ground truth context from dataset
                print("  Using ground truth context...")
                for entry in method_entries:
                    if entry.gt_context:
                        entry.retrieved_docs = [entry.gt_context]
                        entry.retrieved_doc_ids = ["gt_context"]
                    else:
                        entry.retrieved_docs = []
                        entry.retrieved_doc_ids = []
            elif ret_type == "ground_truth_noise":
                # Ground truth + BM25 noise documents
                noise_k = retrieval_cfg.get("noise_k", 5)
                print(f"  Using ground truth + {noise_k-1} BM25 noise docs...")

                if retriever:
                    from .query_formatter import format_and_validate_query
                    queries = [format_and_validate_query(e) for e in method_entries]
                    query_ids = [e.id for e in method_entries]
                    results = retriever.batch_retrieve(queries=queries, query_ids=query_ids, top_k=noise_k)

                    for entry in method_entries:
                        bm25_results = results.get(entry.id, [])
                        bm25_docs = [d.content for d in bm25_results]
                        bm25_doc_ids = [d.doc_id for d in bm25_results]
                        gt_doc = entry.gt_context

                        if not gt_doc:
                            entry.retrieved_docs = bm25_docs[:noise_k]
                            entry.retrieved_doc_ids = bm25_doc_ids[:noise_k]
                        else:
                            # Check if GT is in BM25 results
                            gt_in_bm25 = any(
                                gt_doc in doc or doc in gt_doc or gt_doc == doc
                                for doc in bm25_docs[:noise_k]
                            )
                            if gt_in_bm25:
                                entry.retrieved_docs = bm25_docs[:noise_k]
                                entry.retrieved_doc_ids = bm25_doc_ids[:noise_k]
                            else:
                                entry.retrieved_docs = [gt_doc] + bm25_docs[:noise_k-1]
                                entry.retrieved_doc_ids = ["gt_context"] + bm25_doc_ids[:noise_k-1]
                else:
                    # No retriever, just use GT
                    for entry in method_entries:
                        if entry.gt_context:
                            entry.retrieved_docs = [entry.gt_context]
                            entry.retrieved_doc_ids = ["gt_context"]
                        else:
                            entry.retrieved_docs = []
                            entry.retrieved_doc_ids = []
            elif ret_type == "ground_truth_random_noise":
                # Ground truth + random documents as noise
                import random
                noise_k = retrieval_cfg.get("noise_k", 5)
                print(f"  Using ground truth + {noise_k-1} random noise docs...")

                # Collect all gt_contexts as potential noise pool
                all_gt_contexts = [(e.id, e.gt_context) for e in method_entries if e.gt_context]

                for entry in method_entries:
                    gt_doc = entry.gt_context

                    if not gt_doc:
                        entry.retrieved_docs = []
                        entry.retrieved_doc_ids = []
                    else:
                        # Get random docs from other entries
                        noise_pool = [(eid, ctx) for eid, ctx in all_gt_contexts if eid != entry.id]

                        if len(noise_pool) >= noise_k - 1:
                            random_samples = random.sample(noise_pool, noise_k - 1)
                        else:
                            random_samples = noise_pool

                        # Combine GT + random noise and shuffle
                        docs = [gt_doc] + [ctx for _, ctx in random_samples]
                        doc_ids = ["gt_context"] + [f"rand_{eid}" for eid, _ in random_samples]

                        combined = list(zip(docs, doc_ids))
                        random.shuffle(combined)
                        entry.retrieved_docs = [d for d, _ in combined]
                        entry.retrieved_doc_ids = [did for _, did in combined]
            elif retriever:
                print("  Running BM25 retrieval...")
                top_k = retrieval_cfg.get("top_k", 10)
                rerank_top_k = retrieval_cfg.get("rerank_top_k") if "rerank" in ret_type else None

                from .query_formatter import format_and_validate_query
                queries = [format_and_validate_query(e) for e in method_entries]
                query_ids = [e.id for e in method_entries]

                results = retriever.batch_retrieve(
                    queries=queries,
                    query_ids=query_ids,
                    top_k=top_k,
                    rerank_top_k=rerank_top_k,
                )

                for entry in method_entries:
                    docs = results.get(entry.id, [])
                    entry.retrieved_docs = [d.content for d in docs]
                    entry.retrieved_doc_ids = [d.doc_id for d in docs]

        # Run highlighting if needed
        if highlighter_model and highlighter_manager:
            print(f"  Running highlighter: {highlighter_model.split('/')[-1]}...")
            threshold = highlighter_cfg.get("threshold", 0.5)

            for entry in method_entries:
                if entry.retrieved_docs:
                    from .query_formatter import format_and_validate_query
                    query = format_and_validate_query(entry)
                    pruned = highlighter_manager.prune_documents(
                        model_id=highlighter_model,
                        query=query,
                        documents=entry.retrieved_docs,
                        threshold=threshold,
                    )
                    entry.highlighted_docs = pruned
                else:
                    entry.highlighted_docs = []

        # Build contexts
        for entry in method_entries:
            docs = entry.highlighted_docs if entry.highlighted_docs is not None else entry.retrieved_docs
            entry.context = format_context(docs or [])

        # Check cache for existing results
        print("  Checking cache for existing results...")
        cache_lookup = [(entry.id, entry.context) for entry in method_entries]
        cached_results = llm_cache.get_batch(
            method_name, cache_lookup,
            model_name=batch_client.model_name, sampling_params=sampling_params
        )
        print(f"    Found {len(cached_results)} cached results")

        # Filter to uncached entries
        entries_to_submit = [e for e in method_entries if e.id not in cached_results]
        print(f"    {len(entries_to_submit)} entries need generation")

        if not entries_to_submit:
            print("  All entries cached, skipping batch submission")
            continue

        # Build messages for batch submission
        print("  Preparing batch request...")
        messages_list = []
        custom_ids = []
        batch_entries = []  # For saving to cache tracking

        for entry in entries_to_submit:
            messages = build_messages(
                entry=entry,
                system_prompt=system,
                user_template=user_template,
                context=entry.context,
            )
            messages_list.append(messages)
            custom_id = f"{method_name}_{entry.id}"
            custom_ids.append(custom_id)
            batch_entries.append((custom_id, entry.id, entry.context))

        # Split into chunks based on per_batch_num
        per_batch_num = batch_cfg.get("per_batch_num", 100)
        total_requests = len(messages_list)
        num_batches = (total_requests + per_batch_num - 1) // per_batch_num

        print(f"  Submitting {total_requests} requests in {num_batches} batch(es) (max {per_batch_num} per batch)...")

        for batch_idx in range(num_batches):
            start_idx = batch_idx * per_batch_num
            end_idx = min(start_idx + per_batch_num, total_requests)

            chunk_messages = messages_list[start_idx:end_idx]
            chunk_custom_ids = custom_ids[start_idx:end_idx]
            chunk_entries = batch_entries[start_idx:end_idx]

            try:
                batch_info = batch_client.submit_batch(
                    messages_list=chunk_messages,
                    sampling_params=sampling_params,
                    custom_ids=chunk_custom_ids,
                    description=f"Eval: {method_name} [{batch_idx+1}/{num_batches}] ({len(chunk_messages)} requests)",
                )
                print(f"    Batch {batch_idx+1}/{num_batches}: {batch_info.batch_id} ({len(chunk_messages)} requests)")

                # Save batch job to cache for later retrieval
                llm_cache.save_batch_job(
                    batch_id=batch_info.batch_id,
                    method_name=method_name,
                    entries=chunk_entries,
                    status=batch_info.status,
                    model_name=batch_client.model_name,
                    sampling_params=sampling_params,
                )

            except Exception as e:
                print(f"    ERROR submitting batch {batch_idx+1}: {e}")
                continue

    # Flush cache and unload highlighter
    if highlighter_manager:
        highlighter_manager.flush_cache()
        highlighter_manager.unload()

    # Show summary
    print("\n" + "=" * 70)
    print("BATCH SUBMISSION COMPLETE")
    print("=" * 70)
    print("\nTo check status:")
    print("  python -m evaluation.run_eval --batch-status")
    print("\nTo retrieve results when complete:")
    print("  python -m evaluation.run_eval --batch-retrieve <batch_id>")
    print("  python -m evaluation.run_eval --batch-retrieve all")


def run_batch_status(config: Dict[str, Any], test_mode: bool = False):
    """Show status of all batch jobs."""
    from .highlighter import LLMCache
    from .llm_client import QwenBatchClient

    print("\n" + "=" * 70)
    print("BATCH JOB STATUS" + (" [TEST MODE]" if test_mode else ""))
    print("=" * 70)

    # Initialize cache
    llm_cfg = config.get("llm", {})
    llm_cache_path = llm_cfg.get("cache_path", "./evaluation/cache/llm_cache.db")
    llm_cache = LLMCache(llm_cache_path)

    # Get all batch jobs from cache
    jobs = llm_cache.get_all_batch_jobs()

    if not jobs:
        print("\nNo batch jobs found in cache.")
        return

    # Try to update status from API for pending jobs
    batch_cfg = llm_cfg.get("batch", {})
    batch_client = None
    api_key = batch_cfg.get("api_key") or os.environ.get("DASHSCOPE_API_KEY")
    if api_key:
        try:
            batch_client = QwenBatchClient(
                api_key=api_key,
                model_name=batch_cfg.get("model_name", "qwen-turbo"),
                test_mode=test_mode,
            )
        except Exception:
            pass

    print(f"\n{'Batch ID':<40} {'Method':<30} {'Status':<12} {'Progress':<15} {'Created'}")
    print("-" * 120)

    for job in jobs:
        batch_id = job["batch_id"]
        status = job["status"]

        # Update status from API if pending/running
        if batch_client and status in ("pending", "running"):
            try:
                api_status = batch_client.check_batch_status(batch_id)
                if api_status.status != status:
                    llm_cache.update_batch_status(
                        batch_id,
                        api_status.status,
                        api_status.completed_requests,
                        api_status.error_message,
                    )
                    status = api_status.status
                    job["completed_requests"] = api_status.completed_requests
            except Exception:
                pass

        progress = f"{job['completed_requests']}/{job['total_requests']}"
        method = job["method_name"][:28] + ".." if len(job["method_name"]) > 30 else job["method_name"]

        print(f"{batch_id:<40} {method:<30} {status:<12} {progress:<15} {job['created_at']}")

        if job.get("error_message"):
            print(f"  Error: {job['error_message']}")

    print("-" * 120)

    # Show pending jobs summary
    pending = [j for j in jobs if j["status"] in ("pending", "running")]
    completed = [j for j in jobs if j["status"] == "completed"]
    failed = [j for j in jobs if j["status"] == "failed"]

    print(f"\nSummary: {len(pending)} pending/running, {len(completed)} completed, {len(failed)} failed")

    if pending:
        print("\nTo retrieve completed results:")
        print("  python -m evaluation.run_eval --batch-retrieve <batch_id>")
        print("  python -m evaluation.run_eval --batch-retrieve all")


def run_batch_repopulate(config: Dict[str, Any], batch_id: str):
    """
    Repopulate batch_requests for a batch job.

    This is useful when batch_requests was deleted but you need to re-import results.
    It re-runs the retrieval/highlighting to rebuild the context and context_hash.
    """
    from .pipeline import load_qa_dataset, expand_methods, EvalEntry
    from .retriever import create_retriever
    from .highlighter import LLMCache, create_highlighter_manager, _hash_text

    print("\n" + "=" * 70)
    print("BATCH REPOPULATE")
    print("=" * 70)

    # Initialize cache
    llm_cfg = config.get("llm", {})
    llm_cache_path = llm_cfg.get("cache_path", "./evaluation/cache/llm_cache.db")
    llm_cache = LLMCache(llm_cache_path)

    # Get batch job info
    batch_job = llm_cache.get_batch_job(batch_id)
    if not batch_job:
        print(f"Error: Batch job {batch_id} not found")
        return

    method_name = batch_job["method_name"]
    print(f"  Batch ID: {batch_id}")
    print(f"  Method: {method_name}")

    # Check if batch_requests already populated
    existing = llm_cache.get_batch_requests(batch_id)
    if existing:
        print(f"  batch_requests already has {len(existing)} entries")
        return

    # Load dataset
    dataset_cfg = config.get("dataset", {})
    qa_path = dataset_cfg.get("qa_path")
    field_mappings = dataset_cfg.get("fields", {})
    eval_samples = dataset_cfg.get("eval_samples")

    print(f"\nLoading dataset...")
    base_entries = load_qa_dataset(qa_path, field_mappings)
    if eval_samples:
        base_entries = base_entries[:eval_samples]
    print(f"  Loaded {len(base_entries)} entries")

    # Find the method config
    all_methods = expand_methods(config)
    method_cfg = None
    for m in all_methods:
        if m.get("name") == method_name:
            method_cfg = m
            break

    if not method_cfg:
        print(f"Error: Method {method_name} not found in config")
        return

    retrieval_cfg = method_cfg.get("retrieval")
    highlighter_model = method_cfg.get("highlighter_model")
    is_rfc = method_cfg.get("reason_from_context", False)

    # Use appropriate dataset for reason_from_context methods
    if is_rfc:
        from .pipeline_stepwise import load_reason_from_context_dataset
        print("  Loading reason_from_context dataset...")
        source_entries = load_reason_from_context_dataset(config, field_mappings)
    else:
        source_entries = base_entries

    # Create entries
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
        for e in source_entries
    ]

    # Run retrieval
    retriever = None
    if retrieval_cfg:
        ret_type = retrieval_cfg.get("type", "bm25")

        if ret_type == "ground_truth":
            print("  Using ground truth context...")
            for entry in method_entries:
                if entry.gt_context:
                    entry.retrieved_docs = [entry.gt_context]
                else:
                    entry.retrieved_docs = []
        elif ret_type == "ground_truth_noise":
            print("  Using ground truth + BM25 noise...")
            retriever = create_retriever(config.get("retrieval", {}))
            noise_k = retrieval_cfg.get("noise_k", 5)
            from .query_formatter import format_and_validate_query
            queries = [format_and_validate_query(e) for e in method_entries]
            query_ids = [e.id for e in method_entries]
            results = retriever.batch_retrieve(queries=queries, query_ids=query_ids, top_k=noise_k)

            for entry in method_entries:
                bm25_results = results.get(entry.id, [])
                bm25_docs = [d.content for d in bm25_results]
                gt_doc = entry.gt_context

                if not gt_doc:
                    entry.retrieved_docs = bm25_docs[:noise_k]
                else:
                    gt_in_bm25 = any(gt_doc in doc or doc in gt_doc for doc in bm25_docs[:noise_k])
                    if gt_in_bm25:
                        entry.retrieved_docs = bm25_docs[:noise_k]
                    else:
                        entry.retrieved_docs = [gt_doc] + bm25_docs[:noise_k-1]
        elif ret_type == "ground_truth_random_noise":
            import random
            noise_k = retrieval_cfg.get("noise_k", 5)
            print(f"  Using ground truth + {noise_k-1} random noise...")
            all_gt_contexts = [(e.id, e.gt_context) for e in method_entries if e.gt_context]

            for entry in method_entries:
                gt_doc = entry.gt_context
                if not gt_doc:
                    entry.retrieved_docs = []
                else:
                    noise_pool = [(eid, ctx) for eid, ctx in all_gt_contexts if eid != entry.id]
                    if len(noise_pool) >= noise_k - 1:
                        random_samples = random.sample(noise_pool, noise_k - 1)
                    else:
                        random_samples = noise_pool
                    docs = [gt_doc] + [ctx for _, ctx in random_samples]
                    random.shuffle(docs)
                    entry.retrieved_docs = docs
        else:
            print(f"  Running BM25 retrieval...")
            retriever = create_retriever(config.get("retrieval", {}))
            top_k = retrieval_cfg.get("top_k", 10)
            from .query_formatter import format_and_validate_query
            queries = [format_and_validate_query(e) for e in method_entries]
            query_ids = [e.id for e in method_entries]
            results = retriever.batch_retrieve(queries=queries, query_ids=query_ids, top_k=top_k)

            for entry in method_entries:
                docs = results.get(entry.id, [])
                entry.retrieved_docs = [d.content for d in docs]

    # Run highlighter if needed
    if highlighter_model:
        print(f"  Running highlighter: {highlighter_model.split('/')[-1]}...")
        highlighter_manager = create_highlighter_manager(config)
        highlighter_cfg = config.get("highlighter", {})
        threshold = highlighter_cfg.get("threshold", 0.5)

        for entry in method_entries:
            if entry.retrieved_docs:
                from .query_formatter import format_and_validate_query
                query = format_and_validate_query(entry)
                pruned = highlighter_manager.prune_documents(
                    model_id=highlighter_model,
                    query=query,
                    documents=entry.retrieved_docs,
                    threshold=threshold,
                )
                entry.highlighted_docs = pruned
            else:
                entry.highlighted_docs = []

        highlighter_manager.flush_cache()
        highlighter_manager.unload()

    # Build contexts
    prompts_cfg = config.get("prompts", {})
    context_format = prompts_cfg.get("context_format", "Retrieved Document {rank}: {content}")
    context_separator = prompts_cfg.get("context_separator", "\n\n")

    def format_context(docs):
        if not docs:
            return ""
        parts = []
        for rank, content in enumerate(docs, start=1):
            if content.strip():
                parts.append(context_format.format(rank=rank, content=content))
        return context_separator.join(parts)

    for entry in method_entries:
        docs = entry.highlighted_docs if entry.highlighted_docs is not None else entry.retrieved_docs
        entry.context = format_context(docs or [])

    # Save to batch_requests
    print(f"\n  Saving {len(method_entries)} entries to batch_requests...")
    conn = llm_cache._get_conn()

    for entry in method_entries:
        custom_id = f"{method_name}_{entry.id}"
        context_hash = _hash_text(entry.context)
        conn.execute(
            """INSERT OR REPLACE INTO batch_requests
               (batch_id, custom_id, method_name, question_id, context, context_hash)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (batch_id, custom_id, method_name, entry.id, entry.context, context_hash)
        )
    conn.commit()

    print(f"  Done! Now run: python -m evaluation.run_eval --batch-retrieve {batch_id}")


def run_delete_batch_history(config: Dict[str, Any]):
    """Delete all batch jobs and requests history."""
    from .highlighter import LLMCache

    print("\n" + "=" * 70)
    print("DELETE BATCH HISTORY")
    print("=" * 70)

    llm_cfg = config.get("llm", {})
    llm_cache_path = llm_cfg.get("cache_path", "./evaluation/cache/llm_cache.db")
    llm_cache = LLMCache(llm_cache_path)

    conn = llm_cache._get_conn()

    # Count before delete
    cursor = conn.execute("SELECT COUNT(*) FROM batch_jobs")
    jobs_count = cursor.fetchone()[0]
    cursor = conn.execute("SELECT COUNT(*) FROM batch_requests")
    requests_count = cursor.fetchone()[0]

    print(f"  Found {jobs_count} batch jobs and {requests_count} batch requests")

    if jobs_count == 0 and requests_count == 0:
        print("  Nothing to delete.")
        return

    # Delete
    conn.execute("DELETE FROM batch_requests")
    conn.execute("DELETE FROM batch_jobs")
    conn.commit()

    print(f"  Deleted {jobs_count} batch jobs and {requests_count} batch requests")
    print("\n  Note: LLM cache entries are preserved. Only batch tracking data was deleted.")
    print("  You can now submit fresh batch jobs with --batch-submit")


def run_batch_retrieve(config: Dict[str, Any], batch_id: str, test_mode: bool = False):
    """Retrieve batch results and import to cache."""
    from .highlighter import LLMCache
    from .llm_client import QwenBatchClient

    print("\n" + "=" * 70)
    print("BATCH RETRIEVE" + (" [TEST MODE]" if test_mode else ""))
    print("=" * 70)

    # Initialize cache
    llm_cfg = config.get("llm", {})
    llm_cache_path = llm_cfg.get("cache_path", "./evaluation/cache/llm_cache.db")
    llm_cache = LLMCache(llm_cache_path)

    # Initialize batch client
    batch_cfg = llm_cfg.get("batch", {})
    api_key = batch_cfg.get("api_key") or os.environ.get("DASHSCOPE_API_KEY")
    try:
        batch_client = QwenBatchClient(
            api_key=api_key,
            model_name=batch_cfg.get("model_name", "qwen-turbo"),
            test_mode=test_mode,
        )
    except Exception as e:
        print(f"Error: Could not initialize batch client: {e}")
        return

    # Get jobs to retrieve
    if batch_id.lower() == "all":
        # Get all completed jobs
        all_jobs = llm_cache.get_all_batch_jobs()
        jobs_to_retrieve = []

        for job in all_jobs:
            if job["status"] in ("pending", "running"):
                # Check if actually completed
                try:
                    api_status = batch_client.check_batch_status(job["batch_id"])
                    if api_status.status == "completed":
                        jobs_to_retrieve.append(job)
                        llm_cache.update_batch_status(
                            job["batch_id"],
                            api_status.status,
                            api_status.completed_requests,
                        )
                except Exception:
                    pass
            elif job["status"] == "completed":
                # Check if already imported (has requests in batch_requests)
                requests = llm_cache.get_batch_requests(job["batch_id"])
                if requests:
                    jobs_to_retrieve.append(job)

        if not jobs_to_retrieve:
            print("\nNo completed batch jobs ready for retrieval.")
            return

        print(f"\nFound {len(jobs_to_retrieve)} completed jobs to retrieve")

    else:
        # Get specific job
        job = llm_cache.get_batch_job(batch_id)
        if not job:
            print(f"Error: Batch job {batch_id} not found in cache")
            return

        # Check status from API
        try:
            api_status = batch_client.check_batch_status(batch_id)
            if api_status.status != "completed":
                print(f"Batch job not ready. Status: {api_status.status}")
                print(f"Progress: {api_status.completed_requests}/{api_status.total_requests}")
                return
        except Exception as e:
            print(f"Error checking batch status: {e}")
            return

        jobs_to_retrieve = [job]

    # Retrieve results for each job
    total_imported = 0

    for job in jobs_to_retrieve:
        job_batch_id = job["batch_id"]
        method_name = job["method_name"]

        print(f"\nRetrieving: {job_batch_id}")
        print(f"  Method: {method_name}")

        try:
            # Download results
            raw_results = batch_client.retrieve_batch_results(job_batch_id)
            print(f"  Downloaded {len(raw_results)} results")

            # Parse results
            parsed = batch_client.parse_batch_results(raw_results)

            # Convert to format for import
            results_for_import = {}
            for custom_id, response in parsed.items():
                results_for_import[custom_id] = (
                    response.text,
                    response.input_tokens,
                    response.output_tokens,
                )

            # Import to cache
            imported = llm_cache.import_batch_results(job_batch_id, results_for_import)
            print(f"  Imported {imported} results to cache")
            total_imported += imported

            # Clean up batch job data (optional - keep for audit)
            # llm_cache.delete_batch_job(job_batch_id)

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    print(f"\n{'='*70}")
    print(f"Total imported: {total_imported} results")
    print("='*70")
    print("\nYou can now run evaluation with cached results:")
    print("  python -m evaluation.run_eval")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluation Pipeline for QA with Retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Main arguments
    parser.add_argument(
        "--config",
        type=str,
        default="evaluation/config.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
        help="Specific methods to run (default: all)",
    )

    # Run modes
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration and what would be done without running",
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Run on a small number of samples for testing",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=5,
        help="Number of samples for test run (default: 5)",
    )
    parser.add_argument(
        "--no-stepwise",
        action="store_true",
        help="Use original method-by-method execution instead of step-wise (default: step-wise)",
    )
    parser.add_argument(
        "--highlighter-only",
        action="store_true",
        help="Run only BM25 retrieval and highlighting to build cache (no LLM/judge needed)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download highlighter models from HuggingFace (ignores cached model code)",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Enable caching in test run (by default, test runs disable cache)",
    )

    # Batch API modes
    parser.add_argument(
        "--batch-submit",
        action="store_true",
        help="Submit batch job to Qwen API (exits after submission)",
    )
    parser.add_argument(
        "--batch-status",
        action="store_true",
        help="Show status of all batch jobs",
    )
    parser.add_argument(
        "--batch-retrieve",
        type=str,
        metavar="BATCH_ID",
        help="Retrieve batch results and import to cache (use 'all' for all completed)",
    )
    parser.add_argument(
        "--batch-test",
        action="store_true",
        help="Use batch-test-model for testing batch workflow (max 1MB file, 100 rows)",
    )
    parser.add_argument(
        "--batch-repopulate",
        type=str,
        metavar="BATCH_ID",
        help="Repopulate batch_requests for a batch job (useful after accidental deletion)",
    )
    parser.add_argument(
        "--delete-batch-history",
        action="store_true",
        help="Delete all batch jobs and requests history (allows fresh batch submissions)",
    )

    # Evaluation modes
    parser.add_argument(
        "--judge",
        type=str,
        metavar="PREDICTIONS_FILE",
        help="Run LLM judge on existing predictions file",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        metavar="PREDICTIONS_FILE",
        help="Compute metrics only (no LLM) on existing predictions",
    )

    # Config overrides
    parser.add_argument(
        "--llm-mode",
        type=str,
        choices=["local", "server", "qwen"],
        help="Override LLM mode (qwen uses DashScope real-time API)",
    )
    parser.add_argument(
        "--judge-mode",
        type=str,
        choices=["local", "server", "qwen"],
        help="Override judge LLM mode (qwen uses DashScope real-time API)",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        help="Override vLLM server URL",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Override model name",
    )
    parser.add_argument(
        "--index-path",
        type=str,
        help="Override retrieval index path",
    )
    parser.add_argument(
        "--qa-path",
        type=str,
        help="Override QA dataset path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory",
    )
    parser.add_argument(
        "--judge-output",
        type=str,
        help="Output path for judge results",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Send Telegram notification when complete",
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(str(config_path))
    else:
        print(f"Warning: Config file not found at {config_path}, using defaults")
        config = {}

    # Apply overrides
    config = merge_config_overrides(config, args)

    # Load notification module if needed
    notify_func = None
    notify_error_func = None
    notify_step_func = None
    if args.notify:
        try:
            from .notify import notify_eval_summary, notify_error, notify_step, load_env
            load_env()
            notify_func = notify_eval_summary
            notify_error_func = notify_error
            notify_step_func = notify_step
            print("[Notifications enabled]")
        except Exception as e:
            print(f"Warning: Could not enable notifications: {e}")

    # Dispatch to appropriate mode
    if args.dry_run:
        run_dry_run(config, args.methods, highlighter_only=args.highlighter_only)
    elif args.batch_submit:
        run_batch_submit(config, args.methods, test_mode=args.batch_test)
    elif args.batch_status:
        run_batch_status(config, test_mode=args.batch_test)
    elif args.batch_retrieve:
        run_batch_retrieve(config, args.batch_retrieve, test_mode=args.batch_test)
    elif args.batch_repopulate:
        run_batch_repopulate(config, args.batch_repopulate)
    elif args.delete_batch_history:
        run_delete_batch_history(config)
    elif args.test_run:
        run_test_run(config, args.methods, args.test_samples, notify=notify_func, notify_error=notify_error_func, force_download=args.force_download, use_cache=args.use_cache)
    elif args.metrics:
        run_metrics_only(args.metrics)
    elif args.judge:
        run_judge(args.judge, config, args.judge_output)
    else:
        stepwise = not args.no_stepwise
        highlighter_only = args.highlighter_only

        if highlighter_only and not stepwise:
            print("Error: --highlighter-only requires stepwise execution (don't use --no-stepwise)")
            return

        run_pipeline(
            config,
            args.methods,
            notify=notify_func,
            notify_error=notify_error_func,
            notify_step=notify_step_func,
            stepwise=stepwise,
            highlighter_only=highlighter_only,
            force_download=args.force_download,
        )


if __name__ == "__main__":
    main()
