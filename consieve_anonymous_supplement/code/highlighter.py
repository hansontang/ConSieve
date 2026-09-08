"""
Highlighter integration for evaluation pipeline.

Provides:
- HighlighterCache: Persistent cache for (model, query, doc) -> pruned text
- HighlighterManager: Load/manage multiple highlighter models with caching
"""

import hashlib
import json
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


def _hash_text(text: str) -> str:
    """Create a short hash of text for cache keys."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:32]


class HighlighterCache:
    """
    SQLite-based cache for highlighter results.

    Cache key: (model_id, query_hash, doc_hash) -> pruned_text

    This is persistent across runs and handles concurrent access.
    Supports buffered writes to reduce I/O overhead.
    """

    def __init__(
        self,
        cache_path: str = "./evaluation/cache/highlighter_cache.db",
        write_interval: int = 100,
    ):
        """
        Args:
            cache_path: Path to SQLite database file
            write_interval: Number of entries to buffer before committing to DB.
                           Set to 1 for immediate writes, higher for better performance.
        """
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self.write_interval = write_interval
        self._pending_writes: List[tuple] = []
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.cache_path), timeout=30.0)
            self._local.conn.execute("PRAGMA journal_mode=WAL")
        return self._local.conn

    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS highlighter_cache (
                model_id TEXT NOT NULL,
                query_hash TEXT NOT NULL,
                doc_hash TEXT NOT NULL,
                threshold REAL NOT NULL,
                pruned_text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (model_id, query_hash, doc_hash, threshold)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_query
            ON highlighter_cache(model_id, query_hash)
        """)
        conn.commit()

    def get(
        self,
        model_id: str,
        query: str,
        document: str,
        threshold: float,
    ) -> Optional[str]:
        """Get cached result if exists."""
        query_hash = _hash_text(query)
        doc_hash = _hash_text(document)

        conn = self._get_conn()
        cursor = conn.execute(
            """SELECT pruned_text FROM highlighter_cache
               WHERE model_id = ? AND query_hash = ? AND doc_hash = ? AND threshold = ?""",
            (model_id, query_hash, doc_hash, threshold)
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def get_batch(
        self,
        model_id: str,
        query: str,
        documents: List[str],
        threshold: float,
        sql_batch_size: int = 500,
    ) -> Dict[int, str]:
        """Get cached results for multiple documents. Returns {idx: pruned_text}."""
        query_hash = _hash_text(query)
        doc_hashes = [_hash_text(doc) for doc in documents]

        conn = self._get_conn()
        hash_to_text = {}

        # Process in batches to avoid SQLite variable limit
        for i in range(0, len(doc_hashes), sql_batch_size):
            batch_hashes = doc_hashes[i:i + sql_batch_size]
            placeholders = ",".join("?" * len(batch_hashes))
            cursor = conn.execute(
                f"""SELECT doc_hash, pruned_text FROM highlighter_cache
                   WHERE model_id = ? AND query_hash = ? AND threshold = ?
                   AND doc_hash IN ({placeholders})""",
                (model_id, query_hash, threshold, *batch_hashes)
            )
            hash_to_text.update({row[0]: row[1] for row in cursor.fetchall()})

        # Map back to indices
        results = {}
        for idx, doc_hash in enumerate(doc_hashes):
            if doc_hash in hash_to_text:
                results[idx] = hash_to_text[doc_hash]

        return results

    def put(
        self,
        model_id: str,
        query: str,
        document: str,
        threshold: float,
        pruned_text: str,
    ):
        """Store a result in cache with validation."""
        # Validate: pruned text should not be longer than original
        if pruned_text is not None and len(pruned_text) > len(document) + 10:
            logger.warning(f"Highlighter cache put rejected: pruned ({len(pruned_text)}) > doc ({len(document)})")
            return

        query_hash = _hash_text(query)
        doc_hash = _hash_text(document)

        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO highlighter_cache
               (model_id, query_hash, doc_hash, threshold, pruned_text)
               VALUES (?, ?, ?, ?, ?)""",
            (model_id, query_hash, doc_hash, threshold, pruned_text or "")
        )
        conn.commit()

    def put_batch(
        self,
        model_id: str,
        query: str,
        documents: List[str],
        threshold: float,
        pruned_texts: List[str],
    ):
        """Store multiple results in cache with validation (buffered)."""
        if len(documents) != len(pruned_texts):
            logger.warning(f"Highlighter cache put_batch rejected: mismatched lengths")
            return

        query_hash = _hash_text(query)

        # Filter out invalid entries
        rejected = 0
        for doc, pruned in zip(documents, pruned_texts):
            # Validate: pruned text should not be longer than original
            if pruned is not None and len(pruned) > len(doc) + 10:
                rejected += 1
                continue
            self._pending_writes.append(
                (model_id, query_hash, _hash_text(doc), threshold, pruned or "")
            )

        if rejected > 0:
            logger.warning(f"Highlighter cache put_batch: rejected {rejected} invalid entries")

        # Flush if buffer exceeds threshold
        if len(self._pending_writes) >= self.write_interval:
            self.flush()

    def flush(self):
        """Flush pending writes to database."""
        if not self._pending_writes:
            return

        conn = self._get_conn()
        conn.executemany(
            """INSERT OR REPLACE INTO highlighter_cache
               (model_id, query_hash, doc_hash, threshold, pruned_text)
               VALUES (?, ?, ?, ?, ?)""",
            self._pending_writes
        )
        conn.commit()
        flushed_count = len(self._pending_writes)
        self._pending_writes = []
        logger.debug(f"Highlighter cache flushed {flushed_count} entries")

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT COUNT(*) FROM highlighter_cache")
        total = cursor.fetchone()[0]

        cursor = conn.execute(
            "SELECT model_id, COUNT(*) FROM highlighter_cache GROUP BY model_id"
        )
        by_model = {row[0]: row[1] for row in cursor.fetchall()}

        return {"total_entries": total, "by_model": by_model}

    def clear(self, model_id: Optional[str] = None):
        """Clear cache, optionally only for a specific model."""
        conn = self._get_conn()
        if model_id:
            conn.execute("DELETE FROM highlighter_cache WHERE model_id = ?", (model_id,))
        else:
            conn.execute("DELETE FROM highlighter_cache")
        conn.commit()


def _hash_settings(model_name: str, sampling_params: Dict[str, Any]) -> str:
    """Create a hash of model name and sampling settings for cache key."""
    # Include only relevant sampling params that affect output
    relevant_params = {
        "model": model_name or "",
        "temperature": sampling_params.get("temperature", 0.0),
        "max_tokens": sampling_params.get("max_tokens", 512),
        "top_p": sampling_params.get("top_p", 1.0),
    }
    settings_str = json.dumps(relevant_params, sort_keys=True)
    return hashlib.sha256(settings_str.encode('utf-8')).hexdigest()[:16]


class LLMCache:
    """
    SQLite-based cache for LLM generation results.

    Cache key: (method_name, question_id, context_hash, settings_hash) -> (answer, input_tokens, output_tokens)

    The settings_hash includes model_name and sampling parameters (temperature, max_tokens, top_p).
    Different models or settings will have different cache entries.

    This allows resuming interrupted evaluation runs by caching LLM responses.
    """

    def __init__(self, cache_path: str = "./evaluation/cache/llm_cache.db"):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.cache_path), timeout=30.0)
            self._local.conn.execute("PRAGMA journal_mode=WAL")
        return self._local.conn

    @staticmethod
    def _is_error_answer(answer: str) -> bool:
        """Provider failures are retryable events, never cached predictions."""
        return str(answer).lstrip().startswith("[ERROR:")

    def _init_db(self):
        conn = self._get_conn()
        # Main cache table - now includes settings_hash and model_name
        conn.execute("""
            CREATE TABLE IF NOT EXISTS llm_cache (
                method_name TEXT NOT NULL,
                question_id TEXT NOT NULL,
                context_hash TEXT NOT NULL,
                settings_hash TEXT NOT NULL DEFAULT '',
                model_name TEXT NOT NULL DEFAULT '',
                answer TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (method_name, question_id, context_hash, settings_hash)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_method_question
            ON llm_cache(method_name, question_id)
        """)
        # Add new columns if they don't exist (for migration from old schema)
        try:
            conn.execute("ALTER TABLE llm_cache ADD COLUMN settings_hash TEXT NOT NULL DEFAULT ''")
        except sqlite3.OperationalError:
            pass  # Column already exists
        try:
            conn.execute("ALTER TABLE llm_cache ADD COLUMN model_name TEXT NOT NULL DEFAULT ''")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Batch job tracking table - includes model_name and settings_hash for cache key
        conn.execute("""
            CREATE TABLE IF NOT EXISTS batch_jobs (
                batch_id TEXT PRIMARY KEY,
                method_name TEXT NOT NULL,
                status TEXT NOT NULL,
                total_requests INTEGER NOT NULL,
                completed_requests INTEGER DEFAULT 0,
                model_name TEXT NOT NULL DEFAULT '',
                settings_hash TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                error_message TEXT
            )
        """)
        # Migration: add model_name and settings_hash columns if missing
        try:
            conn.execute("ALTER TABLE batch_jobs ADD COLUMN model_name TEXT NOT NULL DEFAULT ''")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE batch_jobs ADD COLUMN settings_hash TEXT NOT NULL DEFAULT ''")
        except sqlite3.OperationalError:
            pass

        # Pending batch requests table (maps custom_id back to entry info)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS batch_requests (
                batch_id TEXT NOT NULL,
                custom_id TEXT NOT NULL,
                method_name TEXT NOT NULL,
                question_id TEXT NOT NULL,
                context TEXT NOT NULL,
                context_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (batch_id, custom_id),
                FOREIGN KEY (batch_id) REFERENCES batch_jobs(batch_id)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_batch_requests_batch
            ON batch_requests(batch_id)
        """)
        conn.commit()

    def get(
        self,
        method_name: str,
        question_id: str,
        context: str,
        model_name: str = "",
        sampling_params: Dict[str, Any] = None,
    ) -> Optional[Tuple[str, int, int]]:
        """Get cached LLM result if exists.

        Args:
            method_name: Method name (e.g., "bm25_top5_highlighter")
            question_id: Unique question identifier
            context: Context text (will be hashed)
            model_name: LLM model name (e.g., "qwen-turbo")
            sampling_params: Sampling parameters dict (temperature, max_tokens, etc.)

        Returns:
            Tuple of (answer, input_tokens, output_tokens) or None if not cached.
        """
        context_hash = _hash_text(context)
        settings_hash = _hash_settings(model_name, sampling_params or {})

        conn = self._get_conn()
        cursor = conn.execute(
            """SELECT answer, input_tokens, output_tokens FROM llm_cache
               WHERE method_name = ? AND question_id = ? AND context_hash = ? AND settings_hash = ?""",
            (method_name, question_id, context_hash, settings_hash)
        )
        row = cursor.fetchone()
        if row and not self._is_error_answer(row[0]):
            return row[0], row[1], row[2]
        return None

    def get_batch(
        self,
        method_name: str,
        entries: List[Tuple[str, str]],  # List of (question_id, context)
        model_name: str = "",
        sampling_params: Dict[str, Any] = None,
    ) -> Dict[str, Tuple[str, int, int]]:
        """Get cached results for multiple entries.

        Args:
            method_name: Method name
            entries: List of (question_id, context) tuples
            model_name: LLM model name (e.g., "qwen-turbo")
            sampling_params: Sampling parameters dict (temperature, max_tokens, etc.)

        Returns:
            Dict mapping question_id -> (answer, input_tokens, output_tokens)
        """
        if not entries:
            return {}

        context_hashes = {qid: _hash_text(ctx) for qid, ctx in entries}
        settings_hash = _hash_settings(model_name, sampling_params or {})

        conn = self._get_conn()
        # Build list of (method_name, question_id, context_hash) for WHERE clause
        params = [method_name, settings_hash]
        for qid, ctx in entries:
            params.extend([qid, context_hashes[qid]])

        # Construct query with multiple OR conditions
        conditions = " OR ".join(
            "(question_id = ? AND context_hash = ?)" for _ in entries
        )
        cursor = conn.execute(
            f"""SELECT question_id, context_hash, answer, input_tokens, output_tokens
               FROM llm_cache
               WHERE method_name = ? AND settings_hash = ? AND ({conditions})""",
            params
        )

        results = {}
        for row in cursor.fetchall():
            qid, ctx_hash, answer, in_tok, out_tok = row
            # Verify context hash matches (in case of hash collision)
            expected_hash = context_hashes.get(qid)
            if expected_hash == ctx_hash and not self._is_error_answer(answer):
                results[qid] = (answer, in_tok, out_tok)

        return results

    def put(
        self,
        method_name: str,
        question_id: str,
        context: str,
        answer: str,
        input_tokens: int,
        output_tokens: int,
        model_name: str = "",
        sampling_params: Dict[str, Any] = None,
    ):
        """Store an LLM result in cache.

        Args:
            method_name: Method name (e.g., "bm25_top5_highlighter")
            question_id: Unique question identifier
            context: Context text (will be hashed)
            answer: Generated answer
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model_name: LLM model name (e.g., "qwen-turbo")
            sampling_params: Sampling parameters dict (temperature, max_tokens, etc.)
        """
        if self._is_error_answer(answer):
            return
        context_hash = _hash_text(context)
        settings_hash = _hash_settings(model_name, sampling_params or {})

        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO llm_cache
               (method_name, question_id, context_hash, settings_hash, model_name, answer, input_tokens, output_tokens)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (method_name, question_id, context_hash, settings_hash, model_name, answer, input_tokens, output_tokens)
        )
        conn.commit()

    def put_batch(
        self,
        method_name: str,
        entries: List[Tuple[str, str, str, int, int]],  # (question_id, context, answer, input_tokens, output_tokens)
        model_name: str = "",
        sampling_params: Dict[str, Any] = None,
    ):
        """Store multiple LLM results in cache.

        Args:
            method_name: Method name
            entries: List of (question_id, context, answer, input_tokens, output_tokens) tuples
            model_name: LLM model name (e.g., "qwen-turbo")
            sampling_params: Sampling parameters dict (temperature, max_tokens, etc.)
        """
        if not entries:
            return

        settings_hash = _hash_settings(model_name, sampling_params or {})

        data = []
        for qid, context, answer, in_tok, out_tok in entries:
            if self._is_error_answer(answer):
                continue
            context_hash = _hash_text(context)
            data.append((method_name, qid, context_hash, settings_hash, model_name, answer, in_tok, out_tok))

        if not data:
            return

        conn = self._get_conn()
        conn.executemany(
            """INSERT OR REPLACE INTO llm_cache
               (method_name, question_id, context_hash, settings_hash, model_name, answer, input_tokens, output_tokens)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            data
        )
        conn.commit()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT COUNT(*) FROM llm_cache")
        total = cursor.fetchone()[0]

        cursor = conn.execute(
            "SELECT method_name, COUNT(*) FROM llm_cache GROUP BY method_name"
        )
        by_method = {row[0]: row[1] for row in cursor.fetchall()}

        return {"total_entries": total, "by_method": by_method}

    def clear(self, method_name: Optional[str] = None):
        """Clear cache, optionally only for a specific method."""
        conn = self._get_conn()
        if method_name:
            conn.execute("DELETE FROM llm_cache WHERE method_name = ?", (method_name,))
        else:
            conn.execute("DELETE FROM llm_cache")
        conn.commit()

    # =========================================================================
    # Batch Job Management
    # =========================================================================

    def save_batch_job(
        self,
        batch_id: str,
        method_name: str,
        entries: List[Tuple[str, str, str]],  # (custom_id, question_id, context)
        status: str = "pending",
        model_name: str = "",
        sampling_params: Dict[str, Any] = None,
    ):
        """
        Save a batch job and its pending requests.

        Args:
            batch_id: The batch job ID from the API
            method_name: Method name for this batch
            entries: List of (custom_id, question_id, context) tuples
            status: Initial status (default: "pending")
            model_name: LLM model name for cache key
            sampling_params: Sampling parameters for cache key
        """
        conn = self._get_conn()
        settings_hash = _hash_settings(model_name, sampling_params or {})

        # Insert batch job
        conn.execute(
            """INSERT OR REPLACE INTO batch_jobs
               (batch_id, method_name, status, total_requests, completed_requests, model_name, settings_hash)
               VALUES (?, ?, ?, ?, 0, ?, ?)""",
            (batch_id, method_name, status, len(entries), model_name, settings_hash)
        )

        # Insert batch requests
        data = []
        for custom_id, question_id, context in entries:
            context_hash = _hash_text(context)
            data.append((batch_id, custom_id, method_name, question_id, context, context_hash))

        conn.executemany(
            """INSERT OR REPLACE INTO batch_requests
               (batch_id, custom_id, method_name, question_id, context, context_hash)
               VALUES (?, ?, ?, ?, ?, ?)""",
            data
        )
        conn.commit()

    def update_batch_status(
        self,
        batch_id: str,
        status: str,
        completed_requests: int = None,
        error_message: str = None,
    ):
        """Update batch job status."""
        conn = self._get_conn()

        updates = ["status = ?", "updated_at = CURRENT_TIMESTAMP"]
        params = [status]

        if completed_requests is not None:
            updates.append("completed_requests = ?")
            params.append(completed_requests)

        if error_message is not None:
            updates.append("error_message = ?")
            params.append(error_message)

        params.append(batch_id)
        conn.execute(
            f"UPDATE batch_jobs SET {', '.join(updates)} WHERE batch_id = ?",
            params
        )
        conn.commit()

    def get_batch_job(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get batch job info."""
        conn = self._get_conn()
        cursor = conn.execute(
            """SELECT batch_id, method_name, status, total_requests, completed_requests,
                      created_at, updated_at, error_message, model_name, settings_hash
               FROM batch_jobs WHERE batch_id = ?""",
            (batch_id,)
        )
        row = cursor.fetchone()
        if row:
            return {
                "batch_id": row[0],
                "method_name": row[1],
                "status": row[2],
                "total_requests": row[3],
                "completed_requests": row[4],
                "created_at": row[5],
                "updated_at": row[6],
                "error_message": row[7],
                "model_name": row[8] if len(row) > 8 else "",
                "settings_hash": row[9] if len(row) > 9 else "",
            }
        return None

    def get_pending_batch_jobs(self) -> List[Dict[str, Any]]:
        """Get all pending/running batch jobs."""
        conn = self._get_conn()
        cursor = conn.execute(
            """SELECT batch_id, method_name, status, total_requests, completed_requests,
                      created_at, updated_at, error_message
               FROM batch_jobs
               WHERE status IN ('pending', 'running')
               ORDER BY created_at DESC"""
        )
        return [
            {
                "batch_id": row[0],
                "method_name": row[1],
                "status": row[2],
                "total_requests": row[3],
                "completed_requests": row[4],
                "created_at": row[5],
                "updated_at": row[6],
                "error_message": row[7],
            }
            for row in cursor.fetchall()
        ]

    def get_all_batch_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all batch jobs (most recent first)."""
        conn = self._get_conn()
        cursor = conn.execute(
            """SELECT batch_id, method_name, status, total_requests, completed_requests,
                      created_at, updated_at, error_message
               FROM batch_jobs
               ORDER BY created_at DESC
               LIMIT ?""",
            (limit,)
        )
        return [
            {
                "batch_id": row[0],
                "method_name": row[1],
                "status": row[2],
                "total_requests": row[3],
                "completed_requests": row[4],
                "created_at": row[5],
                "updated_at": row[6],
                "error_message": row[7],
            }
            for row in cursor.fetchall()
        ]

    def get_batch_requests(self, batch_id: str) -> List[Dict[str, Any]]:
        """Get all requests for a batch job."""
        conn = self._get_conn()
        cursor = conn.execute(
            """SELECT custom_id, method_name, question_id, context, context_hash
               FROM batch_requests WHERE batch_id = ?""",
            (batch_id,)
        )
        return [
            {
                "custom_id": row[0],
                "method_name": row[1],
                "question_id": row[2],
                "context": row[3],
                "context_hash": row[4],
            }
            for row in cursor.fetchall()
        ]

    def import_batch_results(
        self,
        batch_id: str,
        results: Dict[str, Tuple[str, int, int]],  # custom_id -> (answer, in_tok, out_tok)
    ) -> int:
        """
        Import batch results into the main cache.

        Uses the model_name and settings_hash stored with the batch job
        to ensure proper cache key consistency.

        Args:
            batch_id: The batch job ID
            results: Dict mapping custom_id to (answer, input_tokens, output_tokens)

        Returns:
            Number of results imported
        """
        # Get batch job to get model_name and settings_hash
        batch_job = self.get_batch_job(batch_id)
        if not batch_job:
            return 0

        model_name = batch_job.get("model_name", "")
        settings_hash = batch_job.get("settings_hash", "")

        # Get batch requests to map custom_id back to question_id/context
        requests = self.get_batch_requests(batch_id)
        if not requests:
            return 0

        # Build mapping
        custom_id_to_request = {r["custom_id"]: r for r in requests}

        # Import results
        imported = 0
        conn = self._get_conn()

        for custom_id, (answer, in_tok, out_tok) in results.items():
            request = custom_id_to_request.get(custom_id)
            if not request:
                continue

            conn.execute(
                """INSERT OR REPLACE INTO llm_cache
                   (method_name, question_id, context_hash, settings_hash, model_name, answer, input_tokens, output_tokens)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (request["method_name"], request["question_id"], request["context_hash"],
                 settings_hash, model_name, answer, in_tok, out_tok)
            )
            imported += 1

        conn.commit()

        # Update batch status
        self.update_batch_status(batch_id, "completed", completed_requests=imported)

        # Keep batch_requests for potential re-import (don't delete)
        # Status in batch_jobs tracks completion

        return imported

    def delete_batch_job(self, batch_id: str):
        """Delete a batch job and its requests."""
        conn = self._get_conn()
        conn.execute("DELETE FROM batch_requests WHERE batch_id = ?", (batch_id,))
        conn.execute("DELETE FROM batch_jobs WHERE batch_id = ?", (batch_id,))
        conn.commit()


class BM25Cache:
    """
    SQLite-based cache for BM25 retrieval results.

    Cache key: (index_path_hash, query_hash, top_k) -> (doc_ids, scores)
    """

    def __init__(self, cache_path: str = "./evaluation/cache/bm25_cache.db"):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _normalize_index_path(self, index_path: str) -> str:
        """Normalize index path to ensure consistent cache keys across runs."""
        # Resolve to absolute path and normalize
        return str(Path(index_path).resolve())

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.cache_path), timeout=30.0)
            self._local.conn.execute("PRAGMA journal_mode=WAL")
        return self._local.conn

    def _init_db(self):
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bm25_cache (
                index_hash TEXT NOT NULL,
                query_hash TEXT NOT NULL,
                top_k INTEGER NOT NULL,
                doc_ids TEXT NOT NULL,
                scores TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (index_hash, query_hash, top_k)
            )
        """)
        conn.commit()

    def get(
        self,
        index_path: str,
        query: str,
        top_k: int,
    ) -> Optional[Tuple[List[str], List[float]]]:
        """Get cached BM25 results with validation."""
        normalized_path = self._normalize_index_path(index_path)
        index_hash = _hash_text(normalized_path)
        query_hash = _hash_text(query)

        conn = self._get_conn()
        cursor = conn.execute(
            """SELECT doc_ids, scores FROM bm25_cache
               WHERE index_hash = ? AND query_hash = ? AND top_k = ?""",
            (index_hash, query_hash, top_k)
        )
        row = cursor.fetchone()
        if row:
            try:
                doc_ids = json.loads(row[0])
                scores = json.loads(row[1])

                # Validate cached results
                if len(doc_ids) != len(scores):
                    logger.warning(f"BM25 cache validation failed: doc_ids ({len(doc_ids)}) != scores ({len(scores)})")
                    return None
                if len(doc_ids) > top_k:
                    logger.warning(f"BM25 cache validation failed: results ({len(doc_ids)}) > top_k ({top_k})")
                    return None
                if not all(isinstance(s, (int, float)) for s in scores):
                    logger.warning("BM25 cache validation failed: invalid score types")
                    return None

                return doc_ids, scores
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"BM25 cache decode error: {e}")
                return None
        return None

    def get_batch(
        self,
        index_path: str,
        queries: List[str],
        top_k: int,
        sql_batch_size: int = 500,
    ) -> Dict[str, Tuple[List[str], List[float]]]:
        """
        Batch get cached BM25 results.

        Args:
            index_path: Path to the index
            queries: List of query strings
            top_k: Number of results per query
            sql_batch_size: Max queries per SQL query (SQLite limit is ~999)

        Returns:
            Dict mapping query -> (doc_ids, scores) for cached queries only
        """
        if not queries:
            return {}

        normalized_path = self._normalize_index_path(index_path)
        index_hash = _hash_text(normalized_path)

        # Build query hash -> original query mapping
        query_hash_map = {_hash_text(q): q for q in queries}
        query_hashes = list(query_hash_map.keys())

        conn = self._get_conn()
        results = {}

        # Process in batches to avoid SQLite variable limit
        for i in range(0, len(query_hashes), sql_batch_size):
            batch_hashes = query_hashes[i:i + sql_batch_size]
            placeholders = ",".join("?" * len(batch_hashes))
            cursor = conn.execute(
                f"""SELECT query_hash, doc_ids, scores FROM bm25_cache
                   WHERE index_hash = ? AND top_k = ? AND query_hash IN ({placeholders})""",
                (index_hash, top_k, *batch_hashes)
            )

            for row in cursor.fetchall():
                query_hash, doc_ids_json, scores_json = row
                try:
                    doc_ids = json.loads(doc_ids_json)
                    scores = json.loads(scores_json)

                    # Validate
                    if len(doc_ids) != len(scores) or len(doc_ids) > top_k:
                        continue
                    if not all(isinstance(s, (int, float)) for s in scores):
                        continue

                    # Map back to original query
                    original_query = query_hash_map.get(query_hash)
                    if original_query:
                        results[original_query] = (doc_ids, scores)
                except (json.JSONDecodeError, TypeError):
                    continue

        return results

    def put(
        self,
        index_path: str,
        query: str,
        top_k: int,
        doc_ids: List[str],
        scores: List[float],
    ):
        """Store BM25 results with validation."""
        # Validate before storing
        if len(doc_ids) != len(scores):
            logger.warning(f"BM25 cache put rejected: doc_ids ({len(doc_ids)}) != scores ({len(scores)})")
            return
        if len(doc_ids) > top_k:
            logger.warning(f"BM25 cache put rejected: results ({len(doc_ids)}) > top_k ({top_k})")
            return

        normalized_path = self._normalize_index_path(index_path)
        index_hash = _hash_text(normalized_path)
        query_hash = _hash_text(query)

        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO bm25_cache
               (index_hash, query_hash, top_k, doc_ids, scores)
               VALUES (?, ?, ?, ?, ?)""",
            (index_hash, query_hash, top_k, json.dumps(doc_ids), json.dumps(scores))
        )
        conn.commit()

    def clear(self):
        conn = self._get_conn()
        conn.execute("DELETE FROM bm25_cache")
        conn.commit()


class RerankerCache:
    """
    SQLite-based cache for reranker (cross-encoder) scores.

    Cache key: (model_hash, query_hash, doc_hash) -> score
    """

    def __init__(self, cache_path: str = "./evaluation/cache/reranker_cache.db"):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.cache_path), timeout=30.0)
            self._local.conn.execute("PRAGMA journal_mode=WAL")
        return self._local.conn

    def _init_db(self):
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reranker_cache (
                model_hash TEXT NOT NULL,
                query_hash TEXT NOT NULL,
                doc_hash TEXT NOT NULL,
                score REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (model_hash, query_hash, doc_hash)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_reranker_model_query
            ON reranker_cache(model_hash, query_hash)
        """)
        conn.commit()

    def get_batch(
        self,
        model_name: str,
        query: str,
        doc_contents: List[str],
        sql_batch_size: int = 500,
    ) -> Dict[int, float]:
        """
        Batch get cached reranker scores for a query and multiple documents.

        Args:
            model_name: Reranker model name
            query: Query string
            doc_contents: List of document contents
            sql_batch_size: Max docs per SQL query

        Returns:
            Dict mapping doc_index -> score for cached docs only
        """
        if not doc_contents:
            return {}

        model_hash = _hash_text(model_name)
        query_hash = _hash_text(query)
        doc_hashes = [_hash_text(doc) for doc in doc_contents]

        conn = self._get_conn()
        hash_to_score = {}

        # Process in batches to avoid SQLite variable limit
        for i in range(0, len(doc_hashes), sql_batch_size):
            batch_hashes = doc_hashes[i:i + sql_batch_size]
            placeholders = ",".join("?" * len(batch_hashes))
            cursor = conn.execute(
                f"""SELECT doc_hash, score FROM reranker_cache
                   WHERE model_hash = ? AND query_hash = ? AND doc_hash IN ({placeholders})""",
                (model_hash, query_hash, *batch_hashes)
            )
            hash_to_score.update({row[0]: row[1] for row in cursor.fetchall()})

        # Map back to indices
        results = {}
        for idx, doc_hash in enumerate(doc_hashes):
            if doc_hash in hash_to_score:
                results[idx] = hash_to_score[doc_hash]

        return results

    def put_batch(
        self,
        model_name: str,
        query: str,
        doc_contents: List[str],
        scores: List[float],
    ):
        """Store multiple reranker scores in cache."""
        if not doc_contents or len(doc_contents) != len(scores):
            return

        model_hash = _hash_text(model_name)
        query_hash = _hash_text(query)

        conn = self._get_conn()
        data = [
            (model_hash, query_hash, _hash_text(doc), float(score))
            for doc, score in zip(doc_contents, scores)
        ]
        conn.executemany(
            """INSERT OR REPLACE INTO reranker_cache
               (model_hash, query_hash, doc_hash, score)
               VALUES (?, ?, ?, ?)""",
            data
        )
        conn.commit()

    def clear(self):
        conn = self._get_conn()
        conn.execute("DELETE FROM reranker_cache")
        conn.commit()


@dataclass
class HighlighterConfig:
    """Configuration for a highlighter model."""
    model_id: str  # HuggingFace model ID or local path
    threshold: float = 0.5
    max_length: int = 4096
    batch_size: int = 8
    device: Optional[str] = None  # None for auto


class HighlighterManager:
    """
    Manages multiple highlighter models with caching.

    Features:
    - Lazy loading of models (only load when needed)
    - Model switching (unload old, load new)
    - Result caching via HighlighterCache
    - Batch processing
    - Half-precision inference for speed
    - Sentence split caching (same doc retrieved for multiple queries)
    """

    def __init__(
        self,
        cache: Optional[HighlighterCache] = None,
        device: Optional[str] = None,
        max_length: int = 4096,
        batch_size: int = 8,
        inference_batch_size: int = 1,
        use_fp16: bool = False,
        multi_gpu: bool = False,
        sentence_cache_size: int = 50000,
        num_workers: int = 24,
        debug: bool = False,
        force_download: bool = False,
        model_revisions: Optional[Dict[str, str]] = None,
    ):
        self.cache = cache  # None means no caching (test mode)
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size  # Documents per process() call
        self.inference_batch_size = inference_batch_size  # Fragments per GPU forward pass
        self.use_fp16 = use_fp16
        self.multi_gpu = multi_gpu
        self.num_workers = num_workers  # Number of CPU workers for sentence splitting
        self.debug = debug
        self.force_download = force_download  # Force re-download from HuggingFace
        self.model_revisions = model_revisions or {}

        self._current_model_id: Optional[str] = None
        self._current_pruner = None

        # LRU cache for sentence splits (same doc may be retrieved for many queries)
        # Key: hash of document content, Value: list of (text, start, end) tuples
        self._sentence_cache_size = sentence_cache_size
        self._sentence_split_cache: Dict[str, List[Tuple[str, int, int]]] = {}
        self._sentence_cache_order: List[str] = []  # For LRU eviction
        self._sentence_cache_hits = 0
        self._sentence_cache_misses = 0
        self._sentence_cache_lock = threading.Lock()  # Thread safety for multi-threaded access

        # Cumulative cache stats for progress reporting
        self._total_cache_hits = 0
        self._total_processed = 0

    def _load_model(self, model_id: str, device: Optional[str] = None):
        """Load a highlighter model, unloading any previous one."""
        if self._current_model_id == model_id and self._current_pruner is not None:
            return  # Already loaded

        # Unload current model
        if self._current_pruner is not None:
            logger.info(f"Unloading model: {self._current_model_id}")
            del self._current_pruner
            self._current_pruner = None
            self._current_model_id = None

            # Force garbage collection to free GPU memory
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        # Load new model
        print(f"  Loading highlighter model: {model_id}...", flush=True)
        logger.info(f"Loading highlighter model: {model_id}")

        # Import here to avoid circular imports and allow lazy loading
        import os
        import sys
        from pathlib import Path

        # Show HuggingFace cache location for debugging
        hf_home = os.environ.get("HF_HOME")
        hf_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
        default_cache = os.path.expanduser("~/.cache/huggingface")
        cache_location = hf_home or hf_cache or default_cache
        print(f"  [HighlighterManager] HF cache: {cache_location}", flush=True)
        print(f"  [HighlighterManager] Model ID: {model_id}", flush=True)

        # Clear modules cache if force_download is enabled
        # This MUST happen before any HuggingFace imports that might load cached code
        if self.force_download and "/" in model_id:
            import shutil
            org, model_name = model_id.split("/", 1)
            # HuggingFace converts hyphens to "_hyphen_" in module directory names
            model_name_hf = model_name.replace("-", "_hyphen_")
            modules_cache_paths = [
                os.path.join(cache_location, "modules", "transformers_modules", org, model_name_hf),
                os.path.join(os.path.expanduser("~/.cache/huggingface"), "modules", "transformers_modules", org, model_name_hf),
            ]
            for modules_cache in modules_cache_paths:
                if os.path.exists(modules_cache):
                    print(f"  [HighlighterManager] Clearing modules cache: {modules_cache}", flush=True)
                    try:
                        shutil.rmtree(modules_cache)
                        print(f"  [HighlighterManager] Cleared modules cache successfully", flush=True)
                    except Exception as e:
                        print(f"  [HighlighterManager] Warning: Could not clear cache: {e}", flush=True)

        # Add paths for import
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        if str(project_root / "open_provence") not in sys.path:
            sys.path.insert(0, str(project_root / "open_provence"))

        try:
            from src.inference import DocumentPruner
        except Exception as e:
            print(f"  [HighlighterManager] ERROR importing DocumentPruner: {e}", flush=True)
            raise

        # Determine if it's a HF model or local path
        try:
            model_revision = self.model_revisions.get(model_id)
            if "/" in model_id and not Path(model_id).exists():
                # HuggingFace model
                self._current_pruner = DocumentPruner(
                    model_path=model_id,  # Will be interpreted as HF model
                    device=device or self.device,
                    max_length=self.max_length,
                    use_fp16=self.use_fp16,
                    multi_gpu=self.multi_gpu,
                    batch_size=self.batch_size,
                    force_download=self.force_download,
                    model_revision=model_revision,
                )
            else:
                # Local path
                self._current_pruner = DocumentPruner(
                    model_path=model_id,
                    device=device or self.device,
                    max_length=self.max_length,
                    use_fp16=self.use_fp16,
                    multi_gpu=self.multi_gpu,
                    batch_size=self.batch_size,
                    force_download=self.force_download,
                    model_revision=model_revision,
                )
        except Exception as e:
            # Provide helpful error message for cached model issues
            cache_dir = f"{cache_location}/hub/models--{model_id.replace('/', '--')}"
            print(f"  [HighlighterManager] ERROR loading model: {e}", flush=True)
            print(f"  [HighlighterManager] This may be due to stale cached model code.", flush=True)
            print(f"  [HighlighterManager] Try clearing cache:", flush=True)
            print(f"    rm -rf {cache_dir}", flush=True)
            raise

        self._current_model_id = model_id
        print(f"  Model loaded: {model_id}", flush=True)
        logger.info(f"Model loaded: {model_id}")

    def _get_sentence_splits(self, doc: str) -> List[Tuple[str, int, int]]:
        """Get sentence splits for a document, using cache if available.

        Thread-safe for multi-threaded batch processing.
        """
        doc_hash = _hash_text(doc)

        # Check cache (with lock for thread safety)
        with self._sentence_cache_lock:
            if doc_hash in self._sentence_split_cache:
                self._sentence_cache_hits += 1
                # Move to end for LRU
                if doc_hash in self._sentence_cache_order:
                    self._sentence_cache_order.remove(doc_hash)
                self._sentence_cache_order.append(doc_hash)
                return self._sentence_split_cache[doc_hash]

        # Cache miss - split sentences (outside lock for better parallelism)
        sentence_splitter = self._current_pruner.sentence_splitter
        spans = sentence_splitter.split(doc)
        sentences = [(span.text, span.start, span.end) for span in spans]

        # Cache with LRU eviction (with lock)
        with self._sentence_cache_lock:
            # Double-check in case another thread added it
            if doc_hash not in self._sentence_split_cache:
                self._sentence_cache_misses += 1
                self._sentence_split_cache[doc_hash] = sentences
                self._sentence_cache_order.append(doc_hash)

                # Evict oldest if over capacity
                while len(self._sentence_cache_order) > self._sentence_cache_size:
                    oldest = self._sentence_cache_order.pop(0)
                    self._sentence_split_cache.pop(oldest, None)

        return sentences

    def get_sentence_cache_stats(self) -> Dict[str, Any]:
        """Get sentence cache statistics."""
        total = self._sentence_cache_hits + self._sentence_cache_misses
        hit_rate = self._sentence_cache_hits / total if total > 0 else 0
        return {
            "hits": self._sentence_cache_hits,
            "misses": self._sentence_cache_misses,
            "hit_rate": hit_rate,
            "cached_docs": len(self._sentence_split_cache),
        }

    def _prune_batch_efficient(
        self,
        query: str,
        documents: List[str],
        threshold: float,
        batch_size: int = 8,
        inference_batch_size: int = 1,
    ) -> List[str]:
        """
        Efficiently prune multiple documents using sentence-boundary-aware fragmentation.

        This method handles documents of any length without OOM by:
        1. Splitting into sentences
        2. Grouping sentences into fragments that fit within max_length
        3. Processing each fragment independently (no cross-fragment memory)
        4. Releasing memory between fragments

        Unlike truncation: preserves the ENTIRE document
        Unlike process(): doesn't hold all results in memory

        Args:
            query: Query string
            documents: List of documents to prune
            threshold: Pruning threshold
            batch_size: Unused (kept for API compatibility)
            inference_batch_size: Unused (kept for API compatibility)
        """
        if not documents:
            return []

        if self.debug:
            print(f"      [DEBUG] _prune_batch_efficient: {len(documents)} docs, threshold={threshold}", flush=True)

        # Use DocumentPruner's prune_long_batch which uses sentence-boundary-aware fragmentation
        # This gives full document coverage with memory efficiency
        pruner = self._current_pruner

        all_pruned = pruner.prune_long_batch(
            query=query,
            documents=documents,
            threshold=threshold,
            separator=" ",  # Join sentences with space
            fragment_batch_size=inference_batch_size,  # Control GPU utilization
        )

        if self.debug:
            print(f"      [DEBUG] Total pruned: {len(all_pruned)} results", flush=True)

        return all_pruned

    def prune_documents(
        self,
        model_id: str,
        query: str,
        documents: List[str],
        threshold: float = 0.5,
        use_cache: bool = True,
    ) -> List[str]:
        """
        Prune multiple documents using a highlighter model.

        Uses cache when available to avoid recomputation.

        Args:
            model_id: HuggingFace model ID or local path
            query: Query string
            documents: List of documents to prune
            threshold: Pruning threshold
            use_cache: Whether to use caching

        Returns:
            List of pruned documents
        """
        if not documents:
            return []

        import sys

        results = [None] * len(documents)
        docs_to_process = []
        indices_to_process = []

        # Check cache first
        cache_hits = 0
        cache_errors = 0
        if use_cache and self.cache is not None:
            if self.debug:
                print(f"    [DEBUG] Checking cache for {len(documents)} docs...", end="", flush=True)
            cached = self.cache.get_batch(model_id, query, documents, threshold)
            if self.debug:
                print(f" done ({len(cached)} cached)", flush=True)
            for idx, pruned in cached.items():
                # Validate cached result
                original_doc = documents[idx]
                if pruned is not None:
                    # Check: pruned should not be longer than original
                    if len(pruned) > len(original_doc) + 10:  # small tolerance for whitespace
                        logger.warning(f"Cache validation failed: pruned ({len(pruned)}) > original ({len(original_doc)})")
                        cache_errors += 1
                        continue  # Skip this cached result, recompute
                    # Check: if original is non-empty, pruned being empty is suspicious but OK (could be all filtered)
                    cache_hits += 1
                    results[idx] = pruned

            # Find documents not in cache or with invalid cache
            for idx, doc in enumerate(documents):
                if results[idx] is None:
                    docs_to_process.append(doc)
                    indices_to_process.append(idx)

            if cache_hits > 0 or cache_errors > 0:
                logger.debug(f"Cache: {cache_hits} hits, {cache_errors} errors, {len(docs_to_process)} to process")

            # Update cumulative stats
            self._total_cache_hits += cache_hits
            self._total_processed += len(documents)
        else:
            docs_to_process = documents
            indices_to_process = list(range(len(documents)))
            self._total_processed += len(documents)

        # Process uncached documents
        if docs_to_process:
            if self.debug:
                print(f"    [DEBUG] Processing {len(docs_to_process)} uncached docs", flush=True)
            # Load model if needed
            self._load_model(model_id)

            if self.debug:
                print(f"    [DEBUG] Starting batch inference...", flush=True)
            # Run efficient batched pruning
            pruned_texts = self._prune_batch_efficient(
                query=query,
                documents=docs_to_process,
                threshold=threshold,
                batch_size=self.batch_size,
                inference_batch_size=self.inference_batch_size,
            )

            # Validate and store results
            valid_for_cache = []
            valid_docs_for_cache = []
            for i, (idx, pruned) in enumerate(zip(indices_to_process, pruned_texts)):
                original_doc = docs_to_process[i]
                # Validate: pruned should not be longer than original
                if pruned is not None and len(pruned) > len(original_doc) + 10:
                    logger.warning(f"Pruning result invalid: pruned ({len(pruned)}) > original ({len(original_doc)})")
                    # Use original document as fallback
                    results[idx] = original_doc
                else:
                    results[idx] = pruned
                    # Only cache valid results
                    valid_for_cache.append(pruned)
                    valid_docs_for_cache.append(original_doc)

            if use_cache and self.cache is not None and valid_for_cache:
                self.cache.put_batch(
                    model_id, query, valid_docs_for_cache, threshold, valid_for_cache
                )

        return results

    def get_model_short_name(self, model_id: str) -> str:
        """Get a short display name for a model."""
        # Extract last part of HF model ID
        if "/" in model_id:
            return model_id.split("/")[-1]
        return Path(model_id).name

    def unload(self):
        """Unload current model to free memory."""
        if self._current_pruner is not None:
            del self._current_pruner
            self._current_pruner = None
            self._current_model_id = None

            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def flush_cache(self):
        """Flush any pending cache writes to disk.

        Should be called at the end of evaluation to ensure all results are saved.
        """
        if self.cache is not None:
            self.cache.flush()

    def get_cache_stats(self) -> tuple[int, int]:
        """Get cumulative cache statistics.

        Returns:
            Tuple of (cache_hits, total_processed)
        """
        return self._total_cache_hits, self._total_processed

    def reset_cache_stats(self):
        """Reset cumulative cache statistics."""
        self._total_cache_hits = 0
        self._total_processed = 0


def create_highlighter_manager(
    config: Dict[str, Any],
    skip_cache: bool = False,
    debug: bool = False,
    force_download: bool = False,
) -> HighlighterManager:
    """Create a HighlighterManager from config.

    Args:
        config: Configuration dictionary
        skip_cache: If True, disable result caching (useful for test runs)
        debug: If True, enable verbose debug output
        force_download: If True, force re-download models from HuggingFace (ignores cache)
    """
    highlighter_cfg = config.get("highlighter", {})
    cache_path = highlighter_cfg.get("cache_path", "./evaluation/cache/highlighter_cache.db")
    cache_write_interval = highlighter_cfg.get("cache_write_interval", 100)  # Flush every N entries
    device = highlighter_cfg.get("device")
    max_length = highlighter_cfg.get("max_length", 4096)
    batch_size = highlighter_cfg.get("batch_size", 8)
    inference_batch_size = highlighter_cfg.get("inference_batch_size", 1)  # Fragments per GPU forward pass
    use_fp16 = highlighter_cfg.get("use_fp16", True)  # Default True for bf16 (matches training)
    multi_gpu = highlighter_cfg.get("multi_gpu", False)  # Set to True for DataParallel
    sentence_cache_size = highlighter_cfg.get("sentence_cache_size", 50000)
    num_workers = highlighter_cfg.get("num_workers", 24)  # CPU workers for sentence splitting
    # Allow debug/force_download to be set from config or parameter (parameter takes precedence)
    debug = debug or highlighter_cfg.get("debug", False)
    force_download = force_download or highlighter_cfg.get("force_download", False)
    model_revisions = highlighter_cfg.get("model_revisions", {})

    cache = None if skip_cache else HighlighterCache(cache_path, write_interval=cache_write_interval)
    if skip_cache:
        print("  Highlighter cache disabled (test mode)")
    if force_download:
        print("  Force download enabled - will re-download models from HuggingFace")

    return HighlighterManager(
        cache=cache,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
        inference_batch_size=inference_batch_size,
        use_fp16=use_fp16,
        multi_gpu=multi_gpu,
        sentence_cache_size=sentence_cache_size,
        num_workers=num_workers,
        debug=debug,
        force_download=force_download,
        model_revisions=model_revisions,
    )


def create_bm25_cache(config: Dict[str, Any], skip_cache: bool = False) -> Optional[BM25Cache]:
    """Create a BM25Cache from config.

    Args:
        config: Configuration dictionary
        skip_cache: If True, return None to disable caching (useful for test runs)
    """
    if skip_cache:
        logger.info("BM25 cache disabled (test mode)")
        return None
    cache_path = config.get("retrieval", {}).get("bm25_cache_path", "./evaluation/cache/bm25_cache.db")
    return BM25Cache(cache_path)
