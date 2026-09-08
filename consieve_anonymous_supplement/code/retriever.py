"""
Retriever module: BM25 with optional cross-encoder reranking.
"""

import sys
import re
import json
import sqlite3
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

# Add retrieval module to path
RETRIEVAL_PATH = Path(__file__).parent.parent / "retrieval"
if str(RETRIEVAL_PATH) not in sys.path:
    sys.path.insert(0, str(RETRIEVAL_PATH))


@dataclass
class RetrievedDoc:
    """A retrieved document."""
    doc_id: str
    content: str
    score: float
    rank: int


@dataclass
class RetrievalResult:
    """Retrieval result for a single query."""
    query_id: str
    query: str
    documents: List[RetrievedDoc]


class BM25Retriever:
    """BM25 retriever using Pyserini with optional caching."""

    def __init__(
        self,
        index_path: str,
        k1: float = 0.9,
        b: float = 0.4,
        threads: int = 8,
        cache=None,
    ):
        try:
            from pyserini.search.lucene import LuceneSearcher
        except ImportError:
            raise ImportError("pyserini required. Install with: pip install pyserini")

        self.index_path = index_path
        self.searcher = LuceneSearcher(index_path)
        self.searcher.set_bm25(k1, b)
        self.threads = threads
        self.cache = cache  # BM25Cache instance

    @staticmethod
    def _document_contents(doc) -> str:
        """Read text from either a stored contents field or stored raw JSON.

        The canonical Lucene index was built with ``-storeRaw``.  In that
        layout Pyserini may return ``None`` from ``doc.contents()`` even though
        the raw JSON contains the canonical ``contents`` value.
        """
        if doc is None:
            return ""
        content = doc.contents()
        if isinstance(content, str):
            return content
        raw = doc.raw()
        if not raw:
            return ""
        try:
            payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return str(raw)
        for key in ("contents", "content", "text"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
        return ""

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[RetrievedDoc]:
        """Search for a single query."""
        hits = self.searcher.search(query, k=top_k)
        results = []
        for rank, hit in enumerate(hits, start=1):
            doc = self.searcher.doc(hit.docid)
            content = self._document_contents(doc)
            results.append(RetrievedDoc(
                doc_id=hit.docid,
                content=content,
                score=hit.score,
                rank=rank,
            ))
        return results


    def batch_search(
        self,
        queries: List[str],
        query_ids: List[str],
        top_k: int = 10,
        show_progress: bool = True,
    ) -> Dict[str, List[RetrievedDoc]]:
        """Batch search for multiple queries with caching."""
        results = {}
        queries_to_search = []
        qids_to_search = []

        # Check cache first (batch lookup)
        if self.cache:
            # Batch query the cache
            cached_results = self.cache.get_batch(self.index_path, queries, top_k)
            cache_hits = len(cached_results)

            # Reconstruct documents from cached results with progress bar
            if cached_results:
                cache_iterator = [
                    (q, qid) for q, qid in zip(queries, query_ids) if q in cached_results
                ]
                if show_progress:
                    cache_iterator = tqdm(
                        cache_iterator,
                        desc="  Loading from BM25 cache",
                        leave=True,
                    )

                for query, qid in cache_iterator:
                    doc_ids, scores = cached_results[query]
                    docs = []
                    for rank, (doc_id, score) in enumerate(zip(doc_ids, scores), start=1):
                        doc = self.searcher.doc(doc_id)
                        content = self._document_contents(doc)
                        docs.append(RetrievedDoc(
                            doc_id=doc_id,
                            content=content,
                            score=score,
                            rank=rank,
                        ))
                    results[qid] = docs

            # Collect uncached queries
            for query, qid in zip(queries, query_ids):
                if query not in cached_results:
                    queries_to_search.append(query)
                    qids_to_search.append(qid)

            if cache_hits > 0:
                print(f"  BM25 cache: {cache_hits} hits, {len(queries_to_search)} to search")
        else:
            queries_to_search = queries
            qids_to_search = query_ids

        # Run BM25 for uncached queries
        if queries_to_search:
            hits_dict = self.searcher.batch_search(
                queries_to_search, qids=qids_to_search, k=top_k, threads=self.threads
            )

            # Process results with progress bar
            iterator = zip(queries_to_search, qids_to_search)
            if show_progress:
                iterator = tqdm(
                    list(iterator),
                    desc="  Processing BM25 results",
                    leave=True,
                )

            for query, qid in iterator:
                hits = hits_dict.get(qid, [])
                docs = []
                doc_ids_for_cache = []
                scores_for_cache = []

                for rank, hit in enumerate(hits, start=1):
                    doc = self.searcher.doc(hit.docid)
                    content = self._document_contents(doc)
                    docs.append(RetrievedDoc(
                        doc_id=hit.docid,
                        content=content,
                        score=hit.score,
                        rank=rank,
                    ))
                    doc_ids_for_cache.append(hit.docid)
                    scores_for_cache.append(hit.score)

                results[qid] = docs

                # Store in cache
                if self.cache and doc_ids_for_cache:
                    self.cache.put(self.index_path, query, top_k, doc_ids_for_cache, scores_for_cache)

        return results


class SQLiteFTSBM25Retriever:
    """Smoke-only BM25 backend over a versioned SQLite FTS5 corpus.

    This backend exists to exercise Clean v2 when the canonical Lucene index is
    unavailable. It must not be used for paper results.
    """

    def __init__(self, index_path: str, cache=None, **_: Any):
        path = Path(index_path)
        self.index_path = str(path)
        self.db_path = path / "index.db" if path.is_dir() else path
        manifest = (
            path / "clean_v2_index_manifest.json"
            if path.is_dir()
            else path.parent / "clean_v2_index_manifest.json"
        )
        if not manifest.exists():
            raise FileNotFoundError(f"Missing clean_v2 smoke index manifest: {manifest}")
        self.connection = sqlite3.connect(self.db_path)
        self.cache = cache

    @staticmethod
    def _fts_query(query: str) -> str:
        terms = re.findall(r"[A-Za-z0-9]+", query.casefold())
        return " OR ".join(f'"{term}"' for term in terms) or '"__empty_query__"'

    def search(self, query: str, top_k: int = 10) -> List[RetrievedDoc]:
        rows = self.connection.execute(
            """SELECT d.doc_id, d.content, bm25(documents_fts) AS raw_score
               FROM documents_fts JOIN documents d ON d.rowid = documents_fts.rowid
               WHERE documents_fts MATCH ? ORDER BY raw_score ASC LIMIT ?""",
            (self._fts_query(query), top_k),
        ).fetchall()
        return [
            RetrievedDoc(doc_id=row[0], content=row[1], score=-float(row[2]), rank=rank)
            for rank, row in enumerate(rows, start=1)
        ]

    def batch_search(
        self, queries: List[str], query_ids: List[str], top_k: int = 10,
        show_progress: bool = True,
    ) -> Dict[str, List[RetrievedDoc]]:
        results: Dict[str, List[RetrievedDoc]] = {}
        cached = self.cache.get_batch(self.index_path, queries, top_k) if self.cache else {}
        iterator = list(zip(queries, query_ids))
        if show_progress:
            iterator = tqdm(iterator, desc="  SQLite FTS5 BM25", leave=True)
        for query, query_id in iterator:
            if query in cached:
                doc_ids, scores = cached[query]
                placeholders = ",".join("?" for _ in doc_ids)
                rows = self.connection.execute(
                    f"SELECT doc_id, content FROM documents WHERE doc_id IN ({placeholders})",
                    doc_ids,
                ).fetchall()
                content_by_id = {row[0]: row[1] for row in rows}
                results[query_id] = [
                    RetrievedDoc(
                        doc_id=doc_id, content=content_by_id.get(doc_id, ""),
                        score=score, rank=rank,
                    )
                    for rank, (doc_id, score) in enumerate(zip(doc_ids, scores), start=1)
                ]
                continue
            docs = self.search(query, top_k)
            results[query_id] = docs
            if self.cache:
                self.cache.put(
                    self.index_path, query, top_k,
                    [doc.doc_id for doc in docs], [doc.score for doc in docs],
                )
        return results


class CrossEncoderReranker:
    """Cross-encoder reranker using sentence-transformers with optional caching."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        batch_size: int = 32,
        max_length: int = 512,
        device: str = None,
        cache=None,
    ):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers required. Install with: pip install sentence-transformers"
            )

        self.model_name = model_name
        self.model = CrossEncoder(model_name, max_length=max_length, device=device)
        self.batch_size = batch_size
        self.cache = cache  # RerankerCache instance

    def rerank(
        self,
        query: str,
        documents: List[RetrievedDoc],
        top_k: int = 5,
    ) -> List[RetrievedDoc]:
        """Rerank documents for a single query."""
        if not documents:
            return []

        # Prepare pairs
        pairs = [(query, doc.content) for doc in documents]

        # Score with cross-encoder
        scores = self.model.predict(pairs, batch_size=self.batch_size)

        # Combine with original docs and sort
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Return top-k with updated scores and ranks
        results = []
        for rank, (doc, score) in enumerate(scored_docs[:top_k], start=1):
            results.append(RetrievedDoc(
                doc_id=doc.doc_id,
                content=doc.content,
                score=float(score),
                rank=rank,
            ))
        return results

    def batch_rerank(
        self,
        queries: List[str],
        documents_list: List[List[RetrievedDoc]],
        top_k: int = 5,
        show_progress: bool = True,
    ) -> List[List[RetrievedDoc]]:
        """Batch rerank for multiple queries with caching."""
        if not queries:
            return []

        # Initialize scores matrix: query_doc_scores[q_idx] = [(doc, score), ...]
        query_doc_scores = [[] for _ in queries]

        # Track pairs that need scoring
        pairs_to_score = []  # (query, doc_content)
        pair_locations = []  # (q_idx, d_idx) for each pair to score

        cache_hits = 0
        total_pairs = 0

        for q_idx, (query, docs) in enumerate(zip(queries, documents_list)):
            if not docs:
                continue

            total_pairs += len(docs)
            doc_contents = [doc.content for doc in docs]

            if self.cache:
                # Check cache for this query's documents
                cached_scores = self.cache.get_batch(self.model_name, query, doc_contents)
                cache_hits += len(cached_scores)

                for d_idx, doc in enumerate(docs):
                    if d_idx in cached_scores:
                        # Use cached score
                        query_doc_scores[q_idx].append((doc, cached_scores[d_idx]))
                    else:
                        # Need to score this pair
                        pairs_to_score.append((query, doc.content))
                        pair_locations.append((q_idx, d_idx))
            else:
                # No cache, score all pairs
                for d_idx, doc in enumerate(docs):
                    pairs_to_score.append((query, doc.content))
                    pair_locations.append((q_idx, d_idx))

        # Report cache stats
        if self.cache and cache_hits > 0:
            print(f"    Reranker cache: {cache_hits} hits, {len(pairs_to_score)} to score")

        # Score uncached pairs
        if pairs_to_score:
            scores = self.model.predict(
                pairs_to_score,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
            )

            # Prepare cache entries grouped by query
            cache_entries = {}  # query -> [(doc_content, score), ...]

            for (q_idx, d_idx), score in zip(pair_locations, scores):
                doc = documents_list[q_idx][d_idx]
                query = queries[q_idx]
                query_doc_scores[q_idx].append((doc, float(score)))

                # Group for cache
                if self.cache:
                    if query not in cache_entries:
                        cache_entries[query] = ([], [])
                    cache_entries[query][0].append(doc.content)
                    cache_entries[query][1].append(float(score))

            # Store in cache
            if self.cache and cache_entries:
                for query, (doc_contents, scores_list) in cache_entries.items():
                    self.cache.put_batch(self.model_name, query, doc_contents, scores_list)

        # Sort and return top-k for each query
        results = []
        for scored_docs in query_doc_scores:
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            reranked = []
            for rank, (doc, score) in enumerate(scored_docs[:top_k], start=1):
                reranked.append(RetrievedDoc(
                    doc_id=doc.doc_id,
                    content=doc.content,
                    score=float(score),
                    rank=rank,
                ))
            results.append(reranked)

        return results


class HybridRetriever:
    """
    Hybrid retriever combining BM25 with optional reranking.
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        reranker: Optional[CrossEncoderReranker] = None,
    ):
        self.bm25 = bm25
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        rerank_top_k: Optional[int] = None,
    ) -> List[RetrievedDoc]:
        """
        Retrieve documents for a single query.

        Args:
            query: Search query
            top_k: Number of documents to retrieve from BM25
            rerank_top_k: If set, rerank and return this many docs
        """
        docs = self.bm25.search(query, top_k=top_k)

        if self.reranker and rerank_top_k:
            docs = self.reranker.rerank(query, docs, top_k=rerank_top_k)

        return docs

    def batch_retrieve(
        self,
        queries: List[str],
        query_ids: List[str],
        top_k: int = 10,
        rerank_top_k: Optional[int] = None,
        desc: str = "Retrieving",
        show_progress: bool = True,
    ) -> Dict[str, List[RetrievedDoc]]:
        """
        Batch retrieve documents for multiple queries.
        """
        # BM25 retrieval
        print(f"Running BM25 retrieval (top_k={top_k})...")
        results = self.bm25.batch_search(queries, query_ids, top_k=top_k, show_progress=show_progress)

        # Reranking if enabled
        if self.reranker and rerank_top_k:
            docs_list = [results[qid] for qid in query_ids]
            total_pairs = sum(len(docs) for docs in docs_list)
            print(f"  Reranking {len(queries)} queries ({total_pairs} pairs) to top_k={rerank_top_k}...")
            reranked = self.reranker.batch_rerank(queries, docs_list, top_k=rerank_top_k, show_progress=show_progress)
            results = {qid: docs for qid, docs in zip(query_ids, reranked)}

        return results


def format_context(
    documents: List[RetrievedDoc],
    format_template: str = "Retrieved Document {rank}: {content}",
    separator: str = "\n\n",
) -> str:
    """
    Format retrieved documents into a context string.
    """
    if not documents:
        return ""

    parts = []
    for doc in documents:
        part = format_template.format(
            rank=doc.rank,
            content=doc.content,
            doc_id=doc.doc_id,
            score=doc.score,
        )
        parts.append(part)

    return separator.join(parts)


def create_retriever(config: Dict[str, Any], skip_reranker: bool = False, skip_cache: bool = False) -> Optional[HybridRetriever]:
    """
    Factory function to create retriever based on config.

    Args:
        config: Retrieval configuration
        skip_reranker: If True, skip loading the reranker (for methods that don't need it)
        skip_cache: If True, disable BM25 caching (useful for test runs)
    """
    index_path = config.get("index_path")
    if not index_path:
        return None

    # Create BM25 cache if configured (unless skip_cache is True)
    bm25_cache = None
    if skip_cache:
        print("  BM25 cache disabled (test mode)")
    else:
        cache_path = config.get("bm25_cache_path")
        if cache_path:
            try:
                from .highlighter import BM25Cache
                bm25_cache = BM25Cache(cache_path)
                print(f"  BM25 cache enabled: {cache_path}")
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"BM25 cache not loaded: {e}")

    bm25_cfg = config.get("bm25", {})
    backend = config.get("backend", "pyserini")
    retriever_class = SQLiteFTSBM25Retriever if backend == "sqlite_fts5_smoke" else BM25Retriever
    bm25 = retriever_class(
        index_path=index_path,
        k1=bm25_cfg.get("k1", 0.9),
        b=bm25_cfg.get("b", 0.4),
        threads=config.get("threads", 8),
        cache=bm25_cache,
    )

    reranker = None
    if not skip_reranker:
        reranker_cfg = config.get("reranker")
        if reranker_cfg:
            # Create reranker cache if configured (unless skip_cache is True)
            reranker_cache = None
            if not skip_cache:
                reranker_cache_path = reranker_cfg.get("cache_path") or config.get("reranker_cache_path")
                if reranker_cache_path:
                    try:
                        from .highlighter import RerankerCache
                        reranker_cache = RerankerCache(reranker_cache_path)
                        print(f"  Reranker cache enabled: {reranker_cache_path}")
                    except Exception as e:
                        import logging
                        logging.getLogger(__name__).warning(f"Reranker cache not loaded: {e}")

            try:
                reranker = CrossEncoderReranker(
                    model_name=reranker_cfg.get("model_name", "cross-encoder/ms-marco-MiniLM-L-12-v2"),
                    batch_size=reranker_cfg.get("batch_size", 32),
                    max_length=reranker_cfg.get("max_length", 512),
                    cache=reranker_cache,
                )
            except ImportError as e:
                import logging
                logging.getLogger(__name__).warning(f"Reranker not loaded: {e}")

    return HybridRetriever(bm25=bm25, reranker=reranker)
