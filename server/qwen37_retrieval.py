"""Qwen3.7 embedding retrieval for the live legal RAG demo.

BM25 remains the lexical retriever in Elasticsearch. Dense retrieval uses the
same Qwen3.7 API encoder for queries and the offline Qwen3.7 FAISS indexes for
law and case chunks. BGE and Qwen3-Embedding-8B vectors are intentionally not
mixed with this index.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import requests
from dotenv import load_dotenv
from elasticsearch import Elasticsearch


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parents[1]
load_dotenv(ROOT / ".env", override=False)
load_dotenv(WORKSPACE_ROOT / ".env", override=False)


def _resolve_path(value: str, default: Path) -> Path:
    raw = Path(value) if value else default
    if raw.is_absolute():
        return raw
    for candidate in (ROOT / raw, WORKSPACE_ROOT / raw, raw):
        if candidate.exists():
            return candidate.resolve()
    return (WORKSPACE_ROOT / raw).resolve()


def _read_faiss_index(path: Path):
    """Read FAISS on Windows even when the project path contains CJK characters."""
    path_text = str(path)
    if all(ord(char) < 128 for char in path_text):
        return faiss.read_index(path_text)
    temp_path = Path(tempfile.gettempdir()) / f"qwen37_faiss_{os.getpid()}_{path.stat().st_size}.index"
    shutil.copyfile(path, temp_path)
    try:
        return faiss.read_index(str(temp_path))
    finally:
        try:
            temp_path.unlink()
        except OSError:
            pass


class Qwen37FaissIndex:
    """Load a normalized FAISS index and aligned JSONL metadata."""

    def __init__(self, directory: Path, corpus: str) -> None:
        self.directory = directory
        self.corpus = corpus
        index_path = directory / "index.faiss"
        metadata_path = directory / "meta.jsonl"
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Qwen3.7 index files are missing: {directory}")
        self.index = _read_faiss_index(index_path)
        with metadata_path.open("r", encoding="utf-8") as handle:
            self.metadata = [json.loads(line) for line in handle if line.strip()]
        if self.index.ntotal != len(self.metadata):
            raise RuntimeError(
                f"Qwen3.7 index/metadata mismatch for {directory}: "
                f"index={self.index.ntotal}, metadata={len(self.metadata)}"
            )
        self.dim = self.index.d

    def search(self, vector: np.ndarray, topk: int) -> list[dict[str, Any]]:
        query = np.asarray(vector, dtype="float32").reshape(1, -1)
        if query.shape[1] != self.dim:
            raise ValueError(f"Qwen3.7 query dimension {query.shape[1]} != index dimension {self.dim}")
        scores, ids = self.index.search(query, min(max(1, topk), self.index.ntotal))
        hits: list[dict[str, Any]] = []
        for score, row_id in zip(scores[0], ids[0]):
            if row_id < 0:
                continue
            source = dict(self.metadata[int(row_id)])
            source.setdefault("corpus", self.corpus)
            hits.append({"_id": source.get("chunk_id", str(row_id)), "_score": float(score), "_source": source})
        return hits


class LocalBM25Index:
    """Small in-memory BM25 fallback used when Elasticsearch is unavailable."""

    TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")

    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self.documents = documents
        self.term_postings: dict[str, dict[int, int]] = {}
        self.lengths: list[int] = []
        for doc_id, document in enumerate(documents):
            text = f"{document.get('title', '')} {document.get('text', '')}"
            counts: dict[str, int] = {}
            for token in self.TOKEN_RE.findall(text.lower()):
                counts[token] = counts.get(token, 0) + 1
            self.lengths.append(sum(counts.values()))
            for token, frequency in counts.items():
                self.term_postings.setdefault(token, {})[doc_id] = frequency
        self.average_length = sum(self.lengths) / max(1, len(self.lengths))

    def search(self, query: str, topk: int) -> list[dict[str, Any]]:
        query_tokens = set(self.TOKEN_RE.findall(query.lower()))
        scores: dict[int, float] = {}
        total = len(self.documents)
        for token in query_tokens:
            postings = self.term_postings.get(token)
            if not postings:
                continue
            document_frequency = len(postings)
            inverse_document_frequency = math.log(1.0 + (total - document_frequency + 0.5) / (document_frequency + 0.5))
            for doc_id, term_frequency in postings.items():
                length_norm = 1.2 * (1.0 - 0.75 + 0.75 * self.lengths[doc_id] / max(1.0, self.average_length))
                scores[doc_id] = scores.get(doc_id, 0.0) + inverse_document_frequency * (
                    term_frequency * 2.2 / (term_frequency + length_norm)
                )
        hits: list[dict[str, Any]] = []
        for score, doc_id in heapq.nlargest(max(1, topk), ((score, doc_id) for doc_id, score in scores.items())):
            source = dict(self.documents[doc_id])
            source.setdefault("corpus", "law")
            hits.append({"_id": source.get("chunk_id", str(doc_id)), "_score": float(score), "_source": source})
        return hits


class Qwen37APIEmbedder:
    """DashScope OpenAI-compatible Qwen3.7 query encoder with an in-process cache."""

    def __init__(self) -> None:
        self.model = os.getenv("QWEN37_EMBEDDING_MODEL", "qwen3.7-text-embedding")
        self.dimensions = int(os.getenv("QWEN37_EMBEDDING_DIM", "1024"))
        self.batch_size = int(os.getenv("QWEN37_EMBEDDING_BATCH", "20"))
        self.connect_timeout = float(os.getenv("QWEN37_EMBEDDING_CONNECT_TIMEOUT", "5"))
        self.read_timeout = float(os.getenv("QWEN37_EMBEDDING_READ_TIMEOUT", os.getenv("QWEN37_EMBEDDING_TIMEOUT", "30")))
        self.max_retries = max(0, int(os.getenv("QWEN37_EMBEDDING_MAX_RETRIES", "1")))
        self.trust_env = os.getenv("QWEN37_TRUST_ENV", "false").strip().lower() in {"1", "true", "yes"}
        self.verify_ssl = os.getenv("QWEN37_VERIFY_SSL", "false").strip().lower() in {"1", "true", "yes"}
        if not self.verify_ssl:
            requests.packages.urllib3.disable_warnings()  # type: ignore[attr-defined]
        self.api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is missing; Qwen3.7 query encoding cannot start")
        base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
        self.endpoint = f"{base_url}/embeddings"
        self.http = requests.Session()
        # The workstation's inherited proxy/network settings caused requests to
        # hang while curl and a direct session completed normally. Direct access
        # is the safe default; set QWEN37_TRUST_ENV=true on proxy-only hosts.
        self.http.trust_env = self.trust_env
        self.cache: dict[str, np.ndarray] = {}
        law_default = WORKSPACE_ROOT / "data" / "indexes" / "law_qwen37_api_latest"
        cases_default = WORKSPACE_ROOT / "data" / "indexes" / "cases_qwen37_api_latest"
        self.law_index = Qwen37FaissIndex(_resolve_path(os.getenv("QWEN37_LAW_INDEX_DIR", ""), law_default), "law")
        self.case_index = Qwen37FaissIndex(_resolve_path(os.getenv("QWEN37_CASE_INDEX_DIR", ""), cases_default), "case")
        self._local_bm25: LocalBM25Index | None = None
        if self.law_index.dim != self.dimensions or self.case_index.dim != self.dimensions:
            raise RuntimeError(
                "Qwen3.7 configured dimension does not match the offline index: "
                f"configured={self.dimensions}, law={self.law_index.dim}, cases={self.case_index.dim}"
            )

    @property
    def local_bm25(self) -> LocalBM25Index:
        if self._local_bm25 is None:
            self._local_bm25 = LocalBM25Index(self.law_index.metadata)
        return self._local_bm25

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.http.post(
                    self.endpoint,
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json={"model": self.model, "input": texts, "dimensions": self.dimensions},
                    timeout=(self.connect_timeout, self.read_timeout),
                    verify=self.verify_ssl,
                )
                response.raise_for_status()
                break
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"Qwen3.7 embedding request failed after {attempt + 1} attempt(s): {exc}"
                    ) from exc
                time.sleep(0.25 * (attempt + 1))
        else:  # pragma: no cover - the loop either breaks or raises above
            raise RuntimeError(f"Qwen3.7 embedding request failed: {last_error}")
        payload = response.json()
        rows = sorted(payload.get("data") or [], key=lambda item: int(item.get("index", 0)))
        if len(rows) != len(texts):
            raise RuntimeError(f"Qwen3.7 embedding returned {len(rows)} vectors for {len(texts)} inputs")
        vectors = np.asarray([row["embedding"] for row in rows], dtype="float32")
        if vectors.ndim != 2 or vectors.shape[1] != self.dimensions:
            raise RuntimeError(f"Unexpected Qwen3.7 embedding shape: {vectors.shape}")
        return vectors

    def encode(self, texts: str | list[str], normalize_embeddings: bool = True) -> np.ndarray:
        single = isinstance(texts, str)
        values = [texts] if single else [str(value) for value in texts]
        result: list[np.ndarray | None] = [None] * len(values)
        pending: list[tuple[int, str, str]] = []
        for index, value in enumerate(values):
            key = hashlib.sha256(value.encode("utf-8")).hexdigest()
            cached = self.cache.get(key)
            if cached is not None:
                result[index] = cached.copy()
            else:
                pending.append((index, key, value))
        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            vectors = self._embed_batch([item[2] for item in batch])
            for (index, key, _), vector in zip(batch, vectors):
                if normalize_embeddings:
                    norm = np.linalg.norm(vector)
                    if norm:
                        vector = vector / norm
                self.cache[key] = vector.astype("float32")
                result[index] = self.cache[key].copy()
        matrix = np.asarray([value for value in result if value is not None], dtype="float32")
        return matrix[0] if single else matrix


def retrieve(
    es: Elasticsearch | None, model: Qwen37APIEmbedder, query: str, include_dense: bool = True
) -> tuple[list[dict], list[dict]]:
    topk = int(os.getenv("BM25_TOPK", "10"))
    if es is not None:
        bm25 = es.search(
            index=os.getenv("ES_BM25_INDEX", "law_bm25_latest"),
            size=topk,
            query={"multi_match": {"query": query, "fields": ["title^1.5", "text"]}},
        )["hits"]["hits"]
    else:
        bm25 = model.local_bm25.search(query, topk)
    if not include_dense:
        return bm25, []
    vector = model.encode(query, normalize_embeddings=True)
    dense = model.law_index.search(vector, int(os.getenv("QWEN37_LAW_TOPK", "10")))
    return bm25, dense


def retrieve_cases(model: Qwen37APIEmbedder, query: str) -> list[dict]:
    vector = model.encode(query, normalize_embeddings=True)
    return model.case_index.search(vector, int(os.getenv("QWEN37_CASE_TOPK", "8")))


def rrf_candidates(
    bm25: list[dict], dense: list[dict], case_dense: list[dict] | None = None, rrf_k: int = 60
) -> list[dict]:
    combined: dict[str, dict] = {}
    streams = (("bm25", bm25), ("qwen37", dense), ("case_qwen37", case_dense or []))
    for source_name, hits in streams:
        for rank, hit in enumerate(hits, start=1):
            source = hit.get("_source", {})
            key = source.get("chunk_id") or hit.get("_id")
            item = combined.setdefault(
                key,
                {
                    "chunk_id": key,
                    "title": source.get("title", ""),
                    "text": source.get("text", ""),
                    "corpus": source.get("corpus") or ("case" if source_name == "case_qwen37" else "law"),
                    "ranks": {},
                    "rrf": 0.0,
                },
            )
            item["ranks"][source_name] = rank
            item["rrf"] += 1.0 / (rrf_k + rank)
    return sorted(combined.values(), key=lambda item: item["rrf"], reverse=True)
