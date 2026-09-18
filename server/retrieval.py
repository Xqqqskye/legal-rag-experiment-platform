"""Run one live BM25 + BGE + RRF + semantic-rerank query."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)


def retrieve(es: Elasticsearch, model: SentenceTransformer, query: str) -> tuple[list[dict], list[dict]]:
    topk = int(os.getenv("BM25_TOPK", "10"))
    bm25 = es.search(
        index=os.getenv("ES_BM25_INDEX", "law_bm25"),
        size=topk,
        query={"multi_match": {"query": query, "fields": ["title^1.5", "text"]}},
    )["hits"]["hits"]

    vector = model.encode(query, normalize_embeddings=True).tolist()
    dense = es.search(
        index=os.getenv("ES_BGE_LARGE_INDEX", "law_bge_large"),
        knn={
            "field": os.getenv("ES_VECTOR_FIELD", "vector"),
            "query_vector": vector,
            "k": int(os.getenv("BGE_LARGE_TOPK", "10")),
            "num_candidates": 100,
        },
        source=["chunk_id", "title", "text"],
    )["hits"]["hits"]
    return bm25, dense


def rrf_candidates(bm25: list[dict], dense: list[dict], rrf_k: int = 60) -> list[dict]:
    combined: dict[str, dict] = {}
    for source_name, hits in (("bm25", bm25), ("bge", dense)):
        for rank, hit in enumerate(hits, start=1):
            source = hit["_source"]
            key = source.get("chunk_id") or hit["_id"]
            item = combined.setdefault(
                key,
                {"chunk_id": key, "title": source.get("title", ""), "text": source.get("text", ""), "ranks": {}, "rrf": 0.0},
            )
            item["ranks"][source_name] = rank
            item["rrf"] += 1.0 / (rrf_k + rank)
    return sorted(combined.values(), key=lambda item: item["rrf"], reverse=True)


def semantic_rerank(model: SentenceTransformer, query: str, candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []
    query_vector = model.encode([query], normalize_embeddings=True)[0]
    passage_vectors = model.encode(
        [item["text"] for item in candidates],
        batch_size=16,
        normalize_embeddings=True,
    )
    semantic_scores = np.asarray(passage_vectors) @ np.asarray(query_vector)
    max_rrf = max(item["rrf"] for item in candidates) or 1.0
    for item, semantic_score in zip(candidates, semantic_scores):
        item["semantic_score"] = float(semantic_score)
        item["final_score"] = 0.85 * float(semantic_score) + 0.15 * (item["rrf"] / max_rrf)
    return sorted(candidates, key=lambda item: item["final_score"], reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--retrieval-query", help="Legal rewrite used for retrieval and reranking")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    retrieval_query = (args.retrieval_query or args.query).strip()
    es = Elasticsearch(os.getenv("ES_URL", "http://127.0.0.1:9200"), request_timeout=30)
    if not es.ping():
        raise RuntimeError("Elasticsearch is not reachable")

    model_path = os.getenv("BGE_LARGE_MODEL", "BAAI/bge-large-zh-v1.5")
    model = SentenceTransformer(model_path, device=os.getenv("EMB_DEVICE", "cpu"))
    bm25, dense = retrieve(es, model, retrieval_query)
    ranked = semantic_rerank(model, retrieval_query, rrf_candidates(bm25, dense))

    print(f"Original query: {args.query}")
    print(f"Retrieval query: {retrieval_query}")
    print(f"Candidates: BM25={len(bm25)}, BGE={len(dense)}, fused={len(ranked)}")
    for index, item in enumerate(ranked[: args.topk], start=1):
        excerpt = " ".join(item["text"].split())[:260]
        print(
            f"\n#{index} {item['title']} | score={item['final_score']:.4f} "
            f"| ranks={item['ranks']}\n{excerpt}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
