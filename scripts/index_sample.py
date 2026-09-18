"""Build BM25 and dense-vector Elasticsearch indexes from the public sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)


def load_documents(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    documents: list[dict] = []
    for row in rows:
        title = str(row["law"]).strip()
        article = str(row["article"]).strip()
        text = str(row["text"]).strip()
        fingerprint = hashlib.sha1(f"{title}|{article}|{text}".encode("utf-8")).hexdigest()[:12]
        documents.append(
            {
                "chunk_id": f"sample-{fingerprint}",
                "doc_id": f"sample-{hashlib.sha1(title.encode('utf-8')).hexdigest()[:10]}",
                "title": title,
                "law_group": title,
                "rel_path": "data/samples/law_articles.json",
                "source": f"{title} {article}",
                "text": f"{article} {text}",
                "text_len": len(text),
            }
        )
    return documents


def recreate_bm25(es: Elasticsearch, index_name: str, documents: list[dict]) -> None:
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
    es.indices.create(
        index=index_name,
        settings={"number_of_shards": 1, "number_of_replicas": 0},
        mappings={
            "properties": {
                "chunk_id": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "title": {"type": "text"},
                "law_group": {"type": "keyword"},
                "rel_path": {"type": "keyword"},
                "source": {"type": "text"},
                "text": {"type": "text"},
                "text_len": {"type": "integer"},
            }
        },
    )
    helpers.bulk(
        es,
        ({"_index": index_name, "_id": row["chunk_id"], "_source": row} for row in documents),
    )
    es.indices.refresh(index=index_name)


def recreate_dense(
    es: Elasticsearch,
    index_name: str,
    documents: list[dict],
    model: SentenceTransformer,
) -> None:
    vectors = model.encode(
        [f"{row['title']} {row['source']} {row['text']}" for row in documents],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    dims = int(model.get_sentence_embedding_dimension())
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
    es.indices.create(
        index=index_name,
        settings={"number_of_shards": 1, "number_of_replicas": 0},
        mappings={
            "properties": {
                "chunk_id": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "title": {"type": "text"},
                "law_group": {"type": "keyword"},
                "rel_path": {"type": "keyword"},
                "source": {"type": "text"},
                "text": {"type": "text"},
                "text_len": {"type": "integer"},
                "vector": {"type": "dense_vector", "dims": dims, "index": True, "similarity": "cosine"},
            }
        },
    )
    actions = []
    for row, vector in zip(documents, vectors):
        actions.append({"_index": index_name, "_id": row["chunk_id"], "_source": {**row, "vector": vector.tolist()}})
    helpers.bulk(es, actions)
    es.indices.refresh(index=index_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Index the public five-article sample")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "samples" / "law_articles.json")
    args = parser.parse_args()

    es = Elasticsearch(os.getenv("ES_URL", "http://127.0.0.1:9200"), request_timeout=60)
    if not es.ping():
        raise RuntimeError("Elasticsearch is not reachable. Start docker-compose.es.yml first.")

    documents = load_documents(args.input)
    model_name = os.getenv("BGE_LARGE_MODEL", "BAAI/bge-small-zh-v1.5")
    model = SentenceTransformer(model_name, device=os.getenv("EMB_DEVICE", "cpu"))
    recreate_bm25(es, os.getenv("ES_BM25_INDEX", "law_bm25"), documents)
    recreate_dense(es, os.getenv("ES_BGE_LARGE_INDEX", "law_bge_large"), documents, model)
    print(f"Indexed {len(documents)} public sample articles with {model_name}.")


if __name__ == "__main__":
    main()
