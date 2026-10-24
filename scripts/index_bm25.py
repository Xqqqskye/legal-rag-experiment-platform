"""Build a lexical BM25 index for law or case metadata.

The live demo keeps lexical retrieval separate by corpus: law chunks are written
to ``law_bm25_latest`` and case chunks to ``case_bm25_latest``.  Dense vectors
are not required for this script; it only stores the metadata needed by the
BM25 query path and the later RRF fusion step.

Accepted input formats are a JSON array, JSONL, or a directory containing
JSON/JSONL files.  Each row should contain ``text`` and may contain ``title``;
all other fields are preserved so that citations and parent/child identifiers
remain available to the application.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)


def _iter_paths(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if path.is_dir():
        for candidate in sorted(path.rglob("*")):
            if candidate.suffix.lower() in {".json", ".jsonl", ".ndjson"}:
                yield candidate


def _rows_from_file(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value
        return
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        # Common corpus wrappers: {"data": [...]} or {"items": [...]}.
        value = value.get("data", value.get("items", [value]))
    if isinstance(value, list):
        for row in value:
            if isinstance(row, dict):
                yield row


def load_documents(path: Path, corpus: str) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for file_path in _iter_paths(path):
        for row in _rows_from_file(file_path):
            text = str(row.get("text") or row.get("content") or row.get("body") or "").strip()
            if isinstance(row.get("body"), list):
                text = "\n".join(str(item) for item in row["body"] if item is not None).strip()
            title = str(row.get("title") or row.get("name") or row.get("case_name") or "").strip()
            if not text:
                continue
            source = dict(row)
            raw_id = source.get("chunk_id") or source.get("id") or source.get("case_id")
            if raw_id:
                chunk_id = str(raw_id)
            else:
                fingerprint = hashlib.sha1(f"{title}|{text}".encode("utf-8")).hexdigest()[:16]
                chunk_id = f"{corpus}-{fingerprint}"
            source.update(
                {
                    "chunk_id": chunk_id,
                    "title": title,
                    "text": text,
                    "corpus": corpus,
                    "rel_path": str(file_path),
                    "text_len": len(text),
                }
            )
            documents.append(source)
    if not documents:
        raise RuntimeError(f"No usable rows found under {path}")
    # Duplicate IDs would make bulk indexing silently overwrite records.
    unique: dict[str, dict[str, Any]] = {}
    for document in documents:
        unique[document["chunk_id"]] = document
    return list(unique.values())


def recreate_bm25(es: Elasticsearch, index_name: str, documents: list[dict[str, Any]]) -> None:
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
    es.indices.create(
        index=index_name,
        settings={"number_of_shards": 1, "number_of_replicas": 0},
        mappings={
            "dynamic": True,
            "properties": {
                "chunk_id": {"type": "keyword"},
                "title": {"type": "text"},
                "text": {"type": "text"},
                "corpus": {"type": "keyword"},
                "rel_path": {"type": "keyword"},
                "text_len": {"type": "integer"},
            },
        },
    )
    helpers.bulk(
        es,
        ({"_index": index_name, "_id": row["chunk_id"], "_source": row} for row in documents),
    )
    es.indices.refresh(index=index_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a BM25 index from law/case JSON or JSONL metadata")
    parser.add_argument("--input", type=Path, required=True, help="JSON/JSONL file or directory")
    parser.add_argument("--corpus", choices=("law", "case"), required=True)
    parser.add_argument("--index", default=None, help="Override the Elasticsearch index name")
    args = parser.parse_args()

    documents = load_documents(args.input, args.corpus)
    index_name = args.index or os.getenv(
        "ES_CASE_BM25_INDEX" if args.corpus == "case" else "ES_BM25_INDEX",
        "case_bm25_latest" if args.corpus == "case" else "law_bm25_latest",
    )
    es = Elasticsearch(os.getenv("ES_URL", "http://127.0.0.1:9200"), request_timeout=60)
    if not es.ping():
        raise RuntimeError("Elasticsearch is not reachable. Start docker-compose.es.yml first.")
    recreate_bm25(es, index_name, documents)
    print(f"Indexed {len(documents)} {args.corpus} chunks into {index_name}.")


if __name__ == "__main__":
    main()
