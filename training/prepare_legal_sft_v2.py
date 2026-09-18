#!/usr/bin/env python3
"""Build a conservative legal-only SFT dataset for the AutoDL run.

The original mixed train/validation files contain a large general-purpose
Alpaca component.  This script intentionally starts from the legal source and
the small hand-curated seed set, then removes rows that are especially likely
to teach stale or corrupted legal answers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


REPEALED_OR_REPLACED_LAWS = (
    "侵权责任法",
    "合同法",
    "婚姻法",
    "物权法",
    "民法通则",
    "民法总则",
    "担保法",
    "继承法",
    "收养法",
)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def stable_bucket(key: str, modulo: int = 1000) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_legal = read_json(source_dir / "lawzhidao_best_sft_all.json")
    raw_seed = read_json(source_dir / "seed_sft_200.json")

    candidates = []
    for row in raw_legal:
        candidates.append(
            {
                "instruction": normalize(row.get("instruction")),
                "input": normalize(row.get("input")),
                "output": normalize(row.get("output")),
                "source": "lawzhidao",
            }
        )
    for row in raw_seed:
        candidates.append(
            {
                "instruction": normalize(row.get("query")),
                "input": "",
                "output": normalize(row.get("answer")),
                "source": "curated_seed",
            }
        )

    reasons: Counter[str] = Counter()
    seen = set()
    accepted = []
    for row in candidates:
        question = normalize(row["instruction"] + " " + row["input"])
        answer = row["output"]
        key = (question, answer)
        if not 4 <= len(question) <= 300:
            reasons["question_length"] += 1
            continue
        if not 40 <= len(answer) <= 2500:
            reasons["answer_length"] += 1
            continue
        if "*" in question or "*" in answer:
            reasons["masked_or_corrupted_text"] += 1
            continue
        if any(name in answer for name in REPEALED_OR_REPLACED_LAWS):
            reasons["repealed_or_replaced_law"] += 1
            continue
        if key in seen:
            reasons["duplicate"] += 1
            continue
        seen.add(key)
        accepted.append({k: row[k] for k in ("instruction", "input", "output")})

    # Stable 95/5 split by question so rerunning the script is deterministic.
    train, val = [], []
    for row in accepted:
        key = normalize(row["instruction"] + " " + row["input"])
        (val if stable_bucket(key) < 50 else train).append(row)

    for name, rows in (("train", train), ("val", val)):
        with (output_dir / f"legal_sft_v2_{name}.json").open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

    report = {
        "input_legal_rows": len(raw_legal),
        "input_curated_seed_rows": len(raw_seed),
        "accepted_rows": len(accepted),
        "train_rows": len(train),
        "validation_rows": len(val),
        "rejected_rows": sum(reasons.values()),
        "rejection_reasons": dict(sorted(reasons.items())),
        "filters": {
            "question_chars": [4, 300],
            "answer_chars": [40, 2500],
            "reject_asterisk_masking": True,
            "rejected_law_names": list(REPEALED_OR_REPLACED_LAWS),
        },
    }
    with (output_dir / "quality_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
