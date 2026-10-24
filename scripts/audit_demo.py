"""Smoke-check the running demo; --live also measures two paid LLM requests."""
from __future__ import annotations

import argparse
import json
import time

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8008")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    results = {"endpoints": {}, "queries": []}
    with httpx.Client(base_url=args.base_url, trust_env=False, timeout=240) as client:
        for path in ("/", "/api/healthz", "/api/chat/sessions", "/api/prompts", "/api/evaluation-cases", "/api/batch-experiments"):
            response = client.get(path)
            response.raise_for_status()
            results["endpoints"][path] = response.status_code
        if args.live:
            # Same synthetic question; the second call can reuse the query
            # embedding. Report all stage timings, not a causal speed claim.
            for include_cases in (False, True):
                started = time.perf_counter()
                response = client.post("/api/legal-query", json={
                    "question": "公司拖欠我三个月工资，我应该如何维权？",
                    "include_cases": include_cases, "run_evaluation": False, "skip_rerank": True,
                })
                response.raise_for_status()
                result = response.json()
                assert result["intent"] == "legal_consultation", result.get("analysis")
                assert result.get("answer"), "Missing answer"
                if not include_cases:
                    assert not result["retrieval"]["cases_bm25"]
                    assert not result["retrieval"]["cases_qwen37"]
                row = {
                    "include_cases": include_cases,
                    "wall_sec": round(time.perf_counter() - started, 3),
                    "timings_sec": result["timings_sec"],
                    "hits": {key: len(value) for key, value in result["retrieval"].items()},
                    "case_evidence_count": sum(item.get("corpus") == "case" for item in result["contexts"]),
                }
                results["queries"].append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
