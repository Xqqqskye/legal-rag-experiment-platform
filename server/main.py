"""FastAPI backend for the live legal RAG demo.

The process keeps Elasticsearch, the embedding model, and the LLM client warm so
the browser can submit arbitrary questions instead of matching cached examples.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import statistics
import time
import uuid
import weakref
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from elasticsearch import Elasticsearch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from server.pipeline import (
    DEFAULT_PROMPTS,
    analyze_query,
    complete_json,
    generate_answer,
    llm_client,
    rerank_candidates,
)
from server.qwen37_retrieval import Qwen37APIEmbedder, retrieve_evidence, rrf_candidates
from server.experiment_store import ExperimentStore
from server.chat_store import ChatStore
from server.online_legal_search import search_official_legal_sources


ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "web"
LOGGER = logging.getLogger("legal-rag-demo")
ALLOWED_MODES = {"BM25 + Qwen3.7 向量 + RRF", "仅 BM25", "仅 Qwen3.7 向量"}
PROMPT_STAGES = ("analysis", "rerank", "answer", "online", "evaluation")
PROMPT_STAGE_META = {
    "analysis": {"label": "意图识别与 Query 改写", "step": 1, "input": "原始问题", "output": "意图、改写、争点、证据缺口"},
    "rerank": {"label": "法条适用性重排", "step": 5, "input": "问题分析、候选法条", "output": "保留/淘汰证据及理由"},
    "answer": {"label": "受约束法律回答", "step": 8, "input": "原问题、入选证据、联网证据", "output": "结构化回答与逐项引用"},
    "online": {"label": "联网法源核验", "step": 6, "input": "原问题、问题分析、当前日期", "output": "官方来源与效力摘要"},
    "evaluation": {"label": "回答质量评估", "step": 9, "input": "问题、回答、入选证据", "output": "主张核验、评分与 Bad case"},
}
DASHSCOPE_MODEL_OPTIONS = {
    "analysis": (
        "qwen3.8-flash",
        "qwen3.7-flash",
        "qwen3.5-flash",
        "deepseek-v4-flash",
        "glm-5.2",
    ),
    "rerank": (
        "qwen3.8-flash",
        "qwen3.8-27b",
        "qwen3.7-plus",
        "qwen3.7-max",
        "qwen3.8-max",
        "deepseek-v4-flash",
        "deepseek-v4-pro",
        "glm-5.2",
    ),
    "answer": (
        "qwen3.8-flash",
        "qwen3.8-27b",
        "qwen3.7-plus",
        "qwen3.7-max",
        "qwen3.8-max",
        "qwen3.5-plus",
        "deepseek-v4-flash",
        "deepseek-v4-pro",
        "glm-5.2",
    ),
    "evaluation": (
        "qwen3.7-flash",
        "qwen3.7-plus",
        "qwen3.7-max",
        "qwen3.8-max",
        "deepseek-v4-pro",
        "glm-5.2",
    ),
    "online": ("qwen-flash", "qwen-plus", "qwen-max", "qwen3.8-max", "qwen3.8-flash", "qwen3.7-plus"),
}
MODEL_PRICES_CNY_PER_MILLION = {
    "qwen3.8-flash": (0.8, 2.7),
    "qwen3.7-flash": (0.2, 0.8),
    "qwen3.5-flash": (0.2, 2.0),
    "qwen3.8-27b": (3.0, 12.0),
    "qwen3.7-plus": (2.0, 8.0),
    "qwen3.7-max": (12.0, 36.0),
    "qwen3.8-max": (12.0, 36.0),
    "deepseek-v4-flash": (1.5, 3.0),
    "deepseek-v4-pro": (12.0, 24.0),
    "glm-5.2": (8.0, 28.0),
}
MODEL_PRICING_SNAPSHOT_DATE = os.getenv("MODEL_PRICING_SNAPSHOT_DATE", "2026-09-11")


class TextInputRequest(BaseModel):
    @field_validator("question", "message", check_fields=False)
    @classmethod
    def nonblank_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("请输入问题或消息")
        return value


class LegalQueryRequest(TextInputRequest):
    question: str = Field(min_length=1, max_length=1200)
    use_rewrite: bool = True
    include_cases: bool = True
    retrieval_mode: str = "BM25 + Qwen3.7 向量 + RRF"
    final_top_k: int = Field(default=5, ge=1, le=5)
    run_evaluation: bool = True
    use_online_search: bool = False
    analysis_model: str | None = Field(default=None, max_length=100)
    rerank_model: str | None = Field(default=None, max_length=100)
    answer_model: str | None = Field(default=None, max_length=100)
    evaluation_model: str | None = Field(default=None, max_length=100)
    online_search_model: str | None = Field(default=None, max_length=100)
    prompt_profile: str = Field(default="默认提示词", max_length=40)
    prompt_overrides: dict[str, str] = Field(default_factory=dict)
    conversation_history: list[dict[str, str]] = Field(default_factory=list, max_length=20)
    case_memory: dict[str, Any] = Field(default_factory=dict)
    skip_rerank: bool = False


class ChatSessionCreateRequest(BaseModel):
    title: str = Field(default="新的法律咨询", max_length=120)


class ChatMessageRequest(TextInputRequest):
    message: str = Field(min_length=1, max_length=1200)
    use_rewrite: bool = True
    include_cases: bool = True
    retrieval_mode: str = "BM25 + Qwen3.7 向量 + RRF"
    final_top_k: int = Field(default=5, ge=1, le=5)
    run_evaluation: bool = False
    use_online_search: bool = False
    analysis_model: str | None = Field(default=None, max_length=100)
    rerank_model: str | None = Field(default=None, max_length=100)
    answer_model: str | None = Field(default=None, max_length=100)
    evaluation_model: str | None = Field(default=None, max_length=100)
    online_search_model: str | None = Field(default=None, max_length=100)
    prompt_profile: str = Field(default="默认提示词", max_length=40)
    prompt_overrides: dict[str, str] = Field(default_factory=dict)
    skip_rerank: bool = True


class ExperimentPipelineConfig(BaseModel):
    label: str = Field(default="方案", min_length=1, max_length=30)
    use_rewrite: bool = True
    include_cases: bool = True
    retrieval_mode: str = "BM25 + Qwen3.7 向量 + RRF"
    final_top_k: int = Field(default=5, ge=1, le=5)
    run_evaluation: bool = True
    use_online_search: bool = False
    analysis_model: str | None = Field(default=None, max_length=100)
    rerank_model: str | None = Field(default=None, max_length=100)
    answer_model: str | None = Field(default=None, max_length=100)
    evaluation_model: str | None = Field(default=None, max_length=100)
    online_search_model: str | None = Field(default=None, max_length=100)
    prompt_profile: str = Field(default="默认提示词", max_length=40)
    prompt_overrides: dict[str, str] = Field(default_factory=dict)


class CompareRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1200)
    pipeline_a: ExperimentPipelineConfig
    pipeline_b: ExperimentPipelineConfig


class EvaluationCaseInput(BaseModel):
    id: str | None = Field(default=None, max_length=120)
    question: str = Field(min_length=2, max_length=1200)
    category: str = Field(default="未分类", max_length=60)
    difficulty: str = Field(default="中", max_length=10)
    tags: list[str] = Field(default_factory=list, max_length=20)
    expected_intent: str = Field(default="法律咨询", max_length=30)
    expected_laws: list[str] = Field(default_factory=list, max_length=20)
    quality_threshold: float = Field(default=80, ge=0, le=100)
    notes: str = Field(default="", max_length=1000)


class EvaluationCaseImportRequest(BaseModel):
    cases: list[EvaluationCaseInput] = Field(min_length=1, max_length=500)


class BatchExperimentRequest(BaseModel):
    name: str = Field(default="批量 A/B 实验", min_length=1, max_length=100)
    case_ids: list[str] = Field(min_length=1, max_length=100)
    pipeline_a: ExperimentPipelineConfig
    pipeline_b: ExperimentPipelineConfig


class Runtime:
    def __init__(self) -> None:
        self.es: Elasticsearch | None = None
        self.embedder: Qwen37APIEmbedder | None = None
        self.client: Any = None
        self.provider = ""
        self.analysis_model = ""
        self.rerank_model = ""
        self.answer_model = ""
        self.evaluation_model = ""
        self.online_search_model = ""
        self.error = ""
        self.semaphore = asyncio.Semaphore(max(1, int(os.getenv("DEMO_MAX_CONCURRENCY", "1"))))

    def initialize(self) -> None:
        self.es = Elasticsearch(os.getenv("ES_URL", "http://127.0.0.1:9200"), request_timeout=30)
        try:
            es_available = self.es.ping()
        except Exception:
            es_available = False
        if not es_available:
            LOGGER.warning("Elasticsearch 未启动，将使用本地 BM25 兜底")
            self.es = None

        self.embedder = Qwen37APIEmbedder()
        if self.es is None:
            # Build once before serving requests instead of adding ~20 seconds
            # to the first legal question. Subsequent lookups reuse these maps.
            self.embedder.local_bm25
            self.embedder.local_case_bm25
        self.client, default_model, self.provider = llm_client()
        if self.provider == "dashscope":
            self.analysis_model = os.getenv("DASHSCOPE_ANALYSIS_MODEL", "qwen3.8-flash")
            self.rerank_model = os.getenv("DASHSCOPE_RERANK_MODEL", "qwen3.7-plus")
            # 默认走 Flash 以降低交互延迟；需要更强答案时可通过环境变量或前端下拉框切换。
            self.answer_model = os.getenv("DASHSCOPE_ANSWER_MODEL", "qwen3.8-flash")
            self.evaluation_model = os.getenv("DASHSCOPE_EVAL_MODEL", "qwen3.8-max")
            self.online_search_model = os.getenv("DASHSCOPE_ONLINE_MODEL", "qwen-flash")
        else:
            self.analysis_model = os.getenv("OPENAI_ANALYSIS_MODEL", default_model)
            self.rerank_model = os.getenv("OPENAI_RERANK_MODEL", default_model)
            self.answer_model = os.getenv("OPENAI_ANSWER_MODEL", default_model)
            self.evaluation_model = os.getenv("OPENAI_EVAL_MODEL", default_model)
            self.online_search_model = os.getenv("OPENAI_ONLINE_MODEL", default_model)

    @property
    def ready(self) -> bool:
        return bool(self.embedder is not None and self.client is not None and not self.error)

    def close(self) -> None:
        if self.es is not None:
            self.es.close()


runtime = Runtime()
experiment_store = ExperimentStore()
chat_store = ChatStore()
batch_tasks: dict[str, asyncio.Task[Any]] = {}
chat_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()


def _model_options() -> dict[str, list[str]]:
    if runtime.provider == "dashscope":
        return {stage: list(models) for stage, models in DASHSCOPE_MODEL_OPTIONS.items()}
    return {
        "analysis": [runtime.analysis_model],
        "rerank": [runtime.rerank_model],
        "answer": [runtime.answer_model],
        "evaluation": [runtime.evaluation_model],
        "online": [runtime.online_search_model],
    }


def _resolve_model(requested: str | None, stage: str) -> str:
    default = {
        "analysis": runtime.analysis_model,
        "rerank": runtime.rerank_model,
        "answer": runtime.answer_model,
        "evaluation": runtime.evaluation_model,
        "online": runtime.online_search_model,
    }[stage]
    model = (requested or default).strip()
    if model not in _model_options()[stage]:
        raise ValueError(f"{stage} 阶段不支持模型：{model}")
    return model


def _resolve_prompts(overrides: dict[str, str] | None) -> dict[str, str]:
    overrides = overrides or {}
    unknown = sorted(set(overrides) - set(PROMPT_STAGES))
    if unknown:
        raise ValueError("不支持的提示词阶段：" + "、".join(unknown))
    resolved = dict(DEFAULT_PROMPTS)
    for stage, value in overrides.items():
        prompt = str(value or "").strip()
        if not prompt:
            continue
        if len(prompt) > 12000:
            raise ValueError(f"{stage} 提示词超过 12000 字符")
        resolved[stage] = prompt
    return resolved


def _prompt_trace(prompts: dict[str, str], overrides: dict[str, str] | None, profile: str) -> dict[str, Any]:
    changed = set((overrides or {}).keys())
    return {
        "profile": profile,
        "stages": {
            stage: {
                **PROMPT_STAGE_META[stage],
                "source": "custom" if stage in changed and str((overrides or {}).get(stage) or "").strip() else "default",
                "version": hashlib.sha256(prompts[stage].encode("utf-8")).hexdigest()[:12],
                "prompt": prompts[stage],
            }
            for stage in PROMPT_STAGES
        },
    }


@asynccontextmanager
async def lifespan(_: FastAPI):
    await asyncio.to_thread(experiment_store.initialize)
    await asyncio.to_thread(chat_store.initialize)
    try:
        await asyncio.to_thread(runtime.initialize)
    except Exception as exc:  # Keep the page available so it can show setup errors.
        runtime.error = str(exc)
        LOGGER.exception("Live demo initialization failed")
    yield
    for task in batch_tasks.values():
        task.cancel()
    runtime.close()


app = FastAPI(
    title="法律智能问答实时 RAG Demo",
    version="1.0.0",
    lifespan=lifespan,
)


def _hit_view(hit: dict[str, Any], rank: int) -> dict[str, Any]:
    source = hit.get("_source", {})
    return {
        "rank": rank,
        "chunk_id": source.get("chunk_id") or hit.get("_id", ""),
        "title": source.get("title", ""),
        "text": source.get("text", ""),
        "score": float(hit.get("_score") or 0.0),
    }


def _single_candidates(hits: list[dict[str, Any]], source_name: str) -> list[dict[str, Any]]:
    rrf_k = int(os.getenv("RRF_K", "60"))
    candidates = []
    for rank, hit in enumerate(hits, start=1):
        source = hit.get("_source", {})
        candidates.append(
            {
                "chunk_id": source.get("chunk_id") or hit.get("_id", ""),
                "title": source.get("title", ""),
                "text": source.get("text", ""),
                "corpus": source.get("corpus", "law"),
                "ranks": {source_name: rank},
                "rrf": 1.0 / (rrf_k + rank),
            }
        )
    return candidates


def _direct_rrf_ranking(candidates: list[dict[str, Any]], final_top_k: int) -> list[dict[str, Any]]:
    """Convert RRF candidates to the same evidence shape as LLM reranking."""
    ranking: list[dict[str, Any]] = []
    for index, item in enumerate(candidates[:final_top_k], start=1):
        ranking.append(
            {
                "id": index,
                "title": item.get("title", ""),
                "text": item.get("text", ""),
                "corpus": item.get("corpus", "law"),
                "retrieval_ranks": item.get("ranks", {}),
                "score": round(float(item.get("rrf") or 0.0) * 100, 4),
                "applicability": "直接 RRF 候选",
                "reason": "未调用 LLM 重排，按 BM25/Qwen3.7 向量 RRF 分数排序。",
                "supports": [],
                "limitations": ["未经过法律适用性精排，需结合回答中的不确定性复核。"],
            }
        )
    return ranking


def _usage_view(usage: Any) -> dict[str, int]:
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    usage = usage if isinstance(usage, dict) else {}
    input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": int(usage.get("total_tokens") or input_tokens + output_tokens),
    }


def _estimated_cost(model: str, usage: dict[str, int], search_requests: int = 0) -> dict[str, Any]:
    rates = MODEL_PRICES_CNY_PER_MILLION.get(model)
    token_cost = None
    if rates:
        token_cost = (
            usage.get("input_tokens", 0) * rates[0] + usage.get("output_tokens", 0) * rates[1]
        ) / 1_000_000
    search_cost = search_requests * 0.004
    total = None if token_cost is None else token_cost + search_cost
    return {
        "model": model,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "estimated_cny": round(total, 6) if total is not None else None,
        "search_requests": search_requests,
        "price_known": rates is not None,
    }


def _cost_summary(stages: dict[str, dict[str, Any]]) -> dict[str, Any]:
    priced = [item["estimated_cny"] for item in stages.values() if item.get("estimated_cny") is not None]
    return {
        "stages": stages,
        "total_tokens": sum(int(item.get("total_tokens") or 0) for item in stages.values()),
        "estimated_cny": round(sum(priced), 6),
        "fully_priced": len(priced) == len(stages),
        "pricing_note": "按华北2（北京）公开原价估算，未计缓存、免费额度、活动与阶梯价格",
    }


def _online_legal_search(
    question: str,
    analysis: dict[str, Any],
    model: str,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    result = search_official_legal_sources(
        client=runtime.client,
        provider=runtime.provider,
        model=model,
        question=question,
        analysis=analysis,
        system_prompt=system_prompt or DEFAULT_PROMPTS["online"],
    )
    if result.get("status") in {"failed", "timed_out"}:
        LOGGER.warning("Online legal verification degraded: %s", result.get("error") or result.get("status"))
    return result


def _merge_ranking(answer_result: dict[str, Any], final_top_k: int) -> list[dict[str, Any]]:
    packed = answer_result.get("candidates") or []
    merged = []
    for item in answer_result.get("ranking") or []:
        try:
            candidate_id = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        if not 1 <= candidate_id <= len(packed):
            continue
        candidate = packed[candidate_id - 1]
        merged.append(
            {
                "id": candidate_id,
                "title": candidate.get("title", ""),
                "text": candidate.get("text", ""),
                "retrieval_ranks": candidate.get("retrieval_ranks", {}),
                "score": float(item.get("score") or 0.0),
                "applicability": item.get("applicability", ""),
                "reason": item.get("reason", ""),
                "supports": item.get("supports") if isinstance(item.get("supports"), list) else [],
                "limitations": item.get("limitations") if isinstance(item.get("limitations"), list) else [],
            }
        )
    return merged[:final_top_k]


def _citation_ids(values: Any, allowed: set[int]) -> list[int]:
    if not isinstance(values, list):
        return []
    result: list[int] = []
    for value in values:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number in allowed and number not in result:
            result.append(number)
    return result


def _normalize_answer_result(
    raw: dict[str, Any],
    selected_ids: set[int],
    online_ids: set[int],
) -> tuple[dict[str, Any], str]:
    def normalize_entry(value: Any) -> dict[str, Any]:
        if isinstance(value, str):
            value = {"text": value}
        value = value if isinstance(value, dict) else {}
        return {
            "text": str(value.get("text") or "").strip(),
            "evidence_ids": _citation_ids(value.get("evidence_ids"), selected_ids),
            "online_evidence_ids": _citation_ids(value.get("online_evidence_ids"), online_ids),
        }

    conclusion = normalize_entry(raw.get("conclusion"))
    sections: list[dict[str, Any]] = []
    for section in raw.get("sections") or []:
        if not isinstance(section, dict):
            continue
        items = [normalize_entry(item) for item in section.get("items") or []]
        items = [item for item in items if item["text"]]
        if items:
            sections.append({"title": str(section.get("title") or "建议").strip(), "items": items})
    uncertainties = [str(item).strip() for item in raw.get("uncertainties") or [] if str(item).strip()][:8]
    structured = {"conclusion": conclusion, "sections": sections, "uncertainties": uncertainties}

    def rendered(entry: dict[str, Any]) -> str:
        cites = "".join(f"[{value}]" for value in entry["evidence_ids"])
        cites += "".join(f"[联网{value}]" for value in entry["online_evidence_ids"])
        return entry["text"] + (f" {cites}" if cites else "")

    lines: list[str] = []
    if conclusion["text"]:
        lines.extend(["初步结论", rendered(conclusion), ""])
    for section in sections:
        lines.append(section["title"])
        lines.extend(f"{index}. {rendered(item)}" for index, item in enumerate(section["items"], start=1))
        lines.append("")
    if uncertainties:
        lines.append("尚待确认")
        lines.extend(f"- {item}" for item in uncertainties)
    fallback = str(raw.get("answer") or "").strip()
    return structured, "\n".join(lines).strip() or fallback


def _citation_audit(answer: str, candidate_count: int, selected_ids: set[int], online_count: int = 0) -> dict[str, Any]:
    cited_ids = [int(value) for value in re.findall(r"\[(\d+)\]", answer)]
    online_ids = [int(value) for value in re.findall(r"\[联网(\d+)\]", answer)]
    unique_ids = sorted(set(cited_ids))
    unique_online_ids = sorted(set(online_ids))
    invalid_ids = [value for value in unique_ids if not 1 <= value <= candidate_count]
    outside_context_ids = [value for value in unique_ids if value not in selected_ids and value not in invalid_ids]
    invalid_online_ids = [value for value in unique_online_ids if not 1 <= value <= online_count]
    return {
        "citation_count": len(cited_ids) + len(online_ids),
        "unique_citation_count": len(unique_ids) + len(unique_online_ids),
        "valid": not invalid_ids and not outside_context_ids and not invalid_online_ids and bool(unique_ids or unique_online_ids),
        "invalid_ids": invalid_ids,
        "outside_context_ids": outside_context_ids,
        "online_ids": unique_online_ids,
        "invalid_online_ids": invalid_online_ids,
        "selected_context_ids": sorted(selected_ids),
    }


def _evaluate_answer(
    question: str,
    answer: str,
    analysis: dict[str, Any],
    ranking: list[dict[str, Any]],
    candidate_count: int,
    corpus_cutoff: str,
    evaluation_model: str,
    online_verification: dict[str, Any] | None = None,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    selected_ids = {int(item["id"]) for item in ranking}
    online_count = len((online_verification or {}).get("official_sources") or [])
    citation_audit = _citation_audit(answer, candidate_count, selected_ids, online_count)
    evidence = [
        {
            "id": item["id"],
            "title": item["title"],
            "text": item["text"],
            "applicability": item["applicability"],
            "supports": item.get("supports") or [],
            "limitations": item.get("limitations") or [],
        }
        for item in ranking
    ]
    user = json.dumps(
        {
            "question": question,
            "query_analysis": analysis,
            "answer": answer,
            "selected_evidence": evidence,
            "online_verification": online_verification,
            "citation_audit": citation_audit,
            "corpus_cutoff": corpus_cutoff,
        },
        ensure_ascii=False,
    )
    try:
        result = complete_json(
            runtime.client,
            evaluation_model,
            system_prompt or DEFAULT_PROMPTS["evaluation"],
            user,
            max_tokens=2048,
        )
        scores = result.get("scores") if isinstance(result.get("scores"), dict) else {}
        normalized_scores = {}
        for name in ("groundedness", "answer_relevance", "legal_applicability", "completeness", "safety", "citation_quality"):
            try:
                normalized_scores[name] = max(0.0, min(100.0, float(scores.get(name, 0))))
            except (TypeError, ValueError):
                normalized_scores[name] = 0.0
        weights = {
            "groundedness": 0.25,
            "answer_relevance": 0.15,
            "legal_applicability": 0.20,
            "completeness": 0.15,
            "safety": 0.10,
            "citation_quality": 0.15,
        }
        overall = sum(normalized_scores[name] * weight for name, weight in weights.items())
        claim_checks = result.get("claim_checks") if isinstance(result.get("claim_checks"), list) else []
        unsupported = [
            item for item in claim_checks
            if isinstance(item, dict) and str(item.get("support") or "").lower() in {"unsupported", "contradicted"}
        ]
        if any(str(item.get("severity") or "").lower() == "high" for item in unsupported):
            overall = min(overall, 59.0)
        elif unsupported:
            overall = min(overall, 74.0)
        if not citation_audit["valid"]:
            overall = min(overall, 60.0)
        result.update(
            {
                "status": "completed",
                "model": evaluation_model,
                "scores": normalized_scores,
                "claim_checks": claim_checks,
                "overall_score": round(overall, 1),
                "citation_audit": citation_audit,
                "corpus_freshness": (
                    "已联网核验；本地库仍有过期风险"
                    if (online_verification or {}).get("official_sources")
                    else ("过期风险高" if corpus_cutoff <= "2022-12-31" else "需核验")
                ),
            }
        )
        return result
    except Exception as exc:
        LOGGER.exception("Answer evaluation failed")
        return {
            "status": "failed",
            "model": evaluation_model,
            "error": str(exc),
            "citation_audit": citation_audit,
            "corpus_freshness": "过期风险高" if corpus_cutoff <= "2022-12-31" else "需核验",
        }


def _canonical_intent(value: Any) -> str:
    """Normalize bilingual model labels before routing the request."""
    raw = str(value or "").strip().lower()
    aliases = {
        "法律咨询": "legal_consultation",
        "法律类": "legal_consultation",
        "法律": "legal_consultation",
        "法律问题": "legal_consultation",
        "legal": "legal_consultation",
        "legal_consultation": "legal_consultation",
        "违规请求": "harmful_request",
        "违规类": "harmful_request",
        "有害请求": "harmful_request",
        "犯罪": "harmful_request",
        "harmful": "harmful_request",
        "harmful_request": "harmful_request",
        "闲聊": "non_legal",
        "闲聊类": "non_legal",
        "非法律": "non_legal",
        "non_legal": "non_legal",
        "ambiguous": "ambiguous",
        "模糊": "ambiguous",
        "信息不足": "ambiguous",
        "需要澄清": "ambiguous",
    }
    return aliases.get(raw, raw or "ambiguous")


def _clarification_result(
    analysis: dict[str, Any],
    question: str,
    models: dict[str, str],
    prompt_trace: dict[str, Any],
    started: float,
    analysis_cost: float = 0,
) -> dict[str, Any]:
    """Stop before retrieval when context is genuinely insufficient."""
    follow_up = [str(item).strip() for item in (analysis.get("missing_facts") or []) if str(item).strip()]
    if not follow_up:
        follow_up = ["请补充你要解决的具体事情，以及涉及的主体或事件。"]
    text = "当前信息不足，暂时无法判断你要咨询的具体法律问题。请补充：" + "；".join(follow_up[:3])
    return {
        "mode": "live",
        "intent": "ambiguous",
        "analysis": analysis,
        "retrieval_query": "",
        "use_rewrite": False,
        "retrieval_mode": "未进入检索",
        "rerank_skipped": True,
        "models": models,
        "prompt_trace": prompt_trace,
        "answer": text,
        "answer_structured": {"conclusion": {"text": text, "evidence_ids": [], "online_evidence_ids": []}, "sections": [], "uncertainties": [], "follow_up_questions": follow_up[:3], "risk_notice": ""},
        "retrieval": {"bm25": [], "qwen37": [], "cases_bm25": [], "cases_qwen37": [], "fusion": []},
        "ranking": [],
        "contexts": [],
        "follow_up_questions": follow_up[:3],
        "risk_notice": "",
        "evaluation": {"status": "not_applicable"},
        "online_verification": {"status": "not_applicable", "sources": []},
        "cost": _cost_summary({"analysis": analysis_cost}),
        "timings_sec": {"analysis": round(time.perf_counter() - started, 3), "total": round(time.perf_counter() - started, 3)},
        "corpus_cutoff": os.getenv("CORPUS_CUTOFF", "2022-06-24"),
    }


def _run_query(payload: LegalQueryRequest) -> dict[str, Any]:
    if not runtime.ready:
        raise RuntimeError(runtime.error or "实时检索服务尚未就绪")
    if payload.retrieval_mode not in ALLOWED_MODES:
        raise ValueError("不支持的检索策略")

    analysis_model = _resolve_model(payload.analysis_model, "analysis")
    rerank_model = _resolve_model(payload.rerank_model, "rerank")
    answer_model = _resolve_model(payload.answer_model, "answer")
    evaluation_model = _resolve_model(payload.evaluation_model, "evaluation")
    online_search_model = _resolve_model(payload.online_search_model, "online")
    prompts = _resolve_prompts(payload.prompt_overrides)
    prompt_trace = _prompt_trace(prompts, payload.prompt_overrides, payload.prompt_profile)

    started = time.perf_counter()
    effective_question = _contextual_question(payload)
    analysis = analyze_query(runtime.client, analysis_model, effective_question, prompts["analysis"])
    analysis_usage = _usage_view(analysis.pop("_usage", None))
    after_analysis = time.perf_counter()
    intent = _canonical_intent(analysis.get("intent"))

    # The model must first decide whether the current utterance is complete.
    # Only unresolved incompleteness should stop the retrieval chain; ordinary
    # legal questions may still omit details that can be requested after retrieval.
    needs_context = bool(analysis.get("needs_context"))
    context_resolution = str(analysis.get("context_resolution") or "").strip().lower()
    if intent == "ambiguous" or (needs_context and context_resolution not in {"resolved", "已补全", "resolved_context"}):
        return _clarification_result(
            analysis,
            payload.question,
            {"analysis": analysis_model, "rerank": rerank_model, "answer": answer_model, "evaluation": evaluation_model, "online": online_search_model},
            prompt_trace,
            started,
            _estimated_cost(analysis_model, analysis_usage),
        )

    if intent != "legal_consultation":
        return {
            "mode": "live",
            "intent": intent,
            "analysis": analysis,
            "models": {"analysis": analysis_model, "rerank": rerank_model, "answer": answer_model, "evaluation": evaluation_model, "online": online_search_model},
            "prompt_trace": prompt_trace,
            "answer": (
                "我不能协助实施伤害、违法行为或规避追查。你可以说明希望解决的纠纷，我可以帮助寻找合法、安全的处理方式。"
                if intent == "harmful_request" else str(analysis.get("direct_response") or "你好，我可以帮助你了解法律问题；也可以继续聊你关心的事情。")
            ),
            "retrieval": {"bm25": [], "qwen37": [], "cases_bm25": [], "cases_qwen37": [], "fusion": []},
            "ranking": [],
            "contexts": [],
            "follow_up_questions": [],
            "risk_notice": "",
            "evaluation": {"status": "not_applicable"},
            "online_verification": {"status": "not_applicable", "sources": []},
            "cost": _cost_summary({"analysis": _estimated_cost(analysis_model, analysis_usage)}),
            "timings_sec": {"analysis": round(after_analysis - started, 3), "total": round(after_analysis - started, 3)},
            "corpus_cutoff": os.getenv("CORPUS_CUTOFF", "2022-06-24"),
        }

    rewrite = str(analysis.get("rewrite_query") or "").strip()
    retrieval_query = rewrite if payload.use_rewrite and rewrite else payload.question
    use_dense = payload.retrieval_mode != "仅 BM25"
    bm25_hits, dense_hits, case_bm25_hits, case_dense_hits, retrieval_timings = retrieve_evidence(
        runtime.es, runtime.embedder, retrieval_query, include_dense=use_dense,
        include_bm25=payload.retrieval_mode != "仅 Qwen3.7 向量", include_cases=payload.include_cases,
    )
    after_retrieval = time.perf_counter()

    candidates = rrf_candidates(
        bm25_hits,
        dense_hits,
        case_dense_hits,
        int(os.getenv("RRF_K", "60")),
        case_bm25=case_bm25_hits,
    )
    candidates = candidates[: int(os.getenv("RRF_TOPK", "20"))]
    after_fusion = time.perf_counter()

    if payload.use_online_search:
        online_verification = _online_legal_search(payload.question, analysis, online_search_model, prompts["online"])
    else:
        online_verification = {"status": "disabled", "model": online_search_model, "sources": [], "official_sources": []}
    after_online = time.perf_counter()
    online_for_answer = {
        "status": online_verification.get("status"),
        "summary": online_verification.get("summary", "") if online_verification.get("official_sources") else "未完成权威核验。",
        "official_sources": online_verification.get("official_sources") or [],
    }

    if payload.skip_rerank:
        rerank_result = {"candidates": [], "ranking": []}
        rerank_usage = _usage_view(None)
        ranking = _direct_rrf_ranking(candidates, payload.final_top_k)
    else:
        rerank_result = rerank_candidates(
            runtime.client,
            rerank_model,
            effective_question,
            analysis,
            candidates,
            prompts["rerank"],
        )
        rerank_usage = _usage_view(rerank_result.pop("_usage", None))
        ranking = _merge_ranking(rerank_result, payload.final_top_k)
    after_rerank = time.perf_counter()

    answer_result = generate_answer(
        runtime.client,
        answer_model,
        effective_question,
        analysis,
        ranking,
        online_for_answer,
        prompts["answer"],
    )
    answer_usage = _usage_view(answer_result.pop("_usage", None))
    answer_structured, rendered_answer = _normalize_answer_result(
        answer_result,
        {int(item["id"]) for item in ranking},
        {int(item["id"]) for item in online_for_answer.get("official_sources") or []},
    )
    after_answer = time.perf_counter()
    corpus_cutoff = os.getenv("CORPUS_CUTOFF", "2022-06-24")
    if payload.run_evaluation:
        evaluation = _evaluate_answer(
            effective_question,
            rendered_answer,
            analysis,
            ranking,
            len(rerank_result.get("candidates") or candidates),
            corpus_cutoff,
            evaluation_model,
            online_for_answer,
            prompts["evaluation"],
        )
    else:
        evaluation = {"status": "disabled"}
    evaluation_usage = _usage_view(evaluation.pop("_usage", None))
    after_evaluation = time.perf_counter()

    cost_stages = {
        "analysis": _estimated_cost(analysis_model, analysis_usage),
        "rerank": _estimated_cost(rerank_model, rerank_usage),
        "answer": _estimated_cost(answer_model, answer_usage),
    }
    if payload.use_online_search and online_verification.get("usage"):
        cost_stages["online_search"] = _estimated_cost(
            online_search_model,
            _usage_view(online_verification.get("usage")),
            int(online_verification.get("search_requests") or 0),
        )
    if payload.run_evaluation:
        cost_stages["evaluation"] = _estimated_cost(evaluation_model, evaluation_usage)

    return {
        "mode": "live",
        "intent": intent,
        "analysis": analysis,
        "retrieval_query": retrieval_query,
        "use_rewrite": payload.use_rewrite,
        "retrieval_mode": payload.retrieval_mode,
        "include_cases": payload.include_cases,
        "rerank_skipped": payload.skip_rerank,
        "models": {"analysis": analysis_model, "rerank": rerank_model, "answer": answer_model, "evaluation": evaluation_model, "online": online_search_model},
        "prompt_trace": prompt_trace,
        "retrieval": {
            "bm25": [_hit_view(hit, rank) for rank, hit in enumerate(bm25_hits, start=1)],
            "qwen37": [_hit_view(hit, rank) for rank, hit in enumerate(dense_hits, start=1)],
            "cases_bm25": [_hit_view(hit, rank) for rank, hit in enumerate(case_bm25_hits, start=1)],
            "cases_qwen37": [_hit_view(hit, rank) for rank, hit in enumerate(case_dense_hits, start=1)],
            "fusion": [
                {
                    "rank": rank,
                    "chunk_id": item.get("chunk_id", ""),
                    "title": item.get("title", ""),
                    "text": item.get("text", ""),
                    "corpus": item.get("corpus", "law"),
                    "ranks": item.get("ranks", {}),
                    "score": float(item.get("rrf") or 0.0),
                }
                for rank, item in enumerate(candidates, start=1)
            ],
        },
        "ranking": ranking,
        "contexts": [
            {"id": item["id"], "title": item["title"], "text": item["text"], "corpus": item.get("corpus", "law")}
            for item in ranking
        ],
        "answer": rendered_answer,
        "answer_structured": answer_structured,
        "follow_up_questions": answer_result.get("follow_up_questions") or [],
        "risk_notice": answer_result.get("risk_notice", ""),
        "evaluation": evaluation,
        "online_verification": online_verification,
        "cost": _cost_summary(cost_stages),
        "timings_sec": {
            "analysis": round(after_analysis - started, 3),
            "retrieval": round(after_retrieval - after_analysis, 3),
            "retrieval_breakdown": retrieval_timings,
            "fusion": round(after_fusion - after_retrieval, 3),
            "online_search": round(after_online - after_fusion, 3),
            "rerank": round(after_rerank - after_online, 3),
            "answer": round(after_answer - after_rerank, 3),
            "rerank_and_answer": round(after_answer - after_online, 3),
            "evaluation": round(after_evaluation - after_answer, 3),
            "total": round(after_evaluation - started, 3),
        },
        "corpus_cutoff": corpus_cutoff,
    }


def _pipeline_request(question: str, config: ExperimentPipelineConfig) -> LegalQueryRequest:
    values = config.model_dump(exclude={"label"})
    return LegalQueryRequest(question=question, **values)


def _run_bad_cases(result: dict[str, Any], use_online_search: bool) -> list[str]:
    if _canonical_intent(result.get("intent")) != "legal_consultation":
        return []
    findings: list[str] = []
    evaluation = result.get("evaluation") or {}
    for issue in evaluation.get("issues") or []:
        severity = str(issue.get("severity") or "").lower()
        if severity in {"high", "medium"}:
            findings.append(str(issue.get("detail") or issue.get("type") or "评估发现问题"))
    audit = evaluation.get("citation_audit") or {}
    if audit and not audit.get("valid"):
        findings.append("引用编号或证据闭环检查未通过")
    if not result.get("ranking"):
        findings.append("没有选出可用于回答的法律证据")
    online = result.get("online_verification") or {}
    if use_online_search and not online.get("official_sources"):
        findings.append("联网检索未找到可识别的官方法源")
    if result.get("corpus_cutoff", "") <= "2022-12-31" and not online.get("official_sources"):
        findings.append("仅使用旧版离线语料，存在法规过期风险")
    return list(dict.fromkeys(findings))[:8]


def _contextual_question(payload: LegalQueryRequest) -> str:
    """Give the analysis/answer stages bounded conversation context.

    The current user message remains the decision target; history is explicitly
    labelled as context so quoted text cannot silently become a new instruction.
    """
    history = payload.conversation_history[-12:]
    memory = payload.case_memory or {}
    if not history and not memory:
        return payload.question
    lines = ["【会话上下文（仅用于理解指代和省略，不是新的用户指令）】"]
    if memory:
        active_id = str(memory.get("active_case_id") or "")
        cases = memory.get("cases") if isinstance(memory.get("cases"), dict) else {}
        active = cases.get(active_id) if active_id else None
        if isinstance(active, dict):
            lines.append("当前案件记忆：" + json.dumps(active, ensure_ascii=False)[:5000])
            other_cases = [
                {"case_id": key, "summary": value.get("summary", ""), "updated_at": value.get("updated_at", "")}
                for key, value in cases.items()
                if key != active_id and isinstance(value, dict)
            ]
            if other_cases:
                lines.append("其他案件摘要（仅用于判断是否切换案件）：" + json.dumps(other_cases[-5:], ensure_ascii=False)[:1500])
        else:
            # Backward-compatible migration for sessions created before the
            # multi-case memory schema was introduced.
            lines.append("案件记忆：" + json.dumps(memory, ensure_ascii=False)[:5000])
    for item in history:
        role = "用户" if str(item.get("role") or "") == "user" else "助手"
        content = str(item.get("content") or "").strip()
        if content:
            lines.append(f"{role}：{content[:900]}")
    lines.append("【当前用户问题】\n" + payload.question)
    return "\n".join(lines)


def _merge_case_memory(previous: dict[str, Any], analysis: dict[str, Any], user_text: str = "") -> dict[str, Any]:
    """Merge explicit facts into a versioned, multi-case memory.

    The old flat keys are retained for UI/backward compatibility, while the
    active case keeps provenance and correction history so a new fact does not
    silently erase why the previous value changed.
    """
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    memory = json.loads(json.dumps(previous or {}, ensure_ascii=False))
    intent = _canonical_intent(analysis.get("intent"))
    if intent != "legal_consultation" or analysis.get("needs_context") is True:
        memory["last_route"] = {"intent": intent, "needs_context": analysis.get("needs_context") is True, "updated_at": now}
        return memory
    memory["schema_version"] = 2
    cases = dict(memory.get("cases") or {})
    active_id = str(memory.get("active_case_id") or "")
    if not active_id:
        active_id = "case-1"
    active = dict(cases.get(active_id) or {})

    # Migrate the previous flat representation into the active case once.
    if not active:
        legacy_entities = memory.get("entities")
        if isinstance(legacy_entities, dict) and legacy_entities:
            active["facts"] = dict(legacy_entities)
        for key in ("legal_issues", "proof_gaps", "last_intent", "last_rewrite_query"):
            if memory.get(key):
                active[key] = memory[key]
        active["case_id"] = active_id

    action = str(analysis.get("case_action") or "continue").strip().lower()
    target_id = str(analysis.get("target_case_id") or "")
    if target_id in cases:
        active_id = target_id
        active = dict(cases[target_id])
    elif action in {"new_case", "switch", "new", "切换案件", "新案件"}:
        if active.get("facts") or active.get("legal_issues"):
            cases[active_id] = active
        next_number = 1
        while f"case-{next_number}" in cases:
            next_number += 1
        active_id = f"case-{next_number}"
        active = {"case_id": active_id, "facts": {}, "legal_issues": [], "proof_gaps": [], "corrections": []}

    facts = dict(active.get("facts") or active.get("entities") or {})
    provenance = dict(active.get("fact_provenance") or {})
    conflicts = list(memory.get("conflicts") or [])
    correction_history = list(active.get("corrections") or [])
    corrections = analysis.get("corrections") if isinstance(analysis.get("corrections"), dict) else {}
    entities = analysis.get("entities")
    if isinstance(entities, dict):
        for key, value in entities.items():
            key = str(key)
            if key in corrections:
                continue
            if value in (None, "", [], {}):
                continue
            old_value = facts.get(key)
            if old_value not in (None, "", [], {}) and old_value != value:
                conflicts.append({"case_id": active_id, "field": key, "old": old_value, "new": value, "resolved_at": now})
            facts[key] = value
            provenance[key] = {"source": "model_extracted_user_statement", "user_text": user_text[:1200], "updated_at": now}

    if isinstance(corrections, dict):
        for key, value in corrections.items():
            key = str(key)
            if value in (None, "", [], {}):
                continue
            old_value = facts.get(key)
            if old_value not in (None, "", [], {}) and old_value != value:
                entry = {"case_id": active_id, "field": key, "old": old_value, "new": value, "resolved_at": now, "type": "user_correction"}
                conflicts.append(entry)
                correction_history.append(entry)
            facts[key] = value
            provenance[key] = {"source": "user_correction", "user_text": user_text[:1200], "updated_at": now}

    active["case_id"] = active_id
    active["facts"] = facts
    active["entities"] = facts  # compatibility with the existing memory panel
    active["fact_provenance"] = provenance
    active["corrections"] = correction_history[-20:]
    for key in ("legal_issues", "proof_gaps", "missing_facts"):
        values = analysis.get(key)
        if isinstance(values, list):
            active[key] = [str(item) for item in values if str(item).strip()][:20]
    active["last_intent"] = _canonical_intent(analysis.get("intent"))
    active["is_complete"] = bool(analysis.get("is_complete", True))
    active["needs_context"] = bool(analysis.get("needs_context", False))
    active["context_resolution"] = str(analysis.get("context_resolution") or "not_needed")
    if analysis.get("rewrite_query"):
        active["last_rewrite_query"] = str(analysis["rewrite_query"])[:600]
    active["summary"] = str(analysis.get("case_summary") or "; ".join(f"{key}：{value}" for key, value in facts.items()) or active.get("summary") or "")[:600]
    active["updated_at"] = now
    cases[active_id] = active

    memory["active_case_id"] = active_id
    memory["cases"] = cases
    memory["entities"] = facts
    memory["legal_issues"] = active.get("legal_issues", [])
    memory["proof_gaps"] = active.get("proof_gaps", [])
    memory["missing_facts"] = active.get("missing_facts", [])
    memory["last_intent"] = active["last_intent"]
    memory["last_rewrite_query"] = active.get("last_rewrite_query", "")
    memory["conflicts"] = conflicts[-30:]
    memory["updated_at"] = now
    return memory


def _run_comparison(payload: CompareRequest) -> dict[str, Any]:
    configs = [payload.pipeline_a, payload.pipeline_b]
    runs = [_run_query(_pipeline_request(payload.question, config)) for config in configs]
    comparable_fields = (
        "use_rewrite",
        "include_cases",
        "run_evaluation",
        "retrieval_mode",
        "final_top_k",
        "use_online_search",
        "analysis_model",
        "rerank_model",
        "answer_model",
        "evaluation_model",
        "online_search_model",
        "prompt_overrides",
    )
    changed_fields = [
        field
        for field in comparable_fields
        if getattr(configs[0], field) != getattr(configs[1], field)
    ]
    score_a = (runs[0].get("evaluation") or {}).get("overall_score")
    score_b = (runs[1].get("evaluation") or {}).get("overall_score")
    quality_delta = round(float(score_b) - float(score_a), 2) if score_a is not None and score_b is not None else None
    latency_a = float((runs[0].get("timings_sec") or {}).get("total") or 0)
    latency_b = float((runs[1].get("timings_sec") or {}).get("total") or 0)
    cost_a = runs[0].get("cost") or {}
    cost_b = runs[1].get("cost") or {}
    cost_delta = None
    if cost_a.get("fully_priced") and cost_b.get("fully_priced"):
        cost_delta = round(float(cost_b.get("estimated_cny") or 0) - float(cost_a.get("estimated_cny") or 0), 6)
    field_names = {
        "use_rewrite": "Query 改写",
        "include_cases": "案例检索",
        "run_evaluation": "质量评估",
        "retrieval_mode": "检索策略",
        "final_top_k": "上下文数量",
        "use_online_search": "联网核验",
        "analysis_model": "分析模型",
        "rerank_model": "重排模型",
        "answer_model": "回答模型",
        "evaluation_model": "评估模型",
        "online_search_model": "联网模型",
        "prompt_overrides": "提示词",
    }
    prompt_changed_stages = [
        stage for stage in PROMPT_STAGES
        if (configs[0].prompt_overrides or {}).get(stage, DEFAULT_PROMPTS[stage]).strip()
        != (configs[1].prompt_overrides or {}).get(stage, DEFAULT_PROMPTS[stage]).strip()
    ]
    if "evaluation" in prompt_changed_stages:
        attribution = "两套方案使用了不同的评估提示词，质量分口径已变化，不能直接归因于回答质量；测试其他环节时请固定评估提示词。"
    elif changed_fields == ["prompt_overrides"] and len(prompt_changed_stages) == 1 and quality_delta is not None:
        stage_label = PROMPT_STAGE_META[prompt_changed_stages[0]]["label"]
        direction = "提升" if quality_delta > 0 else ("下降" if quality_delta < 0 else "不变")
        attribution = f"本次只改变了{stage_label}提示词，方案B质量分{direction} {abs(quality_delta):.2f} 分。"
    elif len(changed_fields) == 1 and quality_delta is not None:
        direction = "提升" if quality_delta > 0 else ("下降" if quality_delta < 0 else "不变")
        attribution = f"本次只改变了{field_names[changed_fields[0]]}，方案B质量分{direction} {abs(quality_delta):.2f} 分。"
    elif changed_fields:
        attribution = "本次同时改变多个环节，不能把质量变化归因于单一因素；建议下一轮只改一个变量。"
    else:
        attribution = "两套配置相同，本轮用于观察模型输出波动，不能作为组件提升结论。"
    return {
        "question": payload.question,
        "execution_mode": "sequential",
        "runs": [
            {"label": configs[index].label, "config": configs[index].model_dump(), "result": run, "bad_cases": _run_bad_cases(run, configs[index].use_online_search)}
            for index, run in enumerate(runs)
        ],
        "comparison": {
            "changed_fields": changed_fields,
            "changed_labels": [field_names[field] for field in changed_fields],
            "changed_prompt_stages": prompt_changed_stages,
            "quality_delta_b_minus_a": quality_delta,
            "latency_delta_b_minus_a": round(latency_b - latency_a, 3),
            "cost_delta_b_minus_a": cost_delta,
            "attribution": attribution,
            "quality_winner": configs[1].label if quality_delta is not None and quality_delta > 0 else (configs[0].label if quality_delta is not None and quality_delta < 0 else "持平/不可比"),
            "speed_winner": configs[0].label if latency_a < latency_b else (configs[1].label if latency_b < latency_a else "持平"),
            "cost_winner": (
                configs[0].label if cost_delta is not None and cost_delta > 0 else (configs[1].label if cost_delta is not None and cost_delta < 0 else "持平/不可比")
            ),
        },
    }


def _benchmark_run(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    expected_intent = str(case.get("expected_intent") or "法律咨询")
    expected_intent_canonical = _canonical_intent(expected_intent)
    actual_intent = _canonical_intent(result.get("intent"))
    intent_match = actual_intent == expected_intent_canonical
    expected_laws = [str(value).strip() for value in case.get("expected_laws") or [] if str(value).strip()]
    retrieved_titles = [str(item.get("title") or "") for item in result.get("ranking") or []]
    matched_laws = [
        law for law in expected_laws
        if any(law.lower() in title.lower() or title.lower() in law.lower() for title in retrieved_titles if title)
    ]
    law_recall = len(matched_laws) / len(expected_laws) if expected_laws else (1.0 if actual_intent != "legal_consultation" else None)
    evaluation = result.get("evaluation") or {}
    score = evaluation.get("overall_score")
    audit = evaluation.get("citation_audit") or {}
    citation_valid = bool(audit.get("valid")) if actual_intent == "legal_consultation" else True
    threshold = float(case.get("quality_threshold") or 0)
    if expected_intent_canonical != "legal_consultation":
        passed = intent_match
    else:
        passed = bool(intent_match and score is not None and float(score) >= threshold and (law_recall is None or law_recall >= 0.5) and citation_valid)
    failures: list[str] = []
    if not intent_match:
        failures.append(f"意图错误：期望 {expected_intent}，实际 {actual_intent or '空'}")
    if expected_laws and (law_recall or 0) < 1:
        missed = [law for law in expected_laws if law not in matched_laws]
        failures.append("未命中期望法源：" + "、".join(missed))
    if score is None and expected_intent_canonical == "legal_consultation":
        failures.append("没有可用的自动质量分")
    elif score is not None and float(score) < threshold:
        failures.append(f"质量分 {float(score):.1f} 低于阈值 {threshold:.0f}")
    if not citation_valid:
        failures.append("引用闭环未通过")
    return {
        "passed": passed,
        "expected_intent": expected_intent_canonical,
        "actual_intent": actual_intent,
        "intent_match": intent_match,
        "expected_laws": expected_laws,
        "matched_laws": matched_laws,
        "law_recall": round(law_recall, 4) if law_recall is not None else None,
        "quality_threshold": threshold,
        "quality_score": float(score) if score is not None else None,
        "citation_valid": citation_valid,
        "failures": failures,
    }


def _augment_case_comparison(case: dict[str, Any], comparison: dict[str, Any]) -> dict[str, Any]:
    comparison["benchmark_case"] = {
        key: case.get(key)
        for key in ("id", "category", "difficulty", "tags", "expected_intent", "expected_laws", "quality_threshold", "notes")
    }
    for run in comparison.get("runs") or []:
        benchmark = _benchmark_run(case, run.get("result") or {})
        run["benchmark"] = benchmark
        run["bad_cases"] = list(dict.fromkeys((run.get("bad_cases") or []) + benchmark["failures"]))[:12]
    return comparison


def _percentile_95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


def _aggregate_pipeline(case_results: list[dict[str, Any]], run_index: int) -> dict[str, Any]:
    runs = [item["result"]["runs"][run_index] for item in case_results if (item.get("result") or {}).get("runs")]
    quality = [float(run["benchmark"]["quality_score"]) for run in runs if run.get("benchmark", {}).get("quality_score") is not None]
    law_recalls = [float(run["benchmark"]["law_recall"]) for run in runs if run.get("benchmark", {}).get("law_recall") is not None]
    latencies = [float((run.get("result", {}).get("timings_sec") or {}).get("total") or 0) for run in runs]
    costs = [
        float((run.get("result", {}).get("cost") or {}).get("estimated_cny") or 0)
        for run in runs if (run.get("result", {}).get("cost") or {}).get("fully_priced")
    ]
    passed = sum(1 for run in runs if run.get("benchmark", {}).get("passed"))
    intent_matches = sum(1 for run in runs if run.get("benchmark", {}).get("intent_match"))
    citation_runs = [run for run in runs if _canonical_intent((run.get("benchmark") or {}).get("expected_intent")) == "legal_consultation"]
    citation_valid = sum(1 for run in citation_runs if run.get("benchmark", {}).get("citation_valid"))
    bad_cases = [
        {
            "case_id": item["case_id"],
            "question": item["question"],
            "category": item["category"],
            "failures": (item["result"]["runs"][run_index].get("benchmark") or {}).get("failures") or item["result"]["runs"][run_index].get("bad_cases") or [],
            "score": (item["result"]["runs"][run_index].get("benchmark") or {}).get("quality_score"),
        }
        for item in case_results
        if (item.get("result") or {}).get("runs") and not item["result"]["runs"][run_index].get("benchmark", {}).get("passed")
    ]
    count = len(runs)
    return {
        "label": runs[0].get("label") if runs else ("方案 A" if run_index == 0 else "方案 B"),
        "case_count": count,
        "average_quality": round(statistics.fmean(quality), 2) if quality else None,
        "pass_rate": round(passed / count, 4) if count else 0,
        "passed_cases": passed,
        "intent_accuracy": round(intent_matches / count, 4) if count else 0,
        "average_expected_law_recall": round(statistics.fmean(law_recalls), 4) if law_recalls else None,
        "citation_valid_rate": round(citation_valid / len(citation_runs), 4) if citation_runs else None,
        "average_latency_sec": round(statistics.fmean(latencies), 3) if latencies else 0,
        "p95_latency_sec": round(_percentile_95(latencies), 3),
        "total_cost_cny": round(sum(costs), 6),
        "average_cost_cny": round(statistics.fmean(costs), 6) if costs else None,
        "cost_per_passed_case_cny": round(sum(costs) / passed, 6) if costs and passed else None,
        "bad_case_count": len(bad_cases),
        "bad_cases": bad_cases,
    }


def _batch_summary(experiment: dict[str, Any]) -> dict[str, Any]:
    case_results = [item for item in experiment.get("case_results") or [] if not (item.get("result") or {}).get("error")]
    pipeline_a = _aggregate_pipeline(case_results, 0)
    pipeline_b = _aggregate_pipeline(case_results, 1)
    quality_delta = None
    if pipeline_a["average_quality"] is not None and pipeline_b["average_quality"] is not None:
        quality_delta = round(pipeline_b["average_quality"] - pipeline_a["average_quality"], 2)
    changed_fields = []
    for field in ("use_rewrite", "include_cases", "retrieval_mode", "final_top_k", "use_online_search", "analysis_model", "rerank_model", "answer_model", "evaluation_model", "online_search_model", "run_evaluation", "prompt_overrides"):
        if experiment["pipeline_a"].get(field) != experiment["pipeline_b"].get(field):
            changed_fields.append(field)
    if len(changed_fields) == 1:
        attribution = f"本批次只改变 {changed_fields[0]}；平均质量变化 {quality_delta:+.2f} 分。" if quality_delta is not None else f"本批次只改变 {changed_fields[0]}，但缺少可比质量分。"
    elif changed_fields:
        attribution = "本批次同时改变多个变量，只能比较整体方案，不能归因到单一环节。"
    else:
        attribution = "两套配置相同，本批次用于测量输出波动，不代表组件增益。"
    return {
        "pipeline_a": pipeline_a,
        "pipeline_b": pipeline_b,
        "delta_b_minus_a": {
            "average_quality": quality_delta,
            "pass_rate": round(pipeline_b["pass_rate"] - pipeline_a["pass_rate"], 4),
            "average_expected_law_recall": (
                round(pipeline_b["average_expected_law_recall"] - pipeline_a["average_expected_law_recall"], 4)
                if pipeline_a["average_expected_law_recall"] is not None and pipeline_b["average_expected_law_recall"] is not None else None
            ),
            "average_latency_sec": round(pipeline_b["average_latency_sec"] - pipeline_a["average_latency_sec"], 3),
            "total_cost_cny": round(pipeline_b["total_cost_cny"] - pipeline_a["total_cost_cny"], 6),
        },
        "changed_fields": changed_fields,
        "attribution": attribution,
        "completed_cases": len(case_results),
        "failed_cases": len(experiment.get("case_results") or []) - len(case_results),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "corpus_cutoff": os.getenv("CORPUS_CUTOFF", "2022-06-24"),
        "pricing_date": MODEL_PRICING_SNAPSHOT_DATE,
    }


async def _execute_batch_experiment(experiment_id: str) -> None:
    try:
        experiment_store.set_experiment_status(experiment_id, "running")
        experiment = await asyncio.to_thread(experiment_store.get_experiment, experiment_id, False)
        if not experiment:
            return
        config_a = ExperimentPipelineConfig(**experiment["pipeline_a"])
        config_b = ExperimentPipelineConfig(**experiment["pipeline_b"])
        for case in experiment["dataset_snapshot"]:
            try:
                payload = CompareRequest(question=case["question"], pipeline_a=config_a, pipeline_b=config_b)
                async with runtime.semaphore:
                    result = await asyncio.to_thread(_run_comparison, payload)
                result = _augment_case_comparison(case, result)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOGGER.exception("Batch case failed: experiment=%s case=%s", experiment_id, case.get("id"))
                result = {"error": str(exc), "runs": []}
            await asyncio.to_thread(experiment_store.add_case_result, experiment_id, case, result)
        completed = await asyncio.to_thread(experiment_store.get_experiment, experiment_id, True)
        if completed:
            summary = _batch_summary(completed)
            await asyncio.to_thread(experiment_store.save_summary, experiment_id, summary)
        experiment_store.set_experiment_status(experiment_id, "completed")
    except asyncio.CancelledError:
        experiment_store.set_experiment_status(experiment_id, "cancelled", error="用户取消了批量实验。")
    except Exception as exc:
        LOGGER.exception("Batch experiment failed: %s", experiment_id)
        experiment_store.set_experiment_status(experiment_id, "failed", error=str(exc))
    finally:
        batch_tasks.pop(experiment_id, None)


@app.get("/api/healthz")
async def healthz() -> JSONResponse:
    try:
        es_ok = bool(runtime.es is not None and await asyncio.to_thread(runtime.es.ping))
    except Exception:
        es_ok = False
    status = 200 if runtime.ready else 503
    return JSONResponse(
        status_code=status,
        content={
            "ready": status == 200,
            "elasticsearch": es_ok,
            "lexical_backend": "elasticsearch" if es_ok else ("local_bm25" if runtime.embedder is not None else None),
            "embedding_model_loaded": runtime.embedder is not None,
            "embedding_model": runtime.embedder.model if runtime.embedder is not None else None,
            "law_dense_index_rows": runtime.embedder.law_index.index.ntotal if runtime.embedder is not None else 0,
            "case_dense_index_rows": runtime.embedder.case_index.index.ntotal if runtime.embedder is not None else 0,
            "provider": runtime.provider or None,
            "analysis_model": runtime.analysis_model or None,
            "rerank_model": runtime.rerank_model or None,
            "answer_model": runtime.answer_model or None,
            "evaluation_model": runtime.evaluation_model or None,
            "online_search_model": runtime.online_search_model or None,
            "online_verification": "dashscope_search_info+national_law_database_exact_match+http_fetch",
            "model_options": _model_options(),
            "pricing_date": MODEL_PRICING_SNAPSHOT_DATE,
            "corpus_cutoff": os.getenv("CORPUS_CUTOFF", "2022-06-24"),
            "error": runtime.error or None,
        },
    )


@app.get("/api/prompts")
async def prompt_catalog() -> dict[str, Any]:
    return {
        "stages": [
            {
                "id": stage,
                **PROMPT_STAGE_META[stage],
                "default_prompt": DEFAULT_PROMPTS[stage],
                "version": hashlib.sha256(DEFAULT_PROMPTS[stage].encode("utf-8")).hexdigest()[:12],
            }
            for stage in PROMPT_STAGES
        ]
    }


@app.post("/api/legal-query")
async def legal_query(payload: LegalQueryRequest, request: Request) -> dict[str, Any]:
    if not runtime.ready:
        raise HTTPException(status_code=503, detail=runtime.error or "实时检索服务尚未就绪")
    async with runtime.semaphore:
        try:
            return await asyncio.to_thread(_run_query, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            LOGGER.exception("Live query failed: client=%s", request.client.host if request.client else "unknown")
            raise HTTPException(status_code=502, detail=f"实时链路执行失败：{exc}") from exc


@app.post("/api/chat/sessions")
async def create_chat_session(payload: ChatSessionCreateRequest) -> dict[str, Any]:
    return await asyncio.to_thread(chat_store.create_session, payload.title)


@app.get("/api/chat/sessions")
async def list_chat_sessions(limit: int = 30) -> dict[str, Any]:
    sessions = await asyncio.to_thread(chat_store.list_sessions, limit)
    return {"sessions": sessions, "count": len(sessions)}


@app.get("/api/chat/sessions/{session_id}")
async def get_chat_session(session_id: str) -> dict[str, Any]:
    session = await asyncio.to_thread(chat_store.get_session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return session


@app.delete("/api/chat/sessions/{session_id}")
async def delete_chat_session(session_id: str) -> dict[str, Any]:
    lock = chat_locks.setdefault(session_id, asyncio.Lock())
    async with lock:
        deleted = await asyncio.to_thread(chat_store.delete_session, session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"deleted": True, "id": session_id}


@app.post("/api/chat/sessions/{session_id}/messages")
async def send_chat_message(session_id: str, payload: ChatMessageRequest, request: Request) -> dict[str, Any]:
    # Serialize context read + generation + commit. Taking the snapshot before
    # acquiring the semaphore allowed simultaneous turns to overwrite memory.
    lock = chat_locks.setdefault(session_id, asyncio.Lock())
    async with lock:
        async with runtime.semaphore:
            return await _send_chat_message(session_id, payload, request)


async def _send_chat_message(session_id: str, payload: ChatMessageRequest, request: Request) -> dict[str, Any]:
    if not runtime.ready:
        raise HTTPException(status_code=503, detail=runtime.error or "实时检索服务尚未就绪")
    session = await asyncio.to_thread(chat_store.get_session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    previous_messages = [
        {"role": item.get("role", ""), "content": item.get("content", "")}
        for item in session.get("messages", [])[-12:]
    ]
    title = payload.message.strip()[:40] if not previous_messages else None
    fast_analysis_model = payload.analysis_model or ("qwen3.8-flash" if runtime.provider == "dashscope" else None)
    fast_answer_model = payload.answer_model or ("qwen3.8-flash" if runtime.provider == "dashscope" else None)
    query = LegalQueryRequest(
        question=payload.message.strip(),
        use_rewrite=payload.use_rewrite,
        include_cases=payload.include_cases,
        retrieval_mode=payload.retrieval_mode,
        final_top_k=payload.final_top_k,
        run_evaluation=payload.run_evaluation,
        use_online_search=payload.use_online_search,
        # The chat path is latency-sensitive; callers can still explicitly
        # override these models, while DashScope defaults to the Flash tier.
        analysis_model=fast_analysis_model,
        rerank_model=payload.rerank_model,
        answer_model=fast_answer_model,
        evaluation_model=payload.evaluation_model,
        online_search_model=payload.online_search_model,
        prompt_profile=payload.prompt_profile,
        prompt_overrides=payload.prompt_overrides,
        conversation_history=previous_messages,
        case_memory=session.get("case_memory") or {},
        skip_rerank=payload.skip_rerank,
    )
    try:
        result = await asyncio.to_thread(_run_query, query)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Chat message failed: session=%s client=%s", session_id, request.client.host if request.client else "unknown")
        raise HTTPException(status_code=502, detail=f"多轮问答失败：{exc}") from exc

    analysis = result.get("analysis") or {}
    case_memory = _merge_case_memory(session.get("case_memory") or {}, analysis, payload.message)
    compact_result = {
        "intent": result.get("intent"),
        "analysis": analysis,
        "contexts": result.get("contexts") or [],
        "online_sources": (result.get("online_verification") or {}).get("official_sources") or [],
        "follow_up_questions": result.get("follow_up_questions") or [],
        "risk_notice": result.get("risk_notice") or "",
        "timings_sec": result.get("timings_sec") or {},
    }
    assistant_message = await asyncio.to_thread(
        chat_store.append_turn,
        session_id,
        payload.message.strip(),
        result.get("answer") or "未生成回答。",
        compact_result,
        case_memory,
        title,
    )
    updated_session = await asyncio.to_thread(chat_store.get_session, session_id)
    return {"session": updated_session, "message": assistant_message, "result": result, "case_memory": case_memory}


@app.post("/api/compare")
async def compare_pipelines(payload: CompareRequest, request: Request) -> dict[str, Any]:
    if not runtime.ready:
        raise HTTPException(status_code=503, detail=runtime.error or "实时检索服务尚未就绪")
    async with runtime.semaphore:
        try:
            return await asyncio.to_thread(_run_comparison, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            LOGGER.exception("Comparison failed: client=%s", request.client.host if request.client else "unknown")
            raise HTTPException(status_code=502, detail=f"对照实验执行失败：{exc}") from exc


@app.get("/api/evaluation-cases")
async def evaluation_cases() -> dict[str, Any]:
    cases = await asyncio.to_thread(experiment_store.list_cases)
    return {"cases": cases, "count": len(cases)}


@app.post("/api/evaluation-cases/import")
async def import_evaluation_cases(payload: EvaluationCaseImportRequest) -> dict[str, Any]:
    result = await asyncio.to_thread(
        experiment_store.upsert_cases,
        [item.model_dump() for item in payload.cases],
    )
    cases = await asyncio.to_thread(experiment_store.list_cases)
    return {**result, "cases": cases, "count": len(cases)}


@app.post("/api/batch-experiments", status_code=202)
async def start_batch_experiment(payload: BatchExperimentRequest) -> dict[str, Any]:
    if not runtime.ready:
        raise HTTPException(status_code=503, detail=runtime.error or "实时检索服务尚未就绪")
    all_cases = await asyncio.to_thread(experiment_store.list_cases)
    by_id = {item["id"]: item for item in all_cases}
    missing = [case_id for case_id in payload.case_ids if case_id not in by_id]
    if missing:
        raise HTTPException(status_code=400, detail="评测样本不存在：" + "、".join(missing[:5]))
    selected = [by_id[case_id] for case_id in dict.fromkeys(payload.case_ids)]
    experiment_id = uuid.uuid4().hex[:16]
    await asyncio.to_thread(
        experiment_store.create_experiment,
        experiment_id,
        payload.name,
        selected,
        payload.pipeline_a.model_dump(),
        payload.pipeline_b.model_dump(),
    )
    task = asyncio.create_task(_execute_batch_experiment(experiment_id), name=f"batch-experiment-{experiment_id}")
    batch_tasks[experiment_id] = task
    return {"id": experiment_id, "status": "queued", "total_cases": len(selected)}


@app.get("/api/batch-experiments")
async def list_batch_experiments(limit: int = 30) -> dict[str, Any]:
    safe_limit = max(1, min(100, limit))
    experiments = await asyncio.to_thread(experiment_store.list_experiments, safe_limit)
    return {"experiments": experiments, "count": len(experiments)}


@app.get("/api/batch-experiments/{experiment_id}")
async def get_batch_experiment(experiment_id: str) -> dict[str, Any]:
    experiment = await asyncio.to_thread(experiment_store.get_experiment, experiment_id, True)
    if experiment is None:
        raise HTTPException(status_code=404, detail="实验记录不存在")
    return experiment


@app.post("/api/batch-experiments/{experiment_id}/cancel")
async def cancel_batch_experiment(experiment_id: str) -> dict[str, Any]:
    experiment = await asyncio.to_thread(experiment_store.get_experiment, experiment_id, False)
    if experiment is None:
        raise HTTPException(status_code=404, detail="实验记录不存在")
    if experiment["status"] not in {"queued", "running", "cancelling"}:
        return {"id": experiment_id, "status": experiment["status"]}
    await asyncio.to_thread(experiment_store.set_experiment_status, experiment_id, "cancelling")
    task = batch_tasks.get(experiment_id)
    if task:
        task.cancel()
    return {"id": experiment_id, "status": "cancelling"}


app.mount("/", StaticFiles(directory=DEMO_DIR, html=True), name="demo")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("DEMO_HOST", "127.0.0.1"), port=int(os.getenv("DEMO_PORT", "8008")))
