"""SQLite persistence for evaluation cases and batch experiment runs."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT / "output" / "evaluation" / "experiments.sqlite3"


SEED_CASES = [
    {
        "id": "personal-injury-boyfriend",
        "question": "我男朋友把我打了，我该如何索赔？",
        "category": "人身损害",
        "difficulty": "高",
        "tags": ["关系边界", "侵权", "安全风险"],
        "expected_intent": "法律咨询",
        "expected_laws": ["民法典", "治安管理处罚法", "刑法"],
        "quality_threshold": 80,
        "notes": "未说明共同生活时，不应直接假定属于家庭成员。",
    },
    {
        "id": "domestic-violence-cohabitation",
        "question": "我和男友共同生活两年，他多次殴打我，我可以申请人身安全保护令吗？",
        "category": "家庭暴力",
        "difficulty": "高",
        "tags": ["共同生活", "保护令", "程序"],
        "expected_intent": "法律咨询",
        "expected_laws": ["反家庭暴力法"],
        "quality_threshold": 82,
        "notes": "应识别共同生活人员适用反家庭暴力法相关保护。",
    },
    {
        "id": "labor-unpaid-wages",
        "question": "公司拖欠我两个月工资，我应该怎么维权？",
        "category": "劳动争议",
        "difficulty": "中",
        "tags": ["欠薪", "仲裁", "证据"],
        "expected_intent": "法律咨询",
        "expected_laws": ["劳动合同法", "劳动法"],
        "quality_threshold": 80,
        "notes": "应给出协商、投诉、仲裁等分层路径。",
    },
    {
        "id": "labor-overtime-comp-time",
        "question": "公司未经我同意用调休抵扣周末加班费，这合法吗？",
        "category": "劳动争议",
        "difficulty": "高",
        "tags": ["加班", "调休", "工资"],
        "expected_intent": "法律咨询",
        "expected_laws": ["劳动法"],
        "quality_threshold": 82,
        "notes": "需要区分休息日加班与法定节假日加班。",
    },
    {
        "id": "work-injury-dispute",
        "question": "我在工作中受伤，公司说不算工伤，我该怎么办？",
        "category": "工伤",
        "difficulty": "中",
        "tags": ["工伤认定", "时限", "材料"],
        "expected_intent": "法律咨询",
        "expected_laws": ["工伤保险条例"],
        "quality_threshold": 80,
        "notes": "应提示申请工伤认定和保留劳动关系证据。",
    },
    {
        "id": "consumer-false-advertising",
        "question": "我看到虚假广告后购买了商品，发现被骗，可以要求赔偿吗？",
        "category": "消费者权益",
        "difficulty": "中",
        "tags": ["虚假宣传", "欺诈", "赔偿"],
        "expected_intent": "法律咨询",
        "expected_laws": ["消费者权益保护法", "广告法"],
        "quality_threshold": 80,
        "notes": "不得在证据不足时直接断言满足惩罚性赔偿条件。",
    },
    {
        "id": "restaurant-slip",
        "question": "我在餐厅滑倒摔断了腿，地板很滑且没放提示牌，可以索赔吗？",
        "category": "安全保障义务",
        "difficulty": "中",
        "tags": ["经营场所", "人身损害", "因果关系"],
        "expected_intent": "法律咨询",
        "expected_laws": ["民法典"],
        "quality_threshold": 80,
        "notes": "应围绕安全保障义务、过错和损失证据回答。",
    },
    {
        "id": "housing-quality",
        "question": "新买的房子墙体渗水，开发商一直不维修，我能要求什么？",
        "category": "房屋买卖",
        "difficulty": "中",
        "tags": ["质量瑕疵", "维修", "违约"],
        "expected_intent": "法律咨询",
        "expected_laws": ["民法典", "建筑法"],
        "quality_threshold": 78,
        "notes": "应区分修复、赔偿和严重影响居住时的解除条件。",
    },
    {
        "id": "parking-lot-theft",
        "question": "我把车停在收费停车场，车窗被砸且财物被盗，停车场要赔吗？",
        "category": "保管与侵权",
        "difficulty": "高",
        "tags": ["停车合同", "保管义务", "举证"],
        "expected_intent": "法律咨询",
        "expected_laws": ["民法典"],
        "quality_threshold": 78,
        "notes": "不能仅因收费就当然认定停车场承担全部赔偿。",
    },
    {
        "id": "privacy-face-recognition",
        "question": "小区物业强制我刷脸才能进入，我不同意该怎么办？",
        "category": "个人信息保护",
        "difficulty": "高",
        "tags": ["敏感个人信息", "人脸识别", "替代方式"],
        "expected_intent": "法律咨询",
        "expected_laws": ["个人信息保护法", "民法典"],
        "quality_threshold": 82,
        "notes": "应关注单独同意、必要性和非刷脸替代通行方式。",
    },
    {
        "id": "telecom-fraud-account",
        "question": "有人让我提供银行卡帮忙转账，我不知道是诈骗资金，会承担刑事责任吗？",
        "category": "刑事风险",
        "difficulty": "高",
        "tags": ["主观明知", "帮助行为", "罪名边界"],
        "expected_intent": "法律咨询",
        "expected_laws": ["刑法", "反电信网络诈骗法"],
        "quality_threshold": 82,
        "notes": "应强调主观明知需结合事实认定，不能直接定罪。",
    },
    {
        "id": "non-legal-weather",
        "question": "明天北京天气怎么样？",
        "category": "路由负样本",
        "difficulty": "低",
        "tags": ["非法律", "意图路由"],
        "expected_intent": "闲聊",
        "expected_laws": [],
        "quality_threshold": 0,
        "notes": "应停止法律检索链路。",
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class ExperimentStore:
    def __init__(self, path: Path = DEFAULT_DB_PATH) -> None:
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS evaluation_cases (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL UNIQUE,
                    category TEXT NOT NULL,
                    difficulty TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    expected_intent TEXT NOT NULL,
                    expected_laws_json TEXT NOT NULL,
                    quality_threshold REAL NOT NULL DEFAULT 80,
                    notes TEXT NOT NULL DEFAULT '',
                    source TEXT NOT NULL DEFAULT 'custom',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    total_cases INTEGER NOT NULL,
                    completed_cases INTEGER NOT NULL DEFAULT 0,
                    dataset_snapshot_json TEXT NOT NULL,
                    pipeline_a_json TEXT NOT NULL,
                    pipeline_b_json TEXT NOT NULL,
                    summary_json TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT
                );
                CREATE TABLE IF NOT EXISTS experiment_case_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    category TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(experiment_id, case_id),
                    FOREIGN KEY(experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_experiments_created_at
                ON experiments(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_experiment_results_experiment
                ON experiment_case_results(experiment_id, id);
                """
            )
            now = utc_now()
            for case in SEED_CASES:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO evaluation_cases
                    (id, question, category, difficulty, tags_json, expected_intent,
                     expected_laws_json, quality_threshold, notes, source, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'seed', ?, ?)
                    """,
                    (
                        case["id"], case["question"], case["category"], case["difficulty"],
                        json_text(case["tags"]), case["expected_intent"], json_text(case["expected_laws"]),
                        case["quality_threshold"], case["notes"], now, now,
                    ),
                )
            connection.execute("PRAGMA optimize")
            connection.execute(
                "UPDATE experiments SET status = 'interrupted', completed_at = ?, error = ? WHERE status IN ('queued', 'running', 'cancelling')",
                (now, "服务重启导致任务中断，可使用相同配置重新运行。"),
            )

    @staticmethod
    def _case_view(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["tags"] = json.loads(item.pop("tags_json"))
        item["expected_laws"] = json.loads(item.pop("expected_laws_json"))
        return item

    def list_cases(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM evaluation_cases ORDER BY source DESC, category, created_at, id"
            ).fetchall()
        return [self._case_view(row) for row in rows]

    def upsert_cases(self, cases: list[dict[str, Any]]) -> dict[str, int]:
        created = updated = 0
        now = utc_now()
        with self._connect() as connection:
            for raw in cases:
                question = str(raw.get("question") or "").strip()
                if len(question) < 2:
                    continue
                case_id = str(raw.get("id") or "").strip() or "custom-" + hashlib.sha256(question.encode("utf-8")).hexdigest()[:12]
                exists = connection.execute("SELECT id FROM evaluation_cases WHERE id = ? OR question = ?", (case_id, question)).fetchone()
                if exists:
                    updated += 1
                    case_id = exists["id"]
                else:
                    created += 1
                connection.execute(
                    """
                    INSERT INTO evaluation_cases
                    (id, question, category, difficulty, tags_json, expected_intent,
                     expected_laws_json, quality_threshold, notes, source, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'custom', ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                      question = excluded.question,
                      category = excluded.category,
                      difficulty = excluded.difficulty,
                      tags_json = excluded.tags_json,
                      expected_intent = excluded.expected_intent,
                      expected_laws_json = excluded.expected_laws_json,
                      quality_threshold = excluded.quality_threshold,
                      notes = excluded.notes,
                      updated_at = excluded.updated_at
                    """,
                    (
                        case_id, question, str(raw.get("category") or "未分类"),
                        str(raw.get("difficulty") or "中"), json_text(raw.get("tags") or []),
                        str(raw.get("expected_intent") or "法律咨询"), json_text(raw.get("expected_laws") or []),
                        float(raw.get("quality_threshold") or 80), str(raw.get("notes") or ""), now, now,
                    ),
                )
        return {"created": created, "updated": updated}

    def create_experiment(
        self,
        experiment_id: str,
        name: str,
        cases: list[dict[str, Any]],
        pipeline_a: dict[str, Any],
        pipeline_b: dict[str, Any],
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO experiments
                (id, name, status, total_cases, dataset_snapshot_json, pipeline_a_json,
                 pipeline_b_json, created_at)
                VALUES (?, ?, 'queued', ?, ?, ?, ?, ?)
                """,
                (experiment_id, name, len(cases), json_text(cases), json_text(pipeline_a), json_text(pipeline_b), utc_now()),
            )

    def set_experiment_status(self, experiment_id: str, status: str, *, error: str | None = None) -> None:
        now = utc_now()
        fields = ["status = ?", "error = ?"]
        values: list[Any] = [status, error]
        if status == "running":
            fields.append("started_at = COALESCE(started_at, ?)")
            values.append(now)
        if status in {"completed", "failed", "cancelled", "interrupted"}:
            fields.append("completed_at = ?")
            values.append(now)
        values.append(experiment_id)
        with self._connect() as connection:
            connection.execute(f"UPDATE experiments SET {', '.join(fields)} WHERE id = ?", values)

    def add_case_result(self, experiment_id: str, case: dict[str, Any], result: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO experiment_case_results
                (experiment_id, case_id, question, category, result_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (experiment_id, case["id"], case["question"], case["category"], json_text(result), utc_now()),
            )
            connection.execute(
                "UPDATE experiments SET completed_cases = (SELECT COUNT(*) FROM experiment_case_results WHERE experiment_id = ?) WHERE id = ?",
                (experiment_id, experiment_id),
            )

    def save_summary(self, experiment_id: str, summary: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute("UPDATE experiments SET summary_json = ? WHERE id = ?", (json_text(summary), experiment_id))

    def get_experiment(self, experiment_id: str, include_results: bool = True) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
            if row is None:
                return None
            item = dict(row)
            for key in ("dataset_snapshot_json", "pipeline_a_json", "pipeline_b_json", "summary_json"):
                value = item.pop(key)
                item[key.removesuffix("_json")] = json.loads(value) if value else None
            if include_results:
                rows = connection.execute(
                    "SELECT case_id, question, category, result_json, created_at FROM experiment_case_results WHERE experiment_id = ? ORDER BY id",
                    (experiment_id,),
                ).fetchall()
                item["case_results"] = [
                    {**{k: row[k] for k in ("case_id", "question", "category", "created_at")}, "result": json.loads(row["result_json"])}
                    for row in rows
                ]
        return item

    def list_experiments(self, limit: int = 30) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, name, status, total_cases, completed_cases, summary_json,
                       created_at, started_at, completed_at, error
                FROM experiments ORDER BY created_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {**{key: row[key] for key in row.keys() if key != "summary_json"}, "summary": json.loads(row["summary_json"]) if row["summary_json"] else None}
            for row in rows
        ]
