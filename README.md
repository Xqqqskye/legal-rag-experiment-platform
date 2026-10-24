# LegalRAG Lab：中文法律 RAG 对照实验平台

> 面向法律问答研发的可观测 RAG 实验平台：支持单轮问答、多轮案件对话、法条/案例混合检索、证据引用、联网法源核验、批量评测和 Prompt/模型 A/B 实验。

本项目不是“输入问题后直接让大模型回答”的聊天页面，而是一条可检查、可对比、可评估的 RAG 链路。页面会展示每一步的输入、Prompt、模型、候选证据、耗时和结果。

你可以用它完成：

- **法律问答**：输入问题，得到带证据和引用的结构化回答；
- **多轮案件咨询**：持续补充事实，系统维护会话上下文和结构化案件记忆；
- **检索实验**：对比 BM25、向量检索、RRF 和 LLM 重排的效果与延迟；
- **法源核验**：对候选法规进行官方来源和时效核验；
- **质量评估**：检查主张是否有证据支持，记录引用闭环、Bad Case、成本和 P95 延迟；
- **工程调试**：查看意图识别、Query 改写、召回、融合、回答和评估的逐步输入输出。

> 法律声明：本项目仅用于检索与评测技术演示，不构成法律意见。上线公开服务前还应增加鉴权、限流、日志脱敏和人工复核机制。

## 核心能力

- 意图识别与 Query 改写：保留主体关系、争点和待证明事实，避免把假设写成结论。
- 混合召回：法条与案例分别建立 BM25 词法索引；同时并行召回 Qwen3.7 法条 Dense Top 10、案例 Dense Top 8，并通过 RRF 融合不同证据域。
- 候选融合：RRF 去重融合，默认保留 Top 20 进入重排。
- 法律适用性重排：LLM 根据主体、构成要件、请求权和程序筛选最终 Top 5。
- 联网法源核验：搜索候选法源，并对全国人大国家法律法规数据库做精确名称匹配；只有通过官方数据库精确核验的记录进入 `official_sources`。
- 受约束回答：每项结论绑定本地证据或联网证据，证据不足时输出不确定性。
- 质量评估：逐主张检查引用支持度，并记录综合评分、成本、延迟和 Bad Case。
- Prompt 与模型对比：五个阶段的 Prompt 可见、可修改；同一问题可做 A/B 对照。
- 批量实验：内置评测样例，实验配置和逐题结果持久化到本地 SQLite。
- 多轮法律对话：会话、消息和版本化的多案件记忆持久化到 `output/evaluation/chat.sqlite3`，服务重启后可恢复；当前问题先做语义完整性判断，不完整时查询当前案件上下文，仍无法补全则返回 `needs_context=true` 并追问。
- 低延迟对话路径：多轮对话默认使用法条/案例 BM25 + Qwen3.7 向量的 RRF 直接排序，不调用 LLM 重排；单次问答页可通过“直接 RRF”开关选择同一策略。
- 可选案例检索：多轮、单次、A/B 与批量页面均可通过“检索参考案例”开关选择是否召回案例；关闭后不运行案例 BM25/向量分支。
- 检索延迟观测：查询向量只编码一次，BM25 与查询编码并行；返回分支耗时，区分首次索引构建、查询编码、法条/案例检索和回答生成。没有 Elasticsearch 时，启动阶段预热本地 BM25。

页面中的成本是按代码内价格快照计算的估算值，不是供应商账单；发布前应根据控制台价格更新费率和 `MODEL_PRICING_SNAPSHOT_DATE`。

## 多轮记忆设计

多轮链路把“对话记录”和“案件记忆”分开处理，避免把整段历史直接拼进 Prompt：

| 层级 | 内容 | 用途与边界 |
|---|---|---|
| 会话消息 | `chat_messages` 中的用户/助手消息、时间和本轮结果 JSON | 服务重启后恢复 UI；每轮只取最近 12 条参与上下文分析，避免上下文无限增长 |
| 案件记忆 | `chat_sessions.case_memory_json` 中按 `active_case_id` 管理的案件集合；每个案件保存事实、来源、纠正记录、冲突、法律争点、证据缺口、最近意图和改写 Query | 处理“那笔押金”“他后来又打我”等指代、省略和事实补充；支持新案件切换与用户纠正，记录事实来源，不把模型回答当作事实真相 |
| 当前输入 | 本轮用户消息单独标记为 `当前问题` | 优先级高于历史内容；历史消息只提供语境，不能覆盖本轮纠正或新案件切换 |

每轮请求的顺序是：读取当前会话 → 取最近消息和当前案件记忆 → 判断当前句是否完整 → 完整句进入 `legal_consultation / harmful_request / non_legal` 路由，不完整句先尝试上下文补全 → 仍不完整则返回 `needs_context=true` 并追问 → 已补全的法律问题生成改写 Query → 分别检索法条/案例 BM25 与 Qwen3.7 向量 → RRF/重排 → 生成回答 → 持久化本轮消息和案件记忆。记忆只保存在本机 SQLite（`output/evaluation/chat.sqlite3`），不同会话相互隔离，不会自动上传到 GitHub，也没有跨用户长期画像。用户删除会话时，消息和对应案件记忆一并删除。

## 链路

```text
用户问题
  -> 语义完整性判断 + 意图识别 / Query 改写
  -> 法条 BM25 Top 10 + 案例 BM25 Top 10
  -> Qwen3.7 法条 Top 10 + Qwen3.7 案例 Top 8
  -> RRF 去重融合 Top 20
  -> LLM 法律适用性重排 Top 5
  -> 官方法源联网核验（可选）
  -> 证据约束回答
  -> LLM-as-a-Judge + 规则检查
  -> A/B 实验记录与 Bad Case
```

## 当前模型配置

默认值只是实验配置，实际可用模型以你的 API 账号、区域和供应商控制台为准，均可通过 `.env` 和前端下拉框替换。

| 阶段 | 默认模型 | 选择理由 |
|---|---|---|
| 意图识别、改写 | `qwen3.8-flash` | 结构化任务优先低延迟与成本 |
| 法条重排 | `qwen3.7-plus` | 需要更强的法律适用性判断 |
| 最终回答 | `qwen3.8-flash` | 默认低延迟回答；可在前端切换到更强但更慢的模型 |
| 质量评估 | `qwen3.8-max` | Judge 应尽量强于生成模型，减少漏判 |
| 联网搜索 | `qwen-flash` | 搜索阶段以检索覆盖与响应速度为主 |
| 向量模型 | `qwen3.7-text-embedding` | 查询通过同一 DashScope API 编码，法条和案例使用对应的 1024 维本地 FAISS 索引 |

## 快速启动真实链路

环境建议：Python 3.11、Docker Desktop、至少 8 GB 内存。CPU 可以运行公开样例，首次下载向量模型需要联网。

### 1. 安装依赖

```bash
git clone https://github.com/Xqqqskye/legal-rag-experiment-platform.git
cd legal-rag-experiment-platform
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

macOS / Linux：

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

在 `.env` 中填写你自己的 `DASHSCOPE_API_KEY`，或把 `LLM_PROVIDER` 改为 `openai` 并填写对应密钥。不要把 `.env` 提交到 Git。

### 2. 启动 Elasticsearch 并建立公开样例索引

```bash
docker compose -f docker-compose.es.yml up -d
python scripts/index_sample.py
```

仓库只包含 5 条公开演示样例，用于验证端到端代码。完整实验需要自行准备来源明确、授权允许的法律语料并按相同字段建立索引。

如果已经有按 chunk 生成的完整法条/案例 JSON 或 JSONL 元数据，可分别建立词法索引：

```powershell
python scripts/index_bm25.py --corpus law --input E:\path\to\law_chunks.jsonl
python scripts/index_bm25.py --corpus case --input E:\path\to\case_chunks.jsonl
```

脚本只写入 Elasticsearch 的倒排字段；案例 BM25 与案例 Qwen3.7 向量索引在运行时按 `chunk_id` 进入同一套 RRF，不会把案例误当作法条，也不会要求重新生成向量。

### 数据来源与引用

演示法条片段的来源、适用范围和再分发注意事项见 [`data/samples/SOURCES.md`](data/samples/SOURCES.md)。完整法条与案例均来自 GitHub 上游 [`lttxzmj/chinese-law-corpus`](https://github.com/lttxzmj/chinese-law-corpus)。已核对的语料版本、上游 commit、记录数量和索引模型见公开的[数据版本与索引清单](data/samples/dataset_manifest.json)；完整原始语料和向量索引不随 GitHub 仓库发布。核心法源核验入口是[全国人大国家法律法规数据库](https://flk.npc.gov.cn/)，回答中应保留法源名称、条号和核验日期。`legal_eval_sample.jsonl` 是项目自编测试样例，不是真实用户数据或第三方数据集。

本地完整 Demo 使用法条与案例各自的 Qwen3.7 FAISS 索引，以及 BM25 词法召回。完整 FAISS 索引（约 31k 法条块与 31k 案例块）不随 GitHub 仓库提交：除体积外，原始语料和派生索引的再分发许可仍需逐项确认。运行完整 Demo 时，将 `law_qwen37_api_latest/` 与 `cases_qwen37_api_latest/` 放入工作区 `data/indexes/`，并在 `.env` 配置 DashScope API Key。仓库中的 `scripts/index_sample.py` 用于对 5 条公开法条样例建立 Elasticsearch BM25 和 BGE 对照索引；完整法条/案例的 BM25 索引可使用 `scripts/index_bm25.py` 生成。没有 Elasticsearch 时，服务会从对应 Dense 索引的元数据在本机建立 BM25 回退索引。

### 3. 启动 Demo

Windows：

```powershell
.\scripts\start_demo.ps1
```

macOS / Linux：

```bash
./scripts/start_demo.sh
```

打开 <http://127.0.0.1:8008>。健康检查地址为 <http://127.0.0.1:8008/api/healthz>。

进入页面的“多轮对话”标签即可新建或恢复会话。每轮会把最近消息和案情记忆交给意图分析、检索和回答阶段；会话记录保存在本机，不会自动上传到 GitHub。

## 配置说明

关键检索参数位于 `.env`：

```dotenv
BM25_TOPK=10
CASE_BM25_TOPK=10
QWEN37_LAW_TOPK=10
QWEN37_CASE_TOPK=8
RRF_TOPK=20
RRF_K=60
RERANK_TOPK_FINAL=5
```

这里的含义是法条 BM25 召回 10 条、案例 BM25 召回 10 条、Qwen3.7 法条向量召回 10 条、案例向量召回 8 条；RRF 按 `chunk_id` 去重后排序；LLM 最终选出不超过 5 条证据。法条和案例使用各自的 BM25 语料域；Dense 查询和离线索引必须使用同一套 Qwen3.7 Embedding，不与 BGE 或 Qwen3-Embedding-8B 向量混用。

## 训练代码状态

`training/` 提供 QLoRA 配置、数据清洗和启动脚本，便于复现实验设计；公开仓库没有训练后的 Adapter、checkpoint 或真实训练日志，也不声称当前 API 模型经过本项目 LoRA。Demo 当前使用供应商 API 模型完成分析、重排、生成和评估。

在执行训练前，请先核对模型标识、LLaMA-Factory 模板、显卡显存和训练数据许可证。真实训练结果应至少报告独立测试集、随机种子、数据版本、Prompt 版本、专家抽检、成本和延迟。

## 目录

```text
server/                 真实 FastAPI/RAG 后端
web/                    当前单页实验台（含单轮、多轮和实验页面）
scripts/index_sample.py 公开样例索引脚本
data/samples/           可公开的小样本
tests/                  不联网单元测试
training/               QLoRA 实验配置与数据处理脚本
artifacts/              标明来源的历史实验快照
docs/                   评测、模型与发布安全说明
```

## 哪些内容不能上传

API 接入代码和空白配置模板可以公开；密钥、令牌、私有端点和调用数据不能公开。完整清单见 [GitHub 发布与敏感信息清单](docs/GITHUB_PUBLISHING.md)。仓库的 `.gitignore` 已默认排除 `.env`、模型权重、缓存、私有数据、SQLite 实验库和 Elasticsearch 数据卷，但推送前仍必须做人工复核。

## 测试

2026-10-24 的修复范围、实际计时和复测边界见 [Demo 检查记录](docs/DEMO_AUDIT_20261024.md)。

```bash
python -m compileall -q app server scripts training tests
python -m unittest discover -s tests -v
python training/train.py --dry-run
```

## 许可证

代码使用 MIT License。法律文本、训练数据、第三方模型和 API 输出分别受其来源许可证与服务条款约束，MIT License 不自动授予这些内容的再分发权。
