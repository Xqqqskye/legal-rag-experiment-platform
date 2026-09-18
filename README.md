# LegalRAG Lab：中文法律 RAG 对照实验平台

> 面向 AI 产品经理和算法工程师的法律检索实验台：对同一法律问题进行多模型、多检索方案和多 Prompt 的可观测 A/B 测试，量化质量、引用闭环、延迟与成本，并沉淀 Bad Case。

本项目不是“输入问题后直接让大模型回答”的聊天页面，而是一条可检查、可对比、可评估的 RAG 链路。页面会展示每一步的输入、Prompt、模型、候选证据、耗时和结果。

> 法律声明：本项目仅用于检索与评测技术演示，不构成法律意见。上线公开服务前还应增加鉴权、限流、日志脱敏和人工复核机制。

## 核心能力

- 意图识别与 Query 改写：保留主体关系、争点和待证明事实，避免把假设写成结论。
- 混合召回：BM25 Top 10 与 BGE Dense Top 10 并行召回。
- 候选融合：RRF 去重融合，默认保留 Top 20 进入重排。
- 法律适用性重排：LLM 根据主体、构成要件、请求权和程序筛选最终 Top 5。
- 联网法源核验：搜索候选法源，并对全国人大国家法律法规数据库做精确名称匹配；只有通过官方数据库精确核验的记录进入 `official_sources`。
- 受约束回答：每项结论绑定本地证据或联网证据，证据不足时输出不确定性。
- 质量评估：逐主张检查引用支持度，并记录综合评分、成本、延迟和 Bad Case。
- Prompt 与模型对比：五个阶段的 Prompt 可见、可修改；同一问题可做 A/B 对照。
- 批量实验：内置评测样例，实验配置和逐题结果持久化到本地 SQLite。

页面中的成本是按代码内价格快照计算的估算值，不是供应商账单；发布前应根据控制台价格更新费率和 `MODEL_PRICING_SNAPSHOT_DATE`。

## 链路

```text
用户问题
  -> 意图识别 / Query 改写
  -> BM25 Top 10 + BGE Top 10
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
| 最终回答 | `qwen3.7-plus` | 在质量、速度和成本之间取平衡 |
| 质量评估 | `qwen3.8-max` | Judge 应尽量强于生成模型，减少漏判 |
| 联网搜索 | `qwen-flash` | 搜索阶段以检索覆盖与响应速度为主 |
| 向量模型 | `BAAI/bge-small-zh-v1.5` | 公开样例可在 CPU 运行；正式实验可换 large |

## 快速启动真实链路

环境建议：Python 3.11、Docker Desktop、至少 8 GB 内存。CPU 可以运行公开样例，首次下载向量模型需要联网。

### 1. 安装依赖

```bash
git clone https://github.com/OWNER/REPOSITORY.git
cd REPOSITORY
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

## 配置说明

关键检索参数位于 `.env`：

```dotenv
BM25_TOPK=10
BGE_LARGE_TOPK=10
RRF_TOPK=20
RRF_K=60
RERANK_TOPK_FINAL=5
```

这里的含义是两路分别召回 10 条；RRF 按 `chunk_id` 去重后对最多 20 个候选排序；LLM 最终选出不超过 5 条证据。公开样例只有 5 条，因此界面数量会小于上述上限。

## 历史检索实验快照

`artifacts/retrieval_metrics.json` 保存的是原项目已有评测文件的汇总，并非当前 5 条公开样例，也不是本次 GitHub 整理过程中重新跑出的结果：

| 方案 | Top K | Context Recall | Context Precision |
|---|---:|---:|---:|
| BM25 | 10 | 0.4204 | 0.4181 |
| BGE-large | 10 | 0.5613 | 0.7226 |
| RRF | 20 | **0.6667** | 0.5516 |
| RRF + BGE reranker-large | 10 | 0.6042 | **0.7593** |

这些结果支持“Dense 在该历史评测集上优于 BM25；RRF 提高候选覆盖；reranker 提高最终精度”的局部结论，不能直接外推为所有法律问题或当前线上配置的效果。

## 训练代码状态

`training/` 提供 QLoRA 配置、数据清洗和启动脚本，便于复现实验设计；公开仓库没有训练后的 Adapter、checkpoint 或真实训练日志，也不声称当前 API 模型经过本项目 LoRA。Demo 当前使用供应商 API 模型完成分析、重排、生成和评估。

在执行训练前，请先核对模型标识、LLaMA-Factory 模板、显卡显存和训练数据许可证。真实训练结果应至少报告独立测试集、随机种子、数据版本、Prompt 版本、专家抽检、成本和延迟。

## 目录

```text
server/                 真实 FastAPI/RAG 后端
web/                    当前单页实验台
scripts/index_sample.py 公开样例索引脚本
data/samples/           可公开的小样本
tests/                  不联网单元测试
training/               QLoRA 实验配置与数据处理脚本
artifacts/              标明来源的历史实验快照
docs/                   评测、模型与发布安全说明
```

## 测试

```bash
python -m compileall -q app server scripts training tests
python -m unittest discover -s tests -v
python training/train.py --dry-run
```

## 许可证

代码使用 MIT License。法律文本、训练数据、第三方模型和 API 输出分别受其来源许可证与服务条款约束，MIT License 不自动授予这些内容的再分发权。
