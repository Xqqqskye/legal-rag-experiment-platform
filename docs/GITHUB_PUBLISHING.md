# GitHub 发布与敏感信息清单

## 可以公开

- FastAPI、检索、RRF、重排、评估和前端代码；
- `.env.example` 中的变量名和空值；
- 来源、许可证明确的小型演示数据；
- 不包含真实用户内容的测试用例；
- 明确标注来源和实验条件的指标快照；
- 未携带权重、凭据或个人数据的训练配置。

## 不能公开

1. **API 凭据**：`DASHSCOPE_API_KEY`、`OPENAI_API_KEY`、AutoDL 登录信息、SSH 私钥、Access Token、Cookie、Webhook 和云厂商凭据。
2. **真实 `.env`**：即使密钥暂时失效，也不能提交；Git 历史仍会保留旧值。
3. **用户查询与实验库**：`output/evaluation/experiments.sqlite3` 可能包含真实法律咨询、模型输出、成本和配置。
4. **调用日志**：请求体、响应体、异常堆栈和调试日志可能含个人身份、案情、密钥或内部 URL。
5. **完整原始数据**：未经许可的法律语料、问答数据、SFT/DPO 数据、人工标注文件及用户上传内容。
6. **Elasticsearch 数据与向量索引**：不仅体积大，还可能是受限原文的衍生物。
7. **模型资产**：checkpoint、LoRA Adapter、`*.safetensors`、`*.bin`、`*.pt`、`*.gguf` 和模型缓存；只有许可证允许且使用 Git LFS 时才考虑单独发布。
8. **本机与云主机信息**：绝对路径、内网地址、端口映射、AutoDL 实例连接命令、账号名和密码。
9. **临时文件**：虚拟环境、缓存、压缩包、截图、Notebook 输出及崩溃转储。

## API 相关结论

- **可以上传**：SDK 调用方式、Base URL 的公开官方地址、模型名、超时和重试逻辑。
- **不能上传**：真实 Key、鉴权 Header、账户 ID、请求日志、余额/账单明细和私有网关地址。
- GitHub Actions 如需部署，只能把 Key 放入仓库 `Settings -> Secrets and variables -> Actions`，代码中通过环境变量读取。
- 前端绝不能直接保存或调用供应商 Key；Key 只能位于后端环境变量。

## 推送前检查

```bash
git status --short
git diff --cached
git ls-files
gitleaks detect --source . --redact
```

如果本机尚未安装 Gitleaks，至少应搜索 `API_KEY`、`TOKEN`、`PASSWORD`、`PRIVATE KEY` 等变量，并逐项确认只有空值或占位符；不要把扫描输出粘贴到公开 Issue。

如果密钥曾经进入 Git 历史，仅删除文件不够：应立即轮换密钥，并用 `git filter-repo` 或 BFG 清理历史后再发布。

## GitHub 仓库信息

推荐仓库名：`legal-rag-experiment-platform`

推荐 Description：

> 可观测、可评测的中文法律 RAG 实验平台：支持 BM25+BGE+RRF、LLM 重排、官方法源联网核验、Prompt/模型 A/B 测试、批量评测、成本延迟与 Bad Case 分析。

推荐 Topics：

`legal-ai` `rag` `qwen` `bge` `elasticsearch` `fastapi` `rrf` `llm-evaluation` `prompt-engineering` `qlora`

## 首次推送

先在 GitHub 新建一个空仓库，不要勾选自动生成 README，然后在本地运行：

```bash
git remote add origin https://github.com/YOUR_NAME/legal-rag-experiment-platform.git
git push -u origin main
```

推送后把 README 中的 `OWNER/REPOSITORY` 替换为真实路径，并在仓库页面添加 Description 与 Topics。
