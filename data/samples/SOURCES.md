# Demo 数据来源与引用

## `law_articles.json`

该文件只包含 5 条用于端到端演示的法条片段，按法律名称和条号整理，**不是完整法律数据库**。使用和核验时应以全国人大国家法律法规数据库的现行原文为准：

- 全国人大国家法律法规数据库：<https://flk.npc.gov.cn/>
- 数据库运营方：全国人民代表大会常务委员会办公厅
- 覆盖示例：`中华人民共和国民法典`、`中华人民共和国反家庭暴力法`、`中华人民共和国劳动争议调解仲裁法`
- 许可证/再分发说明：本仓库不声明对法律原文拥有版权，也不把 MIT License 扩展到法律文本；公开发布前请核对官方数据库的使用条款，并在产品中保留法源名称、条号和核验日期。
- 时效说明：法律、司法解释和行政法规可能修订、废止或调整施行日期；Demo 不保证 5 条样例在未来仍然有效。

## `legal_eval_sample.jsonl`

该文件的 3 条问题是本项目为检索链路演示编写的测试样例，不来自真实用户，也不宣称来自第三方数据集。`expected_laws` 只用于离线评测期望命中法源，不是法律结论。

## 完整法条和案例索引

完整法条和案例原始数据均来自 GitHub 上游仓库 [`lttxzmj/chinese-law-corpus`](https://github.com/lttxzmj/chinese-law-corpus)。本仓库提供不含原文和本机路径的 [数据版本与索引清单](dataset_manifest.json)，便于区分公开样例、完整语料和本地索引状态。

| 数据类型 | 工作区本地数据版本 | 上游来源 | 本项目处理方式 |
|---|---|---|---|
| 法条 | `chinese-law-corpus_B_v2_20261018`、`chinese-law-corpus_B_v3_20261020_verified` | [`lttxzmj/chinese-law-corpus`](https://github.com/lttxzmj/chinese-law-corpus) | 清洗和版本核对后，按法条结构递归分块；分别生成 BM25 词法索引和 Qwen3.7 Embedding FAISS 索引 |
| 案例 | `chinese-law-corpus_cases_latest_20261018`、`chinese-law-corpus_cases_latest_20261020` | [`lttxzmj/chinese-law-corpus`](https://github.com/lttxzmj/chinese-law-corpus) | 保留指导性案例、公报案例原始字段，按案件/章节建立父块，再生成摘要块和原文 child；使用 Qwen3.7 Embedding 案例索引 |

本机工作区的详细数据清单记录下载目录和其他训练/评测语料，不公开发布。完整 Qwen3.7 FAISS 索引、原始法条/案例文件和聊天数据库也不随仓库发布。它们可能包含受限来源、衍生向量或用户输入；使用者应自行确认上游 LICENSE/README、个人信息处理和再分发权限。仓库代码只依赖索引的 `index.faiss` 与 `meta.jsonl` 字段约定。

## 引用建议

引用本 Demo 的代码时，请同时说明：

1. 代码仓库：<https://github.com/Xqqqskye/legal-rag-experiment-platform>
2. 法条与案例上游仓库：<https://github.com/lttxzmj/chinese-law-corpus>
3. 法源核验入口：<https://flk.npc.gov.cn/>
4. 使用的语料版本/上游 commit、索引生成日期、嵌入模型和检索配置。
