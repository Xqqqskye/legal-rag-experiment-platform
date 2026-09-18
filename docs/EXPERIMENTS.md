# 实验与产物规范

## 产物状态

- `measured`：脚本真实运行生成，并记录输入数据与环境。
- `historical_artifact`：来自旧项目的真实文件，保留来源路径。
- `simulated_snapshot`：只用于 UI 演示，不得写成真实训练结果。

## 推荐实验矩阵

| 实验 | 变量 | 固定项 | 主要指标 |
|---|---|---|---|
| Retrieval | BM25 / Dense / RRF | Query、语料 | Recall@K、MRR、nDCG |
| Rerank | cosine / BGE reranker / LLM | 候选集合 | Precision@K、nDCG |
| Prompt | default / strict citation | 模型、证据 | 忠实度、引用闭环 |
| Generator | API base / QLoRA | 检索证据 | 专家通过率、成本、延迟 |
| Training | base / SFT / DPO | 测试集 | win rate、法律适用性 |

## 禁止做法

- 用训练集或验证集作为最终测试集；
- 把 loss 下降写成法律准确率提升；
- 把预测区间写成真实指标；
- 不记录 Judge 模型和 Prompt；
- 让同一个模型生成并评审全部偏好数据，却不做人审抽检。

