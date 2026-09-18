# Contributing

1. 不要提交 API Key、私有法律咨询记录、完整训练集、Elasticsearch 数据或模型权重。
2. 新增指标时必须标注 `measured`、`historical_artifact` 或 `simulated_snapshot`。
3. 真实实验应记录数据版本、模型版本、Prompt 哈希、随机种子、延迟与成本。
4. 提交前运行：

```bash
python -m compileall -q app server scripts training tests
python -m unittest discover -s tests -v
python training/train.py --dry-run
```

真实链路的集成测试会产生 API 费用，因此 CI 默认只运行不联网的单元测试。提交实验指标时，请附数据版本、模型版本、Prompt 哈希和可复现实验命令。
