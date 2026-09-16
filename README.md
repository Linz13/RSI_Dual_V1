# DualRSI：TTS–Captioning 递归训练实验

研究目标：通过语音描述模型（Captioner）与语音合成模型（TTS）之间的循环重建，让两个模型在迭代训练中改善。主要训练模型为 MiDashengLM-7B 与 Qwen3-TTS VoiceDesign。

本仓库包含训练实现、评测脚本、实验配置、结果和诊断记录。当前收录 V5 完整双循环框架、V6 四属性 audio-only 框架，以及独立的 **Captioner–TTS 双向评分能力测试**；早期版本保留实验报告和对照成绩。快照日期：2026-09-16。

## 阅读入口

| 想了解什么 | 入口 |
|---|---|
| 新实验：模型能否为对方提供有用的排序反馈 | [双向评分能力测试与 pilot 结果](experiments/bidirectional_scoring/README.md) |
| V5 / V6 的训练思路和区别 | [训练框架](docs/TRAINING.md) |
| 从哪些函数开始读 | [代码索引](docs/CODE_MAP.md) |
| 做过哪些实验、结论是什么 | [实验索引](experiments/INDEX.md) |
| Captioner 各轮 benchmark | [Captioner 总表](results/CAPTIONER.md) |
| TTS DSD benchmark | [TTS 总表](results/TTS_DSD.md) |
| 数据、模型、环境和 API 依赖 | [运行依赖](docs/DEPENDENCIES.md) |
| 代码和结果来自哪个原始文件 | [来源清单](provenance/files.json) |

## 目录

```text
frameworks/v5/      完整双循环训练源码，以 FastResume 实现为基础
frameworks/v6/      四属性 audio-only 训练源码及 API 重试扩展
labeling/          打标适配器、专家调用与配置示例
evaluation/        Captioner、TTS benchmark 与诊断实验脚本
experiments/       实验索引、配置、轮次统计、少量输出样例、历史报告
results/           可读总表、CSV、原始评测汇总
docs/              方法、代码阅读入口、外部依赖说明
provenance/        导出来源、文件哈希、V5 迁移记录
tools/             导出、结果汇总、仓库检查工具
```

V5 仅保留一套源码。LabelRobust 和 FastResume 是这条实现线上的修订：第 1 轮由 LabelRobust 运行，之后切换 FastResume；实际 run 目录仍沿用 LabelRobust 名称。源码中的 `orchestrator_v4.py` 与 `orchestrator_v5.py` 通过继承构成同一条 V5 流程。

V6 已提交 10 轮、Captioner 三个 benchmark 共 30 项评测；V6 DSD 收录前 4 轮。V5 这里收录的主 run 有 3 个已提交轮次，其中前 2 轮有完整 Captioner / DSD 评测。训练统计与外部 benchmark 分开保存，不能把训练奖励上涨当作泛化能力提升。

独立双向评分 pilot 不训练：A 有效 25 条，正确情绪描述严格第一 36%；B 有人工参照 23 组，命中 56.5%，对应随机基线 33.7%。当前 B 有初步正向信号，A 尚未验证可靠选优；双向评分有效性及互相训练收益均未被同时证明。人工排除、原始分数和数据限制完整保留。

## 使用范围

代码和结果以便于检查、讨论和追溯的形式保存。模型权重、原始数据集、全量音频、环境、完整缓存和训练日志不在仓库中。少量 JSONL 样例仅展示数据结构，不代表整体分布。

原始源码保留其服务器路径和实现；克隆后需要配置外部依赖，不能直接拿这个导出目录恢复旧 run。历史 README 中的命令和进度是当时记录，当前方法以本页、`docs/TRAINING.md` 与实际代码为准。具体见 [运行依赖](docs/DEPENDENCIES.md)。

使用 Python 3.10+，无需 GPU 或 API 即可重新汇总表格和检查导出内容：

```bash
python3 tools/summarize_results.py
python3 tools/verify_snapshot.py
python3 -B tools/verify_bidirectional_results.py
```

在聊天中讨论时，先指定版本和实验，例如：“结合 `frameworks/v6/dual_isl_train/reward_v6.py`、V6 训练统计和 Captioner 总表，分析奖励与 benchmark 的差异。”
