# Captioner–TTS 双向评分能力测试

这是一个独立的前置验证实验，分别判断两个基础模型是否能为对方提供有用的排序反馈。本次没有训练，没有多轮循环，也不把旧实验的 reward 或自动 caption 当真值。

## 研究问题与当前结论

| 方向 | 固定条件与候选 | 主评分 | 独立参照 |
|---|---|---|---|
| A：TTS → Captioner | 真实语音、核验转写固定，只改变候选情绪描述 | Qwen3-TTS 对真实 codec 的 teacher-forced 平均 log-probability，全部 16 码本，不含 EOS | 人工核验实际情绪与转写 |
| B：Captioner → TTS | 同一文本和风格条件下的 4 条生成语音 | MiDashengLM 对目标情绪描述正文的平均文本 log-probability | 不显示模型分数的人工盲听最佳集合，允许并列／全部不合格／无法判断 |

当前 pilot 只检验情绪（开心、悲伤、愤怒、平静），没有检验语速。B 的目标描述只作为 assistant 被评分正文，不作为 user 提示中已知的答案。

### pilot30：2026-09-16 核验完成

| 项目 | 有效主比较 | 结果 | 随机选择 |
|---|---:|---:|---:|
| A 正确描述严格第一 | 25 条 | **9/25 = 36.0%** | 四选一 25.0% |
| B 选中人工最佳集合 | 23 组 | **13/23 = 56.5%** | 按每组最佳集合大小计算，33.7% |

- A 准备 30 条，其中 5 条人工无法判断，未评分；正确描述逐对比较 51 胜、24 负、0 平。情绪间差异明显，25 条中 13 条把平静排第一，人工平静只有 3 条。
- B 评分 30 组，其中 3 组全部不合格、4 组无法判断，仅保存排序，不进入命中率分母。23 组中有 7 组人工并列最佳，没有四条全并列最佳的组。
- B 相对随机提升 22.83 个百分点，描述性组级 bootstrap 95% 区间为 +1.09 至 +43.48 个百分点；区间不包含标注者差异和全部数据来源相关性。
- 两侧首样本数值检查通过。A 使用 FP32 talker/code predictor、B 使用 BF16，均固定 math SDPA。逐样本数据与分数复算通过。

**结论：B 有初步正向信号；A 有部分成对区分信号，但尚未验证可靠选优。双向评分有效性尚未同时成立，互相训练能否提升尚未检验。**

重要限制：A 有 10 条转写保留方括号标记，其中 3 条含笑／哭标记，可能在文本条件中额外带入风格信息。影响未经清理对照验证；保留原文和原结果，没有事后删样或更改标签。A 来自历史反复评测集，B 复用历史训练候选；独立人工参照不等于全新未暴露测试集。

[完整分析与成功／失败例子](results/pilot30/reports/v1_review/analysis.md) · [汇总 JSON](results/pilot30/reports/v1/summary.json) · [A 逐对 CSV](results/pilot30/reports/v1/a_pairs.csv) · [B 排序 CSV](results/pilot30/reports/v1/b_ranking.csv)

### smoke01：历史流程验证

旧 4 条 A、4 组 B 用于检查执行链路。A 最终 `v3_fp32` 为 4/4 正确第一，但其中一条几乎并列；B `v2` 命中 50%，随机 62.5%。不把这批较小样本与 pilot30 合并，不挑选较好的结果代表总体。

初始 BF16 数值核验失败后停止；固定 math SDPA 后 B 通过，A 继续改用同一份已加载权重的 FP32 计算通过。未安装依赖或修改权重。保存初始失败的状态、错误信息和后续审计，不把失败项记为零分。

[smoke 解读](results/smoke01/reports/v3_review/analysis.md) · [数值修复记录](checks/NUMERIC_RETRY.md) · [最终汇总](results/smoke01/reports/v3_combined/summary.json)

## 目录与复核入口

```text
PROTOCOL.md                  评分、人工参照、排除及统计协议
src/                         原始脚本快照，评分逻辑未因导出而修改
src/web/                     可播放、保存、导出的盲听标注页面
src/tests/                   轻量流程和评分契约测试
checks/                      当时保存的 CPU/processor 检查记录
results/smoke01/              标注、清单、失败记录与最终 smoke 结果
results/pilot30/              标注、清单、全部评分、逐 token 依据、汇总
```

`results/*/scores/` 保留所有候选分数和每个目标 token 的 log-probability，以及模型文件哈希、代码身份和数值审计。`results/*/manifest.json` 保留样本来源、真实音频哈希、候选映射和生成条件；结果已解除盲态，仅用于标注完成后的复核。

### 克隆后：不加载模型地复算结果

在仓库根目录，使用 Python 3.10+ 和标准库：

```bash
python3 -B tools/verify_bidirectional_results.py
python3 -B tools/verify_snapshot.py
```

第一个命令核验来源哈希、人工排除、逐 token 均值、A/B 指标、平局敏感性和保存的 bootstrap 区间。无需音频、torch、GPU 或 API。它检查的是保存的文本证据；实际音频 SHA256 校验结果来自原服务器，不能在没有音频的克隆中重新声称已核验音频字节。

轻量测试（临时合成夹具，不加载真实模型；HTTP 测试只绑定本机回环地址）：

```bash
python3 -B -m unittest discover -s experiments/bidirectional_scoring/src/tests -p 'test_pilot.py'
(cd experiments/bidirectional_scoring/src && node tests/test_ui.js)
```

Node 只用于已有前端测试；标注服务器本身使用 Python 标准库。不要运行 `check_processor_contract.py` 或 `check_numeric_backend.py` 来代替上述纯标准库检查，它们分别需要本地 processor／torch 环境。

### 原共享服务器：标注与真实模型命令

源码保持原服务器目录布局假设，尤其 `src/common.py` 中的 `CAPTION`、`ENV_ROOT` 和外部数据路径。GitHub 的 `src/` 是可检查的源码快照，不能在缺少外部资源的克隆中原样启动真实评分。

已有完整资源位于 `/data/L202500147/Caption/BidirectionalScoringPilot`。本批已完成，以下仅记录原运行方式，不要求重跑：

```bash
cd /data/L202500147/Caption/BidirectionalScoringPilot
# 标注服务器；通过 SSH/VS Code 转发 8765 后，在本地打开 http://127.0.0.1:8765
python3 -B pilot.py serve --run runs/pilot30 --port 8765

# 真实模型由用户在共享存储的模型服务器执行，0 替换为已分配 GPU 编号
bash run_pilot30.sh 0
```

标注自动保存在原服务器 `runs/pilot30/annotations.json`，页面支持“下一条”保存、返回修改及导出。模型日志、评分、报告分别在同一 run 的 `logs/`、`scores/`、`reports/`。新服务器需先配置外部数据、基础模型和既有模型环境；导出过程不会安装依赖或下载模型。

## 数据和发布边界

仓库不包含音频、试听 ZIP、模型权重、codec 缓存、完整日志或标注编辑历史。它保留这 30+30 及 4+4 个样本的最终人工标签和小规模转写／条件，用于核验结果，不分发完整原始数据集。GitHub 上的标注源码和清单无法直接播放服务器本地音频。

源码和证据的逐文件来源见 [独立来源清单](../../provenance/bidirectional_scoring.json)。原实验目录、旧训练框架和原标注均不由本次发布修改。原报告的 Markdown 仅增加归档提示、移除未上传试听包的链接，其数据和结论保持不变。

后续若对转写或评分机制做对照，使用新结果版本，保留本次人工标签和结果；不自动扩大实验或进入训练。
