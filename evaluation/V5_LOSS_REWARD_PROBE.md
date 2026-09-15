# V5 loss 奖励八卡补算实验

本次补算第二轮 `round_001` audio-only 的 432 条有效候选，固定原音频 codec、原音频缓存转录和语言，候选之间只改变声音指令。加上对照共 704 次评分，八卡各 88 次。无需训练、无需音频生成、无需付费 API。

评分模型固定为第二轮开始时的 `round_000/checkpoints/tts_final`，与历史重建分使用的 TTS 一致；不会使用已经训练过这些第二轮样本的 `round_001/tts_final`。policy 和 reference 两份 adapter 都从这个 checkpoint 精确加载，防止已有 worker 的 reference 模式意外退回 base。

在八卡服务器的 tmux 终端执行：

```bash
cd /data/L202500147/Caption/benchmark
bash run_v5_loss_reward_probe.sh run \
  --gpus 0,1,2,3,4,5,6,7 \
  --gpu-memory-gib 26
```

`run` 自动先在各卡评分一条较长样本作为 smoke，再完成全量，最后在 CPU 上生成报告。每卡至少 28 GiB 空闲；26 GiB 是 PyTorch 分配上限，另外留 CUDA 等开销。每卡逐样本执行 teacher-forcing forward；八卡并行，不进行自回归音频生成。

中断后重复原命令可续跑。只运行 CPU 准备检查可使用 `check`；全量评分完成后只重算分析可使用 `analyze`。

输出目录：

```text
/data/L202500147/Caption/benchmark/reports/v5_loss_reward_probe_round001_run01/
```

主要文件：

- `report.md`：三组权重的平均贡献跨度、第一名改变的组数、属性替换对照。
- `summary.json`：loss 分布、固定 sigmoid 参数、完整统计。
- `candidates.csv` / `candidates.json`：每条候选的 loss、分数、原始/固定转录、TTS 指令。
- `changed_winners.json`：加入 loss 后第一名改变的组，供检查排序变化是否合理。
- `logs/`：八卡执行日志；`scores/` 为逐条可续跑结果及 adapter、重复评分审计。

loss 沿用训练公式 `主码本平均 NLL + 0.3 × 其余 15 个码本平均 NLL`；模型 eval、关闭梯度，提示词不计入目标位置。每个 worker 的第一条评分会重复一次，若不一致超过 1e-5 则报错。

预先按组划分约 25% 用于校准，剩余组用于分析。sigmoid 以校准候选 loss 的 P10 对应 0.9、P90 对应 0.1；分析组不参与拟合参数。比较重建/loss/格式权重 0.8/0.1/0.1、0.7/0.2/0.1、0.6/0.3/0.1，相对于旧 0.9/0/0.1 的排序变化。

对每个有效组额外评分“只有固定文本、无声音指令”，另选最多 32 个分析组评分参考属性指令，以及只替换性别或情绪的指令。属性替换对照只使用通过 schema 校验的参考指令，避免引入 neutral + high 等组合矛盾；这不影响 432 条候选的补算。参考属性来自缓存打标而非人工复核，不能把对照结果视为绝对正确率。

重要范围：这次复用历史属性重建分，历史生成音频中的转录尚未替换，也未加新的转录准入规则。报告用于检查 loss 的尺度与指令敏感性，不代表完整新方案的训练效果；分数排序改变不自动意味着质量提升。

耗时参考：旧 V4 同类八卡 teacher-forcing 评分阶段（含加载）约 116～121 秒。本次还有八卡 smoke、原音频/codec/checkpoint 校验及更多对照，预计 5～15 分钟，首次环境/共享盘较慢可留 20 分钟；这是估算，需要目标服务器的实际运行确认。
