# EmotionTalk 第二轮评测：八卡断点续跑

目标目录为 `midasheng_v5_round001_caption_3gpu_run01`，固定第二轮 `round_001/caption_final`，batch=1、SDPA、max_new_tokens=128、原提示词和 standard_public 评分口径。

先等 TTS DSD 的 generate 结束；API judge 可以继续运行，不占 GPU。然后在原 Captioner 评测终端按 Ctrl+C，等原启动器退出、回到 shell 提示符后，再在 tmux 中执行：

```bash
cd /data/L202500147/Caption/benchmark
bash run_emotiontalk_resume_8gpu.sh run \
  --gpus 0,1,2,3,4,5,6,7 \
  --gpu-memory-gib 26
```

无需额外指定原目录或模型。每卡需要至少 28 GiB 空闲；26 GiB 是 PyTorch 分配上限，CUDA 等额外开销另留余量。脚本不会停止别的任务。原启动器仍持有锁时，本脚本会拒绝运行。

脚本自动执行：

1. 验证原评测 inventory、checkpoint、提示词、数据、batch 和生成参数。
2. 原样备份当时的 `events.jsonl`；保留所有成功的 `(id, task)` 结果。只允许恢复 Ctrl+C 导致的最后一行未写完整情况，不忽略中间损坏记录。
3. 将剩余请求按音频分配给八个进程，各卡独立写入分片目录；同一音频的几个任务尽量放在一起，各卡请求量最多相差四条。
4. 校验八份生成身份、样本覆盖、重复和缺失后，将已有结果与新结果按原数据顺序合并。原事件备份和分片元数据保留用于审计。
5. 自动在 GPU 0 计算完整指标，更新原 `summary.md`，不会重跑已经完成的 StyleCap 和 ParaSpeechCaps。

新日志目录：

```text
midasheng_v5_round001_caption_3gpu_run01/emotiontalk/full/midasheng_rewardv5_r1/resume_8gpu/logs/
```

每约 30 秒显示总进度，包含旧结果。中断后重复同一条命令即可继续；如果推理已合并而评分中断，会复用全部预测并重新评分。分片固定为八份，重新运行不要修改代码、batch 或 checkpoint。

只做 CPU 检查（原评测运行中也可执行，不建立快照、不改动评测结果）：

```bash
bash run_emotiontalk_resume_8gpu.sh check
```

本次只加速剩余推理，最终指标计算仍是一个进程。开发机器没有可用 GPU，CPU 验证不能替代目标服务器上的实际显存验证。
