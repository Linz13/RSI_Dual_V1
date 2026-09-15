# V5 第二轮评测：8 卡，每卡约 32 GiB 可用显存

固定评测 `DualISL_Train_RewardV5_LabelRobust/runs/midasheng_v5_label_robust_10rounds_8gpu_run01/round_001`。
轮次从零编号，所以 `--round-index 1` 是第二轮。两个模型均使用已提交的 `*_final`，启动时验证 checkpoint 哈希。

| 项目 | GPU | 设置 |
|---|---|---|
| Captioner：EmotionTalk、StyleCap、ParaSpeechCaps | 0,1,2 | 三个 benchmark 各用一张卡，推理 batch=1 |
| TTS：InstructTTSEval DSD | 3,4,5,6,7 | 5 个分片，batch=8，共 2000 条音频 |
| DSD API 评审 | 不使用 GPU | Gemini-2.5-Pro，32 worker |

每个 GPU 进程的 PyTorch allocator 上限设为 26 GiB，启动时要求至少 28 GiB 空闲。CUDA 上下文和部分库的显存不受 PyTorch 上限控制，因此留出额外余量；这不是对 `nvidia-smi` 总占用的硬限制。GPU 编号是 `nvidia-smi` 中的物理编号。

## tmux 窗口一：Captioner 三个 benchmark

```bash
cd /data/L202500147/Caption/benchmark
bash run_v5_caption_round.sh run \
  --round-index 1 --gpus 0,1,2 --batch-size 1 --gpu-memory-gib 26
```

每个 benchmark 自动先运行小规模 smoke；通过后进入全量。模型、数据、提示词和评分口径沿用第一轮评测。为控制显存，EmotionTalk 和 StyleCap 的 batch 从 4 降到 1，ParaSpeechCaps 仍按单样本推理。

汇总：`midasheng_v5_round001_caption_3gpu_run01/summary.md`；详细日志位于同目录 `logs/`。

## tmux 窗口二：TTS DSD

```bash
cd /data/L202500147/Caption/benchmark
export DSD_GPUS=3,4,5,6,7
export DSD_BATCH_SIZE=8
export DSD_GPU_MEMORY_GIB=26

bash run_v5_tts_dsd.sh smoke --round-index 1 && \
bash run_v5_tts_dsd.sh generate --round-index 1 && \
bash run_v5_dsd_audio_api.sh judge --round-index 1 --workers 32
```

先生成 40 条 smoke 音频并离线检查流程（不调用 API），通过后生成全量 2000 条，最后自动启动付费 API 评分。API key 和调用方式来自 `/data/L202500147/api/gemini_audio.py`，模型固定为 `models/gemini-2.5-pro`。对照分数读取同一 API 下重新评测的 base 报告。

汇总：`InstructTTSEval-public/qwen3_voice_design/runs/v5_midasheng_round001_dsd_seed42/evaluation_gemini_2_5_pro_audio_api/dsd_summary.json`。

如果 API 阶段中断，单独重跑最后一条 `judge` 命令即可；已成功评分的样本会复用。生成中断可重跑原命令，保留原分片数和 batch。第一轮结果使用独立目录，不会覆盖。

DSD 仍使用 seed=42 和之前的解码参数，但分片数由 4 改成 5 会改变按 batch 派生的随机种子和采样结果；跨轮小幅分数变化也可能含采样及评审波动。

## CPU 预检查（不启动 GPU、不发送 API 请求）

```bash
bash run_v5_caption_round.sh check --round-index 1 --gpus 0,1,2 --batch-size 1 --gpu-memory-gib 26
bash run_v5_tts_dsd.sh check --round-index 1 --gpus 3,4,5,6,7 --batch-size 8 --gpu-memory-gib 26
```

开发机器没有可用 GPU，实际显存和模型执行需要由上述 GPU smoke 验证。
