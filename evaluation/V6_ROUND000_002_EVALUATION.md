# V6 前三轮评测

固定 `/data/L202500147/Caption/DualRSI_Train_V6_AudioOnly4Attr/runs/midasheng_v6_8gpu_run01` 的 `round_000`、`round_001`、`round_002`，分别代表第 1、2、3 轮。校验已提交的 Captioner/TTS checkpoint 哈希，不自动追踪正在增长的 latest，不修改训练目录。

| 模型 | 评测 | 每轮规模 / 口径 |
|---|---|---|
| Captioner | EmotionTalk | 7,716 条任务请求；standard_public，SPIDEr/FENSE |
| Captioner | ParaSpeechCaps | 140 条样本；Scheme A attr6 |
| Captioner | StyleCap | 3,112 道题；speaker-open MCQ |
| TTS | InstructTTSEval DSD | 中文 1,000 条、英文 1,000 条 |

使用完整 benchmark 的原提示和属性，不把评测裁剪成四属性。四属性训练在其他能力上的变化也会反映在分数中。TTS 沿用之前 DSD 数据集、解码设置以及 Gemini-2.5-Pro 裁判、提示和 REST API（凭证从 `api/gemini_audio.py` 读取，不输出到日志）。这是与本项目 base 重评分和 V5 对齐的配置，不声称与官方所有裁判版本/服务配置完全相同。

## 使用另一台 8 卡服务器

每卡有 32 GiB 可用时，进程分配器上限设 26 GiB，留出额外开销。Captioner batch=4，TTS batch=8；推理与 GRPO 的采样概率校验不同，此处不因训练阶段串行回退而强制评测 batch=1。真实显存/吞吐以 GPU smoke 为准。不要在正在跑训练的那台服务器上执行这些默认 GPU 编号。

```bash
cd /data/L202500147/Caption/benchmark

# 不用 GPU、不调用 API；当前已经通过 CPU 预检。
bash run_v6_benchmarks.sh check --gpus 0,1,2,3,4,5,6,7 --gpu-memory-gib 26

# 小规模真实 GPU 推理/指标验证；DSD 使用离线模拟裁判，不产生 API 费用。
bash run_v6_benchmarks.sh smoke --gpus 0,1,2,3,4,5,6,7 --gpu-memory-gib 26

# smoke 完成后，tmux 前台正式执行；包含付费 DSD 裁判调用。
bash run_v6_benchmarks.sh run --gpus 0,1,2,3,4,5,6,7 --gpu-memory-gib 26
```

默认只评测 `--rounds 0,1,2`。正式流程先将 9 个 Captioner 任务分配到 GPU 队列，每张卡同时最多一个任务；随后按轮次使用全部 8 卡分片生成 DSD，每轮生成完成后用 32 个 API worker 打分。短任务做完后可能有 GPU 等待长任务，不承诺全过程八卡满载。DSD 共生成 6,000 条音频，正常需要 6,000 次付费评审，失败重试另计。

重复执行同一命令会复用已完成结果。不要更改同一输出目录的 checkpoint、代码、batch 或 GPU 分片数。GPU 物理编号可更换，但分片数要一致。按 Ctrl+C 结束时只停止本入口创建的子进程，已经写出的音频/预测/评审会保留。

## 分阶段与结果

```bash
# 仅 Captioner 三个 benchmark
bash run_v6_benchmarks.sh captioner --gpus 0,1,2,3,4,5,6,7 --gpu-memory-gib 26

# 仅 TTS：生成并评审
bash run_v6_benchmarks.sh tts --gpus 0,1,2,3,4,5,6,7 --gpu-memory-gib 26

# TTS 也可拆成纯 GPU 生成、纯 API 评审
bash run_v6_benchmarks.sh tts-generate --gpus 0,1,2,3,4,5,6,7 --gpu-memory-gib 26
bash run_v6_benchmarks.sh tts-judge --gpus 0,1,2,3,4,5,6,7 --gpu-memory-gib 26 --workers 32

bash run_v6_benchmarks.sh status
```

同一输出根目录一次只能运行一个评测入口。API 失败保留逐条 checkpoint，结果未齐不显示为完整分数；可用 `tts-judge` 重试，无需重跑 GPU 生成。`--workers 64` 可提高 DSD 评审并发，但应根据该 API 服务的限流情况设置；训练中的 Qwen 打标不会用于 DSD 评分。

输出根目录：`/data/L202500147/Caption/benchmark/v6_round000_002_benchmarks_run01`

- 总表：`summary.md`、`summary.json`。
- Captioner：`captioner/{emotiontalk,paraspeechcaps,stylecap}/full/midasheng_v6_r{0,1,2}/`。
- DSD：`tts/round_00{0,1,2}/evaluation_gemini_2_5_pro_audio_api/dsd_summary.json`。
- DSD smoke 位于单独的 `smoke/` 子目录，模拟分数不会混入正式总表。
- 配置与身份：`inventory.json`；CPU 预检：`cpu_preflight.json`。

CPU 验证涵盖轮次选择、8 卡任务调度、显存参数、checkpoint 不匹配拒绝、DSD 合并/模拟评审/断点复用；真实 GPU 和真实 API 调用尚需用户服务器运行。
