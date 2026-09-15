# StyleCap / PromptSpeech Speaker-open MCQ Benchmark

本目录将 StyleCap 官方 speaker-open test split 的 778 条 LibriTTS 音频，按照
PromptSpeech Real 的四个结构化属性转换为 3112 道独立选择题。它不是
PromptSpeech 原始的 1305 条 test set，也不训练或复现 StyleCap 模型。

## 官方来源与字段

- StyleCap 官方 split：<https://ntt-hilab-gensp.github.io/icassp2024stylecap/train_dev_test_set.zip>
- PromptSpeech Real training metadata：<https://speechresearch.github.io/dataset/promptspeech/Real_training.zip>
- LibriTTS test-clean：<https://www.openslr.org/resources/60/test-clean.tar.gz>

StyleCap 使用 PromptSpeech Real **training** 的 26,588 条样本重新划分为
24,953/857/778，三个 split 分别有 1,113/40/38 位 speaker。PromptSpeech CSV
中的音量字段实际名为 `energy`；本 benchmark 仅作确定性字段重命名
`energy -> volume`，不会从 `style_prompt` 或 reference caption 推断 GT。

LibriTTS 官方页面将语料标为 CC BY 4.0。PromptSpeech 和 StyleCap 下载页面没有
声明单独的数据许可证，因此这里不自行补充许可证结论。

## 构建与验证

首次构建会下载约 1.23 GB 的官方 LibriTTS `test-clean.tar.gz`，但只提取目标
778 个 WAV：

```bash
cd /data/L202500147/Caption/benchmark/stylecap_promptspeech_mcq
python3 build_benchmark.py
python3 validate_benchmark.py
```

如果服务器已有官方 LibriTTS，可避免下载 archive：

```bash
python3 build_benchmark.py --libritts-root /path/to/LibriTTS
```

构建器可重复运行，缓存下载会经过固定 checksum 校验。输出为：

- `sources/`：官方 CSV、下载 archive 和来源/hash 清单；
- `data/test_audio.jsonl`：778 条音频级记录；
- `data/benchmark.jsonl`：3112 道选择题；
- `data/summary.json`：数量、分布和输出 hash；
- `audio/LibriTTS/test-clean/`：实际使用的 778 个 WAV。

## 固定题型与分布

每条音频按 `gender`、`pitch`、`speaking_speed`、`volume` 顺序生成四题，选项
顺序不随机。当前官方数据的分布为：

| 属性 | 分布 |
| --- | --- |
| Gender | Male 266，Female 512 |
| Pitch | Low 266，Normal 285，High 227 |
| Speaking Speed | Slow 278，Normal 263，Fast 237 |
| Volume | Low 318，Normal 234，High 226 |

题目记录示例：

```json
{"question_id":"stylecap-1089_134686_000003_000000-pitch","audio_id":"1089_134686_000003_000000","speaker_id":"1089","audio_path":"/data/L202500147/Caption/benchmark/stylecap_promptspeech_mcq/audio/LibriTTS/test-clean/1089/134686/1089_134686_000003_000000.wav","relative_audio_path":"audio/LibriTTS/test-clean/1089/134686/1089_134686_000003_000000.wav","task":"pitch","question":"请听这段语音。说话者整体的音高属于哪一类？","choices":{"A":"Low","B":"Normal","C":"High"},"answer":"C","label":"high"}
```

## 评测预测

预测文件必须完整包含 3112 个唯一 `question_id`，最小格式为：

```json
{"question_id":"stylecap-1089_134686_000003_000000-pitch","prediction":"C"}
```

`prediction` 可以是精确选项字母，也可以是精确标签文本（如 `high`）。不解析
“The answer is C”之类自由文本。缺失、重复或额外 ID 会直接报错；存在但无法解析
的预测计为错误。

```bash
python3 evaluate.py \
  --predictions /path/to/predictions.jsonl \
  --output /path/to/evaluation_summary.json
```

输出 Gender、Pitch、Speaking Speed、Volume Accuracy 及四项未加权 Macro Average。

## 使用 Qwen3-Omni / MiDasheng 实际评测

`run_midasheng.py`（保留历史文件名）现在同时支持 Qwen3-Omni 和 MiDasheng，
以确定性 greedy decoding 对
`benchmark.jsonl` 的 3112 道题批量推理。每批结果都会立即追加到
`generations.jsonl`，因此可以用 `--resume` 从服务器中断处继续；原始回复不会被
模糊解析，只有精确选项字母或精确标签会被接受。

本服务器的一键脚本默认使用：

- Base：`/data/L202500147/Caption/models/MiDashengLM-7B-1021-BF16`
- Adapter：`dual_recursive_4gpu_h100_midasheng_20260830_run01` 的
  `round_002/checkpoints/caption_final`
- GPU：物理 GPU 0
- 输出：`runs/midasheng4_round2_full_20260831_run01/`

完整运行和自动评分：

```bash
cd /data/L202500147/Caption/benchmark/stylecap_promptspeech_mcq
bash run_midasheng_evaluation.sh
```

可以通过环境变量选择 GPU、adapter 和输出目录：

```bash
MIDASHENG_GPU=0 \
MIDASHENG_ADAPTER_DIR=/path/to/caption_adapter \
MIDASHENG_OUTPUT_DIR=/path/to/run \
bash run_midasheng_evaluation.sh
```

直接调用 Python 时，可用 `--adapter-dir none` 测试未挂载 LoRA 的 base 模型。

两个 Captioner base 及两条训练分支的 r0/r1/r2 共八个 case，可在单节点
8 卡 H100 上并发运行：

```bash
cd /data/L202500147/Caption/benchmark
EVAL_GPUS=0,1,2,3,4,5,6,7 \
  bash run_stylecap_instructtts_8gpu.sh stylecap-smoke
EVAL_GPUS=0,1,2,3,4,5,6,7 \
  bash run_stylecap_instructtts_8gpu.sh stylecap-full
```

所有 case 都使用独立输出目录和 `--resume`。已有的 MiDasheng r2 完整结果会
按原 identity 复用，不会重复计算。统一入口中 Qwen3-Omni 和 MiDasheng 默认都
使用 batch size 4；可分别通过 `STYLECAP_QWEN_BATCH_SIZE` 和
`STYLECAP_MIDASHENG_BATCH_SIZE` 调整。批次失败时会自动二分，隔离 OOM 或坏样本。

## 单元测试

```bash
python3 -m unittest discover -s tests -v
```
