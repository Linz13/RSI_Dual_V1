# EmotionTalk Speech Captioning Benchmark

本目录只复现 EmotionTalk 论文 Table 5 的 **speech-only Emotional Speaker Style Captioning** benchmark。它不训练模型，也不复现 Text / Visual / Multimodal emotion recognition。

数据划分以官方仓库 `audio_label.npz:test_corpus` 为唯一真值：1,929 条 speaker-independent test utterances，来自 G00003（793 条）和 G00015（1,136 条）。代码不会重新划分、改写 caption、用 caption 生成 GT，或将任务改成选择题。

## 数据来源与固定版本

- 论文：[EmotionTalk: An Interactive Chinese Multimodal Emotion Dataset With Rich Annotations](https://aclanthology.org/2026.findings-acl.440/)
- 官方代码：[NKU-HLT/EmotionTalk](https://github.com/NKU-HLT/EmotionTalk)，固定 commit `cb8397e226ce7c1fccee41ea21161a3f98f578e1`
- 官方数据：[BAAI/Emotiontalk](https://huggingface.co/datasets/BAAI/Emotiontalk)，固定 revision `adbc17fc944e8cf2873643906160c6ca0259ab61`
- 仅下载 `Audio.tar` 和 `Text.tar`，不下载 Video/Multimodal。

HF 数据集是 gated dataset。先在网页接受条款，再执行：

```bash
hf auth login
python scripts/download_official_data.py
python scripts/prepare_benchmark.py
python validate_benchmark.py
pytest -q
```

`data/provenance.json` 记录固定 revision、源文件大小/SHA-256、官方 split ID hash、字段映射、speaker 集合及实际重叠。

## Qwen base smoke 闭环

统一入口 `run_emotiontalk_smoke.sh` 固定 Qwen3-Omni base、GPU 0、1 条官方 test 音频、四个 task、greedy decoding 和 `max_new_tokens=128`。它不会启动 1,929×4 全量推理，也不会接入 LoRA/MiDasheng。新结果写入 `runs/smoke_qwen3_base_20260901_run04`，并设置 `umask 000` 以便跨节点续跑。四个 prompt 要求模型仅依据音频输出一句简短中文描述：

```bash
cd /data/L202500147/Caption/benchmark/emotiontalk_speech_captioning
bash run_emotiontalk_smoke.sh prepare-data
bash run_emotiontalk_smoke.sh check
bash run_emotiontalk_smoke.sh prepare-metrics
bash run_emotiontalk_smoke.sh smoke-local
bash run_emotiontalk_smoke.sh smoke-score
bash run_emotiontalk_smoke.sh resume-check
```

也可以一次执行 `bash run_emotiontalk_smoke.sh smoke`。`smoke-score` 带有 `--require-all-metrics`：BLEU-4、ROUGE-L、METEOR、SPIDEr、FENSE、Chinese BERTScore、MS-CLAP 任一不可用都会在写出报告后以非零状态退出。重复运行 `smoke-local` 会校验运行身份并复用已完成事件。

## 八模型、八卡评测

上层入口 `../run_emotiontalk_8gpu.sh` 复用 StyleCap/PromptSpeech MCQ 的同一组八个候选，并固定一张物理 GPU 对应一个模型进程：

| GPU 顺序 | 运行名 | Base model | Adapter |
|---:|---|---|---|
| 0 | `qwen_base` | Qwen3-Omni-30B-A3B-Captioner | 无 |
| 1–3 | `qwen8_round0/1/2` | Qwen3-Omni-30B-A3B-Captioner | Qwen 8-card round 0/1/2 |
| 4 | `midasheng_base` | MiDashengLM-7B-1021-BF16 | 无 |
| 5–7 | `midasheng4_round0/1/2` | MiDashengLM-7B-1021-BF16 | MiDasheng 4-card round 0/1/2 |

在目标八卡服务器先做路径、数据、环境、adapter 身份及单测检查，再跑八模型 smoke；确认八个模型均得到四条结果和七项可用指标后才启动全量：

```bash
cd /data/L202500147/Caption/benchmark
export EVAL_GPUS=0,1,2,3,4,5,6,7
export EMOTIONTALK_RUN_TAG=20260901_run01

bash run_emotiontalk_8gpu.sh prepare
bash run_emotiontalk_8gpu.sh smoke
bash run_emotiontalk_8gpu.sh full
```

`smoke` 是每模型 1 条音频 × 4 个 task；`full` 是每模型 1,929 条音频 × 4 个 task。推理按模型独立追加到 `events.jsonl`，重复原命令会校验身份并只补未完成项。默认 Qwen/MiDasheng batch size 都为 4；显存不足时应在新 run tag 下用 `EMOTIONTALK_QWEN_BATCH_SIZE` 或 `EMOTIONTALK_MIDASHENG_BATCH_SIZE` 调小。模型、环境、训练 run 和总输出根目录都可用脚本开头对应的环境变量覆盖。

八模型结果分别位于 `runs/eight_models_${EMOTIONTALK_RUN_TAG}/{smoke,full}/<model>/`，汇总表为同级 `comparison.md` 和 `comparison.json`，日志集中在 `logs/`。评分进程同样一模型一 GPU，并为 AAC Metrics 使用隔离的临时目录。

## 实际字段映射

官方 `audio.csv` 的顶层字段实际为 `file_name,emotion,content`；`content` 是 Python dict 字符串，准备脚本通过 `ast.literal_eval` 解析。映射如下：

| Benchmark 字段 | 官方来源 |
|---|---|
| `speaker_caption` | `audio.csv content.spe_cap` |
| `style_caption` | `audio.csv content.style_cap` |
| `emotion_caption` | `audio.csv content.emo_cap` |
| `overall_captions` | `audio.csv content.caption_1..caption_5` |
| `transcript` | `transcription.csv chinese`，仅分析 |
| `speaker_id` | `Text.tar` sample JSON 的官方 `speaker_id` |

所有 caption 字符串逐字保存，不 strip、不去重、不改写。`Text.tar:speaker_id` 会逐条与 utterance 文件名第三段 token 核对；只有该字段缺失时才使用文件名 token，并写明 `derived_from_official_filename`。

## 四个任务和输入隔离

`data/test_references.jsonl` 每个 utterance 一行，包含 GT、speaker、analysis-only transcript 和相对音频路径。`data/test_inference.jsonl` 每个 utterance 四行，只允许以下字段：

```json
{"id":"...","task":"style","audio_path":"audio/test/...wav","prompt":"..."}
```

四任务为：

- `speaker`：单参考 `speaker_caption`
- `style`：单参考 `style_caption`
- `emotion`：单参考 `emotion_caption`
- `overall`：一次传入全部五个 `overall_captions`

prompt 原文保存在 `config/prompts.json`。推理 manifest 绝不含 transcript、GT 或 reference；backend 接收的请求也只有 `id/task/audio_path/prompt`。

## 推理

先用 mock backend 调试接口：

```bash
python run_inference.py \
  --backend mock \
  --tasks all \
  --max-samples 1 \
  --output-dir runs/smoke_mock
```

本地 Qwen3 Captioner 的 1 条音频 × 4 task smoke：

```bash
/data/L202500147/miniconda3/envs/qwen3-captioner/bin/python run_inference.py \
  --backend qwen3_omni \
  --tasks all \
  --max-samples 1 \
  --gpu 0 \
  --model-path /data/L202500147/Caption/models/Qwen3-Omni-30B-A3B-Captioner \
  --output-dir runs/smoke_qwen3
```

成功输出 `predictions.jsonl`，严格为 `{id,task,prediction}`。错误和 traceback 写入 `events.jsonl`，运行身份写入 `run_metadata.json`，不会污染 prediction schema。重复执行时加 `--resume`；manifest、prompt、模型路径、任务或解码参数变化会拒绝续跑。

未来全量推理（本次部署不执行）：

```bash
python run_inference.py \
  --backend qwen3_omni \
  --tasks all \
  --gpu 0 \
  --output-dir runs/qwen3_full
```

可用 `module:Class` 注册自定义 `CaptionBackend`，方便接入自研 captioner 或其他 audio-language model。内置 Qwen3-Omni 与 MiDasheng backend 均支持 batch 推理及可选 PEFT adapter。

## 指标

作者官方仓库未发布 Table 5 的评分脚本、中文分词配置、FENSE/CLAP checkpoint 配置，因此这里只提供明确标记为 **非论文精确复现** 的 `standard_public` profile：

- `aac-metrics==0.6.0`：BLEU-4、ROUGE-L、METEOR、SPIDEr、FENSE
- `google-bert/bert-base-chinese`：BERTScore，多参考取 max
- `MS-CLAP-2023`：audio–prediction CLAP cosine similarity
- OpenJDK 11：PTB/METEOR/SPICE 依赖

安装独立指标环境：

```bash
conda env create -f environment.metrics.yml
conda activate emotiontalk-metrics
# 资源统一放在 benchmark/cache，不使用 Captioner 环境中的临时 venv
bash run_emotiontalk_smoke.sh prepare-metrics
```

Smoke 评分（严格要求 7 个指标均可用）：

```bash
python evaluate.py \
  --predictions runs/smoke_qwen3_base_20260901_run04/predictions.jsonl \
  --profile standard_public \
  --allow-partial \
  --require-all-metrics \
  --output-dir runs/smoke_qwen3_base_20260901_run04/metrics_standard_public
```

未来全量评分不加 `--allow-partial`；程序会严格要求每个 task 各 1,929 条：

```bash
python evaluate.py \
  --predictions runs/qwen3_full/predictions.jsonl \
  --profile standard_public \
  --output-dir runs/qwen3_full/metrics
```

每个指标独立执行。缺少依赖、模型下载失败或实现不适用时，输出 `unavailable`、版本和原始异常，不使用替代公式。FENSE、SPICE/SPIDEr 与 MS-CLAP 默认模型并非面向中文 emotional speech，结果不可直接与 Table 5 数值比较。

## 校验内容

全量 `python validate_benchmark.py` 会检查：

- 1,929 references、7,716 inference requests、ID 唯一、四任务覆盖；
- ID 集合和顺序与官方 `test_corpus` 完全一致；
- G00003/G00015 的 793/1,136 构成；
- train/test speaker 不重叠，同时如实记录 validation/test overlap，绝不修改官方 split；
- 1,929 个 WAV 均存在、可解码，且帧数/采样率/channel 有效；
- 三种单参考 caption 非空、Overall 恰有五个非空原始参考；
- inference manifest 不出现 transcript、caption 或 reference 字段。

## 目录结构

```text
emotiontalk_speech_captioning/
├── README.md
├── config/{prompts.json,metrics_standard_public.json}
├── source/{official_repo,hf}
├── audio/test/{G00003,G00015}
├── data/{test_references.jsonl,test_inference.jsonl,provenance.json}
├── backends/{base.py,mock.py,qwen3_omni.py,midasheng.py}
├── scripts/{download_official_data.py,prepare_benchmark.py}
├── validate_benchmark.py
├── run_inference.py
├── evaluate.py
├── summarize_eight_models.py
├── environment.metrics.yml
├── tests/
└── runs/
```
