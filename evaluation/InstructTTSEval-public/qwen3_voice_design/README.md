# Qwen3-TTS deployment for InstructTTSEval

This directory adds a pinned, restartable Qwen3-TTS VoiceDesign pipeline to the
official InstructTTSEval repository. The official evaluation prompt remains
unchanged.

## Quick start

Run from `/data/L202500147/Caption/benchmark`:

```bash
bash run_instructtts_evaluations.sh prepare-data
bash run_instructtts_evaluations.sh check
bash run_instructtts_evaluations.sh smoke-local qwen8_r1
bash run_instructtts_evaluations.sh smoke-judge-dry-run qwen8_r1
```

The smoke run creates six files: the first EN and ZH sample, each synthesized
with APS, DSD, and RP instructions. Its outputs are under:

```text
qwen3_voice_design/runs/qwen8_r1/smoke_bilingual_seed42/
├── audios/{en,zh}/{APS,DSD,RP}/*.wav
├── generation_manifest.json
└── dry_run/{judge_results.jsonl,judge_metadata.json,summary.json}
```

The dry-run constructs Gemini inline-audio payloads and parses simulated
responses. It does not read credentials, initialize a Gemini client, make a
network request, or produce an official benchmark score.

## Registered checkpoints

`base`, `qwen8_r0`, `qwen8_r1`, `qwen8_r2`, `midasheng4_r0`,
`midasheng4_r1`, and `midasheng4_r2` are registered by the top-level driver.
Generation manifests include hashes for the dataset manifests, base model,
adapter configuration, and adapter weights. Reusing an output directory with a
different identity is rejected.

## Full and paid stages

Full local generation is deliberately separate from Gemini evaluation:

```bash
bash run_instructtts_evaluations.sh full-local qwen8_r1
```

Real API requests require both a paid subcommand and explicit confirmation:

```bash
CONFIRM_PAID=YES bash run_instructtts_evaluations.sh smoke-paid qwen8_r1
CONFIRM_PAID=YES bash run_instructtts_evaluations.sh full-paid qwen8_r1
```

The judge reads `JUDGER_API_KEY` or `GENAI_API_KEY`, with a fallback to the
existing private EmergentTTS-Eval judger configuration. A custom base URL uses
inline WAV data; the Google endpoint uses the official Files API. Keys are
never copied into this repository or printed.

共享 `emergent-tts-eval` 环境固定使用 manylinux2014 版 `cryptography==43.0.3`
（最高 GLIBC 要求 2.17），以兼容不同系统版本的 GPU/API 节点。Google SDK 采用
延迟导入，因此本地生成、CPU 测试和离线 dry-run 不依赖它的动态链接库。

Paid results checkpoint each `(language, id, task)` independently. After all
retries are exhausted, isolated request failures no longer stop the remaining
model cases. Scoring uses successful judge results only and keeps
`complete=false`/`official=false` whenever an expected record is missing or
failed. `summary.json` records the expected and scored counts, coverage,
per-cell denominators, and the failed sample keys and errors. Token usage and a
configurable nominal cost estimate are included as well. Rerunning the same
paid command skips successful and already-failed records so a completed retry
cycle is not repeated. Set `INSTRUCTTTS_RETRY_FAILED=YES` to retry recorded
failures deliberately.

`INSTRUCTTTS_JUDGE_MODEL`、`INSTRUCTTTS_JUDGE_BACKEND` 和
`INSTRUCTTTS_JUDGE_TAG` 可分别固定 judge 模型、传输 backend 和隔离输出目录。
论文使用的 `gemini-2.5-pro-preview-05-06` 已被 Google 下线，因此当前统一入口
默认使用 `models/gemini-2.5-pro`，并将结果标为非严格官方复现。

单节点 8 卡批量评测 base、Qwen r0/r1/r2、MiDasheng r0/r1/r2 时，从 benchmark
根目录运行：

```bash
EVAL_GPUS=0,1,2,3,4,5,6,7 \
  bash run_stylecap_instructtts_8gpu.sh instruct-smoke-local
EVAL_GPUS=0,1,2,3,4,5,6,7 \
  bash run_stylecap_instructtts_8gpu.sh instruct-full-local
CONFIRM_PAID=YES bash run_stylecap_instructtts_8gpu.sh instruct-full-paid
```

TTS base 在两条训练分支间相同，因此只评一次；本地完整生成共 7 个 case、
42,000 个 WAV，随后 Gemini 阶段对应 42,000 次可断点续跑的裁判请求。
本地生成默认在每张 GPU 上使用 batch size 8，可用 `INSTRUCTTTS_BATCH_SIZE`
调整。统一入口先让 StyleCap 占满 8 卡，结束后再启动 7 个 InstructTTSEval
case，避免两个 benchmark 抢显存。Gemini judge 默认每次只处理一个模型 case，
该 case 内开 128 个请求 worker（`INSTRUCTTTS_GEMINI_WORKERS` 可调整），因此
全局并发上限是 128，而不是 7 × 128。
