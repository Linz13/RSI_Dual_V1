# labeling2

This directory contains the resumable audio-attribute labeling pipeline. It
does not copy audio files and does not modify the Caption_Bench experiments.

## Manifest

One JSON object per line is required:

```json
{"sample_id":"demo-001","audio_path":"/absolute/path/demo.wav","dataset":"demo","metadata":{},"transcript":"可选，仅供语速专家使用","language_hint":"Chinese"}
```

`sample_id` must be unique. `audio_path` is resolved and checked during
preflight. Metadata and transcript are never included in general-model
prompts; transcript is copied to the speed expert task only. The machine-
readable row contract is in [`manifest_schema.json`](manifest_schema.json).

## Commands

```bash
cd /F00120250029/lixiang_share/Audio_caption_share/Experiment/labeling2

python run_labeling.py preflight \
  --manifest /path/to/data.jsonl \
  --config default_config.json

python run_labeling.py run \
  --manifest /path/to/data.jsonl \
  --config default_config.json \
  --run-dir runs/first_run \
  --max-samples 1 \
  --resume \
  --allow-partial

python run_labeling.py finalize \
  --manifest /path/to/data.jsonl \
  --run-dir runs/first_run
```

## `test_labeling/v1` presets

The repository test set is available without copying or editing its manifests:

- `test-labeling-smoke`: `../data/test_labeling/v1/manifests/smoke_manifest.jsonl` (24 samples)
- `test-labeling-full`: `../data/test_labeling/v1/manifests/labeling_manifest.jsonl` (300 samples)

Preset runs default to `runs/test_labeling_v1/smoke` and
`runs/test_labeling_v1/full`.  `--manifest` and `--preset` are mutually
exclusive.  Use `--models` and `--experts` to run only selected backends; when
omitted, all configured backends are used.

Install the Chinese rate expert dependency in its configured environment and
provide API credentials before running:

```bash
/F00120250029/lixiang_share/Audio_caption_share/miniconda3/envs/paraspeechcaps_rate/bin/python \
  -m pip install -r /F00120250029/lixiang_share/Audio_caption_share/Experiment/labeling2/requirements.txt
export GEMINI_API_KEY='...'
export QWEN_API_KEY='...'
```

Run the non-StepAudio smoke stages:

```bash
python run_labeling.py preflight \
  --preset test-labeling-smoke \
  --models gemini qwen35 qwen3_captioner kimi_audio \
  --experts volume emotion accent_en accent_zh rate_en rate_zh

python run_labeling.py run \
  --preset test-labeling-smoke \
  --stage general \
  --models gemini qwen35 qwen3_captioner kimi_audio \
  --resume

python run_labeling.py run \
  --preset test-labeling-smoke \
  --stage experts \
  --experts volume emotion accent_en accent_zh rate_en rate_zh \
  --resume
```

Start StepAudio manually in a separate terminal, keeping its GPU reserved:

```bash
CUDA_VISIBLE_DEVICES=0 \
/F00120250029/lixiang_share/Audio_caption_share/miniconda3/envs/stepaudio/bin/python \
  /F00120250029/lixiang_share/Audio_caption_share/lzy/Step-Audio-R1/serve_step_audio_r1_1.py \
  --host 127.0.0.1 --port 9999 --max-model-len 4096 \
  --max-num-seqs 1 --gpu-memory-utilization 0.92
```

After the service is healthy, run its client and finalize:

```bash
python run_labeling.py run \
  --preset test-labeling-smoke \
  --stage general \
  --models step_audio_r1_1 \
  --resume

python run_labeling.py finalize --preset test-labeling-smoke

python audit_run.py \
  --run-dir runs/test_labeling_v1/smoke \
  --expected-samples 24
```

Replace `test-labeling-smoke` with `test-labeling-full` for the 300-sample
run.  Existing run directories require `--resume`; the pipeline refuses to
combine a different manifest snapshot with an existing run.

For the full preset, audit with `--run-dir runs/test_labeling_v1/full
--expected-samples 300` after finalization.

`--allow-partial` is useful for development when optional model environments
or API credentials are not installed. A production run should pass preflight
without it. Step-Audio-R1.1 still needs its server started separately.

Each run writes append-only raw and expert predictions, then deterministic
`final/labels.jsonl`, `final/provenance.jsonl`, `final/open_candidates.jsonl`,
`final/review_queue.jsonl`, and `run_summary.json`.

English rate uses the existing ParaSpeechCaps/DataSpeech thresholds with
G2P plus Brouhaha VAD. Chinese rate is explicitly the paper-inspired
initial/final pinyin-rate deployment approximation described in the plan.
Finalization leaves unresolved multi-model open fields in the review queue;
it does not silently invent a consensus.

Resolve those multi-model open fields with the text judge configured in
`/F00120250029/lixiang_share/Audio_caption_share/lzy/api/gpt_text.py`:

```bash
python run_labeling.py resolve-open \
  --preset test-labeling-full \
  --resume
```

The helper is read as configuration only and is never imported or executed.
The original `final/labels.jsonl` remains unchanged. Complete postprocessed
labels are written to `final/labels_open_resolved.jsonl`, with matching
provenance and review files plus append-only responses in `open_resolution/`.
Derived outputs can be rebuilt without API calls:

```bash
python run_labeling.py finalize-open --preset test-labeling-full
```

## API-only staged runs

When GPU/local backends are intentionally deferred, run only the remote audio
models, resolve their shared open fields, and export a resumable intermediate
result:

```bash
python run_labeling.py run \
  --manifest /path/to/manifest.jsonl \
  --run-dir runs/api_stage \
  --stage general --models gemini qwen35 --resume

python run_labeling.py finalize \
  --manifest /path/to/manifest.jsonl --run-dir runs/api_stage
python run_labeling.py resolve-open \
  --manifest /path/to/manifest.jsonl --run-dir runs/api_stage --resume
python run_labeling.py finalize-api-stage \
  --manifest /path/to/manifest.jsonl --run-dir runs/api_stage
```

`final/labels_api_stage.jsonl` keeps valid API values, marks values awaiting
GPU consensus as `api_provisional`, and represents GPU-only fields as JSON
`null` with `not_run` provenance. Later local-model and expert stages can use
the same run directory with `--resume`; the ordinary finalizer remains the
authoritative full-pipeline output.

Run tests from the repository root:

```bash
python -m unittest discover -s Experiment/labeling2/tests -p 'test_*.py'
```
