# Portable local-model tmux runs

The three local models can be resumed independently into the same `RUN_DIR`.
The manifest must contain audio paths that exist on the destination server.
Use one physical GPU per concurrently loaded model.

Set paths for the destination server:

```bash
export AUDIO_CAPTION_ROOT=/path/to/Audio_caption_share
export MANIFEST=/path/to/labeling_manifest.jsonl
export RUN_DIR=/path/to/labeling2_full_run
export CONFIG="$AUDIO_CAPTION_ROOT/Experiment/labeling2/default_config.json"

export LABELING2_QWEN3_PYTHON=/path/to/qwen3-captioner/bin/python
export LABELING2_QWEN3_MODEL_DIR=/path/to/Qwen3-Omni-30B-A3B-Captioner

export LABELING2_KIMI_PYTHON=/path/to/kimi_audio/bin/python
export LABELING2_KIMI_MODEL_DIR=/path/to/Kimi-Audio-7B-Instruct
export LABELING2_KIMI_TOKENIZER_DIR=/path/to/glm-4-voice-tokenizer

export LABELING2_STEP_PYTHON=/path/to/stepaudio/bin/python
export STEP_MODEL_DIR=/path/to/Step-Audio-R1.1
```

## Qwen3-Captioner on GPU 0

```bash
tmux new-session -d -s labeling_qwen3 \
  "cd '$AUDIO_CAPTION_ROOT' && \
   CUDA_VISIBLE_DEVICES=0 \
   AUDIO_CAPTION_ROOT='$AUDIO_CAPTION_ROOT' \
   LABELING2_QWEN3_PYTHON='$LABELING2_QWEN3_PYTHON' \
   LABELING2_QWEN3_MODEL_DIR='$LABELING2_QWEN3_MODEL_DIR' \
   python Experiment/labeling2/run_labeling.py run \
     --manifest '$MANIFEST' --config '$CONFIG' --run-dir '$RUN_DIR' \
     --stage general --models qwen3_captioner --resume \
   2>&1 | tee '$RUN_DIR/qwen3_captioner.log'"
```

## Kimi-Audio on GPU 1

```bash
tmux new-session -d -s labeling_kimi \
  "cd '$AUDIO_CAPTION_ROOT' && \
   CUDA_VISIBLE_DEVICES=1 \
   AUDIO_CAPTION_ROOT='$AUDIO_CAPTION_ROOT' \
   LABELING2_KIMI_PYTHON='$LABELING2_KIMI_PYTHON' \
   LABELING2_KIMI_MODEL_DIR='$LABELING2_KIMI_MODEL_DIR' \
   LABELING2_KIMI_TOKENIZER_DIR='$LABELING2_KIMI_TOKENIZER_DIR' \
   python Experiment/labeling2/run_labeling.py run \
     --manifest '$MANIFEST' --config '$CONFIG' --run-dir '$RUN_DIR' \
     --stage general --models kimi_audio --resume \
   2>&1 | tee '$RUN_DIR/kimi_audio.log'"
```

Qwen3-Captioner and Kimi-Audio can run at the same time when GPU 0 and GPU 1
are distinct physical GPUs with enough free memory. On a one-GPU server, run
them sequentially.

## Step-Audio-R1.1 on GPU 2

Start its server in one tmux session:

```bash
tmux new-session -d -s step_server \
  "cd '$AUDIO_CAPTION_ROOT' && \
   CUDA_VISIBLE_DEVICES=2 '$LABELING2_STEP_PYTHON' \
     lzy/Step-Audio-R1/serve_step_audio_r1_1.py \
     --python '$LABELING2_STEP_PYTHON' \
     --model '$STEP_MODEL_DIR' \
     --host 127.0.0.1 --port 9999 \
     --max-model-len 4096 --max-num-seqs 1 \
     --gpu-memory-utilization 0.92 \
   2>&1 | tee '$RUN_DIR/step_audio_server.log'"
```

After port 9999 is ready, start its labeling client:

```bash
tmux new-session -d -s labeling_step \
  "cd '$AUDIO_CAPTION_ROOT' && \
   CUDA_VISIBLE_DEVICES=2 \
   AUDIO_CAPTION_ROOT='$AUDIO_CAPTION_ROOT' \
   LABELING2_STEP_PYTHON='$LABELING2_STEP_PYTHON' \
   LABELING2_STEP_API_URL='http://127.0.0.1:9999/v1/chat/completions' \
   python Experiment/labeling2/run_labeling.py run \
     --manifest '$MANIFEST' --config '$CONFIG' --run-dir '$RUN_DIR' \
     --stage general --models step_audio_r1_1 --resume \
   2>&1 | tee '$RUN_DIR/step_audio_r1_1.log'"
```

Attach to a progress display with `tmux attach -t labeling_qwen3`,
`tmux attach -t labeling_kimi`, or `tmux attach -t labeling_step`. Detach with
`Ctrl-b d`. Every local worker prints completed count, percentage, elapsed
time, and ETA after each sample. All commands use resumable append-only output.

Outputs are written to:

```text
$RUN_DIR/raw_predictions/qwen3_captioner.jsonl
$RUN_DIR/raw_predictions/kimi_audio.jsonl
$RUN_DIR/raw_predictions/step_audio_r1_1.jsonl
```
