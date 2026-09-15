# Speech similarity experiment environment

The comparison uses the existing `/data/L202500147/miniconda3/envs/qwen3-tts`
environment. No new conda environment or license prompt is needed.

The two frozen local encoders are:

- `models/WavLM-Large-SpeechMetrics`: `microsoft/wavlm-large`, revision
  `c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c`, weights verified by SHA256.
- `cache/labeling2/huggingface/hub/models--facebook--wav2vec2-large-xlsr-53/…`:
  the existing pinned `facebook/wav2vec2-large-xlsr-53` asset.

The upstream `discrete-speech-metrics` source is installed in the environment.
`speech_similarity_env.py` imports only its official `speechbertscore.py`, so
optional MCD/PESQ dependencies are not part of this experiment.

CPU asset check:

```bash
cd /data/L202500147/Caption/benchmark
/data/L202500147/miniconda3/envs/qwen3-tts/bin/python speech_similarity_env.py
```

The actual comparison should run on the other GPU server. It will keep both
encoders frozen, use the same reference/generated audio pairs, and write only
an evaluation report; it will not touch training checkpoints.
