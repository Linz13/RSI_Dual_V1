# DualISL V5 FastResume

This release resumes the stopped eight-GPU LabelRobust run after its first committed round. Training data stays in the original run directory; launch commands must use this source directory.

## Changes

- TTS rollout uses `rollout_batch_size: 1`, avoiding failed four-candidate batch attempts and serial regeneration. It calls the existing serial rollout once per rank, retaining its probability checks and reference replay reuse.
- Audio-only `generate-audio` is launched through torchrun when distributed execution is enabled.
- Synthesis assigns intact global batches to GPUs with greedy text-length balancing. Each batch preserves the existing length/ID ordering, four-candidate membership and batch RNG seed. CPU tests establish those contracts; exact waveform equality across GPU devices is not asserted.
- Synthesis keeps `synthesis_batch_size: 4`; Captioner generation remains batch 4 with its existing fallback.
- Newly saved checkpoint files receive shared read/write permissions at the worker-stage boundary, because safetensors can create mode-0600 files even with a shared umask.

GRPO, SFT, reward formulas, evaluator rules and checkpoint contents are preserved. Core changes are confined to `stages.py`, `workers/tts_batch.py`, and the synthesis dispatch branch of `workers/qwen_voice_design.py`. The original LabelRobust source remains available for the four-GPU task.

## Resume the eight-GPU run

The migration report is `reports/EIGHT_GPU_MIGRATION.json`. It records the source/config hashes, original checkpoint hashes, cache identities and backup directory.

```bash
cd /data/L202500147/Caption/DualISL_Train_RewardV5_FastResume
DUALISL_GPUS=0,1,2,3,4,5,6,7 \
bash scripts/run_v5.sh resume \
  /data/L202500147/Caption/DualISL_Train_RewardV5_LabelRobust/runs/midasheng_v5_label_robust_10rounds_8gpu_run01
```

The saved configuration retains 10 total rounds and paired anchor enabled. `current.round=0` means round one has committed; resume starts at `round_001` (round two), leaving nine rounds. Do not start a new `train` command on this run or use the original source to resume it after migration.

No batch environment override is needed for resume. A fresh run's generic `DUALISL_INFERENCE_BATCH` override still changes TTS rollout as well, so leave it unset to retain this release's serial TTS rollout default.

The four-GPU no-anchor task continues from `DualISL_Train_RewardV5_LabelRobust` with its existing run configuration. This migration applies only to the stopped eight-GPU run.

## Preservation and recovery

The first committed round, `latest.json`, checkpoints, calibration and stage outputs remain in place. Unfinished `round_001` files and its interrupted log are archived inside the migration backup. Round two starts fresh using round-one checkpoints.

All 1282 remote audio-label responses, 641 complete audio-label records and 387 GPT text judgments are imported under a new cache identity. Evaluator code and settings are checked for equivalence; values and scores are retained unchanged. The original cache remains in place. No API requests are made by the migration. Completed labels include local expert results, so original audio does not need to be labeled again.

Migration audits the old source lock, committed checkpoint hashes, completed stage artifacts, frozen external evaluators and cache provenance. Config/state backups are written before activation; recoverable write errors roll back both config and state and restore archived files. The migration refuses unsupported run states or changed evaluator/training methods.

## Validation

`reports/CPU_TESTS.log`: 183 passed, 2 skipped, 1 deselected. Coverage includes real CPU/Gloo eight-process shard merging, empty ranks, journal correspondence, global batch/seed preservation, serial rollout dispatch, migration and rollback, evaluator/cache contracts, and existing GRPO/SFT contracts. Optional dashboard dependencies are skipped; the optional GPU smoke contract is excluded on this CPU node.

The original first-round artifacts also pass the full V5 verifier (`reports/ROUND0_VERIFICATION_BEFORE_MIGRATION.json`). Its batching warnings describe the first round before this change.

The new GPU synthesis path has not been executed on this node. On the GPU server, round-two synthesis will write per-rank counts and `synthesis_placement` in:

`round_001/collections/audio_attribute_synthesis/round_001_audio_attribute_synthesis.output.jsonl.metrics.json`

Expect `distributed=true`, `world_size=8`, and `global_batch_membership_preserved=true`. TTS rollout should use the serial path without `batch_fallback` groups. Stage timing is saved in each `.stage.json`; actual speedup should be assessed there after round two.
