# 代码阅读顺序

## V5

1. [启动脚本](../frameworks/v5/scripts/run_v5.sh) → [launcher](../frameworks/v5/scripts/v5_launcher.py) → [配置](../frameworks/v5/configs/v5_midasheng.yaml)。
2. [orchestrator_v4.py](../frameworks/v5/dual_isl_train/orchestrator_v4.py) 的 `train()` 看轮次与更新顺序，再看 [orchestrator_v5.py](../frameworks/v5/dual_isl_train/orchestrator_v5.py) 的覆盖方法。
3. audio-only 看 [attribute_reward.py](../frameworks/v5/dual_isl_train/attribute_reward.py)；caption-only 看 [rewards.py](../frameworks/v5/dual_isl_train/rewards.py)。
4. [partial_caption.py](../frameworks/v5/dual_isl_train/partial_caption.py) 看字段准入；[labeling.py](../frameworks/v5/dual_isl_train/labeling.py) 看打标、文本评审、缓存和失败处理。
5. [MiDasheng worker](../frameworks/v5/scripts/midasheng_captioner_candidate.py) 与 [TTS worker](../frameworks/v5/dual_isl_train/workers/qwen_voice_design.py) 看真实推理和 loss。
6. [stages.py](../frameworks/v5/dual_isl_train/stages.py) 与 [checkpoints.py](../frameworks/v5/dual_isl_train/checkpoints.py) 看多卡阶段、断点与提交。

`orchestrator.py` 是入口别名，`orchestrator_v4.py` 是 V5 调度继承的实现；不用把这三个文件当作三个独立训练版本。源码目录中保留被复用的历史 worker/config/test，先沿上述入口阅读即可。

## V6

1. [v6.yaml](../frameworks/v6/configs/v6.yaml) → [launcher](../frameworks/v6/scripts/v6_launcher.py)。
2. [orchestrator_v6.py](../frameworks/v6/dual_isl_train/orchestrator_v6.py)：`train()` / `collect()`。
3. [schema_v6.py](../frameworks/v6/dual_isl_train/schema_v6.py)：五字段和格式评分；[content_v6.py](../frameworks/v6/dual_isl_train/content_v6.py)：内容门槛。
4. [reward_v6.py](../frameworks/v6/dual_isl_train/reward_v6.py)：重建分、GRPO 优势、TTS SFT 选择。
5. [labeling_v6.py](../frameworks/v6/dual_isl_train/labeling_v6.py) 与 [API 扩展](../frameworks/v6/runtime_api_workers_v6.py)：固定裁判与实际并发重试。
6. [TTS worker](../frameworks/v6/dual_isl_train/workers/qwen_voice_design.py)：codec teacher forcing 和 SFT。

## 评测与诊断

- [v6_caption_all_rounds.py](../evaluation/v6_caption_all_rounds.py)：十轮 Captioner 评测的调度与结果复用。
- [v6_round_benchmarks.py](../evaluation/v6_round_benchmarks.py)：Captioner / TTS 多轮评测。
- [v5_dsd_audio_api_judge.py](../evaluation/v5_dsd_audio_api_judge.py)：已有 DSD 音频 API 评分。
- [v5_loss_reward_probe.py](../evaluation/v5_loss_reward_probe.py)：固定目标音频/转录的 likelihood 奖励探测。
- [speech_similarity_probe.py](../evaluation/speech_similarity_probe.py)：SpeechBERTScore 与平均特征 cosine 对照。
