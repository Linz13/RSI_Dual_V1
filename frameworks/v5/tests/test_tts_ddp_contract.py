from __future__ import annotations

import inspect

from dual_isl_train.workers.qwen_voice_design import QwenVoiceDesignWorker


def test_incremental_grpo_enters_standard_ddp_forward():
    ensure_source = inspect.getsource(QwenVoiceDesignWorker._ensure_ddp)
    replay_source = inspect.getsource(QwenVoiceDesignWorker.incremental_candidate_logprobs)
    grpo_source = inspect.getsource(QwenVoiceDesignWorker.grpo_update)
    assert "DistributedDataParallel" in ensure_source
    assert "self.replay_module" in ensure_source
    assert "self.ddp_replay" in replay_source
    assert '"incremental", candidate["request"]' in replay_source
    assert "self.incremental_candidate_logprobs" in grpo_source
    assert "self.ddp_replay.no_sync()" in grpo_source
    assert "candidate_loss * backward_scale" in grpo_source
    assert "incremental_group_logprobs(train_candidates)" not in grpo_source
    assert "ddp_talker" not in ensure_source + replay_source + grpo_source
