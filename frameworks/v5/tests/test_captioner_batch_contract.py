from __future__ import annotations

import inspect

from dual_isl_train.workers.qwen3_captioner import Qwen3CaptionerWorker, _batches


def test_batch_slicing_preserves_order_and_remainder():
    assert list(_batches([0, 1, 2, 3], 4)) == [[0, 1, 2, 3]]
    assert list(_batches([0, 1, 2], 2)) == [[0, 1], [2]]


def test_captioner_rollout_batches_candidates_and_both_frozen_replays():
    source = inspect.getsource(Qwen3CaptionerWorker.rollout)
    assert "generate_batch" in source
    assert source.count("sampled_token_logprobs_batch") == 2
    assert "sampled_token_logprobs(" not in source


def test_captioner_grpo_batches_current_policy_replay_without_dropping_candidates():
    update_source = inspect.getsource(Qwen3CaptionerWorker.grpo_update)
    replay_source = inspect.getsource(Qwen3CaptionerWorker._native_candidates_backward)
    assert "_native_candidates_backward" in update_source
    assert "candidate_count=len(candidates)" in update_source
    assert "num_return_sequences=len(candidates)" in replay_source
    assert "candidate_count * row_sampled.numel()" in replay_source
