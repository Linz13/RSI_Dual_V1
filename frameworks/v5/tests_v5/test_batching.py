"""CPU tensor/transport doubles exercise batching, EOS and candidate correspondence."""
from types import SimpleNamespace
import contextlib
import json
import pytest


def test_midasheng_shared_audio_encoding_and_independent_eos():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from dual_isl_train.workers.midasheng_batch import generate_batch
    calls = []
    def prepare(**kwargs):
        calls.append("audio_encoding")
        return torch.zeros((1, 2, 3))
    class Decoder:
        def generate(self, inputs_embeds, logits_processor, **kwargs):
            assert inputs_embeds.shape == (4, 2, 3)
            seq = torch.zeros((4, 0), dtype=torch.long)
            finished = torch.zeros(4, dtype=torch.bool)
            for step in range(4):
                scores = torch.zeros((4, 8))
                for row in range(4):
                    scores[row, 7 if step >= row else row+1] = 5
                chosen = logits_processor(seq, scores).argmax(-1)
                chosen = torch.where(finished, 0, chosen)
                seq = torch.cat([seq, chosen[:, None]], dim=1)
                finished |= chosen == 7
            return SimpleNamespace(sequences=seq)
    base = SimpleNamespace(_prepare_inputs_embeds=prepare, decoder=Decoder(), generation_config=None)
    worker = SimpleNamespace(torch=torch, device="cpu",
        cfg={"generation": {"temperature": 0, "max_new_tokens": 4}},
        model=SimpleNamespace(set_adapter=lambda x: None, eval=lambda: None, get_base_model=lambda: base),
        freeze_batchnorm=lambda: None, _inputs=lambda *args: {"input_ids": torch.tensor([[9, 9]])},
        _warped_logprobs=lambda ids,scores,temp: torch.log_softmax(scores,dim=-1),
        eos_token_ids=[7], eos_token_id=7, pad_token_id=0,
        tokenizer=SimpleNamespace(decode=lambda ids,**kwargs: str(ids)))
    rows = generate_batch(worker, "fixture.wav", "prompt", [1,2,3,4])
    assert calls == ["audio_encoding"]
    assert [r["sampled_token_ids"] for r in rows] == [[7],[2,7],[3,3,7],[4,4,4,7]]
    assert all(len(r["old_token_logprobs"]) == r["generated_token_count"] for r in rows)
    assert all(r["terminated_by_eos"] and r["post_eos_token_count"] == 0 for r in rows)


def test_tts_pure_synthesis_batches_preserve_ids_and_oom_fallback(tmp_path):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    from dual_isl_train.workers.tts_batch import BatchedTTSMixin
    class OOM(RuntimeError):
        pass
    class Worker(BatchedTTSMixin):
        def _v5_generate(self, jobs, capture=False):
            assert not capture  # No rollout-score/replay overhead for audio-only synthesis.
            if len(jobs) > 2:
                raise OOM("injected batch capacity")
            return ([int(r["id"]) for r in jobs], None), 123
    w = Worker()
    w.cfg = {"generation": {"synthesis_batch_size": 4, "batch_fallback": True}}
    w.torch = SimpleNamespace(no_grad=contextlib.nullcontext, cuda=SimpleNamespace(OutOfMemoryError=OOM, empty_cache=lambda: None))
    w.policy = SimpleNamespace(eval=lambda: None)
    w.distributed = SimpleNamespace(rank=0)
    def decode(codes):
        return [np.full(100, c["audio_codes"]/10, dtype=np.float32) for c in codes], 24000
    w.core = SimpleNamespace(speech_tokenizer=SimpleNamespace(decode=decode))
    rows = [{"id": str(i), "candidate_id": str(i), "request": {"text": "a" * (6-i)}, "generation_seed": i}
            for i in (1,2,3,4,5)]
    output = w.generate_audio(rows, str(tmp_path / "output.jsonl"))
    assert [r["id"] for r in output] == [r["id"] for r in rows]
    for row in output:
        wav, sr = sf.read(row["audio_path"])
        assert wav.mean() == pytest.approx(int(row["id"])/10, abs=0.0001)
        assert sr == 24000
    journal = [json.loads(l) for l in (tmp_path / "rank_000.ready.jsonl").read_text().splitlines()]
    assert {r["id"] for r in journal} == {r["id"] for r in rows}
    assert all(r["synthesis_batch_size"] == 1 for r in output)


@pytest.mark.parametrize("batch_matches,malformed_caption,allow_fallback", [
    (False, False, True), (True, False, True), (True, True, True), (False, False, False),
])
def test_midasheng_rollout_retries_bad_batch_trajectories_only(
        monkeypatch, batch_matches, malformed_caption, allow_fallback):
    torch = pytest.importorskip("torch")
    from scripts.midasheng_captioner_candidate import MiDashengCaptionerWorker
    from dual_isl_train.workers import midasheng_batch
    from dual_isl_train.dual_space import project_synth_caption, synth_caption_json
    from dual_isl_train.workers.mock import mock_caption
    calls = []
    raw = "{" if malformed_caption else synth_caption_json(project_synth_caption(mock_caption(0)))
    def sample(seed, logp, size):
        return {"raw_text": raw, "sampled_token_ids": [seed, 99], "old_token_logprobs": [logp, logp],
                "finish_reason": "eos", "terminal_token_id": 99, "terminated_by_eos": True,
                "post_eos_token_count": 0, "generated_token_count": 2, "eos_token_id": 99,
                "generation_batch_size": size}
    def batch(worker, audio_path, prompt, seeds, temperature=None):
        calls.append(("batch", audio_path, seeds))
        return [sample(seed, -0.3 if batch_matches else -0.5, len(seeds)) for seed in seeds]
    class Worker(MiDashengCaptionerWorker):
        def __init__(self):
            self.torch, self.seed = torch, 1
            self.cfg = {"generation": {"rollout_batch_size": 4, "batch_fallback": allow_fallback}}

        def _serial_generate_batch(self, audio_path, prompt, seeds, *, temperature=None):
            calls.append(("serial", audio_path, seeds))
            return [sample(seed, -0.3, 1) for seed in seeds]

        def sampled_token_logprobs_batch(self, audio, prompt, ids, **kwargs):
            return [torch.full((len(row),), -0.3) for row in ids]
    monkeypatch.setattr(midasheng_batch, "generate_batch", batch)
    jobs = [{"id": name, "audio_path": name + ".wav", "prompt": "fixture", "caption_schema": "synth_v1",
             "candidate_seeds": [1, 2, 3, 4], "group_size": 4} for name in ("a", "b")]
    worker = Worker()
    if not batch_matches and not allow_fallback:
        with pytest.raises(RuntimeError, match="batched rollout failed trajectory"):
            worker.rollout(jobs)
        assert [call[0] for call in calls] == ["batch"]
        return
    groups = worker.rollout(jobs)
    assert [call[0] for call in calls] == (["batch", "batch"] if batch_matches else ["batch", "serial", "serial"])
    for group in groups:
        assert [c["generation_seed"] for c in group["candidates"]] == [1, 2, 3, 4]
        assert all(c["trajectory_valid"] for c in group["candidates"])
        assert all(c["old_token_logprobs"] == [-0.3, -0.3] for c in group["candidates"])
        assert all(c["generation_batch_size"] == (4 if batch_matches else 1) for c in group["candidates"])
    if not batch_matches:
        assert groups[0]["rejected_batch_replay_max_abs_error"] > 5e-4
        assert all(c["generation_attempts"] == 2 for c in groups[0]["candidates"])
        assert all(c["generation_batch_fallback"] == "trajectory_mismatch" for c in groups[1]["candidates"])
