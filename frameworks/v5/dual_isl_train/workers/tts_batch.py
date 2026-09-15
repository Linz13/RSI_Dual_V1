"""Batched generation only. GRPO replay and backward stay in qwen_voice_design."""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace

from ..io import stable_hash
from ..trajectory import mark_trajectory
from ..constants import TRAJECTORY_VERSION
from ..reference_replay import RolloutReferenceReplay


def synthesis_assignment(rows, batch_size, world_size):
    """Assign intact global synthesis batches; keep batch RNG identical to one GPU.

    generate_audio sorts each rank's rows by (text length, ID). Assigning whole
    contiguous batches from that same global ordering preserves their members,
    order and seed. The sole short batch is globally last, hence also last on
    its assigned rank. Greedy placement uses text length as a cost estimate.
    """
    if batch_size < 1 or world_size < 1:
        raise ValueError("Synthesis batch size and world size must be positive")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Synthesis job IDs must be unique")
    ordered = sorted(range(len(rows)), key=lambda i: (len(rows[i]["request"]["text"]), rows[i]["id"]))
    batches = [ordered[start:start + batch_size] for start in range(0, len(ordered), batch_size)]
    costs = [max(1, max(len(rows[i]["request"]["text"]) for i in batch)) for batch in batches]
    loads, owners = [0] * world_size, [0] * len(rows)
    for index in sorted(range(len(batches)), key=lambda i: (-costs[i], i)):
        rank = min(range(world_size), key=lambda r: (loads[r], r))
        for i in batches[index]:
            owners[i] = rank
        loads[rank] += costs[index]
    return owners, {"algorithm": "intact_global_batches_length_lpt_v1", "batch_size": batch_size,
                    "batches": len(batches), "estimated_load_per_rank": loads,
                    "global_batch_membership_preserved": True}


class BatchedTTSMixin:
    def _v5_generate(self, jobs, capture=False):
        generation = self.cfg.get("generation", {})
        seed = int(stable_hash([j["generation_seed"] for j in jobs])[:15], 16) % (2**31 - 1)
        self.torch.manual_seed(seed)
        self.torch.cuda.manual_seed_all(seed)
        requests = [j["request"] for j in jobs]
        tokens = [self._tokenize_request(r) for r in requests]
        kwargs = dict(input_ids=[a.to(self.device) for a, b in tokens],
            instruct_ids=[b.to(self.device) if b is not None else None for a, b in tokens],
            languages=[r["language"] for r in requests], non_streaming_mode=True,
            max_new_tokens=int(generation.get("max_new_tokens", 512)), do_sample=True,
            top_k=int(generation.get("top_k", 0)), top_p=float(generation.get("top_p", 1)),
            temperature=float(generation.get("temperature", 1)), subtalker_dosample=True,
            subtalker_top_k=int(generation.get("subtalker_top_k", 0)),
            subtalker_top_p=float(generation.get("subtalker_top_p", 1)),
            subtalker_temperature=float(generation.get("subtalker_temperature", 1)),
            repetition_penalty=float(generation.get("repetition_penalty", 1)))
        with self.torch.no_grad(), self.adapter_context(False):
            result = self._generate_with_main_scores(**kwargs) if capture else self.core.generate(**kwargs)
        return result, seed

    def generate_audio(self, rows, output_path):
        import soundfile as sf
        import numpy as np
        directory = Path(output_path).parent / "audio"
        directory.mkdir(parents=True, exist_ok=True)
        journal = Path(output_path).parent / f"rank_{self.distributed.rank:03d}.ready.jsonl"
        output = []
        size = int(self.cfg.get("generation", {}).get("synthesis_batch_size", 4))
        ordered = sorted(rows, key=lambda r: (len(r["request"]["text"]), r["id"]))
        self.policy.eval()
        def run(jobs):
            failed = False
            try:
                (codes_list, _), seed = self._v5_generate(jobs)
                if len(codes_list) != len(jobs):
                    raise RuntimeError("TTS batch lost candidate correspondence")
                with self.torch.no_grad():
                    wavs, sr = self.core.speech_tokenizer.decode([{"audio_codes": c} for c in codes_list])
            except self.torch.cuda.OutOfMemoryError:
                if len(jobs) == 1 or not self.cfg.get("generation", {}).get("batch_fallback", True):
                    raise
                failed = True
            if failed:
                import gc
                codes_list = wavs = None
                gc.collect()
                self.torch.cuda.empty_cache()
                return sum((run([job]) for job in jobs), [])
            result = []
            for job, wav in zip(jobs, wavs, strict=True):
                if len(wav) == 0 or not np.isfinite(wav).all():
                    raise RuntimeError("TTS produced empty/nonfinite audio: " + job["id"])
                path = directory / (stable_hash(job["id"])[:24] + ".wav")
                sf.write(path, wav, sr)
                row = {**job, "audio_path": str(path), "sample_rate": int(sr), "batch_rng_seed": seed,
                       "synthesis_batch_size": len(jobs), "purpose": "attribute_reward_only"}
                with journal.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                result.append(row)
            return result
        for start in range(0, len(ordered), size):
            output.extend(run(ordered[start:start + size]))
        index = {r["id"]: r for r in output}
        return [index[r["id"]] for r in rows]

    def rollout_v5(self, rows, output_path):
        from .qwen_voice_design import main_generation_logprobs, sub_generation_logprobs
        import soundfile as sf
        directory = Path(output_path).parent / "audio"
        directory.mkdir(parents=True, exist_ok=True)
        reference = RolloutReferenceReplay(self)
        output, fallback_groups = [], 0
        size = int(self.cfg.get("generation", {}).get("rollout_batch_size", 4))
        self.policy.eval()
        for row in rows:
            request = row["request"]
            jobs = [{"id": f"{row['id']}::{i}", "request": request,
                     "generation_seed": row["candidate_seeds"][i]} for i in range(row["group_size"])]
            candidates = []
            fallback_reason = None
            try:
                for start in range(0, len(jobs), size):
                    batch = jobs[start:start + size]
                    ((codes_list, _), main_trace, sub_traces), seed = self._v5_generate(batch, capture=True)
                    for b, (job, codes) in enumerate(zip(batch, codes_list, strict=True)):
                        main = SimpleNamespace(scores=[v[b:b+1] for v in main_trace.scores])
                        sub = [SimpleNamespace(scores=[v[b:b+1] for v in t.scores]) for t in sub_traces]
                        old_main = main_generation_logprobs(self.torch, main, codes).to(self.device)
                        old_sub = sub_generation_logprobs(self.torch, sub, codes).to(self.device)
                        current_main, current_sub = self.incremental_trajectory_logprobs(request, codes)
                        (ref_main, ref_sub), mode = reference.reference(request, codes, (current_main, current_sub))
                        with self.torch.no_grad():
                            wavs, sr = self.core.speech_tokenizer.decode([{"audio_codes": codes}])
                        path = directory / (job["id"].replace("::", "_") + ".wav")
                        sf.write(path, wavs[0], sr)
                        c = mark_trajectory({"candidate_id": job["id"], "audio_path": str(path), "sample_rate": int(sr),
                            "generation_seed": job["generation_seed"], "batch_rng_seed": seed, "request": request,
                            "codec_codes": codes.long().cpu().tolist(), "codec_frames": int(codes.shape[0]), "codebooks": 16,
                            "old_main_logprobs": old_main.cpu().tolist(), "old_sub_logprobs": old_sub.cpu().tolist(),
                            "ref_main_logprobs": ref_main.cpu().tolist(), "ref_sub_logprobs": ref_sub.cpu().tolist(),
                            "main_behavior_replay_max_abs_error": float((old_main-current_main).abs().max().item()),
                            "subtalker_behavior_replay_max_abs_error": float((old_sub-current_sub).abs().max().item()),
                            "main_policy_reference_max_abs_error": float((current_main-ref_main).abs().max().item()),
                            "subtalker_policy_reference_max_abs_error": float((current_sub-ref_sub).abs().max().item()),
                            "behavior_logprob_mode": "processed_generation_scores", "trajectory_version": TRAJECTORY_VERSION,
                            "reference_replay_mode": mode, "generation_batch_size": len(batch)}, "tts")
                        if not c["trajectory_valid"]:
                            raise RuntimeError("batched TTS trajectory validation failed")
                        candidates.append(c)
            except (self.torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                if not self.cfg.get("generation", {}).get("batch_fallback", True):
                    raise
                fallback_reason = type(exc).__name__
            if fallback_reason:
                # Clear batch traces and leave the exception scope before allocating serial replay.
                import gc
                codes_list = codes = main_trace = sub_traces = main = sub = None
                old_main = old_sub = current_main = current_sub = ref_main = ref_sub = None
                gc.collect()
                self.torch.cuda.empty_cache()
                fallback_groups += 1
                serial = self._serial_rollout([row], output_path)[0]
                serial["batch_fallback"] = fallback_reason
                output.append(serial)
                continue
            output.append({**row, "candidates": candidates})
        self.output_metrics["reference_replay"] = reference.metrics
        self.output_metrics["batch_fallback_groups"] = fallback_groups
        return output
