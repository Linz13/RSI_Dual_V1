"""V5 collects new rewards; V4 parameter-update workers are retained."""
from __future__ import annotations
import json
import math
import threading
import time
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from .orchestrator_v4 import DualISLOrchestrator as V4Orchestrator
from .attribute_reward import reconstruction, score_audio_groups, audio_summary, reference_mask
from .partial_caption import admit
from .dual_space import project_synth_caption
from .render import synth_caption_prompt, render_qwen_request
from .io import atomic_json, read_jsonl
from .rewards import score_groups, candidate_components, _statistics, COMPONENTS, CALIBRATION_VERSION


class DualISLOrchestrator(V4Orchestrator):
    def __init__(self, config):
        config = deepcopy(config)
        if not config.get("run", {}).get("inline_mock"):
            from .labeling import evaluator_resources
            resources = evaluator_resources(config["labeling"])
            frozen = config["labeling"].get("frozen_resources")
            if frozen is not None and frozen != resources:
                raise ValueError("Evaluator implementation/config changed; use a new V5 run")
            config["labeling"]["frozen_resources"] = resources
        super().__init__(config)

    def _caption_anchors(self, paired):
        if not self.config["training"].get("paired_anchor_enabled", True):
            return []
        return [row for row in super()._caption_anchors(paired) if row["target_schema"] == "synth_v1"]

    def _tts_anchors(self, paired, codecs):
        return super()._tts_anchors(paired, codecs) if self.config["training"].get("paired_anchor_enabled", True) else []

    def _collect_audio_only(self, round_index, rows, audio_codecs, caption_checkpoint,
                            tts_checkpoint, anchor_tts_checkpoint, directory):
        from .labeling import LabelService
        from .mock_labels import MockLabelService
        service = (MockLabelService if self.config.get("run", {}).get("inline_mock") else LabelService)(self.config)
        try:
            timings = {}
            started = time.perf_counter()
            ref_path = self.config.get("labeling", {}).get("reference_labels_path")
            if ref_path:
                records = {str(r["id"]): r for r in read_jsonl(ref_path)}
                references = {}
                for r in rows:
                    source = records[str(r["id"])]
                    parsed = admit(json.dumps(source.get("attributes", source.get("caption"))))
                    if not parsed["json_parseable"] or any(e != "missing" for e in parsed["field_errors"].values()):
                        raise ValueError("invalid reference attributes: " + str(r["id"]))
                    references[r["id"]] = parsed["caption"]
            else:
                references = service.labels(rows)
            atomic_json(self.run_dir / "prepared/reference_attributes.json", references)
            timings["reference_labels_seconds"] = time.perf_counter() - started
            jobs = [{"id": r["id"], "audio_path": r["audio_path"], "prompt": synth_caption_prompt(),
                "caption_schema": "synth_v1", "group_size": 4,
                "candidate_seeds": [self._seed(round_index, "audio_only", r["id"], i) for i in range(4)]} for r in rows]
            rollout = self._stage(name=f"round_{round_index:03d}_audio_caption_rollout", role="captioner", action="rollout",
                rows=jobs, directory=directory, checkpoint_in=caption_checkpoint, target_checkpoint=caption_checkpoint)
            groups = list(read_jsonl(rollout.output_path))
            requests = []
            for group in groups:
                group["reference_attributes"] = references[group["id"]]
                for c in group["candidates"]:
                    c.update(admit(c["raw_text"]))
                    has_ref = bool(reference_mask(references[group["id"]]))
                    if c["semantic_input_valid"] and c.get("trajectory_valid") and has_ref:
                        requests.append({"id": c["candidate_id"], "candidate_id": c["candidate_id"],
                            "request": render_qwen_request(c["caption"]),
                            "generation_seed": self._seed(round_index, "audio_reconstruction", group["id"],
                                                          int(c["candidate_id"].rsplit("::", 1)[1]))})
                    else:
                        c["attribute_reconstruction"] = {"status": "unrenderable" if has_ref else "no_reference_fields", "score": None}
            generated = []
            if requests:
                ready_dir = directory / "audio_attribute_synthesis"
                ready_dir.mkdir(parents=True, exist_ok=True)
                stop = threading.Event()
                watch = threading.Thread(target=service.watch, args=(ready_dir, stop), daemon=True)
                watch.start()
                try:
                    synthesis = self._stage(name=f"round_{round_index:03d}_audio_attribute_synthesis", role="tts",
                        action="generate-audio", rows=requests, directory=ready_dir,
                        checkpoint_in=tts_checkpoint, target_checkpoint=tts_checkpoint)
                    generated = list(read_jsonl(synthesis.output_path))
                finally:
                    stop.set()
                    watch.join()
            started = time.perf_counter()
            labels = service.labels(generated) if generated else {}
            timings["generated_labels_wait_and_local_seconds"] = time.perf_counter() - started
            index = {r["id"]: r for r in generated}
            def evaluate(item):
                group, c = item
                cid = c["candidate_id"]
                if cid in index:
                    predicted, reference = labels[cid], references[group["id"]]
                    c["generated_attributes"] = predicted
                    c["reconstructed_audio_path"] = index[cid]["audio_path"]
                    c["attribute_reconstruction"] = reconstruction(reference, predicted, service.judge(reference, predicted))
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=int(self.config.get("labeling", {}).get("judge_workers", 4))) as pool:
                list(pool.map(evaluate, [(g, c) for g in groups for c in g["candidates"]]))
            timings["text_judgments_seconds"] = time.perf_counter() - started
            atomic_json(directory / "audio_labeling_metrics.json", timings)
            stage = self._local_stage(name=f"round_{round_index:03d}_audio_collection", inputs=jobs, outputs=groups,
                directory=directory, action="attribute-reconstruction", invocation_material={
                    "version": "v5", "references": references, "labeling": self.config.get("labeling", {})})
            return list(read_jsonl(stage.output_path))
        finally:
            service.close()

    def _collect_sft_calibration_examples(self, round_index, paired, paired_codecs, caption_checkpoint,
                                         tts_checkpoint, anchor_caption_checkpoint, anchor_tts_checkpoint, directory):
        plans, jobs = [], []
        for source in paired:
            caption = project_synth_caption(source["caption"])
            negatives = self._counterfactuals(caption, f"round:{round_index}:paired:{source['id']}")
            if not negatives:
                continue
            ids = [f"sftcal::{source['id']}::positive"] + [f"sftcal::{source['id']}::negative::{i:02d}" for i in range(len(negatives))]
            for cid, target in zip(ids, [caption] + [n["caption"] for n in negatives]):
                jobs.append({"candidate_id": cid, "audio_path": source["audio_path"], "target_caption": target,
                    "prompt": synth_caption_prompt(), "target_schema": "synth_v1", "score_mode": "synth_values_macro"})
            plans.append((source, ids))
        current = self._stage(name=f"round_{round_index:03d}_sftcal_caption_current", role="captioner", action="score",
            rows=jobs, directory=directory, checkpoint_in=caption_checkpoint, target_checkpoint=caption_checkpoint)
        anchor = self._stage(name=f"round_{round_index:03d}_sftcal_caption_anchor", role="captioner", action="score",
            rows=jobs, directory=directory, checkpoint_in=anchor_caption_checkpoint, target_checkpoint=anchor_caption_checkpoint)
        ci, ai = self._index(current.output_path), self._index(anchor.output_path)
        examples = []
        for source, ids in plans:
            for label, cid, negatives in ((1, ids[0], ids[1:]), (0, ids[1], ids[:1])):
                examples.append({"id": cid, "label": label, "candidate": {
                    "candidate_id": cid, "audio_path": source["audio_path"], "trajectory_valid": True,
                    "audio_health": 1.0, "asr_score": 1.0,
                    "caption_target_logprob_macro": ci[cid]["caption_target_logprob_macro"],
                    "anchor_caption_target_logprob_macro": ai[cid]["caption_target_logprob_macro"],
                    "counterfactual_reconstruction": [{"reconstruction": ci[n]["caption_target_logprob_macro"]} for n in negatives],
                    "anchor_counterfactual_reconstruction": [{"reconstruction": ai[n]["caption_target_logprob_macro"]} for n in negatives]}})
        return [], examples

    def _calibration(self, round_index, audio_groups, caption_groups, audio_examples, caption_examples):
        path = self.run_dir / "reward_calibration.json"
        if path.is_file():
            result = json.loads(path.read_text())
            if result.get("v5_scope") != "caption_only":
                raise ValueError("V5 requires caption-only calibration")
            return result
        if round_index != 0:
            raise RuntimeError("Missing frozen caption-only round0 calibration")
        valid = [raw for g in caption_groups for c in g["candidates"]
                 for ok, raw in [candidate_components("caption_only", c, self._quality_config("caption_only"))] if ok]
        if not valid:
            valid = [raw for e in caption_examples for ok, raw in [candidate_components(
                "caption_only", e["candidate"], self._quality_config("caption_only"))] if ok]
        stats = {component: _statistics([v[component] for v in valid]) for component in COMPONENTS}
        result = {"version": CALIBRATION_VERSION, "method": "round0_dual_counterfactual_zscore", "v5_scope": "caption_only",
            "fitted_round": 0, "frozen_across_rounds": True, "loops": {"caption_only": stats}}
        atomic_json(path, result)
        return result

    def _fit_sft_thresholds(self, round_index, audio_examples, caption_examples, calibration, directory):
        from .rewards import score_calibration_examples, fit_sft_threshold
        values = score_calibration_examples(caption_examples, loop="caption_only", calibration=calibration,
            reward_config=self._loop_reward_config("caption_only"), quality=self._quality_config("caption_only"))
        gate = self.config["reward"]["sft_gate"]
        finite = [v for v in values if isinstance(v.get("score"), (int, float)) and math.isfinite(v["score"])]
        threshold = {"threshold": None, "status": "insufficient_calibration_examples"}
        if any(v["label"] == 1 for v in finite) and any(v["label"] == 0 for v in finite):
            threshold = fit_sft_threshold(values, target_precision=gate["target_precision"], min_recall=gate["min_recall"])
        result = {"diagnostic_only": True, "round": round_index, "loops": {"caption_only": threshold,
            "audio_only": {"threshold": None, "status": "disabled_attribute_reward"}}}
        atomic_json(directory / f"round_{round_index:03d}_sft_thresholds.json", result)
        return result

    def _score_round(self, round_index, audio_raw, caption_raw, calibration, thresholds, directory):
        cfg = self.config["reward"]["audio_only"]
        audio = score_audio_groups(audio_raw, cfg["reconstruction_weight"], cfg["format_weight"])
        caption = score_groups(caption_raw, loop="caption_only", calibration=calibration,
            reward_config=self._loop_reward_config("caption_only"), quality=self._quality_config("caption_only"),
            sft_threshold=thresholds["loops"]["caption_only"].get("threshold"))
        for loop, raw, groups in (("audio", audio_raw, audio), ("caption", caption_raw, caption)):
            self._local_stage(name=f"round_{round_index:03d}_{loop}_rewards", inputs=raw, outputs=groups,
                directory=directory, action="score-v5", invocation_material={"reward": self.config["reward"], "calibration": calibration})
        return audio, caption

    def _base_diagnostics(self, audio_groups, caption_groups):
        candidates = [c for g in audio_groups for c in g["candidates"]]
        pending = [c["candidate_id"] for c in candidates if c.get("semantic_input_valid") and c.get("trajectory_valid")
                   and c.get("attribute_reconstruction", {}).get("status") not in ("complete", "no_reference_fields")]
        report = {"before_any_optimizer_step": True, "scope": "collection_integrity_only",
                  "audio_only": "attribute_and_format", "caption_only": "v4",
                  "audio_candidates": len(candidates), "pending_candidates": pending,
                  "audio_trajectory_valid": sum(bool(c.get("trajectory_valid")) for c in candidates),
                  "caption_trajectory_valid": sum(bool(c.get("trajectory_valid")) for g in caption_groups for c in g["candidates"]),
                  "ok": not pending}
        atomic_json(self.run_dir / "base_diagnostics.json", report)
        if pending:
            raise RuntimeError("Attribute evaluations pending before optimizer update")
        return report

    def _require_trainable(self, loop, groups):
        return audio_summary(groups) if loop == "audio_only" else super()._require_trainable(loop, groups)

    def _reward_anchor_contract(self, caption_checkpoint, tts_checkpoint):
        from .checkpoints import checkpoint_record
        record = {"kind": "caption_only_frozen_reward_anchor", "captioner_adapter": checkpoint_record(caption_checkpoint),
                  "audio_only_frozen_reward_anchor": "disabled"}
        path = self.run_dir / "reward_anchor.json"
        if path.is_file() and json.loads(path.read_text()) != record:
            raise ValueError("frozen caption-only reward anchor mismatch")
        atomic_json(path, record)


DualOrchestrator = DualISLOrchestrator
