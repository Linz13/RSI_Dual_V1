"""Independent attribute reconstruction reward; no intermediate-caption GT reward."""
from __future__ import annotations

import math
import re
import unicodedata
from copy import deepcopy
from difflib import SequenceMatcher
from statistics import fmean, pstdev

from .dual_space import SYNTHESIZABLE_FIELDS, TEXT_FIELD
from .partial_caption import known
from .schema import get_path

SCORE_FIELDS = tuple(f for f in SYNTHESIZABLE_FIELDS if f != TEXT_FIELD)
JUDGE_FIELDS = ("paralinguistic.prosody", "paralinguistic.pause")
EMPHASIS = "paralinguistic.emphasis.emphasized_text"
EVENTS = "paralinguistic.nonverbal_vocalization"
VERSION = "attribute_reconstruction_v5.1"


class EvaluationPending(RuntimeError):
    """Infrastructure/invalid evaluator output must never become reward zero."""


def canonical(value):
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def field_known(label, field):
    if field == EMPHASIS and not known(get_path(label, "paralinguistic.emphasis.level")):
        return False
    return known(get_path(label, field))


def reference_mask(label):
    return [field for field in SCORE_FIELDS if field_known(label, field)]


def set_f1(a, b):
    a, b = set(a), set(b)
    return 1.0 if not a and not b else 2 * len(a & b) / (len(a) + len(b))


def event_set(value):
    return {canonical(v) for v in value if canonical(v) != "none"}


def tokens(text):
    return re.findall(r"[\u3400-\u9fff]|[^\W_]+(?:['’][^\W_]+)*", canonical(text), re.UNICODE)


def emphasis_positions(transcript, phrases):
    words = tokens(transcript)
    selected = set()
    for phrase in phrases:
        part = tokens(phrase)
        found = False
        if part:
            for start in range(len(words) - len(part) + 1):
                if words[start:start + len(part)] == part:
                    selected.update(range(start, start + len(part)))
                    found = True
        if not found:
            selected.add(("unlocated", canonical(phrase)))
    return words, selected


def emphasis_f1(reference, generated):
    a, apos = emphasis_positions(get_path(reference, TEXT_FIELD, ""), get_path(reference, EMPHASIS, []))
    b, bpos = emphasis_positions(get_path(generated, TEXT_FIELD, ""), get_path(generated, EMPHASIS, []))
    mapping = {}
    for block in SequenceMatcher(a=a, b=b, autojunk=False).get_matching_blocks():
        mapping.update({block.b + i: block.a + i for i in range(block.size)})
    mapped = {mapping[p] if isinstance(p, int) and p in mapping else ("generated_unmatched", str(p)) for p in bpos}
    # Unlocated reference phrases cannot accidentally match an unlocated prediction.
    return set_f1(apos, mapped)


def reconstruction(reference, generated, judge_scores=None):
    fields = reference_mask(reference)
    if not fields:
        return {"status": "no_reference_fields", "score": None, "fields": {}, "denominator": 0}
    scores = {}
    for field in fields:
        left, right = get_path(reference, field), get_path(generated, field)
        if not field_known(generated, field):
            value = 0.0
        elif field in JUDGE_FIELDS:
            value = (judge_scores or {}).get(field)
            if type(value) not in (int, float) or value not in (0, 0.5, 1):
                raise EvaluationPending("missing/invalid fixed text judgment: " + field)
        elif field == EVENTS:
            value = set_f1(event_set(left), event_set(right))
        elif field == EMPHASIS:
            value = emphasis_f1(reference, generated)
        else:
            value = float(canonical(left) == canonical(right))
        scores[field] = float(value)
    return {"status": "complete", "score": fmean(scores.values()), "fields": scores,
            "denominator": len(fields), "reference_fields": fields, "version": VERSION}


def score_audio_groups(groups, reconstruction_weight=0.9, format_weight=0.1):
    output = deepcopy(groups)
    for group in output:
        for c in group["candidates"]:
            evaluation = c.get("attribute_reconstruction", {})
            status = evaluation.get("status")
            if c.get("semantic_input_valid") and c.get("trajectory_valid") and status not in ("complete", "no_reference_fields"):
                raise EvaluationPending("candidate evaluation incomplete: " + c["candidate_id"])
            rec = evaluation.get("score")
            complete = status == "complete" and type(rec) in (int, float) and math.isfinite(rec)
            if status == "complete" and (not complete or not 0 <= rec <= 1):
                raise EvaluationPending("invalid reconstruction score: " + c["candidate_id"])
            c.update(reconstruction_score=rec if complete else None,
                     semantic_reward=rec if complete else None,
                     semantic_valid=bool(complete and c.get("semantic_input_valid") and c.get("trajectory_valid")),
                     structural_valid=bool(c.get("semantic_input_valid")),
                     reward=reconstruction_weight * (rec if complete else 0.0) + format_weight * c["format_score"],
                     sft_selected=False, skip_update=True, advantage=0.0,
                     grpo_mode="attribute_and_format", schema_progress=c["format_score"],
                     reward_components_raw={"reconstruction": rec, "format": c["format_score"]})
        active = [c for c in group["candidates"] if c.get("trajectory_valid")]
        if len(active) >= 2:
            mean, std = fmean(c["reward"] for c in active), pstdev(c["reward"] for c in active)
            for c in active:
                c["advantage"] = (c["reward"] - mean) / std if std else 0.0
                c["skip_update"] = std == 0.0
        eligible = [c for c in group["candidates"] if c["semantic_valid"]]
        if eligible:
            max(eligible, key=lambda c: (c["reconstruction_score"], c["candidate_id"]))["sft_selected"] = True
        group["cycle_sft_selection"] = "attribute_reconstruction_top1"
    return output


def audio_summary(groups):
    cs = [c for g in groups for c in g["candidates"]]
    def mean(key):
        values = [c[key] for c in cs if type(c.get(key)) in (int, float) and math.isfinite(c[key])]
        return fmean(values) if values else None
    return {"groups": len(groups), "candidates": len(cs),
            "grpo_usable_groups": sum(any(not c["skip_update"] for c in g["candidates"]) for g in groups),
            "raw_schema_valid_candidates": sum(bool(c.get("raw_schema_valid")) for c in cs),
            "semantic_input_valid_candidates": sum(bool(c.get("semantic_input_valid")) for c in cs),
            "synthesized_candidates": sum(bool(c.get("reconstructed_audio_path")) for c in cs),
            "sft_selected": sum(c["sft_selected"] for c in cs),
            "reconstruction_mean": mean("reconstruction_score"), "format_mean": mean("format_score"),
            "reward_mean": mean("reward"), "cycle_sft_selection": "attribute_reconstruction_top1",
            "per_field_mean": {f: fmean(v) for f in SCORE_FIELDS
                if (v := [c["attribute_reconstruction"]["fields"][f] for c in cs
                          if f in c.get("attribute_reconstruction", {}).get("fields", {})])},
            "generated_unknown_counts": {f: sum(not field_known(c["generated_attributes"], f)
                for c in cs if c.get("generated_attributes") is not None) for f in SCORE_FIELDS}}
