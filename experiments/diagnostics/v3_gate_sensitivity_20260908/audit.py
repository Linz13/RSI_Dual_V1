"""Reselect historical V3 candidates; never modify run artifacts or invoke models."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OUT = Path(__file__).resolve().parent
CAPTION = OUT.parents[2]
RUN = CAPTION / "DualISL_Train_RewardV3/runs/midasheng_7b_reward_v3_10rounds_20260905_run01"
REWARD = CAPTION / "DualISL_Train_RewardV3/dual_isl_train/rewards.py"
TARGETS = (0.9, 0.85, 0.8, 0.75, 0.7)
SOURCES = {}


def read_rows(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            if line.strip():
                yield json.loads(line)
    SOURCES[str(path)] = digest.hexdigest()


def read_json(path):
    data = path.read_bytes()
    SOURCES[str(path)] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def main():
    # Execute only these two pure functions from the actual implementation.
    # This avoids importing model workers or creating package bytecode.
    source = REWARD.read_bytes()
    SOURCES[str(REWARD)] = hashlib.sha256(source).hexdigest()
    tree = ast.parse(source, filename=str(REWARD))
    names = {"_finite", "fit_sft_threshold"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == 2
    namespace = {"math": math, "Any": Any}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(REWARD), "exec"), namespace)
    fit = namespace["fit_sft_threshold"]
    records, details = [], []
    for r in range(10):
        directory = RUN / f"round_{r:03d}" / "rewards"
        saved = read_json(directory / f"round_{r:03d}_sft_thresholds.json")
        for loop, suffix, target_model, expected in (
            ("caption_only", "caption", "Captioner", 196),
            ("audio_only", "audio", "TTS", 184),
        ):
            examples = list(read_rows(directory / f"round_{r:03d}_{loop}_sft_confidence_calibration.output.jsonl"))
            assert len(examples) == 78
            assert sum(e["label"] == 1 for e in examples) == 39
            assert all(e["label"] in (0, 1) and math.isfinite(e["score"]) for e in examples)
            fits = {p: fit(examples, target_precision=p, min_recall=0.25) for p in TARGETS}
            assert fits[0.9] == saved["loops"][loop], (r, loop, "baseline threshold mismatch")
            winners, actual, group_ids = [], [], set()
            for group in read_rows(directory / f"round_{r:03d}_{suffix}_rewards.output.jsonl"):
                assert group["id"] not in group_ids
                group_ids.add(group["id"])
                candidates = group["candidates"]
                assert len(candidates) == 4
                valid = [c for c in candidates if c["semantic_valid"]]
                chosen = [c for c in candidates if c["sft_selected"]]
                assert len(chosen) <= 1 and all(c["semantic_valid"] for c in chosen)
                actual.extend((group["id"], c["candidate_id"]) for c in chosen)
                if valid:
                    assert all(math.isfinite(c["semantic_reward"]) and math.isfinite(c["sft_confidence"]) for c in valid)
                    winner = max(valid, key=lambda c: (float(c["semantic_reward"]), str(c["candidate_id"])))
                    winners.append({"group_id": group["id"], "candidate_id": winner["candidate_id"],
                                    "score": winner["sft_confidence"]})
            assert len(group_ids) == expected
            detail = {"round": r, "loop": loop, "target_model": target_model, "winners": winners, "settings": {}}
            for precision in (*TARGETS, None):
                result = fits[precision] if precision is not None else {"threshold": None, "status": "no_absolute_gate"}
                threshold = result["threshold"]
                selected = [w for w in winners if precision is None or (threshold is not None and w["score"] >= threshold)]
                if precision == 0.9:
                    assert {(w["group_id"], w["candidate_id"]) for w in selected} == set(actual), (r, loop, "baseline selection mismatch")
                key = "no_gate" if precision is None else str(precision)
                detail["settings"][key] = {**result, "selected_group_ids": [w["group_id"] for w in selected]}
                records.append({"round": r, "loop": loop, "target_model": target_model,
                                "target_precision": key, "min_recall": 0.25 if precision is not None else None,
                                "pool_groups": expected, "eligible_groups": len(winners),
                                "selected": len(selected), "actual_selected_90": len(actual),
                                **{k: result.get(k) for k in ("threshold", "status", "empirical_precision", "empirical_recall", "accepted_positives", "accepted_negatives")}})
            details.append(detail)
            print(f"r{r} {target_model}: " + ", ".join(f"{v['target_precision']}={v['selected']}" for v in records[-6:]), flush=True)
    with (OUT / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    evidence = {"generated_utc": datetime.now(timezone.utc).isoformat(), "source_run": str(RUN),
                "method": "Original fit_sft_threshold; fixed historical scores; min_recall=0.25; original semantic_valid and top-1 tie-break.",
                "baseline_validation": "All 20 saved 90% threshold objects and all selected candidate identities reproduced exactly.",
                "limitations": ["Not a retraining simulation; later candidates would change after changed updates.",
                                "Calibration precision is not real pseudo-pair accuracy; paired examples were used in training."],
                "source_sha256": SOURCES, "details": details}
    (OUT / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lookup = {(v["round"], v["target_model"], v["target_precision"]): v for v in records}
    lines = ["# V3 历史分数上的 cycle-SFT gate 敏感性重算", "",
             "仅使用既有十轮校准与候选分数，不生成、不重评分、不训练，不修改训练配置或 run。",
             "调用实际 rewards.py 的纯阈值拟合函数，最低 recall 固定 25%，保留 semantic_valid 与原 top-1 规则。",
             "原 90% 设置的 20 个阈值对象及所有入选候选身份均精确复现。", "",
             "| 轮次 | Captioner 90% | Captioner 80% | TTS 90% | TTS 80% |",
             "|---|---:|---:|---:|---:|"]
    for r in range(10):
        values = [lookup[r, model, p]["selected"] for model, p in (("Captioner", "0.9"), ("Captioner", "0.8"), ("TTS", "0.9"), ("TTS", "0.8"))]
        lines.append(f"| r{r} | " + " | ".join(map(str, values)) + " |")
    lines.extend(["", "## 不同 precision 要求的十轮合计", "", "| 设置 | Captioner | TTS |", "|---|---:|---:|"])
    for p in ("0.9", "0.85", "0.8", "0.75", "0.7", "no_gate"):
        totals = [sum(lookup[r, model, p]["selected"] for r in range(10)) for model in ("Captioner", "TTS")]
        lines.append(f"| {p} | {totals[0]} | {totals[1]} |")
    lines.extend(["", "## 80% 下的校准结果", "", "| 轮次 | 训练对象 | 阈值 | 接受正例/39 | 接受负例/39 | precision | recall |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for v in records:
        if v["target_precision"] != "0.8":
            continue
        if v["threshold"] is None:
            lines.append(f"| r{v['round']} | {v['target_model']} | 无可行阈值 | — | — | — | — |")
        else:
            lines.append(f"| r{v['round']} | {v['target_model']} | {v['threshold']:.6f} | {v['accepted_positives']} | {v['accepted_negatives']} | {v['empirical_precision']:.2%} | {v['empirical_recall']:.2%} |")
    lines.extend(["", "这些数量是固定历史候选的重新筛选，不是修改规则后重新训练十轮的预测。",
                  "校准集 precision 不代表实际伪配对正确率；39 个 paired 正例也用于训练。",
                  "caption-only 训练 Captioner，每轮上限 196；audio-only 训练 TTS，每轮上限 184。",
                  "合计为跨轮训练记录次数，不是新增独立源样本数。", "",
                  "[完整数值](results.csv) · [来源哈希、逐组选择与核验](evidence.json) · [重算脚本](audit.py)"])
    (OUT / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
