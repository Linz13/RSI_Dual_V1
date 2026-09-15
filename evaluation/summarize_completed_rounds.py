"""Build round-by-round tables from validated completed benchmark results."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import later_caption_eval as evaluation


def main():
    root = Path(__file__).resolve().parent
    output = root / "reports/completed_caption_rounds_20260907"
    source = root / "later_caption_eval_runs_20260902_run01"
    errors = evaluation.validate_historical(root.parent)
    if errors:
        raise RuntimeError("\n".join(errors))
    entries = {}
    for profile in ("legacy", "midasheng_rewardv2_v3"):
        os.environ["LATER_CAPTION_PROFILE"] = profile
        for entry in evaluation.summary_entries(root.parent, source):
            entries[(entry["trajectory"], entry["label"], entry["suite"])] = entry
    for suite in evaluation.BENCHMARKS:
        base = entries[("midasheng_v1", "base", suite)]
        for trajectory in ("midasheng_rewardv2", "midasheng_rewardv3"):
            entries[(trajectory, "base", suite)] = dict(base, trajectory=trajectory)
    titles = {
        "qwen_v1": "Qwen3-Omni-30B-A3B / DualISL 原版",
        "midasheng_v1": "MiDasheng-7B / DualISL 原版",
        "midasheng_rewardv2": "MiDasheng-7B / RewardV2",
        "midasheng_rewardv3": "MiDasheng-7B / RewardV3",
    }
    metric_names = ("bleu_4", "rouge_l", "meteor", "spider", "fense", "bertscore", "clapscore")
    rows, details, excluded = [], [], []
    for trajectory in titles:
        labels = {label for t, label, _ in entries if t == trajectory}
        for label in sorted(labels, key=lambda x: -1 if x == "base" else int(x[1:])):
            group = {s: entries.get((trajectory, label, s)) for s in evaluation.BENCHMARKS}
            if any(not e or e["state"] != "complete" for e in group.values()):
                excluded.append([trajectory, label])
                continue
            emotion = group["emotiontalk"]["result"]["tasks"]
            row = {"trajectory": trajectory, "round": label}
            row.update({name: emotion["overall"]["metrics"].get(name, {}).get("value") for name in metric_names})
            row["paraspeechcaps"] = group["paraspeechcaps"]["result"]["final_score"]
            row["stylecap"] = group["stylecap"]["result"]["macro_average_accuracy"]
            rows.append(row)
            for task, data in emotion.items():
                for name, metric in data["metrics"].items():
                    details.append({"trajectory": trajectory, "round": label, "task": task,
                                    "metric": name, "value": metric["value"], "status": metric["status"],
                                    "error": metric.get("error"), "source": group["emotiontalk"]["result_path"]})
    assert len({(r["trajectory"], r["round"]) for r in rows}) == len(rows)
    output.mkdir(parents=True, exist_ok=True)
    for filename, records in (("rounds.csv", rows), ("emotiontalk_all_tasks.csv", details)):
        with (output / filename).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    columns = ("spider", "fense", "paraspeechcaps", "stylecap")
    lines = ["# 已完成 Captioner 逐轮评测", "", f"核验时间：{evaluation.utc_now()}", "",
             "覆盖此前盘点中已完成三套 full benchmark 的四条训练轨迹，不包含待评测的 Qwen2.5、训练 smoke/mock 或未提交轮次。",
             "", "- base 为未训练模型；r0 是第一轮训练后的模型。三个 MiDasheng 分支复用同一份历史 base 评测，不是三次独立基线实验。",
             "- EmotionTalk 主表使用 overall 子任务的 SPIDEr/FENSE 原始分值；ParaSpeechCaps 是 Scheme A 综合分，不是单纯准确率；StyleCap 是宏平均准确率。",
             "- 表内各指标均越高越好，保留六位小数；粗体表示该分支含 base 的列最大值，按未舍入原始值判断。",
             "- CLAPScore unavailable 保留为缺失，不计作 0；BLEU-4 原始分值极小，完整指标表使用科学计数法，不额外缩放。",
             "- 不对不同 benchmark 求总分，不凭单次评测的小幅差异判断统计显著性或归因于某个 reward。",
             "", "## 主表", ""]
    last_rows = []
    for trajectory, title in titles.items():
        subset = [r for r in rows if r["trajectory"] == trajectory]
        if not subset:
            continue
        best = {name: max(r[name] for r in subset if r[name] is not None) for name in columns}
        lines += [f"### {title}", "", "| 轮次 | ET SPIDEr | ET FENSE | ParaSpeechCaps | StyleCap |",
                  "|---|---:|---:|---:|---:|"]
        for row in subset:
            values = [f"{row[name]:.6f}" for name in columns]
            values = [f"**{value}**" if row[name] == best[name] else value for name, value in zip(columns, values)]
            lines.append("| " + " | ".join([row["round"], *values]) + " |")
        base, last = subset[0], subset[-1]
        assert base["round"] == "base"
        last_rows.append({"trajectory": trajectory, "last": last["round"],
                          **{name: last[name] - base[name] for name in columns}})
        lines += ["", "末轮相对 base 的原始分值变化：" + "；".join(f"{name} {last[name] - base[name]:+.6f}" for name in columns) + "。", ""]
    lines += ["## EmotionTalk overall 完整指标", ""]
    for trajectory, title in titles.items():
        lines += [f"### {title}", "", "| 轮次 | BLEU-4 | ROUGE-L | METEOR | SPIDEr | FENSE | BERTScore F1 | CLAPScore |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in (r for r in rows if r["trajectory"] == trajectory):
            values = ["unavailable" if row[name] is None else (f"{row[name]:.6e}" if name == "bleu_4" else f"{row[name]:.6f}") for name in metric_names]
            lines.append("| " + " | ".join([row["round"], *values]) + " |")
        lines.append("")
    lines += ["## 数据文件与来源", "", "- [逐轮原始分值 CSV](rounds.csv)",
              "- [EmotionTalk 四个子任务的全部指标、状态及来源](emotiontalk_all_tasks.csv)",
              "- [核验记录与原始结果路径](sources.json)", ""]
    (output / "results.md").write_text("\n".join(lines), encoding="utf-8")
    provenance = {"created_utc": evaluation.utc_now(), "excluded": excluded,
                  "entries": [{k: e[k] for k in ("trajectory", "label", "suite", "state", "result_path")} for e in entries.values()],
                  "last_minus_base": last_rows}
    (output / "sources.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(output / "results.md"), "rows": rows, "last_minus_base": last_rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
