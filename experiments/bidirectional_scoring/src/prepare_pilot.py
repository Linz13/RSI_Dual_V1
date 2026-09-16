"""Prepare frozen samples and a portable, blinded annotation package; standard library only."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import random
import re
import shutil
import wave
import zipfile

from common import (ROOT, CAPTION, OLD_RUN, PILOT_EMOTIONS, PROTOCOL, EMOTIONS,
                    atomic_json, atomic_text, digest, sha256, read_jsonl, read_json,
                    now, empty_annotations, load_manifest, description, shared_runtime)


def wav_duration(path):
    with wave.open(str(path), "rb") as f:
        return f.getnframes() / f.getframerate()


def balanced_sample(rows, n, seed, label_key, speaker_key=None):
    rng = random.Random(seed)
    pools = {}
    for emotion in PILOT_EMOTIONS:
        bucket = [r for r in rows if label_key(r) == emotion]
        rng.shuffle(bucket)
        if speaker_key:
            speakers = collections.defaultdict(list)
            for row in bucket:
                speakers[speaker_key(row)].append(row)
            bucket = []
            keys = sorted(speakers)
            rng.shuffle(keys)
            while any(speakers.values()):
                for speaker in keys:
                    if speakers[speaker]:
                        bucket.append(speakers[speaker].pop())
        pools[emotion] = bucket
    result = []
    for i in range(n):
        emotion = PILOT_EMOTIONS[i % len(PILOT_EMOTIONS)]
        if not pools[emotion]:
            raise ValueError(f"{emotion} 的合格来源样本不足，停止准备；不会用其他类别悄悄替代")
        result.append(pools[emotion].pop(0))
    return result


def legacy_provenance():
    config = OLD_RUN / "resolved_config.yaml"
    log = OLD_RUN / "logs/round_000_caption_tts_rollout.log"
    with log.open() as f:
        first = f.readline()
    if not first.startswith("COMMAND "):
        raise ValueError("旧生成任务缺少可核验命令")
    command = json.loads(first[len("COMMAND "):])
    if "--checkpoint-in" in command or "--target-checkpoint" in command:
        raise ValueError("预期基础模型首轮生成，实际存在 checkpoint 参数")
    cfg = config.read_text()
    match = re.search(r"(?ms)^tts:\n(.*?)(?=^[A-Za-z_][\w]*:|\Z)", cfg)
    if not match:
        raise ValueError("无法读取旧配置的 TTS 部分")
    tts = match.group(1)
    if not re.search(r"(?m)^  model_path: .*/Qwen3-TTS-12Hz-1.7B-VoiceDesign\s*$", tts):
        raise ValueError("旧生成模型身份不是预期的 VoiceDesign 基础模型")
    if not re.search(r"(?m)^  adapter_path: ['\"]{2}\s*$", tts):
        raise ValueError("旧生成配置含 adapter 或无法核验")
    generation={"max_new_tokens":512,"temperature":1.0,"top_p":1.0,"top_k":0,
                "subtalker_temperature":1.0,"subtalker_top_p":1.0,"subtalker_top_k":0,"repetition_penalty":1.0}
    for name,expected in generation.items():
        scalar=re.search(r"(?m)^    "+name+r": ([\d.]+)\s*$",tts)
        if not scalar or float(scalar.group(1))!=expected:
            raise ValueError("旧生成配置发生变化："+name)
    return {"run": str(OLD_RUN), "round": 0, "model": "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            "checkpoint": "base; no checkpoint-in/target-checkpoint; adapter_path empty",
            "config_path": str(config), "config_sha256": sha256(config),
            "generation_command": command, "generation_log_path": str(log),
            "generation": generation,
            "exposure": "legacy_train_pool; generated before round_000 updates; later used for reward/training"}


def copy_audio(source, run, name):
    target = run / "review/audio" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return {"audio": "audio/" + name, "audio_sha256": sha256(target), "duration": wav_duration(target)}


def public_data(m):
    ident = m["identity"]
    return {"protocol": PROTOCOL, "manifest_id": m["manifest_id"], "name": ident["name"],
            "emotions": EMOTIONS,
            "a": [{"id": r["id"], "audio": r["audio"], "duration": r["duration"],
                   "transcript": r["source_transcript"]} for r in ident["a"]],
            "b": [{"id": r["id"], "text": r["request"]["text"],
                   "target": description(r["emotion"]),
                   "candidates": [{k: c[k] for k in ("blind_id", "audio", "duration")}
                                  for c in r["candidates"]]} for r in ident["b"]]}


def build_review(run, m):
    review = run / "review"
    review.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "app.js", "style.css"):
        shutil.copyfile(ROOT / "web" / name, review / name)
    payload = json.dumps(public_data(m), ensure_ascii=False).replace("<", "\\u003c")
    atomic_text(review / "data.js", "window.PILOT_DATA = " + payload + ";\n")
    atomic_text(review / "使用说明.txt",
                "离线使用：解压整个文件夹，双击 index.html。音频在 audio 文件夹中，请勿移动。\n"
                "离线自动保存仅在本浏览器中；结束时点击导出标注。\n"
                "回到服务器标注页面点击导入标注，选择导出的文件即可。无需编辑 JSON。\n"
                "联网使用推荐服务器启动 serve，通过 SSH 转发在电脑浏览器打开。\n")
    archive = run / "annotation_bundle.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(review.rglob("*")):
            if p.is_file():
                z.write(p, Path("annotation_bundle") / p.relative_to(review))


def exclusion_index(manifests):
    index = {"a_ids": set(), "b_ids": set(), "hashes": set(), "sources": []}
    for path in manifests:
        path = Path(path).resolve()
        _, m = load_manifest(path.parent)
        index["sources"].append({"path": str(path), "manifest_id": m["manifest_id"], "sha256": sha256(path)})
        index["a_ids"].update(r["source_id"] for r in m["identity"]["a"])
        index["b_ids"].update(r["source_id"] for r in m["identity"]["b"])
        index["hashes"].update(r["audio_sha256"] for r in m["identity"]["a"])
        index["hashes"].update(c["audio_sha256"] for r in m["identity"]["b"] for c in r["candidates"])
    return index


def prepare(run, a_n=4, b_n=4, seed=20260915, b_source="existing", a_min_seconds=1.0,
            a_max_seconds=15.0, exclude_manifests=()):
    run = Path(run).resolve()
    if run.exists():
        raise ValueError(f"目录已存在，不覆盖样本或标注：{run}")
    if min(a_n, b_n) < 0 or a_n + b_n == 0:
        raise ValueError("样本数必须非负且至少准备一个方向")
    if not 0 < a_min_seconds <= a_max_seconds <= 60:
        raise ValueError("A 时长范围必须满足 0 < min <= max <= 60 秒")
    excluded_index = exclusion_index(exclude_manifests)
    et = CAPTION / "benchmark/emotiontalk_speech_captioning"
    references = et / "data/test_references.jsonl"
    a_pool, excluded = [], collections.Counter()
    a_hashes = set()
    for r in read_jsonl(references):
        if r["emotion_label"] not in PILOT_EMOTIONS:
            continue
        if r["id"] in excluded_index["a_ids"]:
            excluded["a_previously_reviewed_source"] += 1
            continue
        p = et / r["audio_path"]
        duration = wav_duration(p)
        if not (a_min_seconds <= duration <= a_max_seconds) or not r["transcript"].strip():
            excluded["a_duration_or_empty_transcript"] += 1
            continue
        audio_hash = sha256(p)
        if audio_hash in excluded_index["hashes"] or audio_hash in a_hashes:
            excluded["a_previously_reviewed_or_duplicate_bytes"] += 1
            continue
        a_hashes.add(audio_hash)
        a_pool.append(r)
    a_rows = balanced_sample(a_pool, a_n, seed, lambda x: x["emotion_label"], lambda x: x["speaker_id"])
    random.Random(seed+31).shuffle(a_rows)
    b_rows, provenance = [], None
    if b_source == "existing" and b_n:
        provenance = legacy_provenance()
        rollout = OLD_RUN / "round_000/collections/round_000_caption_tts_rollout.output.jsonl"
        provenance["rollout_sha256"] = sha256(rollout)
        eligible = []
        for r in read_jsonl(rollout):
            emotion = r["source_caption"]["paralinguistic"]["emotion"]
            if emotion not in PILOT_EMOTIONS:
                continue
            if r["source_id"] in excluded_index["b_ids"]:
                excluded["b_previously_reviewed_source"] += 1
                continue
            text = r["request"]["text"]
            # Chinese-only pilot, excluding visible language/text conflicts; not an ASR truth filter.
            cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
            if r["request"]["language"] != "Chinese" or cjk < 3 or cjk < len(re.findall(r"[A-Za-z]", text)):
                excluded["b_language_text_metadata"] += 1
                continue
            cs = r["candidates"]
            if len(cs) != 4 or any(c["request"] != r["request"] for c in cs):
                excluded["b_group_not_four_identical_conditions"] += 1
                continue
            if any(not Path(c["audio_path"]).is_file() for c in cs):
                excluded["b_missing_audio"] += 1
                continue
            if any(not 1 <= wav_duration(c["audio_path"]) <= 20 for c in cs):
                excluded["b_duration"] += 1
                continue
            hashes = {sha256(c["audio_path"]) for c in cs}
            if hashes & excluded_index["hashes"]:
                excluded["b_previously_reviewed_bytes"] += 1
                continue
            eligible.append(r)
        b_rows = balanced_sample(eligible, b_n, seed + 1,
                                lambda x: x["source_caption"]["paralinguistic"]["emotion"])
        random.Random(seed+32).shuffle(b_rows)
    run.mkdir(parents=True)
    ident = {"name": run.name, "protocol": PROTOCOL, "seed": seed, "attribute": "emotion",
             "a": [], "b": [], "b_pending_generation": [],
             "sampling": {"a_n": a_n, "b_groups": b_n, "b_source": b_source,
                          "a_duration_seconds": [a_min_seconds, a_max_seconds], "b_duration_seconds": [1, 20],
                          "excluded_manifests": excluded_index["sources"],
                          "selection": "label-balanced, A speaker-interleaved; shuffled before model scores; Chinese only",
                          "excluded_counts": dict(excluded),
                          "a_available_by_emotion": dict(collections.Counter(r["emotion_label"] for r in a_pool)),
                          "interpretation": "n counts annotation items, not guaranteed usable references or decisive findings"},
             "a_source": {"path": str(references), "sha256": sha256(references),
                          "split": "official speaker-independent test", "exposure": "repeated legacy evaluation",
                          "labels": "official labels; audible emotion and transcript require human verification"},
             "b_source": provenance}
    seen_hashes = set()
    for i, r in enumerate(a_rows):
        aud = copy_audio(et / r["audio_path"], run, f"a_{i+1:03d}.wav")
        if aud["audio_sha256"] in seen_hashes:
            raise ValueError("A 抽样发现重复音频；需检查来源后重新准备新目录")
        seen_hashes.add(aud["audio_sha256"])
        ident["a"].append({"id": f"A{i+1:03d}", **aud, "source_id": r["id"],
                           "speaker_id": r["speaker_id"], "session_id": r["session_id"],
                           "source_audio": str(et / r["audio_path"]),
                           "source_emotion": r["emotion_label"], "source_transcript": r["transcript"]})
    rng = random.Random(seed + 2)
    for i, r in enumerate(b_rows):
        cs = list(r["candidates"])
        rng.shuffle(cs)
        record = {"id": f"B{i+1:03d}", "source_id": r["source_id"], "request": r["request"],
                  "emotion": r["source_caption"]["paralinguistic"]["emotion"], "candidates": []}
        for j, c in enumerate(cs):
            aud = copy_audio(c["audio_path"], run, f"b_{i+1:03d}_{j+1}.wav")
            record["candidates"].append({"blind_id": chr(65 + j), **aud,
                                          "source_candidate_id": c["candidate_id"],
                                          "source_audio": c["audio_path"], "generation_seed": c["generation_seed"]})
        record["duplicate_audio_count"] = 4 - len({c["audio_sha256"] for c in record["candidates"]})
        ident["b"].append(record)
    if b_source == "generate":
        texts = ["我们约好明天下午在门口见面。", "桌上放着一本书和一杯水。", "这条路一直通到前面的广场。", "我把这件事情从头说了一遍。"]
        for i in range(b_n):
            e = PILOT_EMOTIONS[i % 4]
            ident["b_pending_generation"].append({"id": f"B{i+1:03d}", "emotion": e,
                "request": {"text": texts[(i // 4) % len(texts)], "language": "Chinese", "instruct": description(e)},
                "candidate_seeds": [seed + 1000 + 4*i+j for j in range(4)]})
    # Check actual bytes against old train audio; manifests alone cannot establish non-overlap.
    old_hashes = {}
    for p in (CAPTION / "training_data/audio").glob("*"):
        if p.is_file() and p.suffix.lower() in (".wav", ".mp3", ".flac"):
            old_hashes[sha256(p)] = str(p)
    for r in ident["a"]:
        r["legacy_train_byte_overlap"] = old_hashes.get(r["audio_sha256"])
    ident["legacy_train_audio_files_checked"] = len(old_hashes)
    previous=[]
    for manifest_path in sorted((ROOT/"runs").glob("*/manifest.json")):
        if manifest_path.parent.resolve()==run:
            continue
        prior=read_json(manifest_path)
        if prior.get("protocol")!=PROTOCOL:
            continue
        known={r["audio_sha256"] for r in prior["identity"]["a"]}
        known.update(c["audio_sha256"] for r in prior["identity"]["b"] for c in r["candidates"])
        for kind in ("a","b"):
            for row in ident[kind]:
                hashes={row["audio_sha256"]} if kind=="a" else {c["audio_sha256"] for c in row["candidates"]}
                if hashes & known:
                    previous.append({"id":row["id"],"prior_manifest":str(manifest_path),
                                     "reason":"previously prepared pilot audio; inspect prior annotations/results for actual exposure"})
    ident["previous_pilot_matches"] = previous
    ident["overlap_scope"] = "byte SHA256 against training_data/audio; not an acoustic fingerprint or pretraining audit"
    m = {"protocol": PROTOCOL, "created_at": now(), "manifest_id": digest(ident), "identity": ident}
    atomic_json(run / "manifest.json", m)
    atomic_json(run / "annotations.json", empty_annotations(m["manifest_id"]))
    build_review(run, m)
    print(f"已准备 A={len(ident['a'])}、B={len(ident['b'])}，待生成 B={len(ident['b_pending_generation'])}。")
    print(f"样本与来源：{run / 'manifest.json'}")
    print(f"离线听音包：{run / 'annotation_bundle.zip'}")
    return m


def main():
    shared_runtime()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--profile", choices=["smoke", "pilot"], default="smoke")
    p.add_argument("--a-n", type=int)
    p.add_argument("--b-groups", type=int)
    p.add_argument("--seed", type=int, default=20260915)
    p.add_argument("--b-source", choices=["existing", "generate"], default="existing")
    p.add_argument("--a-min-seconds", type=float, default=4.0)
    p.add_argument("--a-max-seconds", type=float, default=15.0)
    p.add_argument("--exclude-manifest", action="append", default=[])
    a = p.parse_args()
    prepare(a.run, a.a_n if a.a_n is not None else (4 if a.profile == "smoke" else 30),
            a.b_groups if a.b_groups is not None else (4 if a.profile == "smoke" else 30), a.seed, a.b_source,
            a.a_min_seconds, a.a_max_seconds, a.exclude_manifest)


if __name__ == "__main__":
    main()
