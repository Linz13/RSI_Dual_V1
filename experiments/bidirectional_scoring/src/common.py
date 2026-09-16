"""Shared, model-free utilities. No torch/transformers imports at module scope."""
from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parent
CAPTION = ROOT.parent
ENV_ROOT = CAPTION.parent / "miniconda3/envs"
TTS_MODEL = CAPTION / "models/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
CAPTION_MODEL = CAPTION / "models/MiDashengLM-7B-1021-BF16"
OLD_RUN = CAPTION / "DualISL_Train_RewardV5_LabelRobust/runs/midasheng_v5_label_robust_10rounds_8gpu_run01"
EMOTIONS = {"happy": "开心", "sad": "悲伤", "angry": "愤怒", "neutral": "平静",
            "surprised": "惊讶", "disgusted": "厌恶", "fearful": "恐惧"}
PILOT_EMOTIONS = ("happy", "sad", "angry", "neutral")
PROTOCOL = "bidirectional-emotion-v1"
PROMPT = "请只用一句简短中文描述这段语音中说话者声音表达的情绪。依据声音判断，不复述说话内容。"


def shared_runtime():
    """User-requested cross-account shared-storage outputs; does not chmod external paths."""
    os.umask(0)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def within(path, root):
    """Path containment, including Python 3.8 on the annotation server."""
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        # User requested writable outputs across accounts on the shared model servers.
        os.fchmod(fd, 0o666)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def write_jsonl(path, rows):
    atomic_text(path, "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))


@contextlib.contextmanager
def lock(path):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError("同一目录已有任务正在写入；请等待该任务完成。") from e
        yield


def load_manifest(run):
    run = Path(run).resolve()
    m = read_json(run / "manifest.json")
    if m.get("protocol") != PROTOCOL or digest(m["identity"]) != m["manifest_id"]:
        raise ValueError("清单身份不匹配或协议不受支持")
    return run, m


def empty_annotations(manifest_id):
    return {"schema": "pilot-annotations-v1", "manifest_id": manifest_id,
            "revision": 0, "a": {}, "b": {}, "updated_at": now()}


def annotations(run, m):
    p = Path(run) / "annotations.json"
    return read_json(p) if p.exists() else empty_annotations(m["manifest_id"])


def validate_annotation_item(kind, item, row, final=None):
    if not isinstance(item, dict):
        raise ValueError("标注必须是对象")
    allowed = {"final", "heard", "status", "emotion", "transcript", "notes", "best", "updated_at", "completion_method"}
    if set(item) - allowed:
        raise ValueError("标注含未知字段")
    if not isinstance(item.get("final", False), bool) or not isinstance(item.get("heard", False), bool):
        raise ValueError("确认状态格式错误")
    for key in ("notes", "transcript"):
        if key in item and (not isinstance(item[key], str) or len(item[key]) > 10000):
            raise ValueError("文本字段格式错误或过长")
    if not (item.get("final") if final is None else final):
        return
    if not item.get("heard"):
        raise ValueError("请确认已听音并完成核验")
    if kind == "a":
        if item.get("status") not in ("verified", "unusable", "uncertain"):
            raise ValueError("请选择 A 核验结论")
        if item["status"] == "verified":
            if item.get("emotion") not in EMOTIONS:
                raise ValueError("请选择明确的可听情绪；混合／不确定请标为无法判定")
            if not item.get("transcript", "").strip():
                raise ValueError("核验通过的样本必须有转写")
    else:
        if item.get("status") not in ("preferred", "all_bad", "uncertain"):
            raise ValueError("请选择 B 盲听结论")
        best = item.get("best", [])
        valid = {c["blind_id"] for c in row["candidates"]}
        if not isinstance(best, list) or any(not isinstance(x, str) for x in best):
            raise ValueError("候选选择格式错误")
        if len(best) != len(set(best)) or not set(best) <= valid:
            raise ValueError("选择包含未知或重复候选")
        if (item["status"] == "preferred") != bool(best):
            raise ValueError("有合格候选时请选出最佳；全部不合格／无法判断时请清空选择")


def validate_annotations(data, m):
    if not isinstance(data, dict) or data.get("schema") != "pilot-annotations-v1":
        raise ValueError("不是本工具导出的标注文件")
    if data.get("manifest_id") != m["manifest_id"]:
        raise ValueError("标注文件不属于当前样本包，拒绝导入")
    for kind in ("a", "b"):
        rows = {r["id"]: r for r in m["identity"][kind]}
        entries = data.get(kind, {})
        if not isinstance(entries, dict) or set(entries) - rows.keys():
            raise ValueError("标注含未知样本")
        for rid, item in entries.items():
            validate_annotation_item(kind, item, rows[rid])
    return data


def save_annotations(run, m, incoming, expected_revision=None):
    validate_annotations(incoming, m)
    with lock(Path(run) / ".annotations.lock"):
        current = annotations(run, m)
        if expected_revision is not None and expected_revision != current["revision"]:
            raise ValueError("标注已被其他页面修改。请先导出本页备份，再刷新或导入合并。")
        result = empty_annotations(m["manifest_id"])
        for kind in ("a", "b"):
            result[kind] = {**current[kind], **incoming.get(kind, {})}
        result["revision"] = current["revision"] + 1
        atomic_json(Path(run) / "annotation_history" / f"revision_{result['revision']:05d}.json", result)
        atomic_json(Path(run) / "annotations.json", result)
        return result


def description(emotion):
    if emotion not in EMOTIONS:
        raise ValueError("未知情绪")
    return f"说话者的情绪是{EMOTIONS[emotion]}。"


def finite(values):
    return bool(values) and all(isinstance(x, (int, float)) and math.isfinite(x) for x in values)


def model_identity(path):
    """Byte hashes of small config/code; weight size+mtime identity avoids reading GB on CPU preflight.
    Full weight SHA256 is added by explicitly invoked model commands before loading.
    """
    path = Path(path).resolve()
    if not path.is_dir() or (path / "adapter_config.json").exists():
        raise ValueError(f"需要本地完整基础模型目录：{path}")
    files = []
    for p in sorted(path.rglob("*")):
        if p.is_file() and p.suffix in (".json", ".py", ".safetensors", ".jinja", ".txt") and "__pycache__" not in p.parts:
            s = p.stat()
            files.append({"path": str(p.relative_to(path)), "size": s.st_size,
                          "mtime_ns": s.st_mtime_ns, "sha256": sha256(p) if p.suffix != ".safetensors" else None})
    if not any(x["path"].endswith(".safetensors") for x in files):
        raise ValueError(f"找不到模型权重：{path}")
    return {"path": str(path), "files": files}


def verified_audio(run, row):
    p = (Path(run) / "review" / row["audio"]).resolve()
    if not within(p, Path(run) / "review/audio") or sha256(p) != row["audio_sha256"]:
        raise ValueError("音频内容或路径与冻结清单不符")
    return p
