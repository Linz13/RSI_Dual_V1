"""Resumable scoring orchestration; loading is guarded by explicit --execute."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import traceback

from common import (ROOT, TTS_MODEL, CAPTION_MODEL, PILOT_EMOTIONS, annotations,
    validate_annotations, atomic_json, load_manifest, read_json, digest, sha256,
    model_identity, verified_audio, description, now, lock, PROMPT, shared_runtime)


def a_inputs(m, ann):
    jobs, skipped, pending = [], [], []
    for r in m["identity"]["a"]:
        label = ann["a"].get(r["id"], {})
        if not label.get("final"):
            pending.append(r["id"])
        elif label["status"] != "verified":
            skipped.append({"id":r["id"],"reason":label["status"]})
        else:
            jobs.append({"id":r["id"],"emotion":label["emotion"],"transcript":label["transcript"].strip(),
                         "audio":r["audio"],"audio_sha256":r["audio_sha256"],"speaker_id":r["speaker_id"]})
    if pending:
        raise ValueError("A 尚有未确认标注："+", ".join(pending)+"。请在页面确认保存，无需编辑文件。")
    return jobs, skipped


def runtime_identity(model_path, kind):
    identity=model_identity(model_path)
    versions={}
    for name in ("torch","transformers","qwen-tts","soundfile","librosa","numpy"):
        try:
            versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name]=None
    scripts=["common.py","scoring.py","model_interfaces.py"]
    if kind=="a":
        scripts.append("qwen_conditioning.py")
    external={}
    if kind=="a":
        distribution=importlib.metadata.distribution("qwen-tts")
        for relative in distribution.files or []:
            if str(relative).endswith(("/modeling_qwen3_tts.py","/qwen3_tts_model.py","/qwen3_tts_tokenizer.py")):
                path=distribution.locate_file(relative)
                external[str(relative)]=sha256(path)
    return {"model":identity,"versions":versions,"installed_source_sha256":external,
            "implementation":{n:sha256(ROOT/n) for n in scripts}}


def weight_hashes(identity, directory):
    # User-invoked model command only. Configs/code are already hashed; weights are read, never rewritten.
    destination=directory/"model_hashes.json"
    key=digest(identity["model"])
    if destination.exists():
        prior=read_json(destination)
        if prior.get("stat_identity")==key:
            return prior
    path=Path(identity["model"]["path"])
    hashes={}
    for f in identity["model"]["files"]:
        if f["path"].endswith(".safetensors"):
            print("核验权重文件："+f["path"],flush=True)
            hashes[f["path"]]=sha256(path/f["path"])
    result={"stat_identity":key,"sha256":hashes}
    atomic_json(destination,result)
    return result


def conditions(emotion):
    negatives=[e for e in PILOT_EMOTIONS if e!=emotion][:3]
    return [{"emotion":e,"is_positive":e==emotion,"description":description(e)}
            for e in [emotion,*negatives]]


def ensure_numeric_audit(path, run_audit, context):
    """Persist evidence before stopping; a cached failed check must never unlock scoring."""
    if path.exists():
        result=read_json(path)
    else:
        try:
            result={**context,**run_audit()}
        except Exception as e:
            atomic_json(path,{**context,"passed":False,"error":str(e),"traceback":traceback.format_exc()})
            raise
        atomic_json(path,result)
    if result.get("passed") is not True:
        metrics={k:v for k,v in result.items() if k.endswith("max_abs") or k.endswith("tolerance")}
        raise ValueError(f"数值核验未通过，未继续候选评分。完整检查已保存：{path}；差异：{metrics}")
    return result


def score_a_job(model,run,row,all_rows,directory,model_key,control):
    audio=verified_audio(run,row)
    codec_path=run/"cache/codecs"/(digest([row["audio_sha256"],model_key])+".json")
    if codec_path.exists():
        cached=read_json(codec_path)
        if cached["audio_sha256"]!=row["audio_sha256"] or cached["model_identity"]!=model_key:
            raise ValueError("codec 缓存身份错误")
        codes=cached["codes"]
    else:
        codes=model.encode(audio).tolist()
        atomic_json(codec_path,{"audio_sha256":row["audio_sha256"],"model_identity":model_key,"codes":codes})
    cs=conditions(row["emotion"])
    request={"language":"Chinese","text":row["transcript"],"instruct":description(row["emotion"])}
    audit_path=directory/"numeric_audit.json"
    if not audit_path.exists():
        print("执行 A 的重复评分、padding、首帧逐码本和未来信息检查…",flush=True)
    ensure_numeric_audit(audit_path,lambda:model.audit(request,codes),{"sample_id":row["id"]})
    output=[]
    for c in cs:
        result=model.score({**request,"instruct":c["description"]},codes)
        output.append({**c,**result})
    controls={}
    if control:
        controls["no_description"]=model.score({**request,"instruct":""},codes)
        different=next((r for r in all_rows if r["transcript"]!=row["transcript"]),None)
        if different:
            controls["wrong_transcript"]={"donor_id":different["id"],"transcript":different["transcript"],
                **model.score({**request,"text":different["transcript"]},codes)}
    return {"id":row["id"],"status":"ok","reference_emotion":row["emotion"],
            "verified_transcript":row["transcript"],"audio":row["audio"],"audio_sha256":row["audio_sha256"],
            "speaker_id":row["speaker_id"],"conditions":output,"controls":controls}


def score_b_job(model,run,row,all_rows,directory,control):
    target=description(row["emotion"])
    values=[]
    for c in row["candidates"]:
        audio=verified_audio(run,c)
        audit_path=directory/"numeric_audit.json"
        if not audit_path.exists():
            print("执行 B 的重复评分、padding 和逐 token 前缀检查…",flush=True)
        ensure_numeric_audit(audit_path,lambda:model.audit(audio,target),{"sample_id":row["id"],"blind_id":c["blind_id"]})
        values.append({"blind_id":c["blind_id"],"audio":c["audio"],"audio_sha256":c["audio_sha256"],
                       **model.score(audio,target)})
    controls={}
    if control:
        first=verified_audio(run,row["candidates"][0])
        wrong=next(e for e in PILOT_EMOTIONS if e!=row["emotion"])
        controls["wrong_description"]={"emotion":wrong,**model.score(first,description(wrong))}
        donor=next((r for r in all_rows if r["id"]!=row["id"] and r["emotion"]!=row["emotion"]),None)
        if donor:
            controls["swapped_audio"]={"donor_group":donor["id"],
                **model.score(verified_audio(run,donor["candidates"][0]),target)}
    return {"id":row["id"],"status":"ok","emotion":row["emotion"],"target":target,
            "candidates":values,"controls":controls}


def execute(kind,args):
    score_dtype=getattr(args,"score_dtype","bfloat16")
    run,m=load_manifest(args.run)
    ann=annotations(run,m)
    validate_annotations(ann,m)
    if kind=="a":
        rows,skipped=a_inputs(m,ann)
    else:
        rows=m["identity"]["b"]
        skipped=[]
        if m["identity"].get("b_pending_generation"):
            raise ValueError("B 候选尚未生成；请先显式运行 generate-b")
    if not rows:
        print("没有可评分的样本。",flush=True)
        return
    for r in rows:
        for a in ([r] if kind=="a" else r["candidates"]):
            verified_audio(run,a)
    model_path=Path(args.model or (TTS_MODEL if kind=="a" else CAPTION_MODEL))
    print(f"{kind.upper()}：待评分 {len(rows)}；人工排除 {len(skipped)}；模型 {model_path}",flush=True)
    if not args.execute:
        model_identity(model_path)
        print("轻量检查通过，未加载模型。使用 pilot.py score-"+kind+" --run ... --gpu GPU编号 执行。",flush=True)
        return
    visible=os.environ.get("CUDA_VISIBLE_DEVICES","")
    if not visible or visible=="-1":
        raise ValueError("必须显式设置可用 GPU；推荐通过 pilot.py --gpu 启动")
    if not args.tag.replace("_","").replace("-","").isalnum():
        raise ValueError("tag 只接受字母、数字、下划线和短横线")
    directory=run/"scores"/kind/args.tag
    directory.mkdir(parents=True,exist_ok=True)
    identity={"manifest_id":m["manifest_id"],"kind":kind,"rows":rows,"skipped":skipped,
              "score_protocol":"all16_mean_raw_no_eos" if kind=="a" else "description_body_mean_raw_no_eos",
              "prompt":None if kind=="a" else PROMPT,"attention":args.attention,"controls":args.controls,
              "sdpa_backend":args.sdpa_backend,"numeric_policy":"tf32_off_bf16_reduction_off_math_upcast",
              **runtime_identity(model_path,kind)}
    if kind=="a":
        identity["score_dtype"]=score_dtype
    identity_key=digest(identity)
    with lock(directory/".score.lock"):
        identity_path=directory/"identity.json"
        if identity_path.exists() and read_json(identity_path)["identity_key"]!=identity_key:
            raise ValueError("标注、代码或模型身份发生变化。保留旧结果；请在命令中添加新的 --tag（如 v2）。")
        atomic_json(identity_path,{"identity_key":identity_key,"identity":identity})
        atomic_json(directory/"status.json",{"status":"running","started_at":now(),"gpu":visible})
        try:
            weight_hashes(identity,directory)
            pending=[]
            for r in rows:
                p=directory/"items"/(r["id"]+".json")
                if p.exists() and read_json(p).get("status")=="ok":
                    continue
                pending.append(r)
            if pending:
                from model_interfaces import QwenScorer, CaptionScorer
                print("加载本地基础模型（无 adapter、无训练）…",flush=True)
                print(f"数值配置：attention={args.attention}，SDPA backend={args.sdpa_backend}，score_dtype={score_dtype}。",flush=True)
                model=(QwenScorer(model_path,args.attention,args.sdpa_backend,score_dtype) if kind=="a"
                       else CaptionScorer(model_path,args.attention,args.sdpa_backend))
                for r in pending:
                    position=next(i for i,x in enumerate(rows) if x["id"]==r["id"])
                    print(f"{kind.upper()} {position+1}/{len(rows)} {r['id']}",flush=True)
                    try:
                        result=(score_a_job(model,run,r,rows,directory,identity_key,position<args.controls)
                                if kind=="a" else score_b_job(model,run,r,rows,directory,position<args.controls))
                    except Exception as e:
                        atomic_json(directory/"items"/(r["id"]+".json"),
                                    {"id":r["id"],"status":"invalid","error":str(e),"traceback":traceback.format_exc()})
                        raise
                    atomic_json(directory/"items"/(r["id"]+".json"),result)
                del model
            atomic_json(directory/"status.json",{"status":"complete","completed_at":now(),"scored":len(rows),"skipped":skipped,"gpu":visible})
            print("评分完成："+str(directory),flush=True)
        except Exception as e:
            atomic_json(directory/"status.json",{"status":"failed","failed_at":now(),"error":str(e),"traceback":traceback.format_exc()})
            raise


def main(kind):
    shared_runtime()
    p=argparse.ArgumentParser(description=f"实验 {kind.upper()} 独立评分；不带 --execute 时只进行轻量检查")
    p.add_argument("--run",required=True)
    p.add_argument("--model")
    p.add_argument("--execute",action="store_true")
    p.add_argument("--tag",default="v1")
    p.add_argument("--attention",choices=["sdpa","eager"],default="sdpa")
    p.add_argument("--sdpa-backend",choices=["math","auto"],default="math")
    p.add_argument("--controls",type=int,default=2)
    if kind=="a":
        p.add_argument("--score-dtype",choices=["bfloat16","float32"],default="bfloat16")
    a=p.parse_args()
    if a.controls<0:
        p.error("controls 必须非负")
    execute(kind,a)
