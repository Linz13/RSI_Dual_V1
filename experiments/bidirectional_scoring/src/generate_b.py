"""Optional user-run B generation, only for a prepared pending-generation package."""
from __future__ import annotations

import argparse
from pathlib import Path
import os
import random

from common import (TTS_MODEL, annotations, load_manifest, lock, model_identity, digest,
                    atomic_json, sha256, now, read_json, empty_annotations, shared_runtime)
from prepare_pilot import build_review, wav_duration


def main():
    shared_runtime()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",required=True)
    p.add_argument("--execute",action="store_true")
    a=p.parse_args()
    run,m=load_manifest(a.run)
    pending=m["identity"].get("b_pending_generation",[])
    if not pending:
        raise ValueError("当前样本包已有候选，不需要生成；不会覆盖已有 B")
    ann=annotations(run,m)
    if ann["a"] or ann["b"]:
        raise ValueError("已有标注，不能改变样本包身份。请为新生成使用独立目录")
    if not a.execute:
        print(f"将生成 {len(pending)} 组 × 4。未加载模型。")
        return
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("请通过 pilot.py generate-b --gpu 显式启动")
    with lock(run/".generation.lock"):
        from scoring import runtime_identity,weight_hashes
        from model_interfaces import QwenScorer
        identity=runtime_identity(TTS_MODEL,"a")
        cache=run/"generation"
        cache.mkdir(exist_ok=True)
        key=digest(identity)
        plan=cache/"identity.json"
        if plan.exists() and read_json(plan)["key"]!=key:
            raise ValueError("生成身份发生改变，不能混用断点")
        atomic_json(plan,{"key":key,"identity":identity})
        weight_hashes(identity,cache)
        model=QwenScorer(TTS_MODEL)
        import soundfile as sf
        rng=random.Random(m["identity"]["seed"]+2)
        groups=[]
        for r in pending:
            order=list(range(4));rng.shuffle(order)
            record={"id":r["id"],"source_id":"new_generation", "request":r["request"],"emotion":r["emotion"],"candidates":[]}
            for j,source_index in enumerate(order):
                marker=cache/(r["id"]+f"_{j}.json")
                audio=run/"review/audio"/(r["id"].lower()+f"_{j+1}.wav")
                if marker.exists():
                    c=read_json(marker)
                    if not audio.exists() or sha256(audio)!=c["audio_sha256"]:
                        raise ValueError("生成断点的音频内容不匹配")
                else:
                    seed=r["candidate_seeds"][source_index]
                    model.torch.manual_seed(seed);model.torch.cuda.manual_seed_all(seed)
                    print(f"生成 {r['id']} 候选 {j+1}/4",flush=True)
                    request=r["request"]
                    wavs,sr=model.wrapper.generate_voice_design(text=request["text"],language=request["language"],
                        instruct=request["instruct"],non_streaming_mode=True,do_sample=True,max_new_tokens=512,
                        temperature=1.0,top_k=0,top_p=1.0,repetition_penalty=1.0,
                        subtalker_dosample=True,subtalker_temperature=1.0,subtalker_top_k=0,subtalker_top_p=1.0)
                    audio.parent.mkdir(parents=True,exist_ok=True)
                    sf.write(str(audio),wavs[0],sr,subtype="PCM_16")
                    c={"blind_id":chr(65+j),"audio":"audio/"+audio.name,"audio_sha256":sha256(audio),
                       "duration":wav_duration(audio),"generation_seed":seed,"source_candidate_id":str(source_index)}
                    atomic_json(marker,c)
                record["candidates"].append(c)
            record["duplicate_audio_count"]=4-len({c["audio_sha256"] for c in record["candidates"]})
            groups.append(record)
        ident=m["identity"]
        ident["b"]=groups
        ident["b_pending_generation"]=[]
        ident["b_source"]={"model":str(TTS_MODEL),"model_identity_key":key,"checkpoint":"base",
            "generation":{"temperature":1.0,"top_k":0,"top_p":1.0,"max_new_tokens":512,
                          "subtalker_temperature":1.0,"subtalker_top_k":0,"subtalker_top_p":1.0,
                          "subtalker_dosample":True,"repetition_penalty":1.0,"non_streaming_mode":True},
            "exposure":"new pilot generation; generated-on-instruction is not a true emotion label"}
        updated={**m,"identity":ident,"manifest_id":digest(ident),"generated_at":now()}
        atomic_json(run/"manifest.json",updated)
        atomic_json(run/"annotations.json",empty_annotations(updated["manifest_id"]))
        build_review(run,updated)
        print("候选已生成。请重新启动标注页面；离线包："+str(run/"annotation_bundle.zip"))


if __name__=="__main__":
    main()
