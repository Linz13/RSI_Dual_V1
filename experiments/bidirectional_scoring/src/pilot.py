#!/usr/bin/env python3
"""User-facing entry point. Only explicit score/generate commands launch an existing model environment."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

from common import (ROOT, ENV_ROOT, load_manifest, annotations, validate_annotations,
                    save_annotations, read_json, atomic_json, now, shared_runtime)


def model_command(args):
    if not re.fullmatch(r"\d+|GPU-[A-Za-z0-9-]+|MIG-[A-Za-z0-9/-]+",args.gpu):
        raise ValueError("--gpu 需要一个已分配的 GPU 编号或 UUID")
    run,m=load_manifest(args.run)
    if args.command=="score-a":
        env_name,script="qwen3-tts","score_a.py"
    elif args.command=="score-b":
        env_name,script="midasheng-captioner","score_b.py"
    else:
        env_name,script="qwen3-tts","generate_b.py"
    python=ENV_ROOT/env_name/"bin/python"
    if not python.is_file():
        raise ValueError(f"现有环境不存在：{python}；不会自动安装")
    command=[str(python),"-B","-u",str(ROOT/script),"--run",str(run),"--execute"]
    if args.command!="generate-b":
        command += ["--tag",args.tag,"--attention",args.attention,"--sdpa-backend",args.sdpa_backend,"--controls",str(args.controls)]
    if args.command=="score-a":
        command += ["--score-dtype",args.score_dtype]
    env=dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=args.gpu,HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
               HF_DATASETS_OFFLINE="1",HF_HUB_DISABLE_TELEMETRY="1",PYTHONDONTWRITEBYTECODE="1",
               TOKENIZERS_PARALLELISM="false",HF_MODULES_CACHE=str(run/"cache/hf_modules"))
    logdir=run/"logs"
    logdir.mkdir(parents=True,exist_ok=True)
    log=logdir/(args.command+"_"+now().replace(":","-")+".log")
    print(f"服务器执行 {args.command}，GPU={args.gpu}。日志：{log}",flush=True)
    with log.open("w",encoding="utf-8") as f:
        f.write("COMMAND "+repr(command)+"\nGPU="+args.gpu+"\n")
        f.flush()
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        try:
            for line in process.stdout:
                print(line,end="",flush=True)
                f.write(line)
                f.flush()
            code=process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
    if code:
        print(f"任务失败，日志已保留：{log}\n请把此日志路径发给我排查。不要修改标注或模型环境。",file=sys.stderr)
        raise SystemExit(code)
    if args.command!="generate-b" and not args.skip_report:
        from summarize import summarize
        summarize(run,args.tag)


def status(run):
    run,m=load_manifest(run)
    ann=annotations(run,m)
    validate_annotations(ann,m)
    for kind in ("a","b"):
        rows=m["identity"][kind]
        done=sum(bool(ann[kind].get(r["id"],{}).get("final")) for r in rows)
        print(f"{kind.upper()} 已完成 {done}/{len(rows)}")
    print("标注："+str(run/"annotations.json"))
    print("离线包："+str(run/"annotation_bundle.zip"))


def main():
    shared_runtime()
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    for name in ("serve","status","prepare","report","import","score-a","score-b","generate-b"):
        s=sub.add_parser(name)
        s.add_argument("--run",default=str(ROOT/"runs/smoke01"))
        if name=="serve":
            s.add_argument("--port",type=int,default=8765)
        if name=="prepare":
            s.add_argument("--profile",choices=["smoke","pilot"],default="smoke")
            s.add_argument("--a-n",type=int)
            s.add_argument("--b-groups",type=int)
            s.add_argument("--b-source",choices=["existing","generate"],default="existing")
            s.add_argument("--seed",type=int,default=20260915)
            s.add_argument("--a-min-seconds",type=float,default=4.0)
            s.add_argument("--a-max-seconds",type=float,default=15.0)
            s.add_argument("--exclude-manifest",action="append",default=[])
        if name=="import":
            s.add_argument("--file",required=True)
        if name in ("report","score-a","score-b"):
            s.add_argument("--tag",default="v1")
        if name=="report":
            s.add_argument("--a-tag")
            s.add_argument("--b-tag")
        if name=="score-a":
            s.add_argument("--score-dtype",choices=["bfloat16","float32"],default="bfloat16")
        if name in ("score-a","score-b","generate-b"):
            s.add_argument("--gpu",required=True,help="用户已分配的一个 GPU 编号")
        if name in ("score-a","score-b"):
            s.add_argument("--attention",choices=["sdpa","eager"],default="sdpa")
            s.add_argument("--sdpa-backend",choices=["math","auto"],default="math")
            s.add_argument("--controls",type=int,default=2)
            s.add_argument("--skip-report",action="store_true",help="由外层脚本显式合并两个方向的结果")
    a=p.parse_args()
    if a.command=="prepare":
        from prepare_pilot import prepare
        prepare(a.run,a.a_n if a.a_n is not None else (4 if a.profile=="smoke" else 30),
                a.b_groups if a.b_groups is not None else (4 if a.profile=="smoke" else 30),a.seed,a.b_source,
                a.a_min_seconds,a.a_max_seconds,a.exclude_manifest)
    elif a.command=="serve":
        from serve import main as serve_main
        sys.argv=["serve.py","--run",a.run,"--port",str(a.port)]
        serve_main()
    elif a.command=="status":
        status(a.run)
    elif a.command=="report":
        from summarize import summarize
        summarize(a.run,a.tag,a_tag=a.a_tag,b_tag=a.b_tag)
    elif a.command=="import":
        run,m=load_manifest(a.run)
        save_annotations(run,m,read_json(a.file))
        status(run)
    else:
        model_command(a)


if __name__=="__main__":
    try:
        main()
    except (ValueError,RuntimeError,FileNotFoundError) as e:
        print("未执行或已停止："+str(e),file=sys.stderr)
        raise SystemExit(1)
