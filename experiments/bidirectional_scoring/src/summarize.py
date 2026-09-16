"""Summaries from frozen scores and independent annotations; no model calls."""
from __future__ import annotations

import argparse
import csv
import html
import io
import math
from pathlib import Path
import random
import zipfile

from common import (load_manifest, annotations, validate_annotations, read_json, atomic_json,
                    atomic_text, finite, description, digest, now, shared_runtime)


def compare(delta, epsilon):
    return "win" if delta>epsilon else "loss" if delta < -epsilon else "tie"


def a_metrics(score, epsilon):
    conditions=score["conditions"]
    positives=[c for c in conditions if c["is_positive"]]
    if len(positives)!=1 or len(conditions)<2 or not finite([c["score"] for c in conditions]):
        raise ValueError("A 条件分数无效")
    pos=positives[0]
    negs=[c for c in conditions if not c["is_positive"]]
    maximum=max(c["score"] for c in conditions)
    top=[c["emotion"] for c in conditions if maximum-c["score"]<=epsilon]
    return {"rank":1+sum(c["score"]>pos["score"]+epsilon for c in negs),
            "strict_top1":all(pos["score"]>c["score"]+epsilon for c in negs),
            "tied_top1":len(top)>1 and pos["emotion"] in top,"top_emotions":top,
            "positive_score":pos["score"],
            "margins":[{"negative_emotion":c["emotion"],"delta":pos["score"]-c["score"],
                        "outcome":compare(pos["score"]-c["score"],epsilon)} for c in negs]}


def b_metrics(score, human, epsilon):
    cs=score["candidates"]
    if len(cs)<2 or not finite([c["score"] for c in cs]):
        raise ValueError("B 候选分数无效")
    ids=[c["blind_id"] for c in cs]
    if len(ids)!=len(set(ids)):
        raise ValueError("重复候选 ID")
    maximum=max(c["score"] for c in cs)
    top=[c["blind_id"] for c in cs if maximum-c["score"]<=epsilon]
    result={"model_top":top,"model_tie":len(top)>1,
            "ranking":[{"blind_id":c["blind_id"],"score":c["score"]} for c in sorted(cs,key=lambda c:-c["score"])],
            "reference_status":"pending","agreement":None,"random_baseline":None}
    if human and human.get("final"):
        result["reference_status"]=human["status"]
        if human["status"]=="preferred":
            h=set(human["best"])
            if not h or not h<=set(ids):
                raise ValueError("人工最佳集合无效")
            # Expected hit if selecting uniformly within model ties; avoids a hidden tie-break advantage.
            result.update(human_best=sorted(h),human_tie=len(h)>1,
                          agreement=len(set(top)&h)/len(top),random_baseline=len(h)/len(cs),
                          rank_informative=len(h)<len(cs),all_candidates_tied_best=len(h)==len(cs))
    return result


def mean(values):
    return sum(values)/len(values) if values else None


def bootstrap_uplift(rows):
    if len(rows)<2:
        return None
    values=[r["agreement"]-r["random_baseline"] for r in rows]
    rng=random.Random(20260915)
    samples=sorted(sum(rng.choices(values,k=len(values)))/len(values) for _ in range(2000))
    return [samples[49],samples[1949]]


def read_scores(run,kind,tag,m):
    directory=run/"scores"/kind/tag
    if not directory.exists():
        return {},"not_run",None
    identity=read_json(directory/"identity.json")
    if identity["identity_key"]!=digest(identity["identity"]) or identity["identity"]["manifest_id"]!=m["manifest_id"]:
        raise ValueError("评分身份与当前样本包不同")
    status=read_json(directory/"status.json")["status"] if (directory/"status.json").exists() else "incomplete"
    scores={}
    for p in sorted((directory/"items").glob("*.json")):
        row=read_json(p)
        if row["id"] in scores:
            raise ValueError("重复评分记录")
        scores[row["id"]]=row
    return scores,status,identity["identity"]


def write_csv(path,rows):
    if not rows:
        atomic_text(path,"")
        return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    output=io.StringIO()
    writer=csv.DictWriter(output,fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path,"\ufeff"+output.getvalue())


def summarize(run,tag="v1",epsilon=1e-5,a_tag=None,b_tag=None):
    source_tags={"a":a_tag or tag,"b":b_tag or tag}
    for value in [tag,*source_tags.values()]:
        if not value.replace("_","").replace("-","").isalnum():
            raise ValueError("tag 只接受字母、数字、下划线和短横线")
    if not math.isfinite(epsilon) or epsilon<0:
        raise ValueError("平局容差必须为有限非负数")
    run,m=load_manifest(run)
    ann=annotations(run,m)
    validate_annotations(ann,m)
    a_scores,a_status,_=read_scores(run,"a",source_tags["a"],m)
    b_scores,b_status,_=read_scores(run,"b",source_tags["b"],m)
    output={"protocol":m["protocol"],"manifest_id":m["manifest_id"],"generated_at":now(),
            "tag":tag,"score_source_tags":source_tags,"tie_epsilon":epsilon,"a":{"run_status":a_status,"rows":[]},
            "b":{"run_status":b_status,"rows":[]},
            "limits":["本次不训练；双向一致不等于正确。","官方标签和生成指令不替代当前人工核验。",
                      "A 来自历史反复评测集；B 复用历史训练候选。","B 随机基线按每组人工最佳集合计算；全部不合格与无法判断单列。",
                      "主要评分为全部目标 token 均值；A 不含 EOS，不是波形概率密度。"]}
    a_csv=[]
    for row in m["identity"]["a"]:
        rid=row["id"];label=ann["a"].get(rid,{})
        result={"id":rid,"audio":row["audio"],"speaker_id":row["speaker_id"],"source_emotion":row["source_emotion"]}
        score=a_scores.get(rid)
        if not label.get("final"):
            result["status"]="annotation_pending"
        elif label["status"]!="verified":
            result["status"]="human_"+label["status"]
        elif not score:
            result["status"]="score_missing"
        elif score.get("status")!="ok":
            result.update(status="score_invalid",error=score.get("error"))
        elif (score["reference_emotion"]!=label["emotion"] or score["verified_transcript"]!=label["transcript"].strip()
              or score["audio_sha256"]!=row["audio_sha256"]):
            result["status"]="stale_after_annotation_change"
        else:
            try:
                result.update(status="scored",emotion=label["emotion"],**a_metrics(score,epsilon))
                for margin in result["margins"]:
                    a_csv.append({"id":rid,"speaker_id":row["speaker_id"],"emotion":label["emotion"],
                                  "rank":result["rank"],"strict_top1":result["strict_top1"],**margin})
            except (ValueError,KeyError,TypeError) as e:
                result.update(status="score_invalid",error=str(e))
        output["a"]["rows"].append(result)
    a_ok=[r for r in output["a"]["rows"] if r["status"]=="scored"]
    output["a"].update(total=len(output["a"]["rows"]),scored=len(a_ok),
        strict_top1_count=sum(r["strict_top1"] for r in a_ok),strict_top1_rate=mean([float(r["strict_top1"]) for r in a_ok]),
        tied_top1_count=sum(r["tied_top1"] for r in a_ok),
        pairwise={name:sum(x["outcome"]==name for r in a_ok for x in r["margins"]) for name in ("win","loss","tie")},
        statuses={status:sum(r["status"]==status for r in output["a"]["rows"]) for status in sorted({r["status"] for r in output["a"]["rows"]})},
        by_emotion={e:{"n":len(group),"strict_top1_rate":mean([float(r["strict_top1"]) for r in group]),
                       "mean_margin":mean([x["delta"] for r in group for x in r["margins"]])}
                    for e in sorted({r["emotion"] for r in a_ok}) if (group:=[r for r in a_ok if r["emotion"]==e])})
    b_csv=[]
    for row in m["identity"]["b"]:
        rid=row["id"];score=b_scores.get(rid)
        result={"id":rid,"emotion":row["emotion"],"status":"score_missing"}
        if score and score.get("status")!="ok":
            result.update(status="score_invalid",error=score.get("error"))
        elif score:
            try:
                expected={c["blind_id"]:c["audio_sha256"] for c in row["candidates"]}
                if {c["blind_id"]:c["audio_sha256"] for c in score["candidates"]}!=expected or score["target"]!=description(row["emotion"]):
                    raise ValueError("B 评分音频或目标描述不匹配")
                result.update(status="scored",**b_metrics(score,ann["b"].get(rid),epsilon))
                for rank,c in enumerate(result["ranking"],1):
                    b_csv.append({"id":rid,"emotion":row["emotion"],"order":rank,**c,
                                  "is_model_top":c["blind_id"] in result["model_top"],
                                  "human_status":result["reference_status"],
                                  "human_best":",".join(result.get("human_best",[]))})
            except (ValueError,KeyError,TypeError) as e:
                result.update(status="score_invalid",error=str(e))
        output["b"]["rows"].append(result)
    b_ok=[r for r in output["b"]["rows"] if r["status"]=="scored"]
    comparable=[r for r in b_ok if r["agreement"] is not None]
    human_statuses={k:sum(bool(v.get("final")) and v.get("status")==k for v in ann["b"].values())
                    for k in ("preferred","all_bad","uncertain")}
    output["b"].update(total=len(output["b"]["rows"]),scored=len(b_ok),comparable_groups=len(comparable),
        agreement=mean([r["agreement"] for r in comparable]),random_baseline=mean([r["random_baseline"] for r in comparable]),
        uplift=mean([r["agreement"]-r["random_baseline"] for r in comparable]),
        descriptive_group_bootstrap_95=bootstrap_uplift(comparable),human_statuses=human_statuses,
        model_tie_groups=sum(r["model_tie"] for r in b_ok),human_tie_groups=sum(r.get("human_tie",False) for r in comparable),
        judgment="pending_independent_reference" if not comparable else "sample_comparison_only",
        statuses={s:sum(r["status"]==s for r in output["b"]["rows"]) for s in sorted({r["status"] for r in output["b"]["rows"]})})
    informative=[r for r in comparable if r["rank_informative"]]
    output["b"]["rank_informative_groups"]=len(informative)
    output["b"]["all_candidates_tied_best_groups"]=sum(r["all_candidates_tied_best"] for r in comparable)
    output["b"]["informative_subset"]={"n":len(informative),
        "agreement":mean([r["agreement"] for r in informative]),
        "random_baseline":mean([r["random_baseline"] for r in informative]),
        "uplift":mean([r["agreement"]-r["random_baseline"] for r in informative])}
    output["b"]["by_emotion"]={e:{"comparable_groups":len(group),
        "agreement":mean([r["agreement"] for r in group]),"random_baseline":mean([r["random_baseline"] for r in group])}
        for e in sorted({r["emotion"] for r in comparable}) if (group:=[r for r in comparable if r["emotion"]==e])}
    output["tie_sensitivity"]=[]
    for eps in sorted({epsilon,1e-5,1e-4,1e-3}):
        av=[a_metrics(a_scores[r["id"]],eps) for r in a_ok]
        bv=[b_metrics(b_scores[r["id"]],ann["b"].get(r["id"]),eps) for r in b_ok]
        output["tie_sensitivity"].append({"epsilon":eps,
            "a_strict_top1_rate":mean([float(r["strict_top1"]) for r in av]),
            "a_tied_top1_count":sum(r["tied_top1"] for r in av),
            "b_agreement":mean([r["agreement"] for r in bv if r["agreement"] is not None]),
            "b_model_tie_groups":sum(r["model_tie"] for r in bv)})
    output["numeric_audits"]={}
    for kind in ("a","b"):
        audit_path=run/"scores"/kind/source_tags[kind]/"numeric_audit.json"
        if audit_path.exists():
            output["numeric_audits"][kind]=read_json(audit_path)
    report=run/"reports"/tag
    atomic_json(report/"summary.json",output)
    write_csv(report/"a_pairs.csv",a_csv)
    write_csv(report/"b_ranking.csv",b_csv)
    controls={"a":{k:v.get("controls",{}) for k,v in a_scores.items() if v.get("status")=="ok"},
              "b":{k:v.get("controls",{}) for k,v in b_scores.items() if v.get("status")=="ok"}}
    atomic_json(report/"condition_checks.json",controls)
    def percent(value):
        return "尚无可计算结果" if value is None else f"{100*value:.1f}%"
    audit_overview={kind:{k:v for k,v in audit.items() if k!="target_logprobs"}
                    for kind,audit in output["numeric_audits"].items()}
    md=(f"# 双向评分报告 · {m['identity']['name']}\n\n"
        f"评分来源：A `{source_tags['a']}`；B `{source_tags['b']}`。\n\n"
        f"## A\n\n已评分 {len(a_ok)} / {len(m['identity']['a'])}；严格第一比例：{percent(output['a']['strict_top1_rate'])}。\n\n"
        f"胜负平：{output['a']['pairwise']}；并列第一：{output['a']['tied_top1_count']}。\n\n"
        f"状态：{output['a']['statuses']}。\n\n"
        f"## B\n\n已评分 {len(b_ok)} 组；有可比较人工偏好的 {len(comparable)} 组。\n\n"
        f"人工最佳命中：{percent(output['b']['agreement'])}；对应随机基线：{percent(output['b']['random_baseline'])}。\n\n"
        f"人工结论：{human_statuses}；模型平局 {output['b']['model_tie_groups']} 组。\n\n"
        f"四条并列最佳 {output['b']['all_candidates_tied_best_groups']} 组；人工确有排序区别 {len(informative)} 组。"
        "全组与有排序区别子集分别报告，不能把前者的全命中当作排序能力。\n\n"
        f"有排序区别子集：命中 {percent(output['b']['informative_subset']['agreement'])}；"
        f"随机基线 {percent(output['b']['informative_subset']['random_baseline'])}。\n\n"
        +( "无独立人工偏好参照，只能输出待核验排序，不能声称评分有效。\n\n" if not comparable else
           "仅比较当前样本。分组 bootstrap 区间不包含训练随机性或全部数据来源相关性；少量试运行用于验证执行链路。\n\n")+
        "## 数值检查\n\n"+str(audit_overview)+"\n\n逐 token 检查明细见 numeric_audit.json 或 summary.json。\n\n平局容差敏感性："+str(output["tie_sensitivity"])+"\n\n"+
        "## 边界\n\n"+"\n".join("- "+x for x in output["limits"])+"\n")
    atomic_text(report/"report.md",md)
    # This unblinded report is a separate artifact, deliberately not served by the review server.
    blocks=["<!doctype html><meta charset='utf-8'><title>双向评分结果</title><style>body{font:16px/1.6 system-ui;max-width:980px;margin:30px auto;padding:20px}article{border-top:1px solid #ccc;margin:20px 0;padding:15px 0}audio{display:block;width:100%;max-width:650px}pre{white-space:pre-wrap}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px}</style>",
            "<h1>评分结果与试听</h1><p>此页面含模型排名，只用于标注完成后的检查。标注页面始终隐藏评分。</p><pre>"+html.escape(md)+"</pre>"]
    for kind,results in (("a",a_scores),("b",b_scores)):
        for row in output[kind]["rows"]:
            blocks.append("<article><h2>"+html.escape(row["id"])+"</h2><pre>"+html.escape(str(row))+"</pre>")
            if row["status"]=="scored":
                score=results[row["id"]]
                audios=[{"audio":score["audio"],"blind_id":"真实音频"}] if kind=="a" else score["candidates"]
                for c in audios:
                    blocks.append("<p>"+html.escape(c["blind_id"])+"</p><audio controls preload='none' src='../../review/"+html.escape(c["audio"],quote=True)+"'></audio>")
                small=({"conditions":[{k:v for k,v in c.items() if k!="token_logprobs"} for c in score["conditions"]]}
                       if kind=="a" else {"raw_scoring":score["candidates"]})
                blocks.append("<details><summary>原始评分依据</summary><pre>"+html.escape(str(small))+"</pre></details>")
            blocks.append("</article>")
    html_text="\n".join(blocks)
    atomic_text(report/"report.html",html_text)
    with zipfile.ZipFile(report/"report_bundle.zip","w",compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("report.html",html_text.replace("../../review/audio/","audio/"))
        for name in ("report.md","summary.json","a_pairs.csv","b_ranking.csv","condition_checks.json"):
            archive.write(report/name,name)
        audio_paths={r["audio"] for r in m["identity"]["a"]}
        audio_paths.update(c["audio"] for r in m["identity"]["b"] for c in r["candidates"])
        for audio in sorted(audio_paths):
            archive.write(run/"review"/audio,audio)
        for kind in ("a","b"):
            for path in sorted((run/"scores"/kind/source_tags[kind]/"items").glob("*.json")):
                archive.write(path,Path("raw_scores")/kind/path.name)
            for name in ("identity.json","numeric_audit.json","status.json"):
                path=run/"scores"/kind/source_tags[kind]/name
                if path.exists():archive.write(path,Path("raw_scores")/kind/name)
    print(md)
    print("完整结果："+str(report))
    return output


def main():
    shared_runtime()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",required=True)
    p.add_argument("--tag",default="v1")
    p.add_argument("--a-tag")
    p.add_argument("--b-tag")
    p.add_argument("--tie-epsilon",type=float,default=1e-5)
    a=p.parse_args()
    summarize(a.run,a.tag,a.tie_epsilon,a.a_tag,a.b_tag)


if __name__=="__main__":
    main()
