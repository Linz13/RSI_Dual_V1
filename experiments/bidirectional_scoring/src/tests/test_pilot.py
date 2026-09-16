from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import wave

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import (PROTOCOL, atomic_json, digest, empty_annotations, save_annotations, sha256)
from model_interfaces import completion_body_span, difference_summary
from prepare_pilot import balanced_sample, build_review, public_data, exclusion_index
from serve import make_handler, ThreadingHTTPServer
from scoring import a_inputs, conditions, ensure_numeric_audit, score_b_job
from summarize import a_metrics,b_metrics,summarize


def fixture(run):
    (run/"review/audio").mkdir(parents=True)
    path=run/"review/audio/a.wav"
    with wave.open(str(path),"wb") as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(8000);f.writeframes(bytes(16000))
    audio={"audio":"audio/a.wav","audio_sha256":sha256(path),"duration":1.0}
    ident={"name":"fixture","a":[{"id":"A001",**audio,"speaker_id":"s1",
            "source_emotion":"sad","source_transcript":"原转写"}],
           "b":[{"id":"B001","emotion":"happy","request":{"text":"文本","instruct":"秘密原指令"},
                  "candidates":[{"blind_id":chr(65+i),**audio,"source_candidate_id":f"SECRET-{i}"} for i in range(4)]}]}
    m={"protocol":PROTOCOL,"manifest_id":digest(ident),"identity":ident}
    atomic_json(run/"manifest.json",m)
    atomic_json(run/"annotations.json",empty_annotations(m["manifest_id"]))
    build_review(run,m)
    return m


class ScoringProtocolTests(unittest.TestCase):
    def test_numeric_difference_keeps_token_error_and_mean_score_drift(self):
        result=difference_summary([[-1,-3],[-2,-4]],[[-1.2,-2.8],[-2,-4]])
        self.assertAlmostEqual(result["max_abs"],.2)
        self.assertAlmostEqual(result["score_delta"],0)
        self.assertAlmostEqual(result["mean_abs"],.1)
        self.assertEqual(result["target_tokens"],4)
        for x,y in [([],[]),([1],[1,2]),([1],[float('nan')])]:
            with self.assertRaises(ValueError):difference_summary(x,y)

    def test_target_boundary_excludes_terminators_and_rejects_bad_prefix(self):
        self.assertEqual(completion_body_span([1,2],[1,2,3,4,99,5],[3,4],[99,5]),(2,4))
        for full in ([2,1,3,4,99,5],[1,2,3,9,4,99,5],[1,2,3,4,99]):
            with self.assertRaises(ValueError):
                completion_body_span([1,2],full,[3,4],[99,5])

    def test_counterfactuals_follow_human_corrected_emotion(self):
        cs=conditions("fearful")
        self.assertEqual(len(cs),4)
        self.assertEqual(sum(x["is_positive"] for x in cs),1)
        self.assertEqual(cs[0]["emotion"],"fearful")

    def test_a_score_direction_and_ties(self):
        def record(pos,neg):
            return {"conditions":[{"emotion":"sad","is_positive":True,"score":pos},
                                  {"emotion":"happy","is_positive":False,"score":neg}]}
        self.assertTrue(a_metrics(record(-1,-2),1e-5)["strict_top1"])
        self.assertEqual(a_metrics(record(-2,-1),1e-5)["rank"],2)
        tied=a_metrics(record(-1,-1+1e-6),1e-5)
        self.assertFalse(tied["strict_top1"])
        self.assertTrue(tied["tied_top1"])
        self.assertEqual(tied["margins"][0]["outcome"],"tie")

    def test_b_human_and_model_ties_adjust_random_baseline(self):
        s={"candidates":[{"blind_id":x,"score":p} for x,p in zip("ABCD",[-1,-1,-2,-3])]}
        human={"final":True,"status":"preferred","best":["A","C"]}
        r=b_metrics(s,human,1e-5)
        self.assertEqual(r["agreement"],.5)
        self.assertEqual(r["random_baseline"],.5)
        for h in (None,{"final":True,"status":"all_bad","best":[]},{"final":True,"status":"uncertain","best":[]}):
            self.assertIsNone(b_metrics(s,h,1e-5)["agreement"])
        tied=b_metrics(s,{"final":True,"status":"preferred","best":list("ABCD")},1e-5)
        self.assertEqual(tied["agreement"],1)
        self.assertEqual(tied["random_baseline"],1)
        self.assertFalse(tied["rank_informative"])

    def test_nan_is_invalid_not_zero(self):
        with self.assertRaises(ValueError):
            b_metrics({"candidates":[{"blind_id":"A","score":float("nan")},{"blind_id":"B","score":-1}]},None,1e-5)

    def test_sampling_balanced_without_score_filtering(self):
        rows=[{"emotion":e,"speaker":str(i%2),"id":e+str(i)} for e in ("happy","sad","angry","neutral") for i in range(10)]
        picked=balanced_sample(rows,8,42,lambda r:r["emotion"],lambda r:r["speaker"])
        self.assertEqual(len({r["id"] for r in picked}),8)
        for e in ("happy","sad","angry","neutral"):
            self.assertEqual(sum(r["emotion"]==e for r in picked),2)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.run=Path(self.tmp.name)
        self.m=fixture(self.run)

    def tearDown(self):
        self.tmp.cleanup()

    def test_public_payload_contains_no_reference_or_candidate_identity(self):
        text=json.dumps(public_data(self.m),ensure_ascii=False)
        for secret in ("source_emotion","source_candidate_id","SECRET-","秘密原指令","reward","score"):
            self.assertNotIn(secret,text)
        self.assertTrue((self.run/"annotation_bundle.zip").exists())

    def test_human_gate_corrections_and_unknown_are_preserved(self):
        ann=empty_annotations(self.m["manifest_id"])
        with self.assertRaises(ValueError):
            a_inputs(self.m,ann)
        ann["a"]["A001"]={"final":True,"heard":True,"status":"verified","emotion":"angry","transcript":"更正转写"}
        saved=save_annotations(self.run,self.m,ann,0)
        jobs,skipped=a_inputs(self.m,saved)
        self.assertEqual(jobs[0]["emotion"],"angry")
        self.assertEqual(jobs[0]["transcript"],"更正转写")
        saved["a"]["A001"]["status"]="uncertain"
        jobs,skipped=a_inputs(self.m,saved)
        self.assertEqual(jobs,[])
        self.assertEqual(skipped[0]["reason"],"uncertain")

    def test_import_wrong_package_and_revision_conflict_rejected(self):
        ann=empty_annotations("wrong")
        with self.assertRaises(ValueError):
            save_annotations(self.run,self.m,ann,0)
        ann=empty_annotations(self.m["manifest_id"])
        save_annotations(self.run,self.m,ann,0)
        with self.assertRaises(ValueError):
            save_annotations(self.run,self.m,ann,0)

    def test_navigation_completion_is_accepted_and_shared_file_writable(self):
        ann=empty_annotations(self.m["manifest_id"])
        ann["a"]["A001"]={"status":"verified","emotion":"sad","transcript":"已核对",
                          "final":True,"heard":True,"completion_method":"navigation_save"}
        result=save_annotations(self.run,self.m,ann,0)
        self.assertEqual(result["a"]["A001"]["completion_method"],"navigation_save")
        self.assertEqual((self.run/"annotations.json").stat().st_mode&0o666,0o666)

    def test_exclusion_uses_source_ids_and_audio_hashes(self):
        self.m["identity"]["a"][0]["source_id"]="source-a"
        self.m["identity"]["b"][0]["source_id"]="source-b"
        self.m["manifest_id"]=digest(self.m["identity"])
        atomic_json(self.run/"manifest.json",self.m)
        idx=exclusion_index([self.run/"manifest.json"])
        self.assertEqual(idx["a_ids"],{"source-a"})
        self.assertEqual(idx["b_ids"],{"source-b"})
        self.assertIn(self.m["identity"]["a"][0]["audio_sha256"],idx["hashes"])

    def test_smoke_launcher_forwards_gpu_and_keeps_b_independent(self):
        # Replace only the launcher's python3 with a recording stub: no model process.
        stub=self.run/"python3"
        stub.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$PILOT_TEST_LOG"\n'
                        'if [ "$3" = "score-a" ]; then exit "$PILOT_TEST_A_EXIT"; fi\nexit 0\n')
        stub.chmod(0o755)
        logfile=self.run/"calls.log"
        env=dict(os.environ,PATH=str(self.run)+os.pathsep+os.environ.get("PATH",""),
                 PILOT_TEST_LOG=str(logfile),PILOT_TEST_A_EXIT="7")
        launcher=Path(__file__).resolve().parents[1]/"run_smoke.sh"
        result=subprocess.run(["/bin/bash",str(launcher),"6"],env=env,capture_output=True,text=True)
        self.assertEqual(result.returncode,1)
        self.assertEqual(logfile.read_text().splitlines(),[
            "-B pilot.py score-a --run runs/smoke01 --gpu 6 --tag v2 --sdpa-backend math",
            "-B pilot.py score-b --run runs/smoke01 --gpu 6 --tag v2 --sdpa-backend math"])
        self.assertIn("A exit=7; B exit=0",result.stdout)
        missing=subprocess.run(["/bin/bash",str(launcher)],env=env,capture_output=True,text=True)
        self.assertEqual(missing.returncode,2)

    def test_a_precision_retry_never_launches_b_and_stops_before_report_on_failure(self):
        stub=self.run/"python3"
        stub.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$PILOT_TEST_LOG"\n'
                        'if [ "$3" = "score-a" ]; then exit "$PILOT_TEST_A_EXIT"; fi\nexit 0\n')
        stub.chmod(0o755)
        logfile=self.run/"calls.log"
        env=dict(os.environ,PATH=str(self.run)+os.pathsep+os.environ.get("PATH",""),
                 PILOT_TEST_LOG=str(logfile),PILOT_TEST_A_EXIT="0")
        launcher=Path(__file__).resolve().parents[1]/"run_a_precision_retry.sh"
        result=subprocess.run(["/bin/bash",str(launcher),"5"],env=env,capture_output=True,text=True)
        self.assertEqual(result.returncode,0)
        calls=logfile.read_text().splitlines()
        self.assertEqual(calls,[
            "-B pilot.py score-a --run runs/smoke01 --gpu 5 --tag v3_fp32 --sdpa-backend math --score-dtype float32 --skip-report",
            "-B pilot.py report --run runs/smoke01 --tag v3_combined --a-tag v3_fp32 --b-tag v2"])
        logfile.unlink();env["PILOT_TEST_A_EXIT"]="7"
        result=subprocess.run(["/bin/bash",str(launcher),"5"],env=env,capture_output=True,text=True)
        self.assertEqual(result.returncode,7)
        self.assertEqual(logfile.read_text().splitlines(),calls[:1])

    def test_report_combines_explicit_a_b_tags_without_copying_scores(self):
        import zipfile
        ann=empty_annotations(self.m["manifest_id"])
        ann["a"]["A001"]={"final":True,"heard":True,"status":"verified","emotion":"sad","transcript":"原转写"}
        ann["b"]["B001"]={"final":True,"heard":True,"status":"preferred","best":["A"]}
        save_annotations(self.run,self.m,ann,0)
        for kind,tag in [('a','v3_fp32'),('b','v2')]:
            identity={"manifest_id":self.m["manifest_id"],"kind":kind}
            d=self.run/'scores'/kind/tag
            atomic_json(d/'identity.json',{'identity_key':digest(identity),'identity':identity})
            atomic_json(d/'status.json',{'status':'complete'})
            atomic_json(d/'numeric_audit.json',{'passed':True,'fixture_tag':tag})
        row=self.m['identity']['a'][0]
        atomic_json(self.run/'scores/a/v3_fp32/items/A001.json',{'id':'A001','status':'ok',
            'audio':row['audio'],'audio_sha256':row['audio_sha256'],'reference_emotion':'sad','verified_transcript':'原转写',
            'conditions':[{'is_positive':True,'emotion':'sad','score':-1},{'is_positive':False,'emotion':'happy','score':-2}]})
        from common import description
        cs=self.m['identity']['b'][0]['candidates']
        bpath=self.run/'scores/b/v2/items/B001.json'
        atomic_json(bpath,{'id':'B001','status':'ok','target':description('happy'),
                          'candidates':[{**c,'score':s} for c,s in zip(cs,[-1,-2,-3,-4])]})
        old_hash=sha256(bpath)
        result=summarize(self.run,'combined',a_tag='v3_fp32',b_tag='v2')
        self.assertEqual(result['a']['scored'],1)
        self.assertEqual(result['b']['agreement'],1)
        self.assertEqual(result['score_source_tags'],{'a':'v3_fp32','b':'v2'})
        self.assertEqual(result['numeric_audits']['b']['fixture_tag'],'v2')
        self.assertEqual(sha256(bpath),old_hash)
        self.assertFalse((self.run/'scores/b/v3_fp32').exists())
        with zipfile.ZipFile(self.run/'reports/combined/report_bundle.zip') as z:
            self.assertEqual(json.loads(z.read('raw_scores/b/B001.json'))['candidates'][0]['score'],-1)
            self.assertEqual(json.loads(z.read('raw_scores/b/numeric_audit.json'))['fixture_tag'],'v2')
        with self.assertRaises(ValueError):summarize(self.run,'bad',a_tag='../bad',b_tag='v2')

    def test_failed_numeric_audit_is_saved_and_cannot_unlock_scoring(self):
        from unittest.mock import Mock
        model=Mock()
        model.audit.return_value={"passed":False,"padding_max_abs":.08232876658439636,
                                  "target_logprobs":{"bulk":[-1],"padded":[-1.08232876658439636]}}
        directory=self.run/"scores/b/v2"
        row=self.m["identity"]["b"][0]
        with self.assertRaises(ValueError):score_b_job(model,self.run,row,[row],directory,False)
        result=json.loads((directory/"numeric_audit.json").read_text())
        self.assertIs(result["passed"],False)
        self.assertEqual(result["sample_id"],"B001")
        self.assertIn("target_logprobs",result)
        model.score.assert_not_called()
        model.audit.reset_mock()
        with self.assertRaises(ValueError):score_b_job(model,self.run,row,[row],directory,False)
        model.audit.assert_not_called()
        model.score.assert_not_called()

    def test_numeric_audit_exception_preserves_evidence_and_pass_can_resume(self):
        from unittest.mock import Mock
        path=self.run/"audit_error.json"
        check=Mock(side_effect=RuntimeError("fixture failure"))
        with self.assertRaises(RuntimeError):ensure_numeric_audit(path,check,{"sample_id":"A001"})
        result=json.loads(path.read_text())
        self.assertFalse(result["passed"])
        self.assertEqual(result["error"],"fixture failure")
        path=self.run/"audit_pass.json"
        check=Mock(return_value={"passed":True})
        ensure_numeric_audit(path,check,{"sample_id":"A001"})
        ensure_numeric_audit(path,check,{"sample_id":"A002"})
        self.assertEqual(check.call_count,1)

    def test_report_without_models_or_human_does_not_claim_validity(self):
        result=summarize(self.run)
        self.assertIsNone(result["a"]["strict_top1_rate"])
        self.assertIsNone(result["b"]["agreement"])
        self.assertEqual(result["b"]["judgment"],"pending_independent_reference")
        self.assertEqual(result["a"]["statuses"],{"annotation_pending":1})

    def test_scored_report_and_stale_annotations(self):
        # Synthetic fixtures live only in a TemporaryDirectory, never in the real run.
        ann=empty_annotations(self.m["manifest_id"])
        ann["a"]["A001"]={"final":True,"heard":True,"status":"verified","emotion":"sad","transcript":"原转写"}
        ann["b"]["B001"]={"final":True,"heard":True,"status":"preferred","best":["A","C"]}
        save_annotations(self.run,self.m,ann,0)
        for kind in ("a","b"):
            d=self.run/"scores"/kind/"v1"
            ident={"manifest_id":self.m["manifest_id"]}
            atomic_json(d/"identity.json",{"identity_key":digest(ident),"identity":ident})
            atomic_json(d/"status.json",{"status":"complete"})
        row=self.m["identity"]["a"][0]
        atomic_json(self.run/"scores/a/v1/items/A001.json",{"id":"A001","status":"ok","audio":row["audio"],
            "audio_sha256":row["audio_sha256"],"reference_emotion":"sad","verified_transcript":"原转写",
            "conditions":[{"is_positive":True,"emotion":"sad","score":-1},{"is_positive":False,"emotion":"happy","score":-2}]})
        from common import description
        cs=self.m["identity"]["b"][0]["candidates"]
        atomic_json(self.run/"scores/b/v1/items/B001.json",{"id":"B001","status":"ok","target":description("happy"),
            "candidates":[{**c,"score":s} for c,s in zip(cs,[-1,-1,-2,-3])]})
        result=summarize(self.run)
        self.assertEqual(result["a"]["strict_top1_rate"],1)
        self.assertEqual(result["b"]["agreement"],.5)
        self.assertEqual(result["b"]["random_baseline"],.5)
        self.assertEqual(len(result["tie_sensitivity"]),3)
        self.assertTrue((self.run/"reports/v1/a_pairs.csv").stat().st_size>0)
        ann["a"]["A001"]["transcript"]="改后的转写"
        save_annotations(self.run,self.m,ann)
        stale=summarize(self.run)
        self.assertEqual(stale["a"]["scored"],0)
        self.assertEqual(stale["a"]["rows"][0]["status"],"stale_after_annotation_change")

    def test_server_audio_seek_save_reload_and_private_paths(self):
        server=ThreadingHTTPServer(("127.0.0.1",0),make_handler(self.run,self.m))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base=f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(base+"/") as r:
                self.assertEqual(r.status,200)
            with urlopen(Request(base+"/audio/a.wav",headers={"Range":"bytes=44-63"})) as r:
                self.assertEqual(r.status,206);self.assertEqual(len(r.read()),20)
            for path in ("/manifest.json","/../manifest.json","/scores/a/v1/items/A001.json"):
                with self.assertRaises(HTTPError) as e:
                    urlopen(base+path)
                self.assertEqual(e.exception.code,404)
            ann=empty_annotations(self.m["manifest_id"])
            ann["b"]["B001"]={"status":"all_bad","final":True,"heard":True,"best":[]}
            body=json.dumps({"annotations":ann,"expected_revision":0}).encode()
            request=Request(base+"/api/annotations",data=body,headers={"Content-Type":"application/json"})
            with urlopen(request) as r:
                self.assertEqual(json.load(r)["revision"],1)
            with urlopen(base+"/api/annotations") as r:
                self.assertEqual(json.load(r)["b"]["B001"]["status"],"all_bad")
            bad=Request(base+"/api/annotations",data=body,headers={"Content-Type":"application/json","Origin":"http://external.invalid"})
            with self.assertRaises(HTTPError) as e:
                urlopen(bad)
            self.assertEqual(e.exception.code,403)
        finally:
            server.shutdown();server.server_close();thread.join()


if __name__=="__main__":
    unittest.main()
