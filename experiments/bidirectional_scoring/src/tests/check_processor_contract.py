"""CPU-only tokenizer/processor contract check; model loading is explicitly prohibited."""
from pathlib import Path
import os
import sys
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ["CUDA_VISIBLE_DEVICES"]=""
from common import ROOT, CAPTION_MODEL, PILOT_EMOTIONS, description, read_json, atomic_json
from model_interfaces import CaptionScorer,QwenScorer


def main():
    import torch
    from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
    with patch.object(AutoModelForCausalLM,"from_pretrained",side_effect=AssertionError("Model loading forbidden")):
        scorer=object.__new__(CaptionScorer)
        scorer.torch=torch
        scorer.device=torch.device("cpu")
        scorer.dtype=torch.float32
        scorer.processor=AutoProcessor.from_pretrained(str(CAPTION_MODEL),trust_remote_code=True,local_files_only=True)
        scorer.tokenizer=AutoTokenizer.from_pretrained(str(CAPTION_MODEL),trust_remote_code=True,local_files_only=True)
        manifest=read_json(ROOT/"runs/smoke01/manifest.json")
        audio=ROOT/"runs/smoke01/review"/manifest["identity"]["b"][0]["candidates"][0]["audio"]
        records=[]
        for emotion in PILOT_EMOTIONS:
            inputs,start,end,body=scorer.prepare(audio,description(emotion))
            assert not torch.cuda.is_initialized()
            assert scorer.tokenizer.decode(body)==description(emotion)
            assert "input_values" in inputs and len(body)==end-start
            records.append({"emotion":emotion,"target_tokens":len(body),"prefix_tokens":start,
                            "suffix_ids":inputs["input_ids"][0,end:].tolist(),"audio_shape":list(inputs["input_values"].shape)})
    # Fixed numeric fixture distinguishes actual token averaging from the old 0.3 weighting.
    fixture=torch.full((2,16),-7.0)
    fixture[:,0]=torch.tensor([-1.,-3.])
    qwen=object.__new__(QwenScorer)
    qwen.arrays=lambda request,codes:(fixture,{"target_tokens":32})
    result=qwen.score({},[])
    assert result["score"]==-107/16
    assert abs(result["legacy_weighted_score"]-(-4.1))<1e-10
    result={"passed":True,"model_loaded":False,"cuda_initialized":torch.cuda.is_initialized(),
            "all16_token_mean_fixture":True,"records":records}
    atomic_json(ROOT/"checks/processor_contract.json",result)
    print(result)


if __name__=="__main__":
    main()
