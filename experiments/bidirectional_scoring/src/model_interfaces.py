"""Pure inference adapters. Imports/models are created only inside explicit constructors."""
from __future__ import annotations

import contextlib
import math

from common import PROMPT, digest
from qwen_conditioning import VoiceDesignConditioning


def difference_summary(reference, alternate):
    """CPU lists only; retain score drift as well as the worst individual token."""
    def flatten(values):
        for x in values:
            if isinstance(x, (list, tuple)):
                yield from flatten(x)
            else:
                yield float(x)
    a, b = list(flatten(reference)), list(flatten(alternate))
    if not a or len(a) != len(b) or not all(math.isfinite(x) for x in a+b):
        raise ValueError("数值检查的目标 token 数不一致、为空或含非有限值")
    delta = [y-x for x,y in zip(a,b)]
    worst = max(range(len(delta)), key=lambda i: abs(delta[i]))
    return {"target_tokens":len(a), "max_abs":abs(delta[worst]),
            "mean_abs":sum(abs(x) for x in delta)/len(delta),
            "score_delta":sum(delta)/len(delta), "worst_flat_index":worst}


class NumericalPolicy:
    def configure_numerics(self, sdpa_backend):
        if sdpa_backend not in ("auto", "math"):
            raise ValueError("sdpa_backend 必须为 auto 或 math")
        torch = self.torch
        self.sdpa_backend = sdpa_backend
        # Process-local settings, no changes to installed environments or model files.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)

    def attention_context(self):
        if self.sdpa_backend == "auto":
            return contextlib.nullcontext()
        return self.torch.nn.attention.sdpa_kernel(self.torch.nn.attention.SDPBackend.MATH)

    def numeric_environment(self):
        torch = self.torch
        return {"sdpa_backend":self.sdpa_backend, "model_dtype":getattr(self,"score_dtype","bfloat16"),
                "torch":torch.__version__, "cuda_runtime":torch.version.cuda,
                "gpu_name":torch.cuda.get_device_name(self.device),
                "tf32_matmul":torch.backends.cuda.matmul.allow_tf32,
                "tf32_cudnn":torch.backends.cudnn.allow_tf32,
                "bf16_reduced_precision_reduction":torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                "math_sdp_low_precision_reduction":torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()}


def completion_body_span(prompt_ids, full_ids, body_ids, suffix_ids):
    """No searching for a matching answer substring: exact prefix, body and chat suffix."""
    start = len(prompt_ids)
    end = start + len(body_ids)
    if not body_ids or full_ids[:start] != prompt_ids:
        raise ValueError("目标描述未保留聊天提示前缀")
    if full_ids[start:end] != body_ids or full_ids[end:] != suffix_ids:
        raise ValueError("正文 token 边界与聊天模板不一致；拒绝扩大目标 mask")
    return start, end


class QwenScorer(VoiceDesignConditioning, NumericalPolicy):
    def __init__(self, path, attention="sdpa", sdpa_backend="math", score_dtype="bfloat16"):
        if score_dtype not in ("bfloat16","float32"):
            raise ValueError("A score_dtype 必须为 bfloat16 或 float32")
        import torch
        from qwen_tts import Qwen3TTSModel
        if not torch.cuda.is_available():
            raise RuntimeError("没有可用 GPU；请由用户在模型服务器显式指定 --gpu")
        self.torch = torch
        self.device = torch.device("cuda:0")
        self.configure_numerics(sdpa_backend)
        self.wrapper = Qwen3TTSModel.from_pretrained(str(path), device_map="cuda:0",
            dtype=torch.bfloat16, attn_implementation=attention, local_files_only=True)
        self.core = self.wrapper.model
        self.core.eval()
        self.core.requires_grad_(False)
        if self.core.tts_model_type != "voice_design" or self.core.config.talker_config.num_code_groups != 16:
            raise ValueError("需要 16 码本 VoiceDesign 模型")
        self.core.speech_tokenizer.model.eval()
        self.set_score_dtype(score_dtype)

    def set_score_dtype(self, score_dtype):
        if score_dtype not in ("bfloat16","float32"):
            raise ValueError("A score_dtype 必须为 bfloat16 或 float32")
        # Promote the SAME loaded weights in RAM. Do not change the codec encoder.
        self.core.talker.to(dtype=getattr(self.torch,score_dtype))
        self.score_dtype=score_dtype

    def alternate_tolerance(self):
        return 1e-4 if self.score_dtype=="float32" else 0.05

    def encode(self, audio):
        with self.torch.inference_mode(), self.attention_context():
            codes = self.core.speech_tokenizer.encode(str(audio)).audio_codes[0]
        if codes.ndim != 2 or codes.shape[1] != 16 or codes.shape[0] == 0:
            raise ValueError("真实音频 tokenizer 输出应为非空 [T,16]")
        return codes.detach().long().cpu()

    def arrays(self, request, codes, pad_frames=0, pad_value=0.0):
        torch = self.torch
        talker = self.core.talker
        codes = torch.as_tensor(codes, dtype=torch.long, device=self.device)
        if codes.ndim != 2 or codes.shape[1] != 16 or codes.shape[0] == 0:
            raise ValueError("非法 codec 形状")
        with torch.inference_mode(), self.attention_context():
            prefix, pad = self._conditioning(request)
            frames = self._frame_embeddings(codes.unsqueeze(0), pad)
            inputs = torch.cat([prefix, frames], dim=1)
            mask = torch.ones(inputs.shape[:2], dtype=torch.long, device=self.device)
            if pad_frames:
                inputs = torch.cat([inputs, torch.full((1,pad_frames,inputs.shape[-1]), pad_value, device=self.device,dtype=inputs.dtype)],dim=1)
                mask = torch.cat([mask,torch.zeros((1,pad_frames),device=self.device,dtype=mask.dtype)],dim=1)
            # Use official talker.forward for its multimodal RoPE position construction.
            output = talker(inputs_embeds=inputs, attention_mask=mask, output_hidden_states=True, use_cache=False)
            start, length = prefix.shape[1]-1, codes.shape[0]
            logits = output.logits[:,start:start+length,:].float()
            main = torch.log_softmax(logits,dim=-1).gather(-1,codes[None,:,0,None]).squeeze(0).squeeze(-1)
            hidden = output.hidden_states[0][-1][:,start:start+length,:].squeeze(0)
            sub = []
            for offset in range(0,length,32):
                targets = codes[offset:offset+32]
                sub_logits,_ = talker.forward_sub_talker_finetune(targets,hidden[offset:offset+32])
                sub.append(torch.log_softmax(sub_logits.float(),dim=-1).gather(-1,targets[:,1:,None]).squeeze(-1))
            values = torch.cat([main[:,None],torch.cat(sub)],dim=1)
            if not torch.isfinite(values).all():
                raise ValueError("音频目标 log-probability 含 NaN/Inf")
            return values.cpu(), {"prefix_tokens": int(prefix.shape[1]), "codec_frames":int(length),
                                  "codebooks":16,"target_tokens":int(length*16),
                                  "audio_eos_scored":False,"raw_logits":True,"reference_audio":False,
                                  "score_dtype":getattr(self,"score_dtype","bfloat16")}

    def score(self, request, codes):
        values, detail = self.arrays(request,codes)
        return {**detail,"score":values.double().mean().item(),
                "per_codebook_mean":values.double().mean(0).tolist(),
                "main_mean":values[:,0].double().mean().item(),
                "sub_mean":values[:,1:].double().mean().item(),
                "legacy_weighted_score":values[:,0].double().mean().item()+0.3*values[:,1:].double().mean().item(),
                "token_logprobs":values.tolist()}

    def audit(self,request,codes):
        # Independent first-frame path: prefix-only main + sequential sub-codebook probabilities.
        torch=self.torch; talker=self.core.talker
        codes=torch.as_tensor(codes,dtype=torch.long,device=self.device)
        bulk,_=self.arrays(request,codes)
        repeat,_=self.arrays(request,codes)
        padded,_=self.arrays(request,codes,pad_frames=3)
        padded_changed,_=self.arrays(request,codes,pad_frames=3,pad_value=1.0)
        with torch.inference_mode(), self.attention_context():
            prefix,_=self._conditioning(request)
            mask=torch.ones(prefix.shape[:2],dtype=torch.long,device=self.device)
            output=talker(inputs_embeds=prefix,attention_mask=mask,output_hidden_states=True,use_cache=False)
            main=torch.log_softmax(output.logits[:,-1,:].float(),-1)[0,codes[0,0]]
            hidden=output.hidden_states[0][-1][:,-1:,:]
            predictor=talker.code_predictor
            embeddings=torch.cat([hidden,talker.get_input_embeddings()(codes[0:1,0:1])],dim=1)
            pred=predictor(inputs_embeds=embeddings,use_cache=True)
            seq=[]
            for j in range(15):
                target=codes[0,j+1]
                seq.append(torch.log_softmax(pred.logits[:,-1,:].float(),-1)[0,target])
                if j<14:
                    pred=predictor(input_ids=target.reshape(1,1),past_key_values=pred.past_key_values,
                        attention_mask=torch.ones((1,j+3),dtype=torch.long,device=self.device),
                        generation_steps=pred.generation_steps,use_cache=True)
            first=torch.stack([main,*seq]).cpu()
            parallel_logits,_=talker.forward_sub_talker_finetune(codes[:1],hidden[:,0,:])
            same_hidden_sub=torch.log_softmax(parallel_logits.float(),-1).gather(-1,codes[:1,1:,None]).reshape(-1).cpu()
        changed=codes.clone()
        boundary=max(1,len(codes)//2)
        if boundary<len(codes):
            changed[boundary:]=codes[0]
        perturbed,_=self.arrays(request,changed)
        alternate_tolerance=self.alternate_tolerance()
        result={"repeat_max_abs":(bulk-repeat).abs().max().item(),
                "padding_max_abs":(bulk-padded).abs().max().item(),
                "first_frame_sequential_max_abs":(bulk[0]-first).abs().max().item(),
                "future_prefix_max_abs":(bulk[:boundary]-perturbed[:boundary]).abs().max().item(),
                "masked_padding_content_max_abs":(padded-padded_changed).abs().max().item(),
                "first_frame_sub_same_hidden_max_abs":(same_hidden_sub-first[1:]).abs().max().item(),
                "repeat_tolerance":1e-5,"alternate_kernel_tolerance":alternate_tolerance,
                "codec_target_sha256":digest(codes.cpu().tolist()),
                "environment":self.numeric_environment(),
                "differences":{"padding":difference_summary(bulk.tolist(),padded.tolist()),
                               "first_frame_sequential":difference_summary(bulk[0].tolist(),first.tolist())},
                "target_logprobs":{"bulk":bulk.tolist(),"repeat":repeat.tolist(),"padded":padded.tolist(),
                    "padded_changed":padded_changed.tolist(),"first_frame_sequential":first.tolist(),
                    "first_frame_parallel_same_hidden_sub":same_hidden_sub.tolist(),
                    "future_changed_prefix":perturbed[:boundary].tolist()},
                "note":"Fixed SDPA math backend; BF16 alternate tolerance 0.05, FP32 1e-4. Same loaded weights; codec encoder not promoted."}
        result["passed"]=(result["repeat_max_abs"]<=1e-5 and all(result[k]<=alternate_tolerance for k in
                            ("padding_max_abs","first_frame_sequential_max_abs","future_prefix_max_abs",
                             "first_frame_sub_same_hidden_max_abs")) and result["masked_padding_content_max_abs"]<=1e-5)
        return result


class CaptionScorer(NumericalPolicy):
    def __init__(self,path,attention="sdpa",sdpa_backend="math"):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
        if not torch.cuda.is_available():
            raise RuntimeError("没有可用 GPU；请由用户显式指定 --gpu")
        self.torch=torch
        self.configure_numerics(sdpa_backend)
        self.model=AutoModelForCausalLM.from_pretrained(str(path),trust_remote_code=True,local_files_only=True,
            dtype=torch.bfloat16,device_map={"":"cuda:0"},attn_implementation=attention,low_cpu_mem_usage=True)
        self.model.requires_grad_(False)
        self.model.eval()
        self.processor=AutoProcessor.from_pretrained(str(path),trust_remote_code=True,local_files_only=True)
        self.tokenizer=AutoTokenizer.from_pretrained(str(path),trust_remote_code=True,local_files_only=True)
        self.device=next(self.model.parameters()).device
        self.dtype=next(self.model.parameters()).dtype

    def inputs(self,audio,completion=None):
        messages=[{"role":"user","content":[{"type":"text","text":PROMPT},{"type":"audio","path":str(audio)}]}]
        if completion is not None:
            messages.append({"role":"assistant","content":[{"type":"text","text":completion}]})
        values=self.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=completion is None,
            add_special_tokens=True,return_dict=True)
        return {k:(v.to(self.device,dtype=self.dtype) if v.is_floating_point() else v.to(self.device))
                if hasattr(v,"to") else v for k,v in values.items()}

    def prepare(self,audio,target):
        prompt=self.inputs(audio)
        full=self.inputs(audio,target)
        p=prompt["input_ids"][0].tolist(); ids=full["input_ids"][0].tolist()
        body=self.tokenizer.encode(target,add_special_tokens=False)
        # The end-of-turn template tail is obtained from an empty assistant message,
        # not guessed from token IDs nor included in the score.
        empty=self.inputs(audio,"")["input_ids"][0].tolist()
        if empty[:len(p)]!=p:
            raise ValueError("空 assistant 模板与提示前缀不同，需修正边界适配")
        start,end=completion_body_span(p,ids,body,empty[len(p):])
        if any(t in set(self.tokenizer.all_special_ids) for t in body):
            raise ValueError("目标正文含特殊 token")
        return full,start,end,body

    def values(self,inputs,start,end):
        with self.torch.inference_mode(), self.attention_context():
            logits=self.model(**inputs,use_cache=False).logits[:,start-1:end-1,:]
            targets=inputs["input_ids"][:,start:end]
            values=self.torch.log_softmax(logits.float(),-1).gather(-1,targets.unsqueeze(-1)).squeeze(0).squeeze(-1)
        if not self.torch.isfinite(values).all():
            raise ValueError("描述目标 log-probability 含 NaN/Inf")
        return values.cpu()

    def score(self,audio,target):
        inputs,start,end,body=self.prepare(audio,target)
        values=self.values(inputs,start,end)
        return {"score":values.double().mean().item(),"target":target,"target_tokens":len(body),
                "target_token_ids":body,"target_token_pieces":self.tokenizer.convert_ids_to_tokens(body),
                "token_logprobs":values.tolist(),"prefix_tokens":start,"target_end":end,
                "excluded_suffix_tokens":int(inputs["input_ids"].shape[1]-end),
                "prompt":PROMPT,"raw_logits":True,"target_in_user_prompt":False}

    def audit(self,audio,target):
        torch=self.torch
        inputs,start,end,body=self.prepare(audio,target)
        values=self.values(inputs,start,end)
        repeated=self.values(inputs,start,end)
        padded=dict(inputs)
        pad_id=self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id=self.tokenizer.eos_token_id
        padded["input_ids"]=torch.cat([inputs["input_ids"],torch.full((1,3),pad_id,device=self.device,dtype=inputs["input_ids"].dtype)],1)
        padded["attention_mask"]=torch.cat([inputs["attention_mask"],torch.zeros((1,3),device=self.device,dtype=inputs["attention_mask"].dtype)],1)
        pad_values=self.values(padded,start,end)
        changed_padding=dict(padded)
        changed_padding["input_ids"]=padded["input_ids"].clone()
        changed_padding["input_ids"][:,-3:]=next(t for t in body if t!=pad_id)
        changed_pad_values=self.values(changed_padding,start,end)
        # One independent forward per target token; only its strictly preceding text is supplied.
        sequential=[]
        for i in range(start,end):
            prefix=dict(inputs)
            prefix["input_ids"]=inputs["input_ids"][:,:i]
            prefix["attention_mask"]=inputs["attention_mask"][:,:i]
            with torch.inference_mode(), self.attention_context():
                logits=self.model(**prefix,use_cache=False).logits[:,-1,:]
                sequential.append(torch.log_softmax(logits.float(),-1)[0,inputs["input_ids"][0,i]].cpu())
        seq=torch.stack(sequential)
        result={"repeat_max_abs":(values-repeated).abs().max().item(),
                "padding_max_abs":(values-pad_values).abs().max().item(),
                "prefix_only_max_abs":(values-seq).abs().max().item(),
                "masked_padding_content_max_abs":(pad_values-changed_pad_values).abs().max().item(),
                "target_tokens":len(body),"repeat_tolerance":1e-5,"alternate_kernel_tolerance":0.05,
                "environment":self.numeric_environment(),
                "differences":{"padding":difference_summary(values.tolist(),pad_values.tolist()),
                               "prefix_only":difference_summary(values.tolist(),seq.tolist())},
                "target_logprobs":{"bulk":values.tolist(),"repeat":repeated.tolist(),"padded":pad_values.tolist(),
                    "padded_changed":changed_pad_values.tolist(),"prefix_only":seq.tolist()}}
        result["passed"]=(result["repeat_max_abs"]<=1e-5 and result["padding_max_abs"]<=0.05
                          and result["prefix_only_max_abs"]<=0.05 and result["masked_padding_content_max_abs"]<=1e-5)
        return result
