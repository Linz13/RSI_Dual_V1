"""CPU tensors only: exercise numeric policy and adapter contexts; never load model weights."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from model_interfaces import NumericalPolicy, CaptionScorer, QwenScorer

assert not torch.cuda.is_initialized()
torch.set_num_threads(1)
policy=NumericalPolicy()
policy.torch=torch
policy.configure_numerics('math')
assert not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
assert not torch.backends.cuda.matmul.allow_tf32
assert not torch.backends.cudnn.allow_tf32
assert not torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()

def check_context():
    assert torch.backends.cuda.math_sdp_enabled()
    assert not torch.backends.cuda.flash_sdp_enabled()
    assert not torch.backends.cuda.mem_efficient_sdp_enabled()
    assert not torch.backends.cuda.cudnn_sdp_enabled()

before=(torch.backends.cuda.flash_sdp_enabled(),torch.backends.cuda.mem_efficient_sdp_enabled())
with policy.attention_context():
    check_context()
    torch.manual_seed(0)
    q=torch.randn(1,2,7,8,dtype=torch.bfloat16)
    k=torch.randn_like(q);v=torch.randn_like(q)
    reference=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True)
    qpad=torch.cat([q,torch.ones(1,2,3,8,dtype=q.dtype)],2)
    kpad=torch.cat([k,torch.ones(1,2,3,8,dtype=k.dtype)],2)
    vpad=torch.cat([v,torch.ones(1,2,3,8,dtype=v.dtype)],2)
    mask=torch.ones(10,10,dtype=torch.bool).tril();mask[:,7:]=False
    padded=torch.nn.functional.scaled_dot_product_attention(qpad,kpad,vpad,attn_mask=mask)
    error=(reference.float()-padded[:,:,:7].float()).abs().max().item()
    assert error<=.05,error
assert before==(torch.backends.cuda.flash_sdp_enabled(),torch.backends.cuda.mem_efficient_sdp_enabled())

# Constructors are deliberately bypassed; these are constant tensor fixtures.
# Actual adapter methods must enter the pinned context for every forward path.
class FakeCaption:
    def __call__(self,**kwargs):
        check_context()
        ids=kwargs['input_ids']
        return SimpleNamespace(logits=torch.zeros(1,ids.shape[1],8))
c=CaptionScorer.__new__(CaptionScorer)
c.torch=torch;c.device=torch.device('cpu');c.sdpa_backend='math';c.model=FakeCaption()
c.tokenizer=SimpleNamespace(pad_token_id=0,eos_token_id=7)
c.prepare=lambda audio,target:({'input_ids':torch.tensor([[1,2,3,4,7]]),'attention_mask':torch.ones(1,5,dtype=torch.long)},2,4,[3,4])
c.numeric_environment=lambda:{'fixture_only':True}
caption_audit=c.audit(None,None)
assert caption_audit['passed']
assert len(caption_audit['target_logprobs']['bulk'])==2

class FakeTalker:
    def __call__(self,inputs_embeds,**kwargs):
        check_context()
        n=inputs_embeds.shape[1]
        return SimpleNamespace(logits=torch.zeros(1,n,8),hidden_states=((torch.zeros(1,n,4),),None))
    def forward_sub_talker_finetune(self,targets,hidden):
        check_context()
        return torch.zeros(targets.shape[0],15,8),None
qsc=QwenScorer.__new__(QwenScorer)
qsc.torch=torch;qsc.device=torch.device('cpu');qsc.sdpa_backend='math'
qsc.core=SimpleNamespace(talker=FakeTalker())
qsc._conditioning=lambda request:(torch.zeros(1,3,4),torch.zeros(1,1,4))
qsc._frame_embeddings=lambda codes,pad:torch.zeros(1,codes.shape[1],4)
values,detail=qsc.arrays({},torch.ones(2,16,dtype=torch.long),pad_frames=3,pad_value=1)
assert values.shape==(2,16) and detail['target_tokens']==32
# Tiny CPU state fixture, no pretrained model. Promotion preserves loaded values
# and leaves the independent audio encoder's dtype/values alone.
tiny_talker=torch.nn.Linear(4,4).to(dtype=torch.bfloat16)
tiny_encoder=torch.nn.Linear(4,4).to(dtype=torch.bfloat16)
before_weights=[p.detach().float().clone() for p in tiny_talker.parameters()]
encoder_weights=[p.detach().clone() for p in tiny_encoder.parameters()]
qsc.core=SimpleNamespace(talker=tiny_talker,speech_tokenizer=SimpleNamespace(model=tiny_encoder))
qsc.set_score_dtype('float32')
assert qsc.alternate_tolerance()==1e-4
for p,before in zip(tiny_talker.parameters(),before_weights):
    assert p.dtype==torch.float32 and torch.equal(p,before)
for p,before in zip(tiny_encoder.parameters(),encoder_weights):
    assert p.dtype==torch.bfloat16 and torch.equal(p,before)
assert not torch.cuda.is_initialized()
print(json.dumps({'torch':torch.__version__,'cuda_initialized':torch.cuda.is_initialized(),
                  'models_loaded':False,'fixture_only':True,'cpu_attention_padding_max_abs':error,
                  'caption_adapter_audit_context_passed':caption_audit['passed'],
                  'tts_adapter_arrays_context_passed':True,'backend_context_restored':True,
                  'fp32_promotion_preserves_weights_and_encoder':True,'fp32_alternate_tolerance':qsc.alternate_tolerance()}))
