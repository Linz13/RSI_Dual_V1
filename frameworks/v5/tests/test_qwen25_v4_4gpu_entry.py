import copy
import json
from pathlib import Path
import pytest
from dual_isl_train.config import load_config,ENV_OVERRIDES,INTEGER_ENV_OVERRIDES
from scripts.qwen25_reward_v4_4gpu_run_guard import run_contract,check_run_directory,MARKER

@pytest.fixture
def cfg(monkeypatch):
    for name in (*ENV_OVERRIDES,*INTEGER_ENV_OVERRIDES): monkeypatch.delenv(name,raising=False)
    return load_config('configs/train_4gpu_h100_qwen2_5_omni_3b_reward_v4.yaml')

def test_qwen_config_identity_and_shared_v4_rules(cfg):
    contract=run_contract(cfg,'train',10)
    assert cfg['captioner']['worker_module']=='scripts.qwen2_5_omni_3b_captioner_candidate'
    assert cfg['captioner']['lora']['r']==8
    assert cfg['captioner']['attn_implementation']=='flash_attention_2'
    assert cfg['reward']['cycle_sft_selection']=='semantic_top1'
    assert cfg['reward']['sft_gate']['diagnostic_only'] is True
    assert cfg['reward']['sft_gate']['target_precision']==.9
    assert 'scripts/qwen2_5_omni_3b_captioner_candidate.py' in contract['code_sha256']
    assert 'scripts/run_qwen25_v4_10rounds_4gpu.sh' in contract['code_sha256']
    assert 'scripts/midasheng_captioner_candidate.py' not in contract['code_sha256']

@pytest.mark.parametrize('section,key,value',[
    ('captioner','adapter_path','/old/checkpoint'),('tts','adapter_path','/old/checkpoint'),
    ('training','round_offset',9),('training','rounds',9),('data','max_records_per_role',8),
    ('captioner','worker_module','scripts.midasheng_captioner_candidate')])
def test_incompatible_config_rejected(cfg,section,key,value):
    cfg[section][key]=value
    with pytest.raises(ValueError):run_contract(cfg,'train',10)

def test_disabled_optimizations_rejected(cfg):
    cfg['tts']['generation']['rollout_schedule']='round_robin'
    with pytest.raises(ValueError,match='optimizations'):run_contract(cfg,'train',10)

def test_existing_run_identity_and_completion(tmp_path,cfg):
    contract=run_contract(cfg,'train',10)
    root=tmp_path/'runs/qwen';root.mkdir(parents=True)
    (root/MARKER).write_text(json.dumps({'model':'midasheng'}))
    with pytest.raises(ValueError,match='identity changed'):check_run_directory(root,contract,project=tmp_path)
    (root/MARKER).write_text(json.dumps(contract));(root/'run_state.json').write_text('{}')
    assert check_run_directory(root,contract,project=tmp_path)=='resume'
    (root/'latest.json').write_text(json.dumps({'round':9}))
    with pytest.raises(ValueError,match='complete'):check_run_directory(root,contract,project=tmp_path)


def test_four_gpu_contract_rejects_eight_gpu_resume(cfg):
    assert cfg['distributed']['world_size'] == 4
    cfg['distributed']['world_size'] = 8
    with pytest.raises(ValueError, match='4-rank'):
        run_contract(cfg, 'train', 10)


def test_four_gpu_only_changes_run_and_world_size(cfg):
    from dual_isl_train.config import public_config
    old = public_config(load_config('configs/train_8gpu_h100_qwen2_5_omni_3b_reward_v4.yaml'))
    new = public_config(cfg)
    old['run']['output_dir'] = new['run']['output_dir']
    old['distributed']['world_size'] = 4
    assert old == new
