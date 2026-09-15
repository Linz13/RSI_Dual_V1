"""One model load, independent per-audio results, direct StepAudio Transformers backend."""
import argparse
from pathlib import Path
import json
import os
import traceback
import time
import torch

from Experiment.labeling2.stepaudio_hf import StepAudioHF


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    tasks = [json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()]
    config = json.loads(args.config.read_text())
    item = config['backends']['step_audio_r1_1']
    source = Path(os.environ['AUDIO_CAPTION_ROOT'])
    print('BACKEND Transformers; physical GPU:',os.environ.get('CUDA_VISIBLE_DEVICES'),flush=True)
    print('torch:',torch.__version__,'CUDA:',torch.version.cuda,flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        model = StepAudioHF(item['model_dir'], source/'lzy/Step-Audio-R1/chat_template_step_audio_r1_1.jinja')
    except Exception as exc:
        traceback.print_exc()
        with args.output.open('w') as out:
            for task in tasks:
                out.write(json.dumps({'sample_id':task['sample_id'],'model_name':'step_audio_r1_1',
                                      'backend':'transformers','status':'error','error':repr(exc),'response_text':''})+'\n')
        return 1
    failed = False
    with args.output.open('w') as out:
        for index, task in enumerate(tasks):
            started = time.monotonic()
            row = dict(sample_id=task['sample_id'],audio_path=task['audio_path'],model_name='step_audio_r1_1',
                       backend='transformers',status='error',response_text='',error='')
            print('START SAMPLE', task['sample_id'],flush=True)
            try:
                result = model.infer(task['audio_path'], task['prompt'],
                                     max_new_tokens=item.get('max_new_tokens',config['run'].get('max_new_tokens',1024)),
                                     temperature=item.get('temperature',0.7),top_p=item.get('top_p',0.9),seed=index)
                row.update(result)
                if not row['response_text'].strip():
                    raise ValueError('Empty text response')
                row['status']='success'
            except Exception as exc:
                traceback.print_exc()
                failed = True
                row['error']=repr(exc)
                torch.cuda.empty_cache()
            row['duration_sec']=round(time.monotonic()-started,3)
            out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush()
            print('END SAMPLE',task['sample_id'],row['status'],row['duration_sec'],flush=True)
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
