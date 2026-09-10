"""Download pinned reference inputs, or verify them through offline task loaders.

Run inside the evaluation image, on the login node. No model engine is started.
The default is a plan; --execute downloads, --offline verifies a prior manifest.
GPQA requires the collaborator's own approved Hugging Face access.
"""
import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path

MODEL = 'Qwen/Qwen3-1.7B-Base'
MODEL_REVISION = 'ea980cb0a6c2ae4b936e82123acc929f1cec04c1'
WEIGHTS_SHA = '6df85b39330e5a425ee36253d0f894e4387e4f0a15b9c53cb467d668e6b3a841'
DATA = {
    'piqa': {'repo':'baber/piqa', 'revision':'142f6d7367fd9877f0fb3b5734ea6a545f54cdd1',
             'subset':None, 'counts':{'train':16113, 'validation':1838}},
    'gsm8k': {'repo':'openai/gsm8k', 'revision':'740312add88f781978c0658806c59bc2815b9866',
              'subset':'main', 'counts':{'train':7473, 'test':1319}},
    'GPQADiamond': {'repo':'Idavidrein/gpqa', 'revision':'633f5ee89ab8ad4522a9f850766b73f62147ffdd',
                    'subset':'gpqa_diamond', 'counts':{'train':198}},
}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def row_hash(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, ensure_ascii=True, separators=(',', ':')).encode())
        digest.update(b'\n')
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--hf-home', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--tasks', nargs='+', choices=['piqa','gsm8k','MATH500','GPQADiamond'],
                        default=['piqa','gsm8k','MATH500'])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--offline', action='store_true')
    parser.add_argument('--ask-hf-token', action='store_true', help='Read token without echo; never save it')
    args = parser.parse_args()
    if args.ask_hf_token and (not args.execute or 'GPQADiamond' not in args.tasks):
        parser.error('--ask-hf-token is only for online GPQA preparation')
    home, model = args.hf_home.resolve(), args.model_dir.resolve()
    plan = {'model': MODEL, 'model_revision': MODEL_REVISION, 'model_dir': str(model),
            'hf_home': str(home), 'tasks': args.tasks,
            'datasets': {k:v for k,v in DATA.items() if k in args.tasks}}
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute and not args.offline:
        return
    if args.execute and args.manifest.exists():
        parser.error('Manifest exists; preserve it and select a new manifest filename')
    # Set paths before importing HF libraries; the batch script uses the same paths.
    os.environ.update(HF_HOME=str(home), HF_HUB_CACHE=str(home/'hub'),
                      HF_DATASETS_CACHE=str(home/'datasets'),
                      HF_HUB_OFFLINE='1' if args.offline else '0',
                      HF_DATASETS_OFFLINE='1' if args.offline else '0',
                      HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer
    token = getpass.getpass('Hugging Face read token (not saved): ') if args.ask_hf_token else False
    if args.execute:
        snapshot_download(MODEL, revision=MODEL_REVISION, local_dir=model, token=False)
    if sha(model/'model.safetensors') != WEIGHTS_SHA:
        raise ValueError('Reference model weights differ from the pinned Qwen revision')
    config = AutoConfig.from_pretrained(model, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    # Check prompts through the same adapters without constructing a GPU engine.
    from lm_eval.models.huggingface import HFLM
    from lm_eval.models.vllm_causallms import VLLM
    hf, vl = HFLM.__new__(HFLM), VLLM.__new__(VLLM)
    hf.tokenizer = vl.tokenizer = tokenizer
    hf.chat_template_args = vl.chat_template_args = {}
    vl.hf_chat_template = tokenizer.chat_template
    vl.enable_thinking = None
    messages = [{'role':'user','content':'Problem: 2 + 2\nAnswer:'}]
    prompt = hf.apply_chat_template(messages)
    assert prompt == vl.apply_chat_template(messages)
    report = dict(plan, weights_sha256=WEIGHTS_SHA, model_type=config.model_type,
                  prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), datasets={},
                  model_files={str(p.relative_to(model)):(WEIGHTS_SHA if p.name == 'model.safetensors' else sha(p))
                               for p in sorted(model.rglob('*')) if p.is_file() and '.cache' not in p.relative_to(model).parts})
    for task in args.tasks:
        if task == 'MATH500':
            from eval.task import TaskManager
            rows = TaskManager(task_list=['MATH500']).get_benchmark('MATH500').load_questions()
            assert len(rows) == 500
            report['datasets'][task] = {'count':500, 'rows_sha256':row_hash(rows)}
            continue
        item = DATA[task]
        options = {'cache_dir':str(home/'hub')} if task == 'GPQADiamond' else {}
        if args.execute:
            options.update(revision=item['revision'], token=token if task == 'GPQADiamond' else False)
        else:
            options['token'] = False
        # Offline verification deliberately omits revision: benchmark code does too.
        dataset = load_dataset(item['repo'], item['subset'], **options)
        for split, count in item['counts'].items():
            if len(dataset[split]) != count:
                raise ValueError(f'Incomplete {task}/{split}')
        report['datasets'][task] = {'revision':item['revision'], 'splits':{
            split:{'count':len(rows), 'rows_sha256':row_hash(rows)} for split,rows in dataset.items()}}
    if args.offline:
        expected = json.loads(args.manifest.read_text())
        if report != expected:
            raise ValueError('Offline task/model inputs differ from the prepared manifest')
        print('Offline model, prompt and complete dataset checks passed.')
    else:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(report,indent=2)+'\n')
        print('Prepared inputs; run --offline next to verify actual task cache lookup.')


if __name__ == '__main__':
    main()
