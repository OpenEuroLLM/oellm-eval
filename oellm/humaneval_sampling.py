"""Versioned HumanEval generation and sampled pass@1; preserves every answer.

Generation uses the validated native vLLM multiprocessing executor. Grading is
a separate CPU command so that generated code never executes in a GPU driver.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    os.replace(tmp,path)


def write_jsonl(path, rows):
    with Path(path).open('x') as stream:
        for row in rows:stream.write(json.dumps(row,ensure_ascii=False)+'\n')


def validate_samples(rows, problem_ids, samples):
    """A missing/duplicate sample is an infrastructure failure, never a failure grade."""
    if any(type(r['completion_id']) is not int for r in rows):raise ValueError('Sample IDs must be integers')
    expected={(task,i) for task in problem_ids for i in range(samples)}
    actual=[(r['task_id'],r['completion_id']) for r in rows]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:
        raise ValueError('Incomplete, duplicated or unexpected HumanEval samples')


def summarize(rows, problem_ids, samples):
    validate_samples(rows,problem_ids,samples)
    groups=defaultdict(list)
    for row in rows:
        if type(row['passed']) is not bool:raise ValueError('Verdicts must be booleans')
        groups[row['task_id']].append(row['passed'])
    per_problem={task:sum(groups[task])/samples for task in sorted(problem_ids)}
    mean=statistics.mean(per_problem.values())
    return dict(metric='pass@1',aggregation=f'avg@{samples}',value=mean,
        problems=len(problem_ids),samples_per_problem=samples,responses=len(rows),
        passed=sum(r['passed'] for r in rows),per_problem=per_problem,
        problem_standard_error=statistics.stdev(per_problem.values())/math.sqrt(len(per_problem))
            if len(per_problem)>1 else None,
        note='Mean correctness across all samples per problem, then across problems; no best-of selection.')


def load_case(path):
    case=json.loads(Path(path).read_text())
    if case['schema']!=1 or case['samples'] not in (1,32):raise ValueError('Unsupported sampling protocol')
    if (case['temperature']==0)!=(case['samples']==1):
        raise ValueError('Greedy requires one sample; avg@32 requires positive temperature')
    if not 0<=case['temperature']<=2 or not 0<case['top_p']<=1:raise ValueError('Invalid sampling settings')
    if case['prompt'] not in ('evalchemy','completion'):raise ValueError('Unknown prompt protocol')
    if case['parser'] not in ('evalchemy','evalplus','nvidia'):raise ValueError('Unknown parser protocol')
    if case['context_length']<=case['max_new_tokens'] or case['max_new_tokens']<1:raise ValueError('Invalid budget')
    if case['dp']<1 or case['tp']<1:raise ValueError('Invalid GPU layout')
    for path,expected in case['input_sha256'].items():
        if sha(path)!=expected:raise ValueError('Changed input: '+path)
    return case


def problems_for(case):
    rows=[]
    for language,path in case['problem_files'].items():
        examples=read_jsonl(path)
        expected={'python':164,'sh':158}[language]
        if len(examples)!=expected or len({r['task_id'] for r in examples})!=expected:
            raise ValueError('Incomplete standard task: '+language)
        rows.extend(dict(r,language=language) for r in examples)
    return rows


def build_prompt(problem, policy):
    if policy=='completion':return problem['prompt']
    language={'python':'Python','sh':'Bash'}[problem['language']]
    return ('Please continue to complete the function. You are not allowed to modify the given code and do the completion only. '
        'Please return all completed function in a codeblock. Here is the given code to do completion:\n'
        f'```{language.lower()}\n{problem["prompt"].strip()}\n```')


def request_seed(base, task_id):
    return (base+int.from_bytes(hashlib.sha256(task_id.encode()).digest()[:4],'little'))%(2**31)


def sampling_parameters(case, task_id):
    return dict(n=case['samples'],temperature=case['temperature'],top_p=case['top_p'],top_k=0,
        min_p=0.0,seed=request_seed(case['seed'],task_id),max_tokens=case['max_new_tokens'],
        repetition_penalty=1.0,presence_penalty=0.0,frequency_penalty=0.0,
        stop=case.get('stop',[]),ignore_eos=False)


def preflight(case):
    from transformers import AutoConfig,AutoTokenizer
    config=AutoConfig.from_pretrained(case['model'],trust_remote_code=True,local_files_only=True)
    tokenizer=AutoTokenizer.from_pretrained(case['model'],trust_remote_code=True,local_files_only=True)
    lengths=[len(tokenizer.encode(build_prompt(r,case['prompt']),add_special_tokens=False)) for r in problems_for(case)]
    if max(lengths)+case['max_new_tokens']>case['context_length']:raise ValueError('Prompt plus output exceeds context')
    save(Path(case['output']).parent/'preflight.json',dict(model_type=config.model_type,
        architectures=config.architectures,problems=len(lengths),max_prompt_tokens=max(lengths),
        context_length=case['context_length'],max_new_tokens=case['max_new_tokens'],status='ready'))


def sampling_worker(case, unused, requests, unused_lora, queue, dp, rank, devices):
    """Run one independent replica; preserve all n outputs and actual sampling evidence."""
    import traceback
    try:
        os.environ['CUDA_VISIBLE_DEVICES']=','.join(devices)
        os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
        for key in ('VLLM_DP_RANK','VLLM_DP_RANK_LOCAL','VLLM_DP_SIZE','VLLM_DP_MASTER_IP','VLLM_DP_MASTER_PORT'):
            os.environ.pop(key,None)
        from vllm import LLM,SamplingParams,TokensPrompt
        trace=[]
        original=SamplingParams.update_from_generation_config
        def checked(self,*args,**kwargs):
            result=original(self,*args,**kwargs)
            actual={key:getattr(self,key) for key in ('n','temperature','top_p','top_k','min_p','seed','max_tokens',
                'repetition_penalty','presence_penalty','frequency_penalty','ignore_eos','stop')}
            expected=sampling_parameters(case,requests[0]['task_id'])
            for key in expected:
                if key!='seed' and actual[key]!=expected[key]:raise ValueError('Engine changed sampling parameter '+key)
            if actual['seed'] not in {request_seed(case['seed'],r['task_id']) for r in requests}:
                raise ValueError('Engine changed sampling seed')
            actual['sampling_type']=self.sampling_type.name
            if (actual['sampling_type']=='GREEDY')!=(case['temperature']==0):raise ValueError('Wrong sampling type')
            trace.append(actual)
            return result
        SamplingParams.update_from_generation_config=checked
        engine=LLM(model=case['model'],dtype='bfloat16',tensor_parallel_size=case['tp'],
            distributed_executor_backend='mp' if case['tp']>1 else 'uni',
            max_model_len=case['context_length'],max_num_seqs=case.get('max_num_seqs',32),
            max_num_batched_tokens=case.get('max_num_batched_tokens',4096),
            gpu_memory_utilization=case.get('gpu_memory_utilization',.9),
            seed=case['seed'],trust_remote_code=True,enable_prefix_caching=True)
        params=[SamplingParams(**sampling_parameters(case,r['task_id'])) for r in requests]
        outputs=engine.generate([TokensPrompt(prompt_token_ids=r['prompt_token_ids']) for r in requests],params)
        if len(outputs)!=len(requests) or len(trace)!=len(requests):raise ValueError('Incomplete engine output/trace')
        rows=[]
        for request,output in zip(requests,outputs,strict=True):
            if list(output.prompt_token_ids)!=request['prompt_token_ids']:raise ValueError('Prompt truncation/reordering')
            if len(output.outputs)!=case['samples']:raise ValueError('Wrong number of engine samples')
            if {o.index for o in output.outputs}!=set(range(case['samples'])):raise ValueError('Sample IDs differ')
            for completion in output.outputs:
                rows.append(dict(task_id=request['task_id'],completion_id=completion.index,language=request['language'],
                    output=completion.text,token_ids=list(completion.token_ids),finish_reason=completion.finish_reason,
                    stop_reason=completion.stop_reason,worker=rank,seed=request_seed(case['seed'],request['task_id'])))
        destination=Path(case['output'])
        write_jsonl(destination/f'worker-{rank}.jsonl',rows)
        save(destination/f'worker-{rank}-sampling.json',dict(devices=devices,rank=rank,requests=len(requests),parameters=trace))
        queue.put((rank,dict(rows=len(rows),requests=len(requests))))
    except BaseException:queue.put((rank,dict(error=traceback.format_exc())))


def generate(case):
    from transformers import AutoTokenizer
    from lm_eval.models.vllm_dp import gpu_groups,run_workers
    destination=Path(case['output']);destination.mkdir(parents=True,exist_ok=False)
    tokenizer=AutoTokenizer.from_pretrained(case['model'],trust_remote_code=True,local_files_only=True)
    rows=problems_for(case);requests=[]
    for row in rows:
        prompt=build_prompt(row,case['prompt'])
        tokens=tokenizer.encode(prompt,add_special_tokens=False)
        if len(tokens)+case['max_new_tokens']>case['context_length']:raise ValueError('Full prompt/output do not fit')
        requests.append(dict(task_id=row['task_id'],language=row['language'],prompt=prompt,
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),prompt_token_ids=tokens))
    write_jsonl(destination/'requests.jsonl',requests)
    devices=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    groups=gpu_groups(devices,case['dp'],case['tp'])
    work=[(case,None,requests[rank::case['dp']],None,case['dp'],rank,devices) for rank,devices in enumerate(groups)]
    run_workers(sampling_worker,work)
    outputs=[r for rank in range(case['dp']) for r in read_jsonl(destination/f'worker-{rank}.jsonl')]
    validate_samples(outputs,{r['task_id'] for r in rows},case['samples'])
    outputs.sort(key=lambda r:(r['task_id'],r['completion_id']))
    write_jsonl(destination/'generations.jsonl',outputs)
    import importlib.metadata
    versions={name:importlib.metadata.version(name) for name in ('vllm','torch','transformers')}
    save(destination/'generation-complete.json',dict(case=case,responses=len(outputs),versions=versions,
        raw_sha256=sha(destination/'generations.jsonl'),requests_sha256=sha(destination/'requests.jsonl'),
        tokenizer_sha256=sha(Path(case['model'])/'tokenizer.json')))


def grade(case):
    """Use the installed audited multilingual execution runtime, on CPU only."""
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None,'','-1'):
        raise ValueError('Grade in the CPU-only contained stage')
    root=Path(case['output']);receipt=json.loads((root/'generation-complete.json').read_text())
    for key in ('samples','prompt','temperature','top_p','seed','max_new_tokens','context_length','problem_files','model_revision'):
        if case[key]!=receipt['case'][key]:raise ValueError('Grading case differs from generation: '+key)
    raw=root/'generations.jsonl'
    if sha(raw)!=receipt['raw_sha256']:raise ValueError('Saved generations changed')
    output=read_jsonl(raw);problems=problems_for(case);by_id={r['task_id']:r for r in problems}
    validate_samples(output,set(by_id),case['samples'])
    grading=root/('grading-'+case['parser']);grading.mkdir(exist_ok=False)
    benchmark=Path(case['evalchemy_benchmark'])
    sys.path.insert(0,str(benchmark))
    from human_eval.evaluation import evaluate_functional_correctness
    import human_eval.evaluation as scorer
    if Path(scorer.__file__).resolve()!=benchmark/'human_eval/evaluation.py':raise ValueError('Unexpected scorer')
    if case.get('test_protocol')=='openai':
        # NVIDIA code_eval executes precisely the sanitized completion and original
        # tests; Evalchemy additionally injects helper imports. Keep those distinct.
        def original_test(sample, problems, *args, **kwargs):
            problem=problems[sample['task_id']]
            return sample['generation']+'\n'+problem['test']+'\ncheck('+problem['entry_point']+')\n'
        scorer.process_humaneval_test=original_test
    spec=importlib.util.spec_from_file_location('human_extract',benchmark/'utils/utils.py')
    extractor=importlib.util.module_from_spec(spec);spec.loader.exec_module(extractor)
    verdicts=[]
    for language,problem_file in case['problem_files'].items():
        samples=[]
        for row in output:
            if row['language']!=language:continue
            sample=dict(by_id[row['task_id']],output=row['output'],completion_id=row['completion_id'])
            if case['parser']=='evalchemy' or language!='python':
                sample=extractor.extract_generation_code(sample,lang_code=language)
            else:
                if case['parser']=='nvidia':
                    spec=importlib.util.spec_from_file_location('nvidia_sanitize',case['nvidia_sanitizer'])
                    sanitizer=importlib.util.module_from_spec(spec);spec.loader.exec_module(sanitizer)
                    sanitize=sanitizer.sanitize
                else:
                    from evalplus.sanitize import sanitize
                import re
                target=sample.get('entry_point') or re.findall(r'^def\s+(\w+)\(',sample['prompt'],re.M)[-1]
                code=sample['prompt']+row['output'] if case['prompt']=='completion' else row['output']
                sample['generation']=sanitize(code,entrypoint=target)
            samples.append(sample)
        generated=grading/f'generated-{language}.jsonl';write_jsonl(generated,samples)
        native=evaluate_functional_correctness(input_file=str(generated),tmp_dir=str(grading),
            problem_file=problem_file,language=language,n_workers=case.get('grading_workers',8),
            timeout=case.get('test_timeout',3.0),k=[1])
        results=read_jsonl(str(generated)+'.graded.jsonl')
        ids={r['task_id'] for r in problems if r['language']==language}
        summary=summarize(results,ids,case['samples'])
        if abs(summary['value']-native['pass@1'])>1e-12:raise ValueError('Native pass@1 disagrees with explicit average')
        save(grading/f'{language}-score.json',summary)
        verdicts.append(dict(language=language,**summary))
    save(grading/'complete.json',dict(scores=verdicts,raw_sha256=sha(raw),case=case))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['preflight','generate','grade'])
    parser.add_argument('--case',type=Path,required=True)
    args=parser.parse_args();case=load_case(args.case)
    {'preflight':preflight,'generate':generate,'grade':grade}[args.action](case)


if __name__=='__main__':main()
