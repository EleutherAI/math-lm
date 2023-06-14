import os
import json
import random
import re
import datasets
import tiktoken
import numpy as np
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm


def make_split(examples, split_ratio):
    split2_len = max(int(split_ratio * len(examples)), 1)
    perm = np.random.permutation(len(examples))
    examples = [examples[i] for i in perm]
    split1 = examples[:len(examples)-split2_len]
    split2 = examples[len(examples)-split2_len:]
    return split1, split2


def _skip(args, dataset, data_dir):
    if args.overwrite:
        return False
    if Path(os.path.join(data_dir, 'train', 'mathdatasets-%s-train-0000.jsonl' % dataset)).exists():
        print("Skipping %s ; use --overwrite to not skip." % dataset)
        return True
    return False


def _save_splits(splits, out_dir, dataset, shard_size=1000000):
    print("Saving split to disk...")
    for split, examples in tqdm(splits.items(), total=len(splits)):
        out_dir_ = os.path.join(out_dir, split)
        Path(out_dir_).mkdir(parents=True, exist_ok=True)
        for i in range(0, len(examples), shard_size):
            num_digits = max(len(str(len(examples)//shard_size+1)), 4)
            out_file = os.path.join(
                out_dir_, 'mathdatasets-%s-%s-%s.jsonl' % (dataset, split, str(i).zfill(num_digits))
            )
            shard = examples[i*shard_size:(i+1)*shard_size]
            with open(out_file, 'w') as f:
                for example in shard:
                    f.write(json.dumps(example))
                    f.write('\n')


def _save_stats(stats_of_lang, lang, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    f_out = '%s/mathdatasets-stats.json' % output_dir
    if os.path.isfile(f_out):
        with open(f_out) as f:
            stats = json.load(f)
    else:
        stats = dict()
    stats[lang] = stats_of_lang
    with open(f_out, 'w') as f:
        json.dump(stats, f, indent=2)


def token_length(examples):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = [
        len(x) for x in tokenizer.encode_batch(
            [x['text'] for x in examples], disallowed_special=()
        )
    ]
    return sum(tokens)


# ---- LILA
def _format_lila(example, args):
    text = """### Problem
%s

### Program:
```python
%s
```
The answer is: \\boxed{%s}.

""" % (
    example['input'],
    example['output_program'],
    example['output_answer']
)
    return text


def lila(args, dataset_name='lila'):
    if _skip(args, dataset_name, args.data_dir):
        return
    splits = defaultdict(list)
    LILA_TASKS = [
        'addsub',
        'amps_algebra',
        'amps_calculus',
        'amps_counting_and_stats',
        'amps_geometry',
        'amps_linear_algebra',
        'amps_number_theory',
        'asdiv',
        'deepmind_mathematics_algebra',
        'deepmind_mathematics_basicmath',
        'deepmind_mathematics_calculus',
        'deepmind_mathematics_muldiv',
        'deepmind_mathematics_numbertheory',
        'draw_structured',
        'GSM8k_structured',
        'MATH_algebra_crowdsourced',
        'MATH_counting_and_probability_crowdsourced',
        'MATH_intermediate_algebra_crowdsourced',
        'mathqa_gain',
        'mathqa_general',
        'mathqa_geometry',
        'mathqa_other',
        'mathqa_physics',
        'mathqa_probability',
        'mbpp_structured',
        'MCTaco_event_duration_structured',
        'MCTaco_event_ordering_structured',
        'MCTaco_event_typical_time_structured',
        'MCTaco_frequency_structured',
        'MCTaco_stationarity_structured',
        'multiarith',
        'Numersense_structured',
        'NumGLUE_Type_1_crowdsourced',
        'NumGLUE_Type_2_crowdsourced',
        'NumGLUE_Type_3_crowdsourced',
        'NumGLUE_Type_4_crowdsourced',
        'NumGLUE_Type_5_crowdsourced',
        'NumGLUE_Type_6_crowdsourced',
        'NumGLUE_Type_7_crowdsourced',
        'NumGLUE_Type_8_crowdsourced',
        'simuleq',
        'singleop',
        'singleq',
        'svamp_structured'
    ]
    split_names = {
        'train': 'train',
        'validation': 'validation',
        'test': 'test'
    }
    for task in LILA_TASKS:
        ds = datasets.load_dataset(
            'allenai/lila', task
        )
        for split in ds:
            split_ = split_names[split]
            for i in range(len(ds[split_])):
                example = ds[split_][i]
                text = _format_lila(example, args)
                splits[split_].append({
                    'text': text,
                    'meta': {'dataset': example['dataset']}
                })

    _save_splits(
        splits,
        out_dir=args.data_dir,
        dataset=dataset_name
    )

    num_tokens = token_length(
        splits['train'] + splits['validation'] + splits['test']
    )
    _save_stats({
        'num_train': len(splits['train']),
        'num_validation': len(splits['validation']),
        'num_test': len(splits['test']),
        'num_tokens': num_tokens,
    }, dataset_name, args.meta_dir)
# ----


# ---- MATH
def _format_math(example, args):
    text = """### Problem
%s

### Solution:
```
%s
```
""" % (
    example['problem'],
    example['solution'],
)
    return text


def math_dataset(args, dataset_name='MATH'):
    if _skip(args, dataset_name, args.data_dir):
        return
    splits = defaultdict(list)

    ds = datasets.load_dataset(
        'competition_math'
    )
    for split in ds:
        for i in range(len(ds[split])):
            example = ds[split][i]
            text = _format_math(example, args)
            splits[split].append({
                'text': text,
                'meta': {
                    'level': example['level'],
                    'type': example['type']
                }
            })

    # MATH only has train and test; create a small validation set from train.
    train, validation = make_split(splits['train'], split_ratio=args.eval_ratio)
    splits = {
        'train': train,
        'validation': validation,
        'test': splits['test']
    }

    _save_splits(
        splits,
        out_dir=args.data_dir,
        dataset=dataset_name
    )

    num_tokens = token_length(
        splits['train'] + splits['validation'] + splits['test']
    )
    _save_stats({
        'num_train': len(splits['train']),
        'num_validation': len(splits['validation']),
        'num_test': len(splits['test']),
        'num_tokens': num_tokens,
    }, dataset_name, args.meta_dir)
# -------


# ------- GSM8k
def _format_gsm8k(example, args):
    def remove_brackets(text):
        return re.sub(r'<<.+?>>', '', text)

    chunks = example['answer'].split('####')
    solution = remove_brackets(chunks[0].strip())
    answer = chunks[1].strip()
    text = """### Problem
%s

### Solution:
```
%s
```
The answer is \\boxed{%s}.
""" % (
    example['question'],
    solution,
    answer
)
    return text


def gsm8k(args, dataset_name='gsm8k'):
    if _skip(args, dataset_name, args.data_dir):
        return

    splits = defaultdict(list)

    ds = datasets.load_dataset(
        'gsm8k', 'main'
    )
    for split in ds:
        for i in range(len(ds[split])):
            example = ds[split][i]
            text = _format_gsm8k(example, args)
            splits[split].append({
                'text': text,
                'meta': {
                }
            })

    # only has train and test; create a small validation set from train.
    train, validation = make_split(splits['train'], split_ratio=args.eval_ratio)
    splits = {
        'train': train,
        'validation': validation,
        'test': splits['test']
    }

    _save_splits(splits, out_dir=args.data_dir, dataset=dataset_name)

    num_tokens = token_length(
        splits['train'] + splits['validation'] + splits['test']
    )
    _save_stats({
        'num_train': len(splits['train']),
        'num_validation': len(splits['validation']),
        'num_test': len(splits['test']),
        'num_tokens': num_tokens,
    }, dataset_name, args.meta_dir)
# ----


# ----- Naturalproofs-gen
def _format_naturalproofs_theorem_proof(example, args):
    text = """### Theorem: %s
%s

### Proof
```
%s
```
""" % (
        example['title'],
        example['text'],
        example['target']
    )
    return text


def _format_naturalproofs_theorem_references_proof(example, args):
    def _unique(ctxs):
        seen = set()
        ctxs_ = []
        for ctx in ctxs:
            if ctx['title'] not in seen:
                ctxs_.append(ctx)
                seen.add(ctx['title'])
        return ctxs_

    def _format_refs(example):
        unique_refs = _unique(example['ctxs'])
        ref_texts = []
        for ref in unique_refs:
            # Note: in NaturalProver, only the reference titles (or no references) were added.
            ref_texts.append('#### Reference: %s\n%s' % (ref['title'], ref['text']))
        text = '\n\n'.join(ref_texts)
        return text

    refs = _format_refs(example)
    text = """### Theorem: %s
%s

### Relevant references
%s

### Proof of %s
```
%s
```
""" % (
        example['title'],
        example['text'],
        refs,
        example['title'],
        example['target']
    )
    return text


def naturalproofs_gen(args, dataset_name='naturalproofs-gen'):
    if _skip(args, dataset_name, args.data_dir):
        return

    splits = defaultdict(list)
    ds = datasets.load_dataset(
        'wellecks/naturalproofs-gen',
    )
    for split in ds:
        for i in range(len(ds[split])):
            example = ds[split][i]
            example['text'] = example['text'].replace('\\n', '\n')
            example['target'] = example['target'].replace('\\n', '\n')
            # version with theorem-proof
            text = _format_naturalproofs_theorem_proof(example, args)
            splits[split].append({
                'text': text,
                'meta': {}
            })

            # version with theorem-refs-proof
            text = _format_naturalproofs_theorem_references_proof(example, args)
            splits[split].append({
                'text': text,
                'meta': {}
            })

    _save_splits(splits, out_dir=args.data_dir, dataset=dataset_name)
    num_tokens = token_length(
        splits['train'] + splits['validation'] + splits['test']
    )
    _save_stats({
        'num_train': len(splits['train']),
        'num_validation': len(splits['validation']),
        'num_test': len(splits['test']),
        'num_tokens': num_tokens,
    }, dataset_name, args.meta_dir)

# ----


def setup(args):
    random.seed(args.seed)
    np.random.seed(args.seed)


def main(args):
    lila(args)
    math_dataset(args)
    gsm8k(args)
    naturalproofs_gen(args)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--data-dir', type=str, default='data_jsonl')
    parser.add_argument('--meta-dir', type=str, default='meta_json')
    parser.add_argument('--eval-ratio', type=int, default=0.005)
    parser.add_argument('--seed', type=int, default=72)

    args = parser.parse_args()
    setup(args)
    main(args)
