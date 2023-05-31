"""
To reproduce, first run a query on BigQuery.
See `bigquery-code/original/coq/query.sql` for an example.

Then export the BigQuery table into .jsonl.gz files, which are the input to this script.
"""

import glob
import gzip
import shutil
import ndjson
import os
import tiktoken
from tqdm import tqdm


def _unzip(f_gz):
    with gzip.open(f_gz, 'rb') as f_in:
        with open(f_gz[:-3], 'wb') as f_out:  # removing .gz
            shutil.copyfileobj(f_in, f_out)
    return f_gz[:-3]


def _read(f_jsonl):
    return ndjson.load(open(f_jsonl))


def _delete(f_jsonl):
    os.remove(f_jsonl)


def _save(filtered, lang, input_dir, shard):
    f_out_jsonl = '%s/processed/%s-%d.jsonl' % (input_dir, lang, shard)
    ndjson.dump(filtered, open(f_out_jsonl, 'w'))


def filter_coq(example):
    keep = _has_coq_keyword(example)
    keep = keep and (not _has_verilog_keyword(example))
    return keep


def _has_coq_keyword(example):
    kws = {'Theorem', 'Proof', 'Qed', 'Inductive', 'Definition', 'Fixpoint'}
    for k in kws:
        if k in example['content']:
            return True
    return False


def _has_verilog_keyword(example):
    kws = {'pragma', 'endmodule', 'posedge', 'negedge', 'wire'}
    for k in kws:
        if k in example['content']:
            return True
    return False


def _filter(examples, filter_fn):
    filtered = []
    for x in tqdm(examples, total=len(examples)):
        if filter_fn(x):
            filtered.append(x)
    return filtered


def filter_duplicate_repos(examples):
    from collections import Counter
    seen_repo = set()
    fullnames = set()

    # sort by number of files in the repo. We'll keep the copy
    # that has the highest number of files.
    fullname_counts = Counter([x['repo_name'] for x in examples]).most_common()
    for fullname, count in fullname_counts:
        author, repo = fullname.split('/')
        if repo not in seen_repo:
            seen_repo.add(repo)
            fullnames.add(fullname)

    # now we only keep examples with a retained fullname
    filtered = []
    for example in examples:
        if example['repo_name'] in fullnames:
            filtered.append(example)
    return filtered


def token_length(examples):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = [
        len(x) for x in tokenizer.encode_batch(
            [x['content'] for x in examples], disallowed_special=()
        )
    ]
    return sum(tokens)


def main(args):
    langs = args.langs
    stats = {'original': 0, 'filtered': 0, 'tokens': 0}
    for lang in langs:
        print("==== %s" % lang)
        files = glob.glob('%s/original/%s/*.gz' % (args.input_dir, lang))

        for shard, f_gz in tqdm(enumerate(files), total=len(files)):
            # Unzip .gz into .jsonl
            f_jsonl = _unzip(f_gz)
            raw = _read(f_jsonl)
            _delete(f_jsonl)  # delete large file (we still have the .gz)

            # Apply filter
            if lang == 'coq':
                filter_fn = filter_coq
            else:
                filter_fn = lambda x: True

            filtered = filter_duplicate_repos(_filter(raw, filter_fn))
            num_tokens = token_length(filtered)

            # Save shard
            _save(filtered, lang, args.input_dir, shard)

            stats['original'] += len(raw)
            stats['filtered'] += len(filtered)
            stats['tokens'] += num_tokens

        for k, v in stats.items():
            print('', k, v, sep='\t')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default='bigquery-code')
    parser.add_argument('--langs', type=str, default=['coq'], nargs='+')

    args = parser.parse_args()
    main(args)
