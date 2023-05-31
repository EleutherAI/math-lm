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
from collections import Counter


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
    # Both Coq and Verilog have .v files; we only want Coq files.
    # Rough heuristic of whether this is a Coq file.
    kws = {'Theorem', 'Proof', 'Qed', 'Inductive', 'Definition', 'Fixpoint'}
    for k in kws:
        if k in example['content']:
            return True
    return False


def _has_verilog_keyword(example):
    # Both Coq and Verilog have .v files; we only want Coq files.
    # Rough heuristic of whether this is a Verilog file.
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


def filter_duplicate_repos(examples, seen_repo, repo_names):
    # BigQuery picks up multiple repo copies; e.g {user1}/coq and {user2}/coq.
    # We group the repos by exact string match, and keep the repo copy with the most files.
    #
    # NOTE: doesn't account for non-exact-string matches (e.g. {user1}/repo {user2}/my_repo).
    # NOTE: we select the repo copy greedily by shard. For instance, a repo with more files
    #       may occur in shard 2, but we will use the most common copy from shard 1.

    # Sort by number of files associated with the repo. We'll keep the copy
    # that has the highest number of files.
    repo_name_counts = Counter([x['repo_name'] for x in examples]).most_common()
    for repo_name, count in repo_name_counts:
        author, repo = repo_name.split('/')
        if repo not in seen_repo:
            seen_repo.add(repo)
            repo_names.add(repo_name)

    filtered = []
    for example in examples:
        if example['repo_name'] in repo_names:
            filtered.append(example)
    return filtered, seen_repo, repo_names


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

        seen_repo = set()
        repo_names = set()
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

            filtered = _filter(raw, filter_fn)
            filtered, seen_repo, repo_names = filter_duplicate_repos(
                filtered, seen_repo, repo_names
            )
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
