import os
import re
import hashlib
from pathlib import Path
from github import Github
from tqdm import tqdm
from datetime import datetime
import ndjson
import tiktoken
import json
import random
import numpy as np
import backoff
import subprocess


GITHUB_ACCESS_TOKEN = os.environ['GITHUB_ACCESS_TOKEN']

TEXT_MAX_SIZE = 1048575  # in bytes
MAX_NUMERICAL_DENSITY = .5


@backoff.on_exception(backoff.expo, subprocess.CalledProcessError)
def _get_dir_from_repo(author, repo, sha, save_path, overwrite):
    if (not overwrite) and Path(save_path).exists():
        return
    Path(save_path).mkdir(parents=True, exist_ok=True)
    archive_path = os.path.join(save_path, "archive.tar.gz")
    tarball_url = (
        "https://github.com/" + author + "/" + repo + "/archive/" + sha + ".tar.gz"
    )

    subprocess.call(['wget', '-O', archive_path, tarball_url])
    subprocess.call(['tar', '-xzf', archive_path, '-C', save_path])


def _delete_files_except_pattern(path, pattern):
    for f in os.listdir(path):
        f_path = os.path.join(path, f)
        if os.path.isfile(f_path):
            if not re.search(pattern, f):
                os.remove(f_path)
        elif os.path.islink(f_path):
            os.remove(f_path)
        elif os.path.isdir(f_path):
            _delete_files_except_pattern(f_path, pattern)


def _get_sha(repo):
    # use the most recent commit
    try:
        commit_obj = repo.get_commits()[0]
        sha = commit_obj.sha
    except IndexError:
        sha = 'master'
    return sha


def get_repos(lang, limit, out_dir):
    g = Github(GITHUB_ACCESS_TOKEN)

    results = g.search_repositories(
        query='language:%s' % lang,
        sort='stars'
    )
    repositories = []
    for repo in tqdm(results, total=limit):
        author, repo_name = repo.full_name.split('/')
        info = {
            'author': author,
            'repo': repo_name,
            'sha': _get_sha(repo),
            'save_path': '%s/%s/%s-%s' % (out_dir, lang, author, repo_name)
        }
        repositories.append(info)

        if len(repositories) == limit:
            break
    return repositories


def _extract(path, pattern, metadata):
    out = []
    base = Path(path)
    for pp in base.rglob(pattern):
        if pp.is_file():
            with pp.open() as f:
                try:
                    text = f.read()
                except UnicodeDecodeError:
                    continue
                out.append({
                    'text': text,
                    'meta': metadata
                })
    return out


def _filter(examples, filter_fn):
    filtered = []
    for x in tqdm(examples, total=len(examples)):
        if filter_fn(x):
            filtered.append(x)
    return filtered


def _save_stats(stats_of_lang, lang, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    f_out = '%s/github-stats.json' % output_dir
    if os.path.isfile(f_out):
        with open(f_out) as f:
            stats = json.load(f)
    else:
        stats = dict()
    stats[lang] = stats_of_lang
    with open(f_out, 'w') as f:
        json.dump(stats, f, indent=2)


def _save_repo_metadata(repos, lang, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    f_out = '%s/%s-repos.jsonl' % (output_dir, lang)
    with open(f_out, 'w') as f:
        ndjson.dump(repos, f)

    f_out = '%s/github_%s_index' % (output_dir, lang)
    with open(f_out, 'w') as f:
        for repo in repos:
            f.write('%s/%s' % (repo['author'], repo['repo']))


def _save_splits(splits, out_dir, lang, shard_size=50000):
    print("Saving split to disk...")
    for split, examples in tqdm(splits.items(), total=len(splits)):
        out_dir_ = os.path.join(out_dir, split)
        Path(out_dir_).mkdir(parents=True, exist_ok=True)
        for i in range(0, len(examples), shard_size):
            num_digits = max(len(str(len(examples)//shard_size+1)), 4)
            out_file = os.path.join(
                out_dir_, 'github-%s-%s-%s.jsonl' % (lang, split, str(i).zfill(num_digits))
            )
            shard = examples[i*shard_size:(i+1)*shard_size]
            with open(out_file, 'w') as f:
                for example in shard:
                    f.write(json.dumps(example))
                    f.write('\n')


def make_splits(examples, eval_ratio):
    test_len = max(int(eval_ratio * len(examples)), 1)
    perm = np.random.permutation(len(examples))
    examples = [examples[i] for i in perm]
    splits = {
        'train': examples[:len(examples)-(2*test_len)],
        'validation': examples[len(examples)-(2*test_len):len(examples)-test_len],
        'test': examples[len(examples)-test_len:],
    }
    for k, v in splits.items():
        print("%s length: %d" % (k, len(v)))
    return splits


def deduplicate(examples, chunk_size=2048):
    # "Chunk-based" deduplication. Iterate over consecutive chunks
    # in a document. If a chunk appears exactly in another document,
    # we consider the documents to be duplicates of one another.
    #
    # When a duplicate is detected (i.e., when a chunk appears in
    # more than one document), we keep only the first document.
    duped = set()
    seen_chunks = set()
    for docid, example in enumerate(examples):
        content = example['text']
        for chunk_start in range(0, len(content), chunk_size):
            chunk = content[chunk_start:chunk_start + chunk_size]
            if (len(chunk) == chunk_size) or (len(chunk) == len(content)):
                h = hashlib.new('sha256')
                h.update(chunk.encode())
                h_chunk = h.hexdigest()
                if h_chunk in seen_chunks:
                    duped.add(docid)
                seen_chunks.add(h_chunk)
    filtered = []
    for docid, example in enumerate(examples):
        if docid not in duped:
            filtered.append(example)
    return filtered


def token_length(examples):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = [
        len(x) for x in tokenizer.encode_batch(
            [x['text'] for x in examples], disallowed_special=()
        )
    ]
    return sum(tokens)


def numerical_density(ex):
    # The ratio of digit non-whitespace characters over non-digit non-whitespace
    # characters in the file
    txt = ''.join(ex["text"].split())
    ntoks = sum(txt.count(c) for c in "0123456789")
    return ntoks / max(1, len(txt))


def standard_filter(
        example,
        max_numerical_density=MAX_NUMERICAL_DENSITY,
        text_max_size=TEXT_MAX_SIZE
):
    """
    Byte length and numerical density filter that is repeated throughout
    this script
    """
    if len(example["text"].encode("utf-8")) > text_max_size:
        return False
    elif numerical_density(example) > max_numerical_density:
        return False
    else:
        return True


# --- Coq-specific
def filter_coq(example):
    def _has_coq_keyword(example):
        # Both Coq and Verilog have .v files; we only want Coq files.
        # Rough heuristic of whether this is a Coq file.
        kws = {'Theorem', 'Proof', 'Qed', 'Inductive', 'Definition', 'Fixpoint'}
        for k in kws:
            if k in example['text']:
                return True
        return False

    def _has_verilog_keyword(example):
        # Both Coq and Verilog have .v files; we only want Coq files.
        # Rough heuristic of whether this is a Verilog file.
        kws = {'pragma', 'endmodule', 'posedge', 'negedge', 'wire'}
        for k in kws:
            if k in example['text']:
                return True
        return False

    def _has_bad_keyword(example):
        kws = {'This file was automatically generated'}
        for k in kws:
            if k in example['text']:
                return True
        return False

    keep = standard_filter(example)
    keep = keep and _has_coq_keyword(example)
    keep = keep and (not _has_verilog_keyword(example))
    keep = keep and (not _has_bad_keyword(example))
    return keep
# ---


def run(lang, file_pattern, filter_fn, limit, overwrite, dedup_chunk_size, data_dir, meta_dir, repos_dir):
    print("Getting repos list...")
    repos = get_repos(lang, limit, repos_dir)
    print("Downloading %d repos..." % (len(repos)))
    for repo in tqdm(repos, total=len(repos)):
        _get_dir_from_repo(**repo, overwrite=overwrite)
        _delete_files_except_pattern(repo['save_path'], r".*\%s" % file_pattern)

    _save_repo_metadata(repos, lang, meta_dir)

    print("Extracting files from repos...")
    examples = []
    for repo in tqdm(repos, total=len(repos)):
        examples_ = _extract(repo['save_path'], "*%s" % file_pattern, metadata=repo)
        examples.extend(examples_)
    print("\t%d files" % len(examples))

    print("Filtering...")
    examples = _filter(examples, filter_fn)
    print("\t%d files" % len(examples))
    examples = deduplicate(examples, chunk_size=dedup_chunk_size)
    print("\t%d files" % len(examples))

    print("Computing tokens...")
    num_tokens = token_length(examples)
    print("\t%d tokens" % (num_tokens))

    _save_splits(
        splits=make_splits(examples, args.eval_ratio),
        out_dir=data_dir,
        lang=lang
    )
    _save_stats({
        'num_repos': args.limit,
        'num_examples': len(examples),
        'num_tokens': num_tokens,
    }, lang, meta_dir)


def coq(args):
    run(
        lang='coq',
        file_pattern='.v',
        filter_fn=filter_coq,
        limit=args.limit,
        overwrite=args.overwrite,
        dedup_chunk_size=args.dedup_chunk_size,
        data_dir=args.data_dir,
        meta_dir=args.meta_dir,
        repos_dir=args.repos_dir,
    )


def setup(args):
    random.seed(args.seed)
    np.random.seed(args.seed)


def main(args):
    if 'coq' in args.langs:
        coq(args)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=250)
    parser.add_argument('--langs', type=str, default=['coq'], nargs='+')
    parser.add_argument('--dedup-chunk-size', type=int, default=2048)
    parser.add_argument('--shard-size', type=int, default=50000)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--data-dir', type=str, default='data_jsonl')
    parser.add_argument('--meta-dir', type=str, default='meta_json')
    parser.add_argument('--repos-dir', type=str, default='github-repos')
    parser.add_argument('--eval-ratio', type=int, default=0.005)
    parser.add_argument('--seed', type=int, default=72)

    args = parser.parse_args()
    setup(args)
    main(args)
