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


GITHUB_ACCESS_TOKEN = os.environ['GITHUB_ACCESS_TOKEN']

TEXT_MAX_SIZE = 1048575  # in bytes
MAX_NUMERICAL_DENSITY = .5


def _get_dir_from_repo(author, repo, sha, repo_dir, save_path, overwrite):
    if (not overwrite) and Path(save_path).exists():
        return
    Path(save_path).mkdir(parents=True, exist_ok=True)
    archive_path = os.path.join(save_path, "archive.tar.gz")
    tarball_url = (
        "https://github.com/" + author + "/" + repo + "/archive/" + sha + ".tar.gz"
    )

    os.system("wget -O " + archive_path + " " + tarball_url)
    os.system("tar -xzf " + archive_path + " -C " + save_path)

    export_name = repo + "-" + sha
    os.system(
        "cp -r " + os.path.join(save_path, export_name, repo_dir, "*") + " " + save_path
    )
    os.system("rm -r " + os.path.join(save_path, export_name) + " " + archive_path)


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
            'repo_dir': '',
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


def _save(filtered, lang, output_dir, shard):
    out_dir = '%s/processed' % output_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    f_out_jsonl = '%s/%s-%d' % (out_dir, lang, shard)
    ndjson.dump(filtered, open(f_out_jsonl, 'w'))


def _save_stats(stats, lang, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    f_out = '%s/stats-%s.json' % (output_dir, lang)
    with open(f_out, 'w') as f:
        json.dump(stats, f)


def _save_repo_metadata(repos, lang, input_dir):
    f_out = '%s/repos-%s.jsonl' % (input_dir, lang)
    with open(f_out, 'w') as f:
        ndjson.dump(repos, f)


def _save_splits(splits, out_dir, lang):
    print("Saving split to disk...")
    for split, examples in tqdm(splits.items(), total=len(splits)):
        out_dir_ = os.path.join(out_dir, 'splits', split)
        Path(out_dir_).mkdir(parents=True, exist_ok=True)
        out_file = os.path.join(
            out_dir_, '%s-%s.jsonl' % (split, lang)
        )
        with open(out_file, 'w') as f:
            for example in examples:
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


def run(lang, file_pattern, filter_fn, limit, overwrite, dedup_chunk_size, out_dir):
    print("Getting repos list...")
    repos = get_repos(lang, limit, out_dir)
    print("Downloading %d repos..." % (len(repos)))
    for repo in tqdm(repos, total=len(repos)):
        _get_dir_from_repo(**repo, overwrite=overwrite)
        _delete_files_except_pattern(repo['save_path'], r".*\%s" % file_pattern)

    _save_repo_metadata(repos, lang, out_dir)

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

    _save(examples, lang, out_dir, shard=0)  # extra copy before splitting; not strictly needed
    _save_splits(
        splits=make_splits(examples, args.eval_ratio),
        out_dir=out_dir,
        lang=lang
    )
    _save_stats({
        'num_repos': args.limit,
        'num_examples': len(examples),
        'num_tokens': num_tokens,
        'timestamp': datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    }, lang, out_dir)


def coq(args):
    run(
        lang='coq',
        file_pattern='.v',
        filter_fn=filter_coq,
        limit=args.limit,
        overwrite=args.overwrite,
        dedup_chunk_size=args.dedup_chunk_size,
        out_dir=args.out_dir
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
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--out-dir', type=str, default='github-code')
    parser.add_argument('--eval-ratio', type=int, default=0.05)
    parser.add_argument('--seed', type=int, default=72)

    args = parser.parse_args()
    setup(args)
    main(args)
