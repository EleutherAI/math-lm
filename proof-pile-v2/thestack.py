from datasets import load_dataset
from itertools import islice
from tqdm import tqdm
import os
import json
import ndjson
import sys
import random
from pathlib import Path
from functools import reduce, partial
from transformers import AutoTokenizer

"""
Just as a reminder, here are the stack keys:

hexsha
size
ext
lang
max_stars_repo_path
max_stars_repo_name
max_stars_repo_head_hexsha
max_stars_repo_licenses
max_stars_count
max_stars_repo_stars_event_min_datetime
max_stars_repo_stars_event_max_datetime
max_issues_repo_path
max_issues_repo_name
max_issues_repo_head_hexsha
max_issues_repo_licenses
max_issues_count
max_issues_repo_issues_event_min_datetime
max_issues_repo_issues_event_max_datetime
max_forks_repo_path
max_forks_repo_name
max_forks_repo_head_hexsha
max_forks_repo_licenses
max_forks_count
max_forks_repo_forks_event_min_datetime
max_forks_repo_forks_event_max_datetime
content
avg_line_length
max_line_length
alphanum_fraction
"""

NUM_PROC = 8

SAVE_DIR = "stack-code"

DATA_DIRS = [
    # numerical computing
    #"matlab",
    #"julia",
    #"r",
    # CAS
    "sage",
    #"mathematica",
    #"maple",
    #"gap",
    # formal math
    "lean",
    #"isabelle",
]

DATA_DIRS_TO_FILTER = [
    #"python",
    #"c",
    #"c++",
    #"tex",
]


def py_filter(example):
    text = example["content"]
    keywords = []
    packages = [
        "numpy",
        "scipy",
        "sympy",
        "sage",
        "numba",
        "numexpr",
    ]
    for pack in packages:
        keywords += [f"import {pack}", f"from {pack}"]

    found = [x for x in keywords if x in text]
    return found


def c_filter(example):
    text = example["content"]
    keywords = [
        "#include <fftw.h>",
        "#include <rfftw.h>",
        "#include <gsl",
        "#include <cblas.h>",
    ]

    found = [x for x in keywords if x in text]
    return found


def cpp_filter(example):
    text = example["content"]
    keywords = [
        "#include <adept_arrays.h>",
        "#include <adept.h>",
        "#include <alglib",
        "#include <boost",
        "#include <armadillo",
        "#include <blitz",
        "#include <Eigen",
        "#include <deal.II",
        "#include <dlib",
        "#include <NTL",
        "#include <mtl",
    ]

    found = [x for x in keywords if x in text]
    return found


def tex_filter(example):
    if example["ext"] != "tex": 
        return False 

    text = example["content"]

    if "gnuplot" in text: 
        return False 

    keywords = [
        "\\chapter{",
        "\\chapter*{",
        "\\section{",
        "\\section*{",
        "\\subsection{",
        "\\subsection*{",
        "\\subsubsection{",
        "\\subsubsection*{",
        "\\paragraph{",
        "\\subparagraph{"
    ]

    found = [x for x in keywords if x in text]
    return found


def token_length(examples, tokenizer):
    return {
        "neox_tokens": [len(x) for x in tokenizer(examples["content"])["input_ids"]]
    }

def batch_loader(ds, size):
    """
    Iterator that takes in a list `seq` and returns
    chunks of size `size` """
    for pos in range(0, len(ds), size):
        if pos + size < len(ds): 
            yield [x for x in ds.select(list(range(pos, pos+size)))]
        else: 
            yield [x for x in ds.select(list(range(pos, len(ds))))]

def main():
    stats = {}

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    for lang in DATA_DIRS + DATA_DIRS_TO_FILTER:
        print(lang.upper() + "#" * 70)

        print(f"loading {lang} data...")
        ds = load_dataset(
            "bigcode/the-stack-dedup", data_dir=f"data/{lang}", split="train"
        )

        # debugging block
        # print("selecting samples from dataset (debugging)...")
        # ds = ds.select(random.sample(range(len(ds)), k=10_000))

        print("filtering dataset...")
        if lang == "python":
            ds = ds.filter(py_filter, num_proc=NUM_PROC)
        elif lang == "c":
            ds = ds.filter(c_filter, num_proc=NUM_PROC)
        elif lang == "c++":
            ds = ds.filter(cpp_filter, num_proc=NUM_PROC)
        elif lang == "tex":
            ds = ds.filter(tex_filter, num_proc=NUM_PROC)
        else:
            print("NO FILTERING APPLICABLE")

        print("calculating tokens...")
        ds = ds.map(
            partial(token_length, tokenizer=tokenizer),
            batched=True,
            num_proc=NUM_PROC,
        )

        for x in islice(ds, 1):
            print(x["content"])

        # counts number of files and dataset byte size and tokens in single loop
        print("calculating dataset statistics...")
        files, size, tokens = reduce(
            lambda x, y: (x[0] + 1, x[1] + y["size"], x[2] + y["neox_tokens"]),
            tqdm(ds),
            (0, 0, 0),
        )

        stats_of_lang = {"files": files, "size": size, "neox_tokens": tokens}

        print("printing stats...")
        print(stats_of_lang)

        print("saving dataset to disk in batches...")
        save_lang = os.path.join(SAVE_DIR, lang)
        Path(save_lang).mkdir(parents=True, exist_ok=True)
        for i, batch in tqdm(enumerate(batch_loader(ds, 100_000))): 
            with open(os.path.join(save_lang, str(i).zfill(7) + ".jsonl"), "w") as f:
                ndjson.dump(batch, f)


        print("saving stats to disk...")
        stats_path = os.path.join(SAVE_DIR, "stats.json")
        if os.path.isfile(stats_path):
            with open(stats_path) as f:
                stats = json.load(f)
        else:
            stats = dict()

        stats[lang] = stats_of_lang
        with open(stats_path, "w") as f:
            f.write(json.dumps(stats, indent=2))



if __name__ == "__main__":
    main()
