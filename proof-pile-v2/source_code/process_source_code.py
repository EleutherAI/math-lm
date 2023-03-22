from datasets import load_dataset
from itertools import islice
from tqdm import tqdm
import os
import json
import ndjson
import sys
import random
import re
from pathlib import Path
import shutil
import nbformat
from functools import reduce, partial
from transformers import AutoTokenizer
from nbconvert.exporters import MarkdownExporter

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

NUM_PROC = 10
SAVE_BATCH_SIZE = 100_000
SAVE_DIR = "stack-code"

DATA_DIRS = [
    # numerical computing
    #"matlab",
    #"r",
    # CAS
    #"sage",
    #"mathematica",
    #"maple",
    #"gap",
    # formal math
    #"lean",
    #"isabelle",
]

DATA_DIRS_TO_FILTER = [
    #"python",
    # "jupyter-notebook",
    "julia",
    #"c",
    #"c++",
    #"tex",
]

def matlab_rexp(example, rexp):
    return bool(rexp.search(example["content"]))

h = re.compile('[a-df-zA-Z]')
matlab_filter = partial(matlab_rexp, rexp=h)

def r_filter(example): 
    return "/* Resource fork" not in example["content"]

def mathematica_filter(example): 
    return example["max_stars_repo_name"] != "dendaxD/QAOA-MaxCut-amplitudes"

def maple_filter(example): 
    return "<?xml" != example["content"][:5]

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


def julia_test_file(ex, ratio=0.1):
    # Whether a file has some minimum ratio of @test statements
    txt = ex["content"]
    kwd = "@test"
    nlines = txt.count("\n") + 1
    return kwd in txt and (txt.count(kwd) / nlines >= ratio)

def julia_numerical_density(ex):
    # The ratio of digit characters over non-digit characters in the file
    txt = ex["content"]
    ntoks = sum(txt.count(c) for c in "0123456789")
    return ntoks / len(txt)

def generated_file(ex):
    #This heuristic happens to be superfluous
    return "generated" in ex["max_stars_repo_name"] or ex["max_stars_repo_name"][0] == "."

def julia_filter(ex):
    if ex["content"][0] in ["%", "{", "["]:
        # Eliminates non-Julia files such as JSON lines (.jl) files 
        return False
    elif ex["size"] >= 1e5:
        # Overly large files are often auto-generated boilerplate and/or mostly
        # contain large arrays of numbers.Thus, we reject such large files unless
        # unless they are test files with low numerical density.
        return julia_test_file(ex) and julia_numerical_density(ex) <= 0.5
    else:
        return True
    
# A list of keywords that make a Julia file interesting
julia_whitelist = [
    # Popular packages for scientific computing
    "LinearAlgebra",
    "DifferentialEquations",
    "Symbolics",
    "Distributions",
    "DataFrames",
    "DynamicalSystems",
    "Turing",
    "Gen",
    "JuMP",
    # Standard mathematical functions
    "sqrt",
    "abs",
    "zeros",
    "ones",
    "sin",
    "cos",
    "tan",
    "log",
    "exp",
    "integrate",
    "likelihood",
    "Matrix",
    "π",
    "pi",
    "rand",
    "grad"
]

julia_whitelist_rexp = re.compile("|".join("(\\W" + kwd + "\\W)" for kwd in julia_whitelist))

def julia_filter_strict(ex):
    # A stricter Julia filter that operates from a whitelist
    return julia_filter(ex) and julia_whitelist_rexp.search(ex["content"])


def tex_filter_rexp(example, rexp):
    if example["ext"] != "tex": 
        return False 

    if "latex/" in example["max_stars_repo_path"]:
        return False 

    text = example["content"]

    if rexp.search(text): 
        return False 

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
        "\\subparagraph{",
    ]

    found = [x for x in keywords if x in text]
    return bool(found)

h = re.compile("[\u0370-\u18aA\u3000-\U0001047f]")
tex_filter = partial(tex_filter_rexp, rexp=h)


def jupyter_notebook_filter(example):
    text = example["content"]
    lower = text.lower()
    keywords = {
        "\\begin{equation}",
        "\\begin{align}",
        "import sympy",
        "from sympy"
    }
    for keyword in keywords:
        if keyword in lower:
            return True
    return False


def _filter_cell_output(output):
    # See https://ipython.org/ipython-doc/3/notebook/nbformat.html
    #   as a reference on formatting.
    # Remove image/png data (a base64 string).
    if ('output_type' in output and
            'data' in output and
            'image/png' in output['data'] and
            len(output['data']['image/png']) > 0):
        return True

    # Remove exceptions.
    if 'ename' in output and 'traceback' in output:
        return True
    return False


def process_jupyter_notebook(example):
    try:
        content = example['content']
        notebook = nbformat.reads(content, as_version=4)

        # Filter output content.
        for cell in notebook.cells:
            if 'outputs' in cell:
                clear = False
                for output in cell['outputs']:
                    if _filter_cell_output(output):
                        clear = True
                        break
                if clear:
                    cell['outputs'] = []

        # Convert to Markdown
        exporter = MarkdownExporter()
        body, resources = exporter.from_notebook_node(notebook)
        example['content'] = body
        example['converted'] = True

    # Mark to discard later if conversion wasn't successful.
    except Exception:
        example['converted'] = False
    return example


def filter_processed_jupyter_notebook(example):
    return example['converted']


def token_length(examples, tokenizer):
    return {
        "neox_tokens": [len(x) for x in tokenizer(examples["content"])["input_ids"]]
    }


def batch_loader(ds, size):
    """
    Iterator that takes in a list `seq` and returns
    chunks of size `size`"""
    for pos in range(0, len(ds), size):
        if pos + size < len(ds):
            yield [x for x in ds.select(list(range(pos, pos + size)))]
        else:
            yield [x for x in ds.select(list(range(pos, len(ds))))]


def main():
    stats = {}

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    for lang in DATA_DIRS + DATA_DIRS_TO_FILTER:
        print(lang.upper() + "#" * 70)
        use_auth_token=None
        if (tok := os.environ.get("HUGGING_FACE_TOKEN")) is not None:
            use_auth_token = tok
        print(f"loading {lang} data...")
        ds = load_dataset(
            "bigcode/the-stack-dedup", data_dir=f"data/{lang}", split="train",
            use_auth_token=use_auth_token
        )

        # debugging block
        # print("selecting samples from dataset (debugging)...")
        # ds = ds.select(random.sample(range(len(ds)), k=10_000))

        print("filtering dataset...")
        if lang=="matlab":
            ds = ds.filter(matlab_filter, num_proc=NUM_PROC)
        elif lang=="r":
            ds = ds.filter(r_filter, num_proc=NUM_PROC)
        elif lang=="mathematica":
            ds = ds.filter(mathematica_filter, num_proc=NUM_PROC)
        elif lang=="maple":
            ds = ds.filter(maple_filter, num_proc=NUM_PROC)
        elif lang == "python":
            ds = ds.filter(py_filter, num_proc=NUM_PROC)
        elif lang == "c":
            ds = ds.filter(c_filter, num_proc=NUM_PROC)
        elif lang == "c++":
            ds = ds.filter(cpp_filter, num_proc=NUM_PROC)
        elif lang == "tex":
            ds = ds.filter(tex_filter, num_proc=NUM_PROC)
        elif lang == "julia":
            ds = ds.filter(julia_filter, num_proc=NUM_PROC)
        elif lang == "jupyter-notebook":
            ds = ds.filter(jupyter_notebook_filter, num_proc=NUM_PROC)
            ds = ds.map(process_jupyter_notebook, num_proc=NUM_PROC)
            ds = ds.filter(filter_processed_jupyter_notebook, num_proc=NUM_PROC)
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
        if Path(save_lang).exists():
            shutil.rmtree(save_lang)
        Path(save_lang).mkdir(parents=True, exist_ok=True)

        for i, batch in tqdm(
            enumerate(batch_loader(ds, SAVE_BATCH_SIZE)),
            total=len(ds) // SAVE_BATCH_SIZE + 1,
        ):
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
