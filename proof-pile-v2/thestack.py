from datasets import load_dataset 
from itertools import islice
from tqdm import tqdm
import os
import json

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

SAVE_DIR = "stack-code"

DATA_DIRS = [
        # numerical computing
        "matlab", 
        "julia", 
        # CAS
        "sage", 
        "mathematica", 
        "maple", 
        "gap", 
        # formal math
        "lean", 
        "isabelle", 
]

DATA_DIRS_TO_FILTER = [
        "python", 
        "c", 
        "c++", 
        "tex", 
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
            "numexpr" ,
            ]
    for pack in packages: 
        keywords += [f"import {pack}", f"from {pack}"]

    return any(keyword in text for keyword in keywords)

def c_filter(example): 
    text = example["content"]
    keywords = [
            "#include <fftw.h>", "#include <rfftw.h>", 
            "#include <gsl", 
            "#include <cblas.h>"
            ]

    found = [x for x in keywords if x in keywords]
    return found

def cpp_filter(example): 
    text = example["content"]
    keywords = [
            "#include <adept_arrays.h>", "#include <adept.h>",
            # how alglib?
            "#include <boost"
            "#include <armadillo", 
            "#include <blitz", 
            "#include <Eigen", 
            "#include <deal.II", 
            "#include <dlib", 
            "#include <NTL", 
            "#include <mtl"
            ]

    found = [x for x in keywords if x in keywords]
    return found

def tex_filter(text):
    keywords = [
            "\\section{", "\\section*{", "\\subsection{", "\\subsection*{", 
            "\\subsubsection{", "\\subsubsection*{", "\\chapter{", "\\chapter*{", 
            "\\paragraph{"
            ]

    found = [x for x in keywords if x in keywords]
    return found



def main(): 
    stats = {}
    for lang in tqdm(DATA_DIRS): 
        ds = load_dataset("bigcode/the-stack", 
                data_dir=f"data/{lang}", split="train")

        print(lang.upper() + "#"*70)

        for x in islice(ds, 1): 
            print(x["content"])

        stats[lang] = {"files": len(ds), "size": ds.data.nbytes }

        print(stats[lang])
        
        print("saving to disk...")
        ds.save_to_disk(os.path.join(SAVE_DIR, f"{lang}.jsonl"))

    for lang in tqdm(DATA_DIRS_TO_FILTER): 
        print(lang.upper() + "#"*70)

        ds = load_dataset("bigcode/the-stack", data_dir=f"data/{lang}", split="train")
        unfiltered_size = ds.data.nbytes
        if lang=="python": 
            ds = ds.filter(py_filter)
        elif lang=="c":
            ds = ds.filter(c_filter)
        elif lang=="c++":
            ds = ds.filter(cpp_filter)
        elif lang=="tex": 
            ds = ds.filter(tex_filter)
        else: 
            raise Exception("DATA_DIRS_TO_FILTER and if statement not synced")

        for x in islice(ds, 1): 
            print(x["content"])

        stats[lang] = {"files": len(ds), "size": ds.data.nbytes, 
                "unfiltered_size": unfiltered_size}

        print(stats[lang])
        
        print("saving to disk...")
        ds.save_to_disk(os.path.join(SAVE_DIR, f"{lang}.jsonl"))

    with open(os.path.join(SAVE_DIR, "stats.json"), "w") as f: 
        f.write(json.dumps(stats, indent=2))

if __name__=="__main__": 
    main()
