import sys
import os
from typing import List, Dict
from concurrent import futures
from urllib.parse import urlparse
from functools import partial
from multiprocessing import Lock
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import httpx
import ndjson
import json

import argparse
from tqdm import tqdm

import tiktoken

sys.path.append("../source_code/")
from process_stack import *
from process_github import setup, transform_lean, transform_isabelle


PYTHON_EXTENSIONS = ["py", "pyw"]
C_EXTENSIONS = ["c", "h"]
CPP_EXTENSIONS = ["cpp", "cxx", "cc", "c++", "hpp", "hxx", "hh", "h++",]
FORTRAN_EXTENSIONS = ["for", "ftn", "f77", "f90", "f95", "f03", "f08",]

EXTENSIONS = [
        # Scientific/statistical computing
            # R
            "r", 
            # MATLAB
            "m",
            # Julia
            "jl",
        # CAS
            # Maple
            "mpl",
        # Formal
            # Lean
            "lean",
            # Isabelle
            "thy",
            # Idris
            "idr", 
            # Coq
            "v",
            # Agda
            "agda",
        # Imperative
            *PYTHON_EXTENSIONS,
            # C
            *C_EXTENSIONS,
            # C++
            *CPP_EXTENSIONS,
            # Fortran
            *FORTRAN_EXTENSIONS,
        # Markup
            # tex
            "tex",
        # Notebooks
            # python jupyter
            "ipynb",
]

def init_pool_processes(the_locks):
    '''Initialize each process with a global variable lock.
    '''
    global locks
    locks = the_locks

class CustomProcessPoolExecutor(ProcessPoolExecutor):
    def __init__(self, max_workers=None, initializer=None, initargs=()):
        super().__init__(
                max_workers=max_workers, initializer=initializer, initargs=initargs
        )

def _get_ext(example):
    _, extension_with_dot = os.path.splitext(example["meta"]["path"])
    return extension_with_dot[1:]

def _convert_to_stack_format(example):
    return {
        "content": example["text"], 
        "size": len(example["text"].encode('utf-8')),
        "max_stars_repo_path": example["meta"]["path"],
        "ext": _get_ext(example),
        **example,
    }

def _convert_to_gh_format(example):
    return {
            "text": example["text"],
            "meta": {
                "repo": example["meta"]["repo_name"], 
                **example["meta"]
            }
    }

def download_jsonl(url: str, filepath: str):
    with open(filepath, "wb") as download_file:
        with httpx.stream("GET", url) as response:
            total = int(response.headers["Content-Length"])

            with tqdm(total=total,unit_scale=True,unit_divisor=1024,unit="B") as progress:
                num_bytes_downloaded = response.num_bytes_downloaded
                for chunk in response.iter_bytes():
                    download_file.write(chunk)
                    progress.update(response.num_bytes_downloaded - num_bytes_downloaded)
                    num_bytes_downloaded = response.num_bytes_downloaded

def get_jsonl(url: str, raw_dir: str) -> List:
    filename = os.path.basename(urlparse(url).path)
    filepath = os.path.join(raw_dir, filename)

    print(f"getting {filename} to {raw_dir}...")

    if not os.path.isfile(filepath):
        print(f"downloading {filename} from web...")
        download_jsonl(url=url, filepath=filepath)
    else:
        print(f"found local copy...")
    
    with open(filepath) as f: 
        data = ndjson.load(f)

    print(f"done getting {filename}")

    return data

def filter_fn(example):
    extension = _get_ext(example)

    if extension not in EXTENSIONS: 
        return False
    elif extension=="r":
        return r_filter(_convert_to_stack_format(example))
    elif extension=="mpl":
        return maple_filter(_convert_to_stack_format(example))
    elif PYTHON_EXTENSIONS:
        return py_filter(_convert_to_stack_format(example))
    elif extension in C_EXTENSIONS:
        return c_filter(_convert_to_stack_format(example))
    elif extension in CPP_EXTENSIONS:
        return cpp_filter(_convert_to_stack_format(example))
    elif extension=="jl":
        return julia_filter(_convert_to_stack_format(example))
    elif extension=="tex":
        return tex_filter(_convert_to_stack_format(example))
    elif extension=="v":
        return filter_coq(_convert_to_gh_format(example))
    elif extension=="lean": 
        return filter_lean(_convert_to_gh_format(example))
    elif extension=="thy": 
        return filter_isabelle(_convert_to_gh_format(example))
    elif extension=="m": 
        return filter_matlab(_convert_to_gh_format(example))
    elif extension=="ipynb":
        "HIT NOTEBOOK"
        print(example["text"])
        sys.exit()
    else: 
        return True

ENC = tiktoken.get_encoding("cl100k_base")
def process(example): 
    extension = _get_ext(example)

    if extension=="thy":
        new, _ = transform_isabelle(example)
    elif extension=="lean":
        new, _ = transform_lean(example)
    else: 
        new = example

    return {**new, "num_tokens": len(ENC.encode(new["text"]))}
    

def get_filter_save(url: str, raw_dir: str, data_dir: str):
    data = get_jsonl(url=url, raw_dir=raw_dir)
    
    print("processing...")
    processed_data = [process(x) for x in tqdm(data) if filter_fn(x)]
    
    token_counts_dict = {}
    for example in processed_data:
        extension = _get_ext(example)
        
        if extension in token_counts_dict: 
            token_counts_dict[extension] += example["num_tokens"]
        else: 
            token_counts_dict[extension] = example["num_tokens"]

        lock = locks[extension]
        lock.acquire()
        try: 
            with open(os.path.join(data_dir, f"{extension}.jsonl"), "a") as f: 
                f.write(json.dumps(example) + "\n")
        finally: 
            lock.release()

    return token_counts_dict

def concurrent_get_filter_save(
        urls: List[str], 
        raw_dir: str, 
        data_dir: str, 
        meta_dir: str,
):
    to_map = partial(get_filter_save, raw_dir=raw_dir, data_dir=data_dir)

    locks = {k: Lock() for k in EXTENSIONS}

    with CustomProcessPoolExecutor(initializer=init_pool_processes, initargs=(locks,)) as executor: 
        token_count_dicts = executor.map(to_map, urls)

    #init_pool_processes(locks)
    
    # token_count_dicts = list(map(to_map, urls))

    print("TOKEN COUNT DICTS...", token_count_dicts)
    token_count_dict = {
            k: sum(tkd[k] for tkd in token_count_dicts if k in tkd)
            for k in set(k for tkd in token_count_dicts for k in tkd)
    }
    
    print("TOKEN DATA...")
    print(token_count_dict)

    with open(os.path.join(meta_dir, "meta.json"), "w") as f:
        json.dump(token_count_dict, f)

def main(args):
    if os.path.isdir(args.data_dir): 
        raise OSError(f"{args.data_dir} already exists")
    Path(args.data_dir).mkdir(exist_ok=True, parents=True)
    Path(args.raw_dir).mkdir(exist_ok=True, parents=True)
    Path(args.meta_dir).mkdir(exist_ok=True, parents=True)

    with open(args.urls) as f: 
        urls = f.read().split()


    concurrent_get_filter_save(
            urls=urls,
            raw_dir=args.raw_dir,
            data_dir=args.data_dir,
            meta_dir=args.meta_dir
    )


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--urls', type=str, default='urls.txt')
    parser.add_argument('--data-dir', type=str, default='data_jsonl/')
    parser.add_argument('--meta-dir', type=str, default='meta_json/')
    parser.add_argument('--raw-dir', type=str, default='raw_rj/')
    parser.add_argument('--seed', type=int, default=72)

    args = parser.parse_args()


    # hack to get setup to work
    args.langs = []
    args.cutoff_date = None
 
    setup(args)
    main(args) 
