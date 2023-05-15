import argparse
import os
import sys
from pathlib import Path

from datasets import load_dataset, load_from_disk
import datasets

import json
import code

from tqdm import tqdm

import ast

def main(args):
    save_dir = "data_jsonl"
    train_path = os.path.join(save_dir, "train/")
    validation_path = os.path.join(save_dir, "validation/")
    test_path = os.path.join(save_dir, "test/")

    Path(train_path).mkdir(exist_ok=True, parents=True)
    Path(validation_path).mkdir(exist_ok=True, parents=True)
    Path(test_path).mkdir(exist_ok=True, parents=True)

    # build index
    index_dir = "../source_code/stack-code/"
    index = []

    print("loading and dedupping index...")
    if args.sample:
        index = ["google/closure-compiler", "openstack/tempest"]
    else:
        for name in tqdm(os.listdir(index_dir)):
            if name.endswith("index"):
                with open(os.path.join(index_dir, name)) as f:
                    index += [x.strip() for x in f.readlines()]

    index = set(index)


    if args.sample:
        issues = load_dataset(
            "CarperAI/pile-v2-small-filtered",
            data_dir="data/GithubDiff_ver2",
            split="train",
        )
        diffs = load_dataset(
            "CarperAI/pile-v2-small-filtered",
            data_dir="data/GithubIssues",
            split="train",
        )
    else:
        issues = load_from_disk(
            "raw_pilev2/GithubIssues",
        )
        diffs = load_from_disk("raw_pilev2/GithubDiff")
    

    print(list(index)[:10])
    print([ast.literal_eval(x["meta"])["repo_name_with_owner"] for x in issues.select(range(10))])
    print([ast.literal_eval(x["meta"])["repo_name"] for x in diffs.select(range(10))])

    assert isinstance(issues[0]["meta"], str)
    assert isinstance(diffs[0]["meta"], str)

    
    print("filtering at repo level")
    filtered_issues = issues.filter(
        lambda x: ast.literal_eval(x["meta"])["repo_name_with_owner"] in index, 
        num_proc=args.cpus,
    )
    filtered_diffs = diffs.filter(
        lambda x: ast.literal_eval(x["meta"])["repo_name"] in index, 
        num_proc=args.cpus
    )

    ds = datasets.concatenate_datasets([filtered_issues, filtered_diffs]).shuffle(
        seed=42
    )

    test_len = max(int(0.005 * len(ds)), 1)

    train = ds.select(range(len(ds) - 2 * test_len))
    validation = ds.select(range(len(ds) - 2 * test_len, len(ds) - test_len))
    test = ds.select(range(len(ds) - test_len, len(ds)))

    print(f"SPLIT: {len(train)}, {len(validation)}, {len(test)}")
    
    print("saving to disk...")
    for save_path, ds in zip(
        [train_path, validation_path, test_path], [train, validation, test]
    ):
        with open(os.path.join(save_path, "issues_diffs.jsonl"), "w") as f:
            for x in tqdm(ds):
                y = {"text": x["text"], "meta": ast.literal_eval(x["meta"])}
                f.write(json.dumps(y))
                f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s", "--sample", action="store_true", help="test script with toy dataset"
    )
    parser.add_argument(
        "-c", "--cpus", type=int, help="number of cpus to parallelize filter over"
    )
    args = parser.parse_args()

    main(args)
