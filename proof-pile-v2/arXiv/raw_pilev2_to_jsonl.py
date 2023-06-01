import sys
import os
import json
import datasets 
import argparse
from tqdm import tqdm
import code
from pathlib import Path
import ast

EVAL_RATIO=0.005
SEED=57

name = "arXiv"
inpath = "raw_pilev2/arXiv"
outdir = "data_jsonl"
shard_size = 10_000
# "id":"0","check_char_repetition_criteria":0.0,"check_flagged_words_criteria":0.0,"check_stop_word_ratio_criteria":0.0}
columns_to_remove = [
        "id", 
        "check_char_repetition_criteria", 
        "check_flagged_words_criteria", 
        "check_stop_word_ratio_criteria", 
]

def fix_meta(example): 
    meta = ast.literal_eval(example["meta"])
    meta["id"] = example["id"]
    return {"meta": meta}


def main(args): 
    Path(os.path.join(outdir, "train")).mkdir(exist_ok=True, parents=True)
    Path(os.path.join(outdir, "validation")).mkdir(exist_ok=True, parents=True)
    Path(os.path.join(outdir, "test")).mkdir(exist_ok=True, parents=True)

    print("loading dataset from disk...")
    ds = datasets.load_from_disk(inpath, keep_in_memory=False)

    # shuffle, just in case upstream data is ordered. Note this makes script way slower, 
    # since we're no longer accessing a contiguous disk location.
    ds = ds.shuffle(seed=SEED)

    test_len = max(int(EVAL_RATIO * len(ds)), 1)

    train = ds.select(range(len(ds)-2*test_len))
    print("TRAIN", len(train))

    print(ds)

    # Train 
    for shard, left in enumerate(range(0, len(ds), shard_size)):
        print(f"saving shard with dataset indices {left}-{left+shard_size}")
        # note we have to convert "meta" column from string to dict
        train.select(range(left, min(len(train), left+shard_size))).map(
            fix_meta, 
            num_proc=args.cpus,
        ).remove_columns(
                columns_to_remove
        ).to_json(
                os.path.join(outdir, "train", f"{name}_{str(shard).zfill(3)}.jsonl"), 
                lines=True,
                num_proc=args.cpus,
        )

    # Validation and test
    validation = ds.select(range(len(ds)-2*test_len, len(ds)-test_len))
    test = ds.select(range(len(ds)-test_len, len(ds)))
    print("VALIDATION", len(validation))
    print("TEST", len(test))

    validation.to_json(
            os.path.join(outdir, "validation", f"{name}.jsonl"), 
            lines=True, 
            num_proc=args.cpus
    )

    test.to_json(
            os.path.join(outdir, "test", f"{name}.jsonl"), 
            lines=True, 
            num_proc=args.cpus
    )

if __name__=="__main__": 
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-c", "--cpus", type=int)

    args = parser.parse_args()
    main(args)
