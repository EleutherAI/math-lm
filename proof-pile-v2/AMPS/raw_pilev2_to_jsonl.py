import sys
import os
import json
import datasets 
import argparse
from tqdm import tqdm
import code
from pathlib import Path
import ast

name = "AMPS"
inpath = "raw_pilev2/AMPS"
outdir = "data_jsonl"
shard_size = 100_000

def main(args): 
    Path(os.path.join(outdir, "train")).mkdir(exist_ok=True, parents=True)

    print("loading dataset from disk...")
    train = datasets.load_from_disk(inpath, keep_in_memory=False)

    print("TRAIN", len(train))

    print(train)

    # Train 
    for shard, left in enumerate(range(0, len(train), shard_size)):
        print(f"saving shard with dataset indices {left}-{left+shard_size}")
        # note we have to convert "meta" column from string to dict
        train.select(range(left, min(len(train), left+shard_size))).map(
            lambda x: {"meta": ast.literal_eval(x["meta"])}, 
            num_proc=args.cpus
        ).to_json(
                os.path.join(outdir, "train", f"{name}_{str(shard).zfill(2)}.jsonl"), 
                lines=True,
                num_proc=args.cpus,
        )
        break

if __name__=="__main__": 
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-c", "--cpus", type=int)

    args = parser.parse_args()
    main(args)
