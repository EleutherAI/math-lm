import os
import json
from copy import deepcopy
import fire
from pathlib import Path

import random 
random.seed(1337)

PATHS = [
    "human-filtered-stack-exchange/cstheory_humanfiltered.jsonl",
    "human-filtered-khan/khan.jsonl",
    "human-edited-hendrycksmath/math.jsonl",
    "human-filtered-stack-exchange/math_humanfiltered.jsonl",
    "human-filtered-stack-exchange/math_overflow_humanfiltered.jsonl",
    "human-filtered-stack-exchange/physics_humanfiltered.jsonl",
    "human-filtered-stack-exchange/proofassistants_humanfiltered.jsonl", 
]

TOTAL_EXAMPLES=1800
PROP_VAL = 1/18
PROP_TEST = 1/18

def process_row(x):
    old_meta = x["meta"]
    new_meta = dict() 
    if "id" in old_meta:
        new_meta["id"] = old_meta["id"]
        if "level" in old_meta:
            new_meta["source"] = "hendrycks_math"
        elif "openai_response" in old_meta:
            new_meta["source"] = "amps_khan_academy"
        else:
            raise ValueError("couldn't classify schema")
    elif "post_id" in old_meta:
        new_meta["id"] = old_meta["post_id"]
        new_meta["source"] = "stack_exchange"
    else:
        raise ValueError("Missing id")

    new_meta["misc"] = str(deepcopy(old_meta))

    x["meta"] = new_meta

    return x
    

def merge(destdir, paths=PATHS):
    all_data = {"train": [], "validation": [], "test": []}
    for path in paths:
        with open(path) as f:
            this_data = [process_row(json.loads(x)) for x in f]
        
        val_boundary = int((1-PROP_VAL-PROP_TEST) * len(this_data))
        test_boundary = int((1-PROP_TEST) * len(this_data))
        
        all_data["train"] += this_data[:val_boundary]
        all_data["validation"] += this_data[val_boundary:test_boundary]
        all_data["test"] += this_data[test_boundary:]

    for k in ("train", "validation", "test"):
        random.shuffle(all_data[k])

    unsplit = [x for k,v in all_data.items() for x in v]

    print(len(unsplit))
    assert len(unsplit)==TOTAL_EXAMPLES

    val_boundary = int((1-PROP_VAL-PROP_TEST) * len(unsplit))
    test_boundary = int((1-PROP_TEST) * len(unsplit))

    all_data["train"] = unsplit[:val_boundary]
    all_data["validation"] = unsplit[val_boundary:test_boundary]
    all_data["test"] = unsplit[test_boundary:]

    for k in ("train", "validation", "test"):
        savepath = Path(destdir) / k
        if os.path.isdir(savepath):
            raise OSError
        savepath.mkdir(parents=True)
        with open(savepath / "math-instruct.jsonl", "w") as f:
            [f.write(json.dumps(x) + "\n") for x in all_data[k]]

if __name__=="__main__":
    fire.Fire(merge)