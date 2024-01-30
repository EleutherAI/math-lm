import json
from copy import deepcopy
import random
from typing import List, Dict

from tqdm import tqdm
import fire

random.seed(133742420)

def process_row(row: List):
    row["input"] = row.pop("problem")
    row["output"] = "\n\n".join(row.pop("hints"))
    row["meta"] = {"id": row.pop("id"), "category": row.pop("category")}

    for key in ("q_hierarchy", "title", "hint_count"):
        if key in row:
            row.pop(key)

    return row

def select_row_of_cat(cat: List[Dict], cat_name: str):
    # eliminates multi-choice problems
    cat = [x for x in cat if "Choose 1 answer" not in x["problem"]]
    for i in range(len(cat)):
        cat[i]["category"] = cat_name
    return random.choice(cat) if cat else []

def prefilter(inputpath: str, destpath: str):
    with open(inputpath) as fle:
        pre_data = json.load(fle)

    rows_data = [
        x for x in 
        (select_row_of_cat(pre_data[cat], cat_name=cat) for cat in tqdm(pre_data))
        if x
        ]

    post_data = [process_row(x) for x in rows_data]

    print(f"Found {len(post_data)} examples")

    with open(destpath, "w") as f:
        [f.write(json.dumps(x) + "\n") for x in post_data]

if __name__=="__main__":
    fire.Fire(prefilter)
