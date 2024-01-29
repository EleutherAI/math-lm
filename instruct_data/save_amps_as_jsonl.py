import fire
import json
import os
from pathlib import Path
from tqdm import tqdm
import math

NUM_SHARDS = 8

def convert_khan_to_jsonl(amps_path: str, dest_dir: str):
    path = Path(amps_path) / "khan"

    data = []

    for file in tqdm(path.rglob('*.json')):
        with open(file) as fle:
            data.append(json.load(fle))
    
    print(f"Found {len(data)} rows")

    prefix = Path(dest_dir)
    chunk_size = math.ceil(len(data)/NUM_SHARDS)

    for i in range(NUM_SHARDS):
        savepath = prefix / f"raw_{i}.jsonl"
        with open(savepath, "w") as fle:
            for row in data[i*chunk_size:(i+1)*chunk_size]:
                fle.write(json.dumps(row) + "\n") 

if __name__=="__main__":
    fire.Fire(convert_khan_to_jsonl)

        
