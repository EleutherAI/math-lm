import json
import argparse
from tqdm import tqdm

import fire

def filter_file(inputpath: str, destpath: str, min_answer_votes: int):
    with open(inputpath) as fle:
        filtered_data = [
            row for row in 
            (json.loads(x) for x in tqdm(fle))
            if row["meta"]["output_score"] > min_answer_votes
            ]
    with open(destpath, "w") as fle:
        for row in filtered_data:
            json.dump(row, fle)
            fle.write("\n")

if __name__=="__main__":
    fire.Fire(filter_file)



    

