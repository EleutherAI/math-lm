import os
import random
import subprocess
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import json
from tqdm import tqdm
import sentencepiece as spm

random.seed(7)

TEMP_FILE = "proofsteps_temp.jsonl"
EVAL_RATIO = 0.005


def row_of_name(name: str, sp):
    out = subprocess.run(
        ["lake", "exe", "training_data", name, "--proofstep"],
        capture_output=True,
        text=True,
        cwd="lean-training-data",
    )
    text = out.stdout
    return {
        "text": text,
        "meta": {"mathlib_filename": name, "llama_tokens": len(sp.encode_as_ids(text))},
    }


def train_test_split(path: str):
    base = "data_jsonl"
    filename = "lean_proofsteps.jsonl"
    cum_tokens = 0

    with open(os.path.join(base, "train", filename), "w") as train_file, \
         open(os.path.join(base, "validation", filename), "w") as val_file, \
         open(os.path.join(base, "test", filename), "w") as test_file:

        with open(path, "r") as f:
            for line in f:
                row = json.loads(line)

                cum_tokens += row['meta']['llama_tokens']

                choice = random.random()
                if choice < EVAL_RATIO:
                    dest = val_file
                elif choice < 2 * EVAL_RATIO:
                    dest = test_file
                else:
                    dest = train_file

                dest.write(json.dumps(row) + "\n")
        
    print(f"TOTAL TOKENS: {cum_tokens}")


def main(args):
    if os.path.isfile(TEMP_FILE):
        raise OSError(f"delete {TEMP_FILE}")

    sp = spm.SentencePieceProcessor(model_file=args.vocab)

    data = []
    basepath = "lean-training-data/lake-packages/mathlib/"

    names = []
    for dirpath, dirnames, filenames in os.walk(
        "lean-training-data/lake-packages/mathlib/Mathlib"
    ):
        dirpath = os.path.relpath(dirpath, basepath)
        for filename in filenames:
            if filename.endswith(".lean"):
                mathlib_name = dirpath.replace("/", ".") + f".{filename[:-5]}"

                names.append(mathlib_name)

    if args.max_docs:
        names = names[: args.max_docs]

    print("num files: ", len(names))

    data = []
    pbar = tqdm(total=len(names))

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(row_of_name, name, sp): name for name in names}
        for future in as_completed(futures):
            result = future.result()

            with open(TEMP_FILE, "a+") as f:
                f.write(json.dumps(result) + "\n")

            pbar.update(1)


    train_test_split(TEMP_FILE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--vocab", type=str, help="Path to a sentencepiece tokenizer model"
    )
    parser.add_argument(
        "--max-docs", type=int, required=False, help="Max number of files to compile"
    )

    args = parser.parse_args()

    main(args)
