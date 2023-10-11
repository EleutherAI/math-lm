# AlgebraicStack

The following contains instructions for reproducing the AlgebraicStack.

## Source Code
By far the largest source of tokens in the AlgebraicStack is the source code dataset, which is partially filtered from [the Stack](https://huggingface.co/datasets/bigcode/the-stack-dedup) and partially downloaded directly from Github. 

Note that the Stack is a gated Huggingface dataset, therefore your `HUGGING_FACE_TOKEN` environment variable must contain an access key that has permission to access the Stack. Moreover, setting the `GITHUB_ACCESS_TOKEN` environment variable to your Github access token will make the Github download significantly faster.

To reproduce the source code portion of the AlgebraicStack, run
```
./get_source.sh $NUM_CPUS
```
from within this directory. This downloads `jsonl` files to `algebraic_stack/data_jsonl/`. Note that this script will NOT create the Lean proofsteps and the Isabelle proofsteps datasets, as creating those requires nonstandard dependencies. Details on how to create the proofsteps datasets are below.

## Lean Proofsteps
Within the `algebraic_stack` directory, run 
```
git clone https://github.com/semorrison/lean-training-data.git
```
Then, `cd` into the `lean-training-data` directory and follow the setup instructions in `README.md`. Finally, run `process_lean_proofsteps.py --vocab /path/to/llama/tokenizer.model` and you will find the preprocessed data in `data_jsonl`.

## Lean Proofsteps
Within the `algebraic_stack` directory, run 
```
git clone https://github.com/semorrison/lean-training-data.git
```
Then, `cd` into the `lean-training-data` directory and follow the setup instructions in `README.md`. Finally, run `process_lean_proofsteps.py --vocab /path/to/llama/tokenizer.model` and you will find the preprocessed data in `data_jsonl`.

## Isabelle Proofsteps
Isabelle Proofsteps are processed in `./process_isabelle_proofsteps.py`.

To process the Isabelle proofsteps, one needs to extract/download the PISA dataset using code available in [this repository](https://github.com/albertqjiang/Portal-to-ISAbelle/), and then run `./process_isabelle_proofsteps.py` with the correct paths to the AFP extractions, the Standard library extractions and PISA's test set (`universal_test_theorems`).
