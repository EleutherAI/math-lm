# Proof Pile v2

This directory contains code for downloading and preprocessing the `proof-pile-v2` dataset. Additionally, the notebook
`analysis.ipynb` contains utilities for visualizing and manually inspecting the data. 

Preprocessed data is hosted [on Huggingface](https://huggingface.co/datasets/zhangirazerbayev/proof-pile-v2-dev). There may be discrepancies between the data on Huggingface and the data produced by these scripts due to one being updated before the other or changes in the upstream data sources.

Instructions for each of the subsets are given below: 
- `AMPS`: Run `download_from_pilev2.sh` (works only on the Stability cluster), then run `python raw_pilev2_to_jsonl.py
  -c $NUM_CPUS`. Data will be saved to the `data_jsonl` directory. 
- `arXiv`: Same as above. 
- `issues_diffs`: Run `download_from_pilev2.sh`, except the Python script is called `filter_issues_diffs.py`. It still
  has a `-c` argument. Note that the issues and diffs subsets must be built AFTER `source_code`.
- `source_code`: Run `python process_source_code.py -c $NUM_CPUS` to build all language subsets. To build only the
  subsets for some specific languages, run `python process_source_code.py -c $NUM_CPUS -l python cpp`, for example. The
  preprocessed data will be saved to `data_jsonl`. The generated files in the `meta_json` directory are
  metadata useful for analysis and building the issues and diffs dataset. 
- `stack_exchange`: Run `python process_stack_exchange.py`. Data is saved in the `data_jsonl` directory.
