# math-lm
Repository for the Math-LM project, an open-source replication of the
[Minerva](https://arxiv.org/abs/2206.14858) model. This repository hosts data and model training code. Evaluation code is hosted in a [fork of the lm-evaluation-harness](https://github.com/wellecks/lm-evaluation-harness).

A WIP build of the proof-pile-v2 dataset is currently hosted [on Huggingface](https://huggingface.co/datasets/zhangirazerbayev/proof-pile-v2-dev).

Note that because this project contains submodules, you should clone this project with the `--recurse-submodules` flag or, alternatively, run `git submodule update --init --recursive` from within the project directory after cloning the project. After running `git pull`, you should also run `git submodule update`.

This project contains the following directories
- `analysis`: scaling law analysis of training runs. 
- `gpt-neox`: git submodule containing a modified branch of `EleutherAI/gpt-neox`
- `proof-pile-v2`: scripts for downloading and preprocessing data. 
- `task-finetunes`: scripts for fine-tuning models on task-specific datasets, such as MATH or GSM8k. 
