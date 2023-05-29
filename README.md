# math-lm
Repository for the Math-LM project, an open-source replication of the
[Minerva](https://arxiv.org/abs/2206.14858) model. This repository hosts data and model training code. Evaluation code is hosted in a [fork of the lm-evaluation-harness](https://github.com/wellecks/lm-evaluation-harness).

This project contains the following directories
- `analysis`: scaling law analysis of training runs. 
- `gpt-neox-1.0`: a fork of GPT-NeoX 1.0. 
- `proof-pile-v2`: scripts for downloading and preprocessing data. 
- `task-finetunes`: scripts for fine-tuning models on task-specific datasets, such as MATH or GSM8k. 
