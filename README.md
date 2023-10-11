# `Llemma`: an open language model for mathematics


Repository for [Llemma: an open language model for mathematics [Azerbayev et al 2023]]().

This repository hosts data and training code related to the following artifacts:

- [**Llemma 7b**]() [TODO: huggingface links when available]
- [**Llemma 34b**]()
- [**Proof-Pile-2**]()
    - [**AlgebraicStack**]()


This repository also contains submodules related to the overlap, fine-tuning, and theorem proving experiments described in the paper.
Additional evaluation code is in a [fork of the Eleuther LM Evaluation Harness](https://github.com/wellecks/lm-evaluation-harness).

## Overview 
This repository contains the following directories
- `proof_pile_2`: scripts for downloading and preprocessing data.
- `gpt-neox`: git submodule containing a modified branch of `EleutherAI/gpt-neox`
- `finetunes`: git submodule containing scripts for the fine-tuning experiments
- `llemma_formal2formal`: git submodule containing scripts for the formal2formal experiments
- `overlap`: git submodule containing the overlap and memorization analysis 

Because this project contains submodules, you should clone this project with the `--recurse-submodules` flag or, alternatively, run `git submodule update --init --recursive` from within the project directory after cloning the project. After running `git pull`, you should also run `git submodule update`.

## Citation
Please cite the following:
```
@article{azerbayev2023llemma,
    title={Llemma: an open language model for mathematics},
    author={Zhangir Azerbayev and Hailey Schoelkopf and Keiran Paster and Marco Dos Santos and Stephen McAleer and Albert Q. Jiang and Jia Deng and Stella Biderman and Sean Welleck},
    eprint={xyz.xyz},
    archivePrefix={arXiv}
    year={2023}
}
```
