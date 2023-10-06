# Proof-Pile-2

This directory contains code for downloading Proof-Pile-2 data from its original source and reapplying our filtering and preprocessing methodology. 
- AlgebraicStack: the `algebraic_stack` directory contains code for reproducing the AlgebraicStack. This includes refiltering the Stack, downloading certain language subsets directly from Github, reproducing the Lean proofsteps dataset, and reproducing the Isabelle proofsteps dataset.
- ArXiv: we do not apply any further preprocessing to the Redpajama ArXiv subset, however, our downloading script replicates the train-validation-test split used in our paper. 
- OpenWebMath: we apply no filtering or preprocessing to OpenWebMath, therefore this repository contains no corresponding code.

