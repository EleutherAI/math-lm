#!/bin/bash
#SBATCH --job-name=buildsourcecodedataset
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/fsx/mathlm0/math-lm/proof-pile-v2/source_code/job.out
#SBATCH --error=/fsx/mathlm0/math-lm/proof-pile-v2/source_code/job.out
#SBATCH --comment=eleuther

# quick script for building the source dataset on the stability cluster

source /fsx/gpt-neox/conda/bin/activate mathlm1

cd /fsx/mathlm0/math-lm/proof-pile-v2/source_code/

python process_source_code.py

