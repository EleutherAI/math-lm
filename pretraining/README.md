# LLeMA Pretraining

This subfolder contains instructions to replicate pretraining of the LLeMA models. 

Training was performed across 256 A100 GPUs using the GPT-NeoX library. We include configuration files and sample SLURM job script for the library to replicate training on a SLURM-managed cluster.


## Replicating Training


### Set up environment

We provide a file containing a dump of our training environment.

You can install all required packages via
```bash
pip install -r env_dump.txt
```
Make sure you are installing https://github.com/EleutherAI/DeeperSpeed/tree/new-fix for your DeepSpeed version and install fused kernels for GPT-NeoX via `python ./megatron/fused_kernels/setup.py install` from within your GPT-NeoX install.


### Converting Llama 2 checkpoints into NeoX format

First, download CodeLlama 7b or 34b from the Meta AI repo and rename the download folder to 7B or 34B within the CodeLlama repository.

Then, to convert either model into the format expected by GPT-NeoX for checkpoints:

Sample command for 7b Meta->NeoX format:
```bash
python convert_raw_llama_weights_to_hf.py --input_dir /path/to/codellama/repo --config_file /path/to/this/repo/math-lm/pretraining/llemma_7b.yml --output_dir /path/to/save/into/ --num_output_shards {TP_DEGREE, we use 2}
```

Sample command for 34b Meta->NeoX format:
(Requires large amounts of GPU VRAM or CPU RAM. Pass `CUDA_VISIBLE_DEVICES=""` to perform conversion on CPU. 34b conversion may take a while)
```bash
CUDA_VISIBLE_DEVICES="" python convert_raw_llama_weights_to_hf.py --input_dir /path/to/codellama/repo --config_file /path/to/this/repo/math-lm/pretraining/llemma_34b.yml --output_dir /path/to/save/into/ --num_output_shards {TP_DEGREE, we use 8}
```


### Check Out Codebase

Next, check out the commit used to train the model you are replicating.

* 7b / 34b: https://github.com/EleutherAI/gpt-neox/commit/e59c873ee779df2d7f182deb6ad34f290a077ea4

### Launching Training

Then, edit the provided YML files to set paths based on your own system's saved locations for checkpoints and data files, and edit the SLURM job script as specified (using ) or run the job across multiple nodes using your own system's orchestration.

**Tip**: Note that the global batch size will be scaled by your number of nodes. Therefore, if running on a number of nodes different from 32 you should scale gradient accumulation steps accordingly. 

We used a batch size of 4M tokens. To calculate global batch size, you should compute `seq_len * num_gpus * ( train_microbatch_size_per_gpu * gradient_accumulation_steps) / (model_parallel_size * max(pipeline_parallel_size, 1))` .


## Contents

The files in this folder are as follows:

* `34b_launch_script.sh` contains a skeleton SLURM job script to launch training with NeoX across 32 nodes.

* `data_mixture.yml` contains a list of the domain weights for the final training run.

* `llemma_7b.yml` is a cleaned-up version of the config file used to train Llemma-7b.

* `llemma_34b.yml` is a cleaned-up version of the config file used to train Llemma-34b.

* `env_dump.txt` is a dump of the virtual environmment used in training, created via `pip freeze`.