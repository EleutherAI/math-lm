import sys
import os
import yaml 
import pathlib
import math

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

from transformers import AdamW
from transformers import get_cosine_schedule_with_warmup

from transformers.trainer_pt_utils import get_parameter_names


class HFTrainSet(torch.utils.data.Dataset): 
    def __init__(self, hf_dataset, tokenizer, max_length):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx): 
        text = self.hf_dataset[idx]["text"]

        encoded = self.tokenizer(text, 
                max_length = self.max_length, 
                padding='max_length', 
                return_tensors='pt', 
                truncation=True, 
                )

        ids = encoded['input_ids'].squeeze()
        mask = encoded['attention_mask'].squeeze()

        return ids.long(), mask.long()

    def __len__(self): 
        return len(self.hf_dataset)

def data_collator(data):
    return {'input_ids': torch.stack([f[0] for f in data]),
            'attention_mask': torch.stack([f[1] for f in data]),
            'labels': torch.stack([f[0] for f in data])
           }

def create_math_train(): 
    data = load_dataset("competition_math", ignore_verifications=True)

    orig_cols = data['train'].column_names

    data = data.map(lambda example: {
        "text": "Problem: "
        + example["problem"]
        + "\nAnswer: "
        + example["solution"],
        })

    text = data.remove_columns(orig_cols)

    return text["train"]

def create_gsm8k_train(): 
    data = load_dataset("gsm8k", "main")

    orig_cols = data['train'].column_names

    data = data.map(lambda example: {
        "text": "Question: "
        + example["question"]
        + "\nAnswer: "
        + example["answer"],
        })

    text = data.remove_columns(orig_cols)

    return text["train"]

def main(): 
    config_path = sys.argv[1]

    with open(config_path) as f: 
        cfg_text = f.read()
        cfg = yaml.safe_load(cfg_text)

    log_dir = os.path.join(cfg["log_dir"], cfg["experiment_name"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_load_path"])
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg["model_load_path"])
    model.resize_token_embeddings(len(tokenizer))

    match cfg["task"]: 
        case "math": 
            hf_data = create_math_train()
            data = HFTrainSet(hf_data, tokenizer, cfg["max_length"])
        case "gsm8k": 
            hf_data = create_gsm8k_train()
            data = HFTrainSet(hf_data, tokenizer, cfg["max_length"])
        case _ : 
            raise NameError(f"task is not defined")

    # don't apply weight decay to biases
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                if n in decay_parameters],
            "weight_decay": cfg["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters() 
                if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg["lr"])

    match cfg["lr_schedule"]:
        case "cosine": 
            scheduler = get_cosine_schedule_with_warmup(
                    optimizer, 
                    cfg["warmup_steps"], 
                    cfg["train_steps"]
                    )
        case _ : raise NameError("Unrecognized lr schedule")

    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=False)
    with open(os.path.join(log_dir, "config.yml"), "w") as f: 
        f.write(cfg_text)

    training_args = TrainingArguments(output_dir=log_dir,
                              max_steps=cfg["train_steps"],
                              per_device_train_batch_size=cfg["batch_size_per_device"],
                              logging_steps=cfg["log_steps"],
                              save_steps=cfg["save_steps"],
                              remove_unused_columns=False,
                              max_grad_norm=1.0,
                              gradient_accumulation_steps=cfg["gradient_accumulation"],
                              fp16=True, 
                              )

    Trainer(
            model=model, 
            args=training_args, 
            train_dataset = data, 
            data_collator = data_collator, 
            optimizers = (optimizer, scheduler)
            ).train()
    
if __name__=="__main__": 
    main()
