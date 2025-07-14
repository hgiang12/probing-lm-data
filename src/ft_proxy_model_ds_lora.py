import os
import json
import torch
import logging
from itertools import chain
from argparse import ArgumentParser, Namespace

import transformers
import datasets
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)

from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

PROMPT = """Here is a statement:

[TEXT]

Is the above statement correct? Answer: """

# set logging level
logging.basicConfig(level=logging.INFO)

def parse_args() -> Namespace:
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", default="pythia-2.8b")
    parser.add_argument("--save_dir", default="./saved_models")
    parser.add_argument("--data_path", default="")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--optimizer", type=str, default="adamw_torch")
    
    # New args for LoRA
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--bias", type=str, default="lora_only")
    
    return parser.parse_args()

def load_dataset(data_path) -> datasets.Dataset:
    dataset = datasets.load_dataset(
        "json",
        data_files=data_path,
        split="train"
    )

    dataset = dataset.filter(
        lambda x: x.get("label", 0) == 1 and isinstance(x.get("text", ""), str) and x["text"].strip()
    )

    dataset = dataset.map(
        lambda x: {"text": PROMPT.replace("[TEXT]", x["text"])},
        remove_columns=["label", "text"]  
    )
    return dataset


def main(args: Namespace) -> None:
    """Main: Training LLM."""
    
    # Create Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, return_dict=True, torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model.config.use_cache = False
    tokenizer.pad_token = tokenizer.eos_token
    model_name = args.model_path.split("/")[-1]
    
    # LoRA 
    # r: rank, smaller means lighter, common values are 4, 8, 16
    # lora_alpha: LoRA adds the adaptation matrix scaled by alpha / r
    # target_modules: also "k_proj", "o_proj", ...

    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        bias=args.bias,
        target_modules=["gate_proj", "down_proj", "up_proj", "q_proj", "v_proj", "k_proj", "o_proj"],  # or adjust based on model
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Create Dataloaders
    train_dataset = load_dataset(args.data_path)
    text_column_name = "text"

    def tokenize_function(example):
        return tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
    
    sft_config = SFTConfig(
        output_dir=os.path.join(args.save_dir, model_name),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        lr_scheduler_type="constant",
        optim=args.optimizer,
        seed=args.seed,
        bf16=torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="no",
        remove_unused_columns=False,
        
        max_seq_length=512,
        packing=False,
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=sft_config,
    )
    
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(os.path.join(args.save_dir, model_name))

if __name__ == "__main__":
    main(parse_args())
