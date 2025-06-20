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
    Trainer,
    TrainingArguments,
)

import wandb

PROMPT = """Here is a statement:

[TEXT]

Is the above statement correct? Answer: """

# set logging level
logging.basicConfig(level=logging.INFO)

# login wandb
with open("config/wandb_key.txt", "r") as f:
    api_key = f.read().strip()
wandb.login(key=api_key)

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
    
    return parser.parse_args()

def load_dataset(data_path) -> datasets.Dataset:
    def gen(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line:
                    sample = json.loads(line.strip())
                    if sample["label"] == 1:
                        yield {"text": PROMPT.replace("[TEXT]", sample["text"])}
    
    dataset = datasets.Dataset.from_generator(gen, gen_kwargs={"data_path": data_path})
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
    
    # Create Dataloaders
    train_dataset = load_dataset(args.data_path)
    text_column_name = "text"
    
    # Tokenize the datasets
    def tokenize_function(examples):
        examples[text_column_name] = [
            line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
        ]
        return tokenizer(examples[text_column_name])
    
    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    training_args = TrainingArguments(
        seed=args.seed,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        output_dir=os.path.join(args.save_dir, model_name),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        optim=args.optimizer,
        lr_scheduler_type="constant",
        dataloader_drop_last=False,
        bf16=False,
        bf16_full_eval=False,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        save_strategy="no",
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_train,
    )
    
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(os.path.join(args.save_dir, model_name))

if __name__ == "__main__":
    main(parse_args())
