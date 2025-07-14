import os
import json
import argparse
import random
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Any, Dict
from datasets import Dataset
from torchvision import transforms

import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    TrainingArguments,
    Trainer,
    set_seed,
)

# Prompt template
PROMPT = """Here is a description of the image:

[TEXT]

Is the above description correct? Answer: """

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--save_dir", type=str, default="./output")
    return parser.parse_args()

def set_random_seed(seed):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_transform():
    return transforms.Compose([
        transforms.Resize((384, 384), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])

def load_dataset(data_path, images_dir):
    def process_example(example):
        # Only process the sample if the label is 1
        if example["label"] != 1:
            return None

        image_path = os.path.join(images_dir, example["image"])
        if not os.path.exists(image_path):
            print(f"[WARN] Missing image: {image_path}")
            return None
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[ERROR] Failed to open {image_path}: {e}")
            return None

        # Replace caption with prompt
        caption = PROMPT.replace("[TEXT]", example["caption"])
        return {"image": image, "caption": caption}

    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = [json.loads(line) for line in f]

    processed = [process_example(e) for e in raw_data]
    processed = [e for e in processed if e is not None]

    if not processed:
        raise ValueError("No valid examples found.")

    print(f"[INFO] Loaded {len(processed)} valid image-caption pairs.")
    return Dataset.from_list(processed)

@dataclass
class DataCollator:
    processor: BlipProcessor

    def __call__(self, batch: list[Dict[str, Any]]) -> Dict[str, Any]:
        images = [x["image"] for x in batch]
        captions = [x["caption"] for x in batch]
        encoding = self.processor(
            images=images,
            text=captions,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        encoding["labels"] = encoding["input_ids"]
        return encoding

def main(args):
    set_random_seed(args.seed)

    processor = BlipProcessor.from_pretrained(args.model_path)
    model = BlipForConditionalGeneration.from_pretrained(args.model_path)
    model_name = args.model_path.split("/")[-1]

    dataset = load_dataset(args.data_path, args.images_dir)
    
    training_args = TrainingArguments(
        output_dir=os.path.join(args.save_dir, model_name),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_steps=10,
        remove_unused_columns=False,
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollator(processor),
    )

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("[WARN] Training was interrupted manually.")
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        return

    # Save model and processor
    print(f"[INFO] Saving model to {args.save_dir}")
    trainer.save_model()
    processor.save_pretrained(os.path.join(args.save_dir, model_name))

if __name__ == "__main__":
    main(parse_args())