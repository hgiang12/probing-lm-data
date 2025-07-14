import os
import json
import torch
import argparse
import shutil
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from transformers import BlipProcessor, BlipForQuestionAnswering

QUESTION_TEXT = "What is shown in the image?"

class Hook:
    def __init__(self):
        self.out = None

    def __call__(self, module, inputs, outputs):
        self.out = outputs[0]  # BLIP vision encoder layer output

def get_activations(images, questions, processor, model, layers):
    hooks, handles = [], []

    for layer in layers:
        hook = Hook()
        handle = model.vision_model.encoder.layers[layer].register_forward_hook(hook)
        hooks.append(hook)
        handles.append(handle)

    # Chỉ lấy pixel values từ processor
    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(model.device)

    # Forward qua vision encoder thôi
    with torch.no_grad():
        model.vision_model(pixel_values=pixel_values)

    activations = {
        layer: hook.out.mean(dim=1).cpu() for layer, hook in zip(layers, hooks)
    }

    for handle in handles:
        handle.remove()

    return activations

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--output_dir", default="./blip_qa_acts")
    parser.add_argument("--dataset", required=True)
    return parser.parse_args()

def load_dataset(jsonl_path, images_dir):
    images, questions = [], []
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
    ])

    with open(jsonl_path, "r") as f:
        for line in tqdm(f, desc="Loading dataset"):
            item = json.loads(line)
            image_path = os.path.join(images_dir, item["image"])
            try:
                img = Image.open(image_path).convert("RGB")
                images.append(transform(img))
                questions.append(QUESTION_TEXT)
            except Exception as e:
                print(f"Skipping {image_path}: {e}")
                continue

    return images, questions

if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BlipForQuestionAnswering.from_pretrained(args.model_path).to(device)
    processor = BlipProcessor.from_pretrained(args.model_path)

    model_name = os.path.basename(args.model_path.rstrip("/"))
    save_dir = os.path.join(args.output_dir, model_name, args.dataset)

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    images, questions = load_dataset(args.dataset_path, args.images_dir)

    layers = list(range(len(model.vision_model.encoder.layers)))

    batch_size = 5

    for idx in tqdm(range(0, len(images), batch_size), desc="Extracting activations"):
        batch_images = images[idx:idx + batch_size]
        batch_questions = questions[idx:idx + batch_size]
        acts = get_activations(batch_images, batch_questions, processor, model, layers)
        for layer, act in acts.items():
            torch.save(act, os.path.join(save_dir, f"layer_{layer}_{idx}.pt"))
