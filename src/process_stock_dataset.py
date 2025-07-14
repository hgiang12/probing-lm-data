from datasets import load_dataset
import os
from PIL import Image
import json
import random

# Number of samples
num_dev = 400   
num_test = 1600      
num_train = 200     

dataset = load_dataset('agentlans/stock-photos-asian-people')

# Output directory
output_dir = 'data/images'
os.makedirs(output_dir, exist_ok=True)

# dev + test 
num_total = num_dev + num_test
samples = []
for idx in range(num_total):
    image_data = dataset['train'][idx]['image']
    caption = dataset['train'][idx]['caption']
    label = idx % 2  # label ratio 1:1

    image_name = f'image_{idx}.jpg'
    image_path = os.path.join(output_dir, image_name)
    image_data.save(image_path)

    samples.append({'image': image_name, 'caption': caption, 'label': label})

label_0_samples = [s for s in samples if s['label'] == 0]
label_1_samples = [s for s in samples if s['label'] == 1]
random.shuffle(label_0_samples)
random.shuffle(label_1_samples)

n_dev_per_label = num_dev // 2
n_test_per_label = num_test // 2

dev_samples = label_0_samples[:n_dev_per_label] + label_1_samples[:n_dev_per_label]
test_samples = label_0_samples[n_dev_per_label:n_dev_per_label + n_test_per_label] + \
               label_1_samples[n_dev_per_label:n_dev_per_label + n_test_per_label]
random.shuffle(dev_samples)
random.shuffle(test_samples)

# train
train_samples = []
start_idx = num_total
for idx in range(start_idx, start_idx + num_train):
    image_data = dataset['train'][idx]['image']
    caption = dataset['train'][idx]['caption']
    label = idx % 2  # label ratio 1:1

    image_name = f'image_{idx}.jpg'
    image_path = os.path.join(output_dir, image_name)
    image_data.save(image_path)

    train_samples.append({'image': image_name, 'caption': caption, 'label': label})

# save
def save_jsonl(samples, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

save_jsonl(dev_samples, 'data/stock_dev.jsonl')
save_jsonl(test_samples, 'data/stock_test.jsonl')
save_jsonl(train_samples, 'data/stock_train.jsonl')

print(f"Saved {len(dev_samples)} dev, {len(test_samples)} test, {len(train_samples)} train samples.")