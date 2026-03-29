import os
import shutil
import random

dataset_dir = "dataset"
train_dir = "dataset/train"
test_dir = "dataset/test"

split_ratio = 0.8

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

for cls in classes:
    class_path = os.path.join(dataset_dir, cls)

    if cls in ["train", "test"]:
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    split_index = int(len(images) * split_ratio)

    train_images = images[:split_index]
    test_images = images[split_index:]

    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

    for img in train_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, cls, img))

    for img in test_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(test_dir, cls, img))

print("Dataset Split Complete")