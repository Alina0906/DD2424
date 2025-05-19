import os
import re

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


def load_data(dataset):
    data = []
    with open(os.path.join('annotations', dataset + '.txt'), 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue    

            parts = re.split(r"\s+", line)

            if len(parts) == 5:
                image_name = "_".join(parts[:2])
                class_id, species, breed_id = parts[2], parts[3], parts[4]
            else:
                image_name, class_id, species, breed_id = parts
            
            data.append({
                "Image": f"images/{image_name}.jpg",
                "CLASS_ID": int(class_id)-1,
                "SPECIES": int(species)-1,
                "BREED_ID": int(breed_id)
            })
    
    dataframe = pd.DataFrame(data)
    return dataframe


def get_processor(processor):
    mean, std, size = processor.image_mean, processor.image_std, processor.size["height"]
    return mean, std, size


def vit_transforms(processor):
    mean, std, size = get_processor(processor)
    
    vit_transform = transforms.Compose([
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return vit_transform


def train_test_shuffle(train, test):
  combined = pd.concat([train, test], ignore_index=True)
  shuffled = combined.sample(frac=1, random_state=42).reset_index(drop=True)

  new_train, new_test = train_test_split(
    shuffled, test_size=0.2, shuffle=True, random_state=42
  )
  
  return new_train, new_test


class VitDataset(Dataset):
    def __init__(self, dataframe, transform=None, task="class"):
        self.dataframe = dataframe
        self.transform = transform
        self.task = task

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = row["Image"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        if self.task == "species":
            label = row["SPECIES"]
        else:
            label = row["CLASS_ID"]
            
        return {
            "pixel_values": image,
            "labels": label
        }