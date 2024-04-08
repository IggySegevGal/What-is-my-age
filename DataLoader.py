import os
from glob import glob
import torch
import pandas as pd
import numpy as np
from PIL import Image


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, transforms=None, num_classes=1):
        self.df = df
        self.transforms = transforms
        self.len = df.shape[0]
        self.base_path = "./data"
        self.num_classes = num_classes
        self.class_size = 50//num_classes
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_idx = row['path']
        # print(image_idx)
        age = row["Patient Age"]
        if self.num_classes > 1:
            age = (age // self.class_size) - (20//self.class_size) # init classes between 0-5 (assuming ages are between 20-70)

        # find image's path:
        parts = image_idx.split('_')
        image_id = parts[0]  # The image id is the part before the first underscore
        folder_id = parts[1].split('.')[0]  # The folder id is the part between the underscore and the period

        # img_path = os.path.join(self.base_path,f"images_{folder_id}","images", image_idx)
        # img_path = f"{self.base_path}/"
        X = Image.open(image_idx).convert('RGB')
        y = torch.tensor(age, dtype=int)


        if self.transforms:
            X = self.transforms(X)
        return X, y


