#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import torch

# Map string labels to integers
label_to_int = {'anger': 0, 'happy': 1, 'engaged': 2, 'neutral': 3}

class FacialEmotionDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.df.iloc[idx, 0])
        print(f"Trying to open {img_name}")  # Print the path being accessed

        # Skip hidden directories and files including .ipynb_checkpoints
        if any(part.startswith('.') for part in img_name.split(os.path.sep)):
            print(f"Skipping hidden file or directory: {img_name}")
            return self.__getitem__((idx + 1) % len(self))

        try:
            image = Image.open(img_name).convert('L')  # Ensure image is grayscale
        except Exception as e:
            print(f"Failed to open {img_name}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        label = self.df.iloc[idx, 1]  # Use the second column for the label
        label = label_to_int[label]  # Convert string label to integer
        label = torch.tensor(label, dtype=torch.long)  # Ensure label is a tensor

        if self.transform:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.Resize((90, 90)),
    transforms.ToTensor()
])

def my_dataset():
    # Create the dataset
    root_dir = "C:\\Users\\WILLIAM\\Desktop\\Comp472\\Comp472AiGithub\\part1\\cleaned_images"  # Correct path to the images
    csv_file = "C:\\Users\\WILLIAM\\Desktop\\Comp472\\Comp472AiGithub\\part1\\Combined_Labels_DataFrame.csv"  # Correct path to the CSV file

    dataset = FacialEmotionDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)

    # Split the dataset into training and validation sets
    train_indices, val_indices = train_test_split(list(range(len(dataset))), test_size=0.25, random_state=42)
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    return train_loader, val_loader