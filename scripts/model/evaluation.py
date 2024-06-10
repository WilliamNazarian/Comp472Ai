import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.cuda as cuda
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split


__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_model(model: nn.Module, dataloader: DataLoader):
    model.eval().to(__device)

    # Initialize variables to track correct predictions and total samples
    correct = 0
    total = 0

    # Disable gradient computation for evaluation
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(__device), labels.to(__device)

            # Forward pass
            outputs = model(images)
            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)
            # Update total number of labels
            total += labels.size(0)
            # Update the number of correct predictions
            correct += (predicted == labels).sum().item()

    # Calculate and print the accuracy
    accuracy = (correct / total) * 100
    print('Test Accuracy of the model on  test images: {:.2f} %'.format(accuracy))
