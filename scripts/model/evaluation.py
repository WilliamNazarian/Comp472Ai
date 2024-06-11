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
from typing import List, Tuple
from scripts.model.types import EvaluationResults


__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_model(model: nn.Module, dataloader: DataLoader) -> EvaluationResults:
    model.eval().to(__device)

    expected_vs_actual_pairs: List[Tuple[int, int]] = []

    # evaluation/testing
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(__device), labels.to(__device)

            # Forward pass
            outputs = model(images)
            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)
            expected = labels.tolist()
            actual = predicted.tolist()
            expected_vs_actual_pairs.extend(list(zip(expected, actual)))

    # generate confusion matrix
    confusion_matrix = np.zeros((4, 4), dtype=int)
    for expected, actual in expected_vs_actual_pairs:
        confusion_matrix[expected, actual] += 1

    return EvaluationResults(raw_tuples=expected_vs_actual_pairs, confusion_matrix=confusion_matrix)
