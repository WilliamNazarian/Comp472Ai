import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # Datasets
    training_set_loader: DataLoader
    validation_set_loader: DataLoader
    testing_set_loader: DataLoader

    # Training hyperparameters
    epochs: int
    learning_rate: float

    classes: List[str]
    model: nn.Module
    criterion: nn.modules.loss._Loss
    optimizer: optim.Optimizer


@dataclass
class EvaluationResults:
    # List of tuples (a, b), where 'a' represents the PREDICTED class, while 'b' represents the ACTUAL class
    true_positives: List[Tuple[int, int]]
    false_positives: List[Tuple[int, int]]
    true_negatives: List[Tuple[int, int]]
    false_negatives: List[Tuple[int, int]]
