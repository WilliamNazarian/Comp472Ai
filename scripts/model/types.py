import torch.nn as nn
import torch.optim as optim
import numpy as np
import numpy.typing as npt

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


@dataclass(frozen=True)  # 'frozen=True' means type is immutable
class EvaluationResults:
    raw_tuples: List[Tuple[int, int]]  # List of tuples (a, b), where 'a' represents the PREDICTED class, while 'b' represents the ACTUAL class
    confusion_matrix: npt.NDArray[int]

    def total_tests(self):
        return len(self.raw_tuples)

    def calculate_metrics(self):
        true_positives = np.diag(self.confusion_matrix)
        false_positives = np.sum(self.confusion_matrix, axis=0) - true_positives
        false_negatives = np.sum(self.confusion_matrix, axis=1) - true_positives
        true_negatives = np.sum(self.confusion_matrix) - (true_positives + false_positives + false_negatives)

        return true_positives, false_positives, true_negatives, false_negatives

    def calculate_precision(self):
        true_positives, false_positives, _, _ = self.calculate_metrics()
        true_positives = np.sum(true_positives)
        false_positives = np.sum(false_positives)

        return true_positives / (true_positives + false_positives)

    def calculate_recall(self):
        true_positives, _, _, false_negatives = self.calculate_metrics()
        true_positives = np.sum(true_positives)
        false_negatives = np.sum(false_negatives)

        return true_positives / (true_positives + false_negatives)

    def calculate_accuracy(self):
        true_positives, false_positives, true_negatives, false_negatives = self.calculate_metrics()
        true_positives = np.sum(true_positives)
        false_positives = np.sum(false_positives)
        true_negatives = np.sum(true_negatives)
        false_negatives = np.sum(false_negatives)

        return (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives)
