import torch.nn as nn
import torch.optim as optim
import numpy as np
import numpy.typing as npt

from torch.utils.data import DataLoader
from typing import List, Tuple
from dataclasses import dataclass, field


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


# "Static" class for reasoning about an N x N confusion matrix
class ConfusionMatrx:
    @classmethod
    def total(cls, confusion_matrix: npt.NDArray[int]):
        return np.sum(confusion_matrix)

    @classmethod
    def calculate_metrics(cls, confusion_matrix: npt.NDArray[int]):
        true_positives = np.diag(confusion_matrix)
        false_positives = np.sum(confusion_matrix, axis=0) - true_positives
        false_negatives = np.sum(confusion_matrix, axis=1) - true_positives
        true_negatives = np.sum(confusion_matrix) - (true_positives + false_positives + false_negatives)
        return true_positives, false_positives, true_negatives, false_negatives

    @classmethod
    def calculate_precision(cls, confusion_matrix: npt.NDArray[int], indices=None):
        true_positives, false_positives, _, _ = cls.calculate_metrics(confusion_matrix)

        if indices is None:
            true_positives = np.sum(true_positives)
            false_positives = np.sum(false_positives)
            return true_positives / (true_positives + false_positives)
        elif isinstance(indices, list) and all(isinstance(item, int) for item in indices):
            precisions_per_class = true_positives / (true_positives + false_positives)
            return precisions_per_class[indices]
        else:
            raise ValueError("Incorrect type, pass one of the following: \"List[int]\", or \"None\"")

    @classmethod
    def calculate_recall(cls, confusion_matrix: npt.NDArray[int], indices=None):
        true_positives, _, _, false_negatives = cls.calculate_metrics(confusion_matrix)

        if indices is None:
            true_positives = np.sum(true_positives)
            false_negatives = np.sum(false_negatives)
            return true_positives / (true_positives + false_negatives)
        elif isinstance(indices, list) and all(isinstance(item, int) for item in indices):
            recalls_per_class = true_positives / (true_positives + false_negatives)
            return recalls_per_class[indices]
        else:
            raise ValueError("Incorrect type, pass one of the following: \"List[int]\", or \"None\"")

    @classmethod
    def calculate_accuracy(cls, confusion_matrix: npt.NDArray[int], indices=None):
        true_positives, false_positives, true_negatives, false_negatives = cls.calculate_metrics(confusion_matrix)

        if indices is None:
            true_positives = np.sum(true_positives)
            false_positives = np.sum(false_positives)
            true_negatives = np.sum(true_negatives)
            false_negatives = np.sum(false_negatives)
            return (true_positives + true_negatives) / (
                    true_positives + false_positives + true_negatives + false_negatives)
        elif isinstance(indices, list) and all(isinstance(item, int) for item in indices):
            accuracy_per_class = (true_positives + true_negatives) / (
                        true_positives + false_positives + true_negatives + false_negatives)
            return accuracy_per_class[indices]
        else:
            raise ValueError("Incorrect type, pass one of the following: \"List[int]\", or \"None\"")

    @classmethod
    def calculate_f1_score(cls, confusion_matrix: npt.NDArray[int], indices=None):
        precision = cls.calculate_precision(confusion_matrix, indices)
        recall = cls.calculate_recall(confusion_matrix, indices)
        f1_score_per_class = 2 * (precision * recall) / (precision + recall)

        if indices is None:
            return np.sum(f1_score_per_class)
        elif isinstance(indices, list) and all(isinstance(item, int) for item in indices):
            return f1_score_per_class[indices]
        else:
            raise ValueError("Incorrect type, pass one of the following: \"List[int]\", or \"None\"")


@dataclass
class TrainingLogger:
    training_confusion_matrix_history: List[npt.NDArray[int]] = field(default_factory=list)
    validation_confusion_matrix_history: List[npt.NDArray[int]] = field(default_factory=list)

    training_precision_history: List[npt.NDArray[int]] = field(default_factory=list)
    training_recall_history: List[npt.NDArray[int]] = field(default_factory=list)
    training_accuracy_history: List[npt.NDArray[int]] = field(default_factory=list)
    training_f1_score_history: List[npt.NDArray[int]] = field(default_factory=list)

    validation_precision_history: List[npt.NDArray[int]] = field(default_factory=list)
    validation_recall_history: List[npt.NDArray[int]] = field(default_factory=list)
    validation_accuracy_history: List[npt.NDArray[int]] = field(default_factory=list)
    validation_f1_score_history: List[npt.NDArray[int]] = field(default_factory=list)


@dataclass(frozen=True)  # 'frozen=True' means type is immutable
class EvaluationResults:
    raw_tuples: List[Tuple[int, int]]
    confusion_matrix: npt.NDArray[int]

    def total_tests(self):
        return ConfusionMatrx.total(self.confusion_matrix)

    def calculate_metrics(self):
        return ConfusionMatrx.calculate_metrics(self.confusion_matrix)

    def calculate_precision(self, indices=None):
        return ConfusionMatrx.calculate_precision(self.confusion_matrix, indices)

    def calculate_recall(self, indices=None):
        return ConfusionMatrx.calculate_recall(self.confusion_matrix, indices)

    def calculate_accuracy(self, indices=None):
        return ConfusionMatrx.calculate_accuracy(self.confusion_matrix, indices)

    def calculate_f1_score(self, indices=None):
        return ConfusionMatrx.calculate_f1_score(self.confusion_matrix, indices)
