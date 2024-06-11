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
    def calculate_confusion_matrix_metrics(cls, confusion_matrix: npt.NDArray[int]):
        true_positives = np.diag(confusion_matrix)
        false_positives = np.sum(confusion_matrix, axis=0) - true_positives
        false_negatives = np.sum(confusion_matrix, axis=1) - true_positives
        true_negatives = np.sum(confusion_matrix) - (true_positives + false_positives + false_negatives)
        return true_positives, false_positives, true_negatives, false_negatives

    @classmethod
    def calculate_per_class_metrics(cls, confusion_matrix: npt.NDArray[int]):
        true_positives, false_positives, true_negatives, false_negatives = (
            cls.calculate_confusion_matrix_metrics(confusion_matrix))

        precisions_per_class = true_positives / (true_positives + false_positives)
        recalls_per_class = true_positives / (true_positives + false_negatives)
        accuracy_per_class = (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives)
        f1_score_per_class = 2 * (precisions_per_class * recalls_per_class) / (precisions_per_class + recalls_per_class)

        return precisions_per_class, recalls_per_class, f1_score_per_class, accuracy_per_class

    class Macro:
        @classmethod
        def calculate_overall_metrics(cls, confusion_matrix: npt.NDArray[int]):
            true_positives, false_positives, false_negatives, false_negatives = (
                ConfusionMatrx.calculate_per_class_metrics(confusion_matrix))

            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
            f1_score = 2 * (precision * recall) / (precision + recall)
            accuracy = np.sum(true_positives) / np.sum(confusion_matrix)

            mean_precision = np.mean(precision)
            mean_recall = np.mean(recall)
            mean_f1_score = np.mean(f1_score)

            return mean_precision, mean_recall, mean_f1_score, accuracy

    class Micro:
        @classmethod
        def calculate_overall_metrics(cls, confusion_matrix: npt.NDArray[int]):
            true_positives, false_positives, false_negatives, _ = (
                ConfusionMatrx.calculate_confusion_matrix_metrics(confusion_matrix))

            tp_sum = np.sum(true_positives)
            fp_sum = np.sum(false_positives)
            fn_sum = np.sum(false_negatives)

            precision = tp_sum / (tp_sum + fp_sum)
            recall = tp_sum / (tp_sum + fn_sum)
            f1_score = 2 * (precision * recall) / (precision + recall)
            accuracy = np.sum(true_positives) / np.sum(confusion_matrix)

            return precision, recall, f1_score, accuracy
        pass


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