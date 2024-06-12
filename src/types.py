import logging
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import numpy.typing as npt

from torch.utils.data import DataLoader
from typing import List, Tuple
from dataclasses import dataclass, field
from typing import Union
from src.utils.confusion_matrix import ConfusionMatrix

cm = ConfusionMatrix
cm_macro = ConfusionMatrix.Macro
cm_micro = ConfusionMatrix.Micro


SchedulerType = Union[
    optim.lr_scheduler.StepLR,
    optim.lr_scheduler.MultiStepLR,
    optim.lr_scheduler.ExponentialLR,
    optim.lr_scheduler.CosineAnnealingLR,
    optim.lr_scheduler.ReduceLROnPlateau,
    optim.lr_scheduler.CyclicLR,
    optim.lr_scheduler.OneCycleLR,
    optim.lr_scheduler.CosineAnnealingWarmRestarts,
    optim.lr_scheduler.LambdaLR
]


@dataclass
class TrainingConfig:
    # Where models will be saved to
    model_name: str
    output_dir: str
    output_logger: Union[logging.Logger, None]

    # Datasets
    training_set_loader: DataLoader
    validation_set_loader: DataLoader
    testing_set_loader: DataLoader

    # Training hyperparameters
    epochs: int
    # learning_rate: float

    classes: List[str]
    model: nn.Module
    criterion: nn.modules.loss._Loss
    optimizer: optim.Optimizer
    scheduler: SchedulerType


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
    confusion_matrix: npt.NDArray[int]

    def get_metrics_table_as_df(self) -> pd.DataFrame:
        macro_precision, macro_recall, macro_f1_score, macro_accuracy = cm_macro.calculate_overall_metrics(
            self.confusion_matrix)
        micro_precision, micro_recall, micro_f1_score, micro_accuracy = cm_micro.calculate_overall_metrics(
            self.confusion_matrix)
        accuracy = (macro_accuracy + micro_accuracy) / 2  # should be the same for both

        data = [
            [macro_precision, macro_recall, macro_f1_score, micro_precision, micro_recall, micro_f1_score, accuracy]]
        tuples = [("macro", "precision"), ("macro", "recall"), ("macro", "f1_score"), ("micro", "precision"),
                  ("micro", "recall"), ("micro", "f1_score"), ("", "accuracy")]

        df = pd.DataFrame(data,
                          index=pd.Index(["model"]),
                          columns=pd.MultiIndex.from_tuples(tuples, names=["", "metrics"]))

        return df

    def get_confusion_matrix_as_df(self) -> pd.DataFrame:
        df = pd.DataFrame(self.confusion_matrix,
                          index=pd.Index(["anger", "engaged", "happy", "neutral"]),
                          columns=pd.Index(["anger", "engaged", "happy", "neutral"]))

        return df

    def get_metrics_per_class_as_df(self) -> pd.DataFrame:
        precisions, recalls, f1_scores, accuracies = cm.calculate_per_class_metrics(self.confusion_matrix)
        array = [precisions, recalls, f1_scores, accuracies]

        df = pd.DataFrame(array,
                          index=pd.Index(["precision", "recall", "f1_score", "accuracy"]),
                          columns=pd.Index(["anger", "engaged", "happy", "neutral"]))
        return df
