import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets

from typing import Type, List
from dataclasses import dataclass
from typing import Union
from src.types import SchedulerType


@dataclass
class KFoldTrainingConfig:
    # Where models will be saved to
    output_dir: str
    output_logger: Union[logging.Logger, None]

    # Datasets
    dataset: datasets.ImageFolder
    classes: List[str]

    # Training hyperparameters
    num_folds: int
    epochs_per_fold: int
    initial_learning_rate: float
    patience: int

    model_type: Type[torch.nn.Module]
    criterion: nn.modules.loss._Loss
    optimizer: optim.Optimizer
    scheduler: SchedulerType
