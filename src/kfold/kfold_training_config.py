import logging
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import numpy.typing as npt

from torch.utils.data import DataLoader
from typing import List, Tuple
from dataclasses import dataclass, field
from typing import Union
from src.types import SchedulerType
from src.utils.confusion_matrix import ConfusionMatrix


@dataclass
class KFoldTrainingConfig:
    # Where models will be saved to
    model_name: str
    output_dir: str
    output_logger: Union[logging.Logger, None]

    # Datasets
    folds: List[torch.utils.data.Subset]
    classes: List[str]

    # Training hyperparameters
    epochs_per_fold: int
    initial_learning_rate: float
    patience: int

    model: nn.Module
    criterion: nn.modules.loss._Loss
    optimizer: optim.Optimizer
    scheduler: SchedulerType
