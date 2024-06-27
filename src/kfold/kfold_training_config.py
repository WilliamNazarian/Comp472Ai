import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets

from typing import Type, List, Callable, Tuple
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

    model_type: Type[torch.nn.Module]

    # Function that returns hyperparameters, since some are tied to the instance of the model itself
    # model -> (criterion, optimizer, scheduler)
    generate_hyper_parameters: Callable[[torch.nn.Module], Tuple[nn.modules.loss._Loss, optim.Optimizer, SchedulerType]]
