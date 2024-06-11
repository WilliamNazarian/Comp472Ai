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

import numpy.typing as npt
from typing import List, Callable, Tuple
from dataclasses import dataclass

from scripts.model.types import TrainingConfig, TrainingLogger, ConfusionMatrx

__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
__num_classes = 4


def train_model(training_config: TrainingConfig) -> TrainingLogger:
    training_logger = TrainingLogger()

    model = training_config.model
    model.to(__device)

    for epoch in range(training_config.epochs):
        model.train()
        training_confusion_matrix = __train(training_config)
        training_logger.training_confusion_matrix_history.append(training_confusion_matrix)

        model.eval()
        validation_confusion_matrix = __validate(training_config)
        training_logger.validation_confusion_matrix_history.append(validation_confusion_matrix)

        __calculate_metrics(training_logger)
        __print(epoch, training_config.epochs, training_logger)

    return training_logger


def __train(training_config: TrainingConfig) -> npt.NDArray[int]:
    iter_loss = 0.0
    correct = 0
    iterations = 0

    confusion_matrix: npt.NDArray[int] = np.zeros((4, 4), dtype=int)

    training_set_loader = training_config.training_set_loader

    # classes = training_config.classes
    model = training_config.model
    criterion = training_config.criterion
    optimizer = training_config.optimizer

    for i, (items, classes) in enumerate(training_set_loader):
        items = Variable(items).to(__device)
        classes = Variable(classes).to(__device)

        optimizer.zero_grad()
        outputs = model(items)
        loss = criterion(outputs, classes)
        iter_loss += loss.item()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        for expected, actual in list(zip(classes.tolist(), predicted.tolist())):
            if expected > 3 or actual > 3:
                print(f"expected: {expected}")
                print(f"actual: {actual}")

            confusion_matrix[expected, actual] += 1

    # return (iter_loss / iterations), (100 * correct / len(training_set_loader.dataset))
    return confusion_matrix


def __validate(training_config: TrainingConfig) -> npt.NDArray[int]:
    loss = 0.0
    correct = 0
    iterations = 0

    confusion_matrix: npt.NDArray[int] = np.zeros((4, 4), dtype=int)

    validation_set_loader = training_config.validation_set_loader

    # classes = training_config.classes
    model = training_config.model
    criterion = training_config.criterion

    for i, (items, classes) in enumerate(validation_set_loader):
        items = Variable(items).to(__device)
        classes = Variable(classes).to(__device)

        outputs = model(items)  # Do the forward pass
        loss += criterion(outputs, classes).item()  # Calculate the loss

        _, predicted = torch.max(outputs.data, 1)
        for expected, actual in list(zip(classes.tolist(), predicted.tolist())):
            if expected > 3 or actual > 3:
                print(f"expected: {expected}")
                print(f"actual: {actual}")

            confusion_matrix[expected, actual] += 1

    # return (loss / iterations), (correct / len(validation_set_loader.dataset) * 100.0)
    return confusion_matrix


def __calculate_metrics(training_logger: TrainingLogger):
    training_confusion_matrix = training_logger.training_confusion_matrix_history[-1]
    validation_confusion_matrix = training_logger.validation_confusion_matrix_history[-1]

    # calculating metrics
    training_precision = ConfusionMatrx.calculate_precision(training_confusion_matrix, indices=list(range(4)))
    training_recall = ConfusionMatrx.calculate_recall(training_confusion_matrix, indices=list(range(4)))
    training_accuracy = ConfusionMatrx.calculate_accuracy(training_confusion_matrix, indices=list(range(4)))
    training_f1_score = ConfusionMatrx.calculate_f1_score(training_confusion_matrix, indices=list(range(4)))

    validation_precision = ConfusionMatrx.calculate_precision(validation_confusion_matrix, indices=list(range(4)))
    validation_recall = ConfusionMatrx.calculate_recall(validation_confusion_matrix, indices=list(range(4)))
    validation_accuracy = ConfusionMatrx.calculate_accuracy(validation_confusion_matrix, indices=list(range(4)))
    validation_f1_score = ConfusionMatrx.calculate_f1_score(validation_confusion_matrix, indices=list(range(4)))

    # storing metrics
    training_logger.training_precision_history.append(training_precision)
    training_logger.training_recall_history.append(training_recall)
    training_logger.training_accuracy_history.append(training_accuracy)
    training_logger.training_f1_score_history.append(training_f1_score)

    training_logger.validation_precision_history.append(validation_precision)
    training_logger.validation_recall_history.append(validation_recall)
    training_logger.validation_accuracy_history.append(validation_accuracy)
    training_logger.validation_f1_score_history.append(validation_f1_score)


def __print(epoch, total_epochs, training_logger: TrainingLogger):

    # pulling metrics
    training_precision = np.average(training_logger.training_precision_history[-1])
    training_recall = np.average(training_logger.training_recall_history[-1])
    training_accuracy = np.average(training_logger.training_accuracy_history[-1])
    training_f1_score = np.average(training_logger.training_f1_score_history[-1])

    validation_precision = np.average(training_logger.validation_precision_history[-1])
    validation_recall = np.average(training_logger.validation_recall_history[-1])
    validation_accuracy = np.average(training_logger.validation_accuracy_history[-1])
    validation_f1_score = np.average(training_logger.validation_f1_score_history[-1])

    print(f'Epoch {epoch + 1}/{total_epochs}:\n'
          f'\tTraining precision: {training_precision:.4f}\n'
          f'\tTraining recall: {training_recall:.4f}\n'
          f'\tTraining accuracy: {training_accuracy:.4f}\n'
          f'\tTraining f1-score: {training_f1_score:.4f}\n\n'
          f'\tValidation precision: {validation_precision:.4f}\n'
          f'\tValidation recall: {validation_recall:.4f}\n'
          f'\tValidation accuracy: {validation_accuracy:.4f}\n'
          f'\tValidation f1-score: {validation_f1_score:.4f}\n\n'
          )
