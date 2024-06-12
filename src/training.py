import os.path

import pickle
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

from src.types import TrainingConfig, TrainingLogger, ConfusionMatrix


# TODO: Add support for tracking the following metrics during training and validation: 'loss'


cm = ConfusionMatrix
cm_macro = ConfusionMatrix.Macro
cm_micro = ConfusionMatrix.Micro


__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
__num_classes = 4


def train_model(training_config: TrainingConfig) -> TrainingLogger:
    best_validation_loss = float("inf")
    patience = 10
    trigger_times = 0

    training_logger = TrainingLogger()

    model = training_config.model
    model.to(__device)
    scheduler = training_config.scheduler

    for epoch in range(training_config.epochs):
        # training
        model.train()
        training_confusion_matrix, training_loss = __train(training_config)
        training_logger.training_confusion_matrix_history.append(training_confusion_matrix)

        # validation
        model.eval()
        validation_confusion_matrix, validation_loss = __validate(training_config)
        training_logger.validation_confusion_matrix_history.append(validation_confusion_matrix)

        # calculate macro/micro metrics
        __calculate_metrics(training_logger)

        # early stopping stuff ig
        scheduler.step(validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            trigger_times = 0

            # save training logger + model
            with open(os.path.join(training_config.models_output_dir, "training_logger.pkl"), "wb") as file:
                pickle.dump(training_logger, file)

            torch.save(model.state_dict(), os.path.join(training_config.models_output_dir, "best_model.pth"))
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping")
                return training_logger

        # print data to stdout
        __print(epoch, training_config.epochs, training_config, training_logger)

    return training_logger


def __train(training_config: TrainingConfig):
    running_loss = 0.0
    confusion_matrix: npt.NDArray[int] = np.zeros((4, 4), dtype=int)

    training_set_loader = training_config.training_set_loader

    model = training_config.model
    criterion = training_config.criterion
    optimizer = training_config.optimizer

    for i, (inputs, labels) in enumerate(training_set_loader):
        inputs = Variable(inputs).to(__device)
        labels = Variable(labels).to(__device)

        # process input/back propagate
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # build confusion matrix
        _, predicted = torch.max(outputs.data, 1)
        for expected, actual in list(zip(labels.tolist(), predicted.tolist())):
            confusion_matrix[expected, actual] += 1

        # update outside values
        running_loss += loss.item() * inputs.size(0)

    total = np.sum(confusion_matrix)
    train_loss = running_loss / total

    return confusion_matrix, train_loss


def __validate(training_config: TrainingConfig):
    validation_loss = 0.0
    confusion_matrix: npt.NDArray[int] = np.zeros((4, 4), dtype=int)

    validation_set_loader = training_config.validation_set_loader

    model = training_config.model

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(validation_set_loader):
            inputs = Variable(inputs).to(__device)
            labels = Variable(labels).to(__device)

            outputs = model(inputs)

            # build confusion matrix
            _, predicted = torch.max(outputs.data, 1)
            for expected, actual in list(zip(labels.tolist(), predicted.tolist())):
                confusion_matrix[expected, actual] += 1

            # update outside values
            loss = F.cross_entropy(outputs, labels)
            validation_loss += loss.item() * inputs.size(0)

    total = np.sum(confusion_matrix)
    validation_loss /= total

    return confusion_matrix, validation_loss


def __calculate_metrics(training_logger: TrainingLogger):
    training_confusion_matrix = training_logger.training_confusion_matrix_history[-1]
    validation_confusion_matrix = training_logger.validation_confusion_matrix_history[-1]

    # calculating metrics
    training_precision, training_recall, training_f1_score, training_accuracy = (
        cm.calculate_per_class_metrics(training_confusion_matrix))

    validation_precision, validation_recall, validation_f1_score, validation_accuracy = (
        cm.calculate_per_class_metrics(validation_confusion_matrix))

    # storing metrics
    training_logger.training_precision_history.append(training_precision)
    training_logger.training_recall_history.append(training_recall)
    training_logger.training_accuracy_history.append(training_accuracy)
    training_logger.training_f1_score_history.append(training_f1_score)

    training_logger.validation_precision_history.append(validation_precision)
    training_logger.validation_recall_history.append(validation_recall)
    training_logger.validation_accuracy_history.append(validation_accuracy)
    training_logger.validation_f1_score_history.append(validation_f1_score)


def __print(epoch, total_epochs, training_config: TrainingConfig, training_logger: TrainingLogger):

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
          f'\tValidation f1-score: {validation_f1_score:.4f}')

    for param_group in training_config.optimizer.param_groups:
        print(f"\tLearning rate: {param_group['lr']}\n\n")
