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

from typing import List, Callable
from dataclasses import dataclass

from scripts.model.types import TrainingConfig


__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
__num_classes = 4


def train_model(training_config: TrainingConfig):
    train_loss: List[float] = []
    train_accuracy: List[float] = []
    valid_loss: List[float] = []
    valid_accuracy: List[float] = []

    model = training_config.model
    model.to(__device)

    for epoch in range(training_config.epochs):
        model.train()
        epoch_train_loss, epoch_train_accuracy = __train(training_config)
        train_loss.append(epoch_train_loss)
        train_accuracy.append(epoch_train_accuracy)

        model.eval()
        epoch_valid_loss, epoch_valid_accuracy = __validate(training_config)
        valid_loss.append(epoch_valid_loss)
        valid_accuracy.append(epoch_valid_accuracy)

        print(f'Epoch {epoch + 1}/{training_config.epochs}, '
              f'Tr Loss: {train_loss[-1]:.4f}, '
              f'Tr Acc: {train_accuracy[-1]:.4f}, '
              f'Val Loss: {valid_loss[-1]:.4f}, '
              f'Val Acc: {valid_accuracy[-1]:.4f}')


def __train(training_config: TrainingConfig) -> (List[float], List[float]):
    iter_loss = 0.0
    correct = 0
    iterations = 0

    training_set_loader = training_config.training_set_loader

    # classes = training_config.classes
    model = training_config.model
    criterion = training_config.criterion
    optimizer = training_config.optimizer

    for i, (items, classes) in enumerate(training_set_loader):
        # Convert torch tensor to Variable
        items = Variable(items).to(__device)
        classes = Variable(classes).to(__device)

        """
        # If we have GPU, shift the data to GPU
        if torch.cuda.is_available():
            items = items.cuda()
            classes = classes.cuda()
        """

        optimizer.zero_grad()
        outputs = model(items)
        loss = criterion(outputs, classes)
        iter_loss += loss.item()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == classes.data).sum().item()
        iterations += 1

    return (iter_loss / iterations), (100 * correct / len(training_set_loader.dataset))


def __validate(training_config: TrainingConfig) -> (List[float], List[float]):
    loss = 0.0
    correct = 0
    iterations = 0

    validation_set_loader = training_config.validation_set_loader

    # classes = training_config.classes
    model = training_config.model
    criterion = training_config.criterion

    for i, (items, classes) in enumerate(validation_set_loader):

        # Convert torch tensor to Variable
        items = Variable(items).to(__device)
        classes = Variable(classes).to(__device)

        """
        # If we have GPU, shift the data to GPU
        if torch.cuda.is_available():
            items = items.cuda()
            classes = classes.cuda()
        """

        outputs = model(items)  # Do the forward pass
        loss += criterion(outputs, classes).item()  # Calculate the loss

        # Record the correct predictions for validation data
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == classes.data).sum().item()

        iterations += 1

    return (loss / iterations), (correct / len(validation_set_loader.dataset) * 100.0)
