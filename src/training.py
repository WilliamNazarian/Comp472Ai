import os.path

import pickle
import torch
import numpy as np
import torch.nn as nn
import numpy.typing as npt
import torch.nn.functional as F

from torch.autograd import Variable
from src.types import TrainingConfig, TrainingLogger
from src.utils.confusion_matrix import ConfusionMatrix


# TODO: Add support for tracking the following metrics during training and validation: 'loss'


cm = ConfusionMatrix
cm_macro = ConfusionMatrix.Macro
cm_micro = ConfusionMatrix.Micro


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 4


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)


def train_model(training_config: TrainingConfig) -> TrainingLogger:
    best_validation_loss = float("inf")
    patience = 10
    trigger_times = 0

    training_logger = TrainingLogger()

    training_set_loader = training_config.training_set_loader
    validation_set_loader = training_config.validation_set_loader

    model = training_config.model
    model.apply(init_weights)
    model.to(device)
    criterion = training_config.criterion
    optimizer = training_config.optimizer
    scheduler = training_config.scheduler

    for epoch in range(training_config.epochs):
        # training
        model.train()
        training_confusion_matrix, training_loss = train(training_set_loader, model, criterion, optimizer)
        training_logger.training_confusion_matrix_history.append(training_confusion_matrix)

        # validation
        model.eval()
        validation_confusion_matrix, validation_loss = validate(validation_set_loader, model)
        training_logger.validation_confusion_matrix_history.append(validation_confusion_matrix)

        # calculate macro/micro metrics
        calculate_metrics(training_logger)

        # early stopping stuff ig
        scheduler.step(validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            trigger_times = 0

            # save training logger + model
            with open(os.path.join(training_config.output_dir, "training_logger.pkl"), "wb") as file:
                pickle.dump(training_logger, file)

            torch.save(model.state_dict(), os.path.join(training_config.output_dir, "best_model.pth"))
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping")
                return training_logger

        # print data to stdout
        print_metrics(epoch, training_config.epochs, training_config, training_logger)

    return training_logger


def train(training_set_loader, model, criterion, optimizer):
    running_loss = 0.0
    confusion_matrix: npt.NDArray[int] = np.zeros((4, 4), dtype=int)

    for i, (inputs, labels) in enumerate(training_set_loader):
        inputs = Variable(inputs).to(device)
        labels = Variable(labels).to(device)

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
            confusion_matrix[actual, expected] += 1

        # update outside values
        running_loss += loss.item() * inputs.size(0)

    total = np.sum(confusion_matrix)
    train_loss = running_loss / total

    return confusion_matrix, train_loss


def validate(validation_set_loader, model):
    validation_loss = 0.0
    confusion_matrix: npt.NDArray[int] = np.zeros((4, 4), dtype=int)

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(validation_set_loader):
            inputs = Variable(inputs).to(device)
            labels = Variable(labels).to(device)

            outputs = model(inputs)

            # build confusion matrix
            _, predicted = torch.max(outputs.data, 1)
            for expected, actual in list(zip(labels.tolist(), predicted.tolist())):
                confusion_matrix[actual, expected] += 1

            # update outside values
            loss = F.cross_entropy(outputs, labels)
            validation_loss += loss.item() * inputs.size(0)

    total = np.sum(confusion_matrix)
    validation_loss /= total

    return confusion_matrix, validation_loss


def calculate_metrics(training_logger: TrainingLogger):
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


def print_metrics(epoch, total_epochs, training_config: TrainingConfig, training_logger: TrainingLogger):
    if training_config.output_logger is None:
        return

    # pulling metrics
    training_precision = np.average(training_logger.training_precision_history[-1])
    training_recall = np.average(training_logger.training_recall_history[-1])
    training_accuracy = np.average(training_logger.training_accuracy_history[-1])
    training_f1_score = np.average(training_logger.training_f1_score_history[-1])

    validation_precision = np.average(training_logger.validation_precision_history[-1])
    validation_recall = np.average(training_logger.validation_recall_history[-1])
    validation_accuracy = np.average(training_logger.validation_accuracy_history[-1])
    validation_f1_score = np.average(training_logger.validation_f1_score_history[-1])

    learning_rates_str = "\n".join(
        f"\tLearning rate for param group \"{i}\": {param_group['lr']}"
        for i, param_group in enumerate(training_config.optimizer.param_groups)
    )

    training_config.output_logger.info(
          f'\nEpoch {epoch + 1}/{total_epochs}:\n'
          f'\tTraining precision: {training_precision:.4f}\n'
          f'\tTraining recall: {training_recall:.4f}\n'
          f'\tTraining accuracy: {training_accuracy:.4f}\n'
          f'\tTraining f1-score: {training_f1_score:.4f}\n\n'
          f'\tValidation precision: {validation_precision:.4f}\n'
          f'\tValidation recall: {validation_recall:.4f}\n'
          f'\tValidation accuracy: {validation_accuracy:.4f}\n'
          f'\tValidation f1-score: {validation_f1_score:.4f}\n'
          f'{learning_rates_str}'
    )
