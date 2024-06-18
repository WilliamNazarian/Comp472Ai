import logging
import torch
import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader
from src.types import EvaluationResults
from src.utils.confusion_matrix import ConfusionMatrix


cm = ConfusionMatrix
cm_macro = ConfusionMatrix.Macro
cm_micro = ConfusionMatrix.Micro


__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_model(logger: logging.Logger, model: nn.Module, dataloader: DataLoader) -> EvaluationResults:
    model.eval().to(__device)

    confusion_matrix = np.zeros((4, 4), dtype=int)

    current_batch = 1

    # evaluation/testing
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(__device), labels.to(__device)

            # Forward pass
            outputs = model(images)
            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)
            expected = labels.tolist()
            actual = predicted.tolist()

            # add to confusion matrix
            for expected, actual in list(zip(expected, actual)):
                confusion_matrix[actual, expected] += 1

            __print_confusion_matrix_metrics(logger, current_batch, confusion_matrix)
            current_batch += 1

    return EvaluationResults(confusion_matrix=confusion_matrix)


def __print_confusion_matrix_metrics(logger: logging.Logger, batch_number: int, confusion_matrix: np.ndarray):
    macro_precision, macro_recall, macro_f1_score, macro_accuracy = cm_macro.calculate_overall_metrics(
        confusion_matrix)
    micro_precision, micro_recall, micro_f1_score, micro_accuracy = cm_micro.calculate_overall_metrics(
        confusion_matrix)
    accuracy = (macro_accuracy + micro_accuracy) / 2  # should be the same for both

    logger.info(
        f'\nTesting batch #{batch_number}:\n'
        f'\tMACRO precision: {macro_precision:.4f}\n'
        f'\tMACRO recall: {macro_recall:.4f}\n'
        f'\tMACRO f1_score: {macro_f1_score:.4f}\n\n'
        f'\tMICRO precision: {micro_precision:.4f}\n'
        f'\tMICRO recall: {micro_recall:.4f}\n'
        f'\tMICRO f1_score: {micro_f1_score:.4f}\n\n'
        f'\tAccuracy: {accuracy:.4f}\n\n'
    )
