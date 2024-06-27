import os
import torch
import pickle
import numpy as np
import pipe as pipe
import src.training as training
import src.evaluation as evaluation

from typing import List, Tuple
from dataclasses import dataclass
from torch.utils.data import Subset
from src.types import TrainingLogger, EvaluationResults
from src.data_loader import create_data_loader
from src.kfold.kfold_training_config import KFoldTrainingConfig
from src.models.main_model import OB_05Model


@dataclass
class SubsetIndices:
    """
    Type containing the indices for the training, validation, and testing partitions of the dataset.
    """
    training_indices: List[int]
    validation_indices: List[int]
    testing_indices: List[int]

    @staticmethod
    def to_data_loaders(image_folder_dataset, subset_indices):
        training_subset = Subset(image_folder_dataset, subset_indices.training_indices)
        validation_subset = Subset(image_folder_dataset, subset_indices.validation_indices)
        testing_subset = Subset(image_folder_dataset, subset_indices.testing_indices)
        return create_data_loader(training_subset), create_data_loader(validation_subset), create_data_loader(
            testing_subset)


def __start_end_ratios_to_indices(image_folder_dataset, start_ratio, end_ratio):
    """
    Helper method converting a tuple of ratios in the form of (x, y) into the corresponding indices in the dataset,
    where x and y are between 0 and 1 inclusive.
    """
    dataset_size = len(image_folder_dataset)
    start_index = int(dataset_size * start_ratio)
    end_index = int(dataset_size * end_ratio)
    return list(range(start_index, end_index))


def __get_subset_indices_per_fold(image_folder_dataset, num_folds):
    """
    Returns a list of `SubsetIndices` objects based on the number of folds.
    """
    subset_length_ratio = 1 / num_folds

    fold_start_end_ratios = list([x for x in range(num_folds)]
                                 | pipe.map(lambda i: i * subset_length_ratio)
                                 | pipe.map(lambda x: (x, x + subset_length_ratio)))

    all_indices = set(range(len(image_folder_dataset)))

    f = __start_end_ratios_to_indices
    return (
        list([x for x in range(num_folds)]
             | pipe.map(lambda i: (fold_start_end_ratios[i], fold_start_end_ratios[i - 1]))
             | pipe.map(lambda pair: (f(image_folder_dataset, pair[0][0], pair[0][1]), f(image_folder_dataset, pair[1][0], pair[1][1])))
             | pipe.map(lambda pair: (pair[0], pair[1], list(all_indices - (set(pair[0]) | set(pair[1])))))
             | pipe.map(lambda ntuple: SubsetIndices(testing_indices=ntuple[0], validation_indices=ntuple[1], training_indices=ntuple[2]))
             ))


def kfold_cross_validation(training_config: KFoldTrainingConfig) -> List[Tuple[TrainingLogger, EvaluationResults]]:
    dataset = training_config.dataset
    criterion = training_config.criterion
    optimizer = training_config.optimizer
    scheduler = training_config.scheduler
    patience = training_config.patience

    num_folds = training_config.num_folds
    subset_indices_list = __get_subset_indices_per_fold(dataset, num_folds)

    results_per_fold: List[Tuple[TrainingLogger, EvaluationResults]] = []

    for current_index, subset_indices in enumerate(subset_indices_list):
        # 1. setups
        fold_output_dir = os.path.join(training_config.output_dir, f"fold_{current_index + 1}")
        if not os.path.exists(fold_output_dir):
            os.makedirs(fold_output_dir)

        training_dataloader, validation_dataloader, testing_dataloader = (
            SubsetIndices.to_data_loaders(dataset, subset_indices))

        model = OB_05Model()
        model.apply(training.init_weights)
        model.to(training.device)

        # 2. initializing values for the fold
        best_validation_loss = float("inf")
        trigger_times = 0
        training_logger = TrainingLogger()

        # 3. perform training & validation for the fold
        for epoch in range(training_config.epochs_per_fold):
            # a. training
            model.train()
            training_confusion_matrix, training_loss = training.train(training_dataloader, model, criterion, optimizer)
            training_logger.training_confusion_matrix_history.append(training_confusion_matrix)

            # b. validation
            model.eval()
            validation_confusion_matrix, validation_loss = training.validate(validation_dataloader, model)
            training_logger.validation_confusion_matrix_history.append(validation_confusion_matrix)

            # c. calculating performance metrics
            training.calculate_metrics(training_logger)

            # d. early stopping stuff
            scheduler.step(validation_loss)

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                trigger_times = 0

                # save training logger + model
                with open(os.path.join(fold_output_dir, "training_logger.pkl"), "wb") as file:
                    pickle.dump(training_logger, file)

                torch.save(model.state_dict(), os.path.join(fold_output_dir, "best_model.pth"))
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    training_config.output_logger.info("Early stopping")
                    break

            # print data to stdout
            __print_metrics(current_index, num_folds, epoch, training_config, training_logger)

        # 4. testing & saving results
        evaluation_results = evaluation.evaluate_model(training_config.output_logger, model, testing_dataloader)
        evaluation_results.print_extensive_summary()

        with open(os.path.join(fold_output_dir, "testing_results.pkl"), "wb") as file:
            pickle.dump(evaluation_results, file)

        results_per_fold.append((training_logger, evaluation_results))

    return results_per_fold


def __print_metrics(current_fold, total_folds, current_epoch,
                    training_config: KFoldTrainingConfig,
                    training_logger: TrainingLogger):
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
        f'\nFold {current_fold + 1}/{total_folds};f Epoch {current_epoch + 1}/{training_config.epochs_per_fold}:\n'
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
