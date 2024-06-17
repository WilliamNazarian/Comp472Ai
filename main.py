import sys
import torch
import logging
import os.path
import scripts.data_loader as data_loader
import src.training as training
import src.evaluation as evaluation

from pick import pick
from rich.prompt import Prompt, Confirm
from src.types import *
from src.models.main_model import OB_05Model
from src.models.main_model_v1 import OB_05Model_Variant1
from src.models.main_model_v2 import OB_05Model_Variant2
from scripts.visualization.model_evaluation import TrainingVisualizations, TestingVisualizations


project_root_directory = os.path.join(os.path.abspath(__file__), "..")
output_directory = r"output"


def main():
    # create output directory for saved models/data, if it doesn't exist already
    output_directory_absolute_path = os.path.join(project_root_directory, output_directory)
    if not os.path.exists(output_directory_absolute_path):
        os.makedirs(output_directory_absolute_path)

    # prompt user
    title = 'Select which model you want to train and evaluate on:'
    options = ['Main model', 'Model (Variant 1)', 'Model (Variant 2)']
    _, index = pick(options, title, indicator='=>', default_index=0)

    model_name: str
    while True:
        model_name = Prompt.ask('Give a name to the model:', default="model")
        if Confirm.ask('Continue?'):
            break

    # initialize output folders
    model_output_dir = os.path.join(output_directory_absolute_path, model_name)
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)

    training_data_output_dir = os.path.join(model_output_dir, "training")
    testing_data_output_dir = os.path.join(model_output_dir, "testing")

    if not os.path.exists(training_data_output_dir):
        os.makedirs(training_data_output_dir)

    if not os.path.exists(testing_data_output_dir):
        os.makedirs(testing_data_output_dir)

    # initialize datasets
    training_dataset, validation_dataset, testing_dataset = data_loader.split_images_dataset()
    torch.save(training_dataset, os.path.join(model_output_dir, "training_dataset.pth"))
    torch.save(validation_dataset, os.path.join(model_output_dir, "validation_dataset.pth"))
    torch.save(testing_dataset, os.path.join(model_output_dir, "testing_dataset.pth"))

    training_set_loader = data_loader.create_data_loader(training_dataset)
    validation_set_loader = data_loader.create_data_loader(validation_dataset)
    testing_set_loader = data_loader.create_data_loader(testing_dataset)

    # initialize training parameters
    model: nn.Module
    if index == 0:
        model = OB_05Model()
    elif index == 1:
        model = OB_05Model_Variant1()
    else:
        model = OB_05Model_Variant2()

    initial_learning_rate = 0.0001
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate, weight_decay=5e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)

    training_log_file_path = os.path.join(training_data_output_dir, "training_log.txt")
    training_logger = initialize_logger("training_logger", training_log_file_path)

    training_config = training.TrainingConfig(
        model_name=model_name,
        output_dir=model_output_dir,
        output_logger=training_logger,

        training_set_loader=training_set_loader,
        validation_set_loader=validation_set_loader,
        testing_set_loader=testing_set_loader,

        epochs=100,

        classes=data_loader.get_trainset().classes,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler
    )

    # training
    print("------------------------------ Training log ------------------------------")
    training_logger = training.train_model(training_config)
    torch.save(model.state_dict(), os.path.join(model_output_dir, "model.pth"))
    print("--------------------------- End of training log ---------------------------\n\n")

    # save training data/visualizations
    fig = TrainingVisualizations.plot_training_metrics(training_logger)
    training_metrics_img_path = os.path.join(training_data_output_dir, "training_metrics.png")
    fig.savefig(training_metrics_img_path)

    print(f"Training log file saved at:\n\t{training_log_file_path}\n")
    print(f"Training metrics visualization saved at:\n\t{training_metrics_img_path}")

    # testing & saving testing data/visualizations
    testing_log_file_path = os.path.join(testing_data_output_dir, "testing_logger.txt")
    testing_logger = initialize_logger("testing_logger", testing_log_file_path)

    print("\n")
    print("------------------------------ Testing log ------------------------------")
    evaluation_results = evaluation.evaluate_model(testing_logger, model, testing_set_loader)
    print("--------------------------- End of Testing log ---------------------------\n\n")

    print(f"Testing log file saved at:\n\t{testing_log_file_path}\n")

    evaluation_results.print_extensive_summary()

    fig_filename_pairs = [
        [TestingVisualizations.plot_metrics_per_class(evaluation_results), "metrics_per_class_bar_graph.png"],
        [TestingVisualizations.generate_metrics_per_class_table(evaluation_results), "metrics_per_class_table.png"],
        [TestingVisualizations.generate_overall_metrics_table(evaluation_results), "overall_metrics_table.png"],
        [TestingVisualizations.generate_confusion_matrix_table(evaluation_results), "confusion_matrix.png"]
    ]

    print("\n\nSaved Visualizations:")
    for fig, file_name in fig_filename_pairs:
        file_path = os.path.join(testing_data_output_dir, file_name)
        fig.savefig(file_path)
        print(f"File \"{file_name}\" saved at:\n\t{file_path}\n")


def initialize_logger(logger_name: str, log_file_path: str):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(log_file_path)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


if __name__ == '__main__':
    main()
