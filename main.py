import torch
import torch.nn as nn

import scripts.data_loader as data_loader
import scripts.model.train_model as train_model

from models.model1 import OB_05Model


def main():
    training_set_loader, validation_set_loader, testing_set_loader = data_loader.split_images_dataset()

    learning_rate = 0.001
    model = OB_05Model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    training_config = train_model.TrainingConfig(
        training_set_loader=training_set_loader,
        validation_set_loader=validation_set_loader,
        testing_set_loader=testing_set_loader,

        epochs=100,
        learning_rate=learning_rate,

        classes=data_loader.get_trainset().classes,
        model=model,
        criterion=criterion,
        optimizer=optimizer
    )

    train_model.train_model(training_config)


if __name__ == '__main__':
    main()
