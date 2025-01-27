import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pandas as pd

from torch.utils.data import DataLoader, random_split

import src.utils.shuffled_image_folder

# Get the current file's directory
__current_file_dir = os.path.dirname(os.path.abspath(__file__))
__project_root = os.path.abspath(os.path.join(__current_file_dir, os.pardir))
greyscale_images_directory = os.path.abspath(os.path.join(__project_root, r".\dataset\cleaned_images"))
colored_images_directory = os.path.abspath(os.path.join(__project_root, r".\dataset\structured_data"))

__mean_gray = 0.1307
__stddev_gray = 0.3081

transform = transforms.Compose([
    transforms.Resize((90, 90)),
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Dataloader settings
shuffle = True
num_workers = 4
pin_memory = True


def get_metadata():
    csv_path = "./dataset/Combined_Labels_DataFrame.csv"
    df = pd.read_csv(csv_path)
    df['path'] = df['path'].apply(lambda path: "./dataset/structured_data/" + path)
    return df


def get_trainset(use_colored=False):
    images_directory = colored_images_directory if use_colored else greyscale_images_directory
    trainset = src.utils.shuffled_image_folder.ShuffledImageFolder(root=images_directory, transform=transform)
    trainset.shuffle()
    return trainset


# Splits the dataset to training, validation, and test sub-datasets
def split_images_dataset(use_colored=False):
    images_directory = colored_images_directory if use_colored else greyscale_images_directory
    trainset = datasets.ImageFolder(root=images_directory, transform=transform)

    training_ratio = 0.7  # x% of the dataset is for training
    validation_ratio = 0.15  # y% of the dataset is for validation
    # (1 - x - y)% for testing

    # calculating the number of images per dataset partition
    training_set_length = int(training_ratio * len(trainset))
    validation_set_length = int(validation_ratio * len(trainset))
    testing_set_length = len(trainset) - training_set_length - validation_set_length

    # splitting the datasets
    lengths = [training_set_length, validation_set_length + testing_set_length]
    training_dataset, validation_and_testing_dataset = random_split(trainset, lengths)

    lengths = [validation_set_length, testing_set_length]
    validation_dataset, testing_dataset = random_split(validation_and_testing_dataset, lengths)

    return training_dataset, validation_dataset, testing_dataset


def create_data_loader(dataset):
    return DataLoader(dataset, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
