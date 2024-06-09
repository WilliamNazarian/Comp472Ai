import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pandas as pd


from torch.utils.data import DataLoader, random_split

# Get the current file's directory
__current_file_dir = os.path.dirname(os.path.abspath(__file__))
__project_root = os.path.abspath(os.path.join(__current_file_dir, os.pardir))
__greyscale_images_directory = os.path.abspath(os.path.join(__project_root, r".\part1\cleaned_images"))
__colored_images_directory = os.path.abspath(os.path.join(__project_root, r".\part1\structured_data"))

__mean_gray = 0.1307
__stddev_gray = 0.3081

__transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((__mean_gray,), (__stddev_gray,))
])


def get_metadata():
    csv_path = "./part1/Combined_Labels_DataFrame.csv"
    df = pd.read_csv(csv_path)
    df['path'] = df['path'].apply(lambda path: "./part1/structured_data/" + path)
    return df


def get_trainset(use_colored=False):
    images_directory = __colored_images_directory if use_colored else __greyscale_images_directory
    return datasets.ImageFolder(root=images_directory, transform=__transform)


def split_images_dataset(use_colored=False) -> (DataLoader, DataLoader, DataLoader):
    images_directory = __colored_images_directory if use_colored else __greyscale_images_directory
    trainset = datasets.ImageFolder(root=images_directory, transform=__transform)

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

    # creating data loaders per partition
    num_workers = 4
    pin_memory = True

    training_set_loader = DataLoader(training_dataset, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    validation_set_loader = DataLoader(validation_dataset, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    testing_set_loader = DataLoader(testing_dataset, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    return training_set_loader, validation_set_loader, testing_set_loader
