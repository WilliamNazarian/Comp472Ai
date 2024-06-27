from torchvision import datasets
import random


class ShuffledImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform, target_transform)
        self.indices = list(range(len(self.samples)))
        random.shuffle(self.indices)

    def __getitem__(self, index):
        # Use the shuffled index to get the item
        shuffled_index = self.indices[index]
        path, target = self.samples[shuffled_index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def shuffle(self):
        # Method to reshuffle the indices
        random.shuffle(self.indices)
