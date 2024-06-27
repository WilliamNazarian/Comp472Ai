from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold


def get_dataset_subsets_per_fold(image_folder_dataset, num_folds):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    indices = list(range(len(image_folder_dataset)))
    folds = list(kf.split(indices))

    subsets = []

    for fold_idx, (train_val_indices, test_indices) in enumerate(folds):
        train_size = int(0.8 * len(train_val_indices))
        # val_size = len(train_val_indices) - train_size
        train_indices, val_indices = train_val_indices[:train_size], train_val_indices[train_size:]

        train_subset = Subset(image_folder_dataset, train_indices)
        val_subset = Subset(image_folder_dataset, val_indices)
        test_subset = Subset(image_folder_dataset, test_indices)

        subsets.append((train_subset, val_subset, test_subset))

    return subsets
