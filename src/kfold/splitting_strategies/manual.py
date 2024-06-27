import random
import pipe as pipe

from src.data_loader import create_data_loader


def __start_end_ratios_to_indices(image_folder_dataset, start_ratio, end_ratio):
    """
    Helper method converting a tuple of ratios in the form of (x, y) into the corresponding indices in the dataset,
    where x and y are between 0 and 1 inclusive.
    """
    dataset_size = len(image_folder_dataset)
    start_index = int(dataset_size * start_ratio)
    end_index = int(dataset_size * end_ratio)
    return list(range(start_index, end_index))


def get_dataset_subsets_per_fold(image_folder_dataset, num_folds):
    """
    """
    random.shuffle(image_folder_dataset.samples)
    subset_length_ratio = 1 / num_folds
    all_indices = set(range(len(image_folder_dataset)))
    f = __start_end_ratios_to_indices

    fold_start_end_ratios = list([x for x in range(num_folds)]
                                 | pipe.map(lambda i: i * subset_length_ratio)
                                 | pipe.map(lambda x: (x, x + subset_length_ratio)))

    return (
        list
        ([x for x in range(num_folds)]
         | pipe.map(lambda i: (fold_start_end_ratios[i], fold_start_end_ratios[i - 1]))
         | pipe.map(lambda pair: (f(image_folder_dataset, pair[0][0], pair[0][1]), f(image_folder_dataset, pair[1][0], pair[1][1])))
         | pipe.map(lambda pair: (pair[0], pair[1], list(all_indices - (set(pair[0]) | set(pair[1])))))
         | pipe.map(lambda ntuple: (create_data_loader(ntuple[2]), create_data_loader(ntuple[1]), create_data_loader(ntuple[0])))
         )
    )
