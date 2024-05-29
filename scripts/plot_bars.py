from typing import List, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipe import *
from scripts import *


def plot_classes_distribution() -> None:
    df = get_metadata()
    df_grouped_by_label = df.groupby('label')

    df_anger = df_grouped_by_label.get_group('anger')
    df_engaged = df_grouped_by_label.get_group('engaged')
    df_happy = df_grouped_by_label.get_group('happy')
    df_neutral = df_grouped_by_label.get_group('neutral')

    classes = ['anger', 'engaged', 'happy', 'neutral']
    images_per_class = list([df_anger, df_engaged, df_happy, df_neutral]
                            | select(lambda _df: _df.shape[0]))

    plt.bar(classes, images_per_class)
    plt.title('Class distribution')
    plt.xlabel('Image classes')
    plt.ylabel('Images in class')
    plt.show()


def main():
    plot_classes_distribution()


if __name__ == "__main__":
    main()
