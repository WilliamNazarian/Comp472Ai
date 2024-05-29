import numpy as np
import pandas as pd
import math

from scripts.data_loader import *
from scripts.plot_histograms import *


def main():
    df = get_metadata()
    df_grouped_by_label = df.groupby('label')

    df_anger = df_grouped_by_label.get_group('anger')
    df_engaged = df_grouped_by_label.get_group('engaged')
    df_happy = df_grouped_by_label.get_group('happy')
    df_neutral = df_grouped_by_label.get_group('neutral')

    plot_aggregate_pixel_intensity_histogram("anger", df_anger['path'].tolist())


if __name__ == '__main__':
    main()
