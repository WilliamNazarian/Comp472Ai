# Script file that randomly samples images and pairs it with its pixel intensity histogram
import os.path
from typing import List, Callable
from collections import deque
from src.data_loader import get_metadata
from src.utils.image import *

import sys
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

import src.visualization.plot_histograms as hist


# Defining aliases
hist_pf_rgb = hist.PlotFormatter.RgbChannelsFormatter


def get_callback_for_class_sampling(class_name: str) -> Callable[[plt.Figure], None]:
    def customize_figure(fig: plt.Figure):
        fig.suptitle(f"15 sampled images from the class \"{class_name}\"")
        hist_pf_rgb.add_legend(fig=fig)

    return customize_figure


def sample_and_get_pixel_intensity_histogram(image_paths: List[str],
                                             callback: Callable[[plt.Figure], None] = (lambda _fig: None)):
    ignore_zeros = True
    sampled_image_paths = random.sample(image_paths, 15)  # selection without duplicates
    image_paths_queue = deque(sampled_image_paths)

    fig = plt.figure(figsize=(19, 11))
    outer_grid = gridspec.GridSpec(5, 3, wspace=0.1, hspace=0.5)

    for i in range(5):
        for j in range(3):
            inner_grid = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_grid[i, j], wspace=0.1, hspace=0.1)

            current_image_path = image_paths_queue.pop()
            image = mpimg.imread(current_image_path)

            # Left cell
            ax_left = plt.Subplot(fig, inner_grid[0])
            ax_left.imshow(image)
            __format_image_cell(ax_left, current_image_path)
            fig.add_subplot(ax_left)

            # Right cell
            ax_right = plt.Subplot(fig, inner_grid[1])
            red_pixels, green_pixels, blue_pixels = (
                hist.Data.calculate_aggregate_rgb_channel_intensities([current_image_path], ignore_zeros=ignore_zeros))

            _ = hist.PlotCreator.generate_rgb_channel_intensities_plot(ax_right, red_pixels, green_pixels, blue_pixels)

            hist_pf_rgb.format_rgb_channel_intensities_plot(ax_right, current_image_path,
                                                            ignore_zeros=ignore_zeros,
                                                            add_title=False)
            fig.add_subplot(ax_right)

    callback(fig)
    plt.show()


def __format_image_cell(ax: plt.Axes, image_path: str):
    file_name = os.path.basename(image_path)
    ax.set_title(f"{file_name}")


def __format_histogram_cell(ax: plt.Axes):
    ax.set_xlabel(f'Pixel intensity (log w/ base {hist.log_base})')
    ax.set_ylabel('Normalized frequency')
    ax.grid(True)


def __sample_from_all_images(df):
    def customize_figure(fig: plt.Figure):
        fig.suptitle(f"15 sampled images from the whole dataset")
        hist_pf_rgb.add_legend(fig=fig)

    sample_and_get_pixel_intensity_histogram(df['path'].tolist(), callback=customize_figure)


def main():
    df = get_metadata()
    if len(sys.argv) > 1:
        arg = sys.argv[1]

        match arg:
            case "anger" | "engaged" | "happy" | "neutral":
                print(f"Sampling 15 images from the class \"{arg}\"")
                df_grouped_by_label = df.groupby('label')
                df_class = df_grouped_by_label.get_group(arg)
                sample_and_get_pixel_intensity_histogram(df_class['path'].tolist(), get_callback_for_class_sampling(arg))
            case _:
                __sample_from_all_images(df)
    else:
        __sample_from_all_images(df)


if __name__ == "__main__":
    main()
