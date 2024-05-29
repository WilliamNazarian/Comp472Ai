from typing import List, Callable

import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import math

from datetime import datetime
from matplotlib.patches import Patch
from pipe import *
from PIL import Image

from scripts import *
from utils.image import *

log_base = 2


def plot_image_dimensions_histogram(image_paths: List[str]) -> None:
    image_info = []
    for path in image_paths:
        with Image.open(path) as image:
            width, height = image.size

            image_info.append({
                'width': width,
                'height': height,
            })
    df = pd.DataFrame(image_info)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 15))

    df['height'].plot(kind='hist', bins=5, ax=axes[0])
    axes[0].set_title('Image height distribution')
    axes[0].set_xlabel('Image Height (px)')
    axes[0].set_ylabel('Number of images')

    df['width'].plot(kind='hist', bins=5, ax=axes[1])
    axes[1].set_title('Image width distribution')
    axes[1].set_xlabel('Image Width (px)')
    axes[1].set_ylabel('Number of images')

    # plt.tight_layout()
    plt.show()


def aggregate_rgb_channel_intensities(image_paths: List[str], ignore_zeros=True) -> (np.ndarray, np.ndarray, np.ndarray):
    red_pixels = []
    green_pixels = []
    blue_pixels = []

    for path in image_paths:
        with Image.open(path) as image:
            red_pixels.extend(get_image_pixel_intensity(image, transform=convert_to_redscale))
            green_pixels.extend(get_image_pixel_intensity(image, transform=convert_to_greenscale))
            blue_pixels.extend(get_image_pixel_intensity(image, transform=convert_to_bluescale))

    red_pixels = np.asarray(red_pixels, dtype=np.uint16).flatten()
    red_pixels = np.log(red_pixels + 1) / np.log(log_base)

    green_pixels = np.asarray(green_pixels, dtype=np.uint16).flatten()
    green_pixels = np.log(green_pixels + 1) / np.log(log_base)

    blue_pixels = np.asarray(blue_pixels, dtype=np.uint16).flatten()
    blue_pixels = np.log(blue_pixels + 1) / np.log(log_base)

    if ignore_zeros:
        red_pixels = red_pixels[red_pixels != 0]
        green_pixels = green_pixels[green_pixels != 0]
        blue_pixels = blue_pixels[blue_pixels != 0]

    return red_pixels, green_pixels, blue_pixels


def calculate_rgb_channel_weights(red_pixels, green_pixels, blue_pixels) -> (np.ndarray, np.ndarray, np.ndarray):
    red_pixels_w = np.empty(red_pixels.shape)
    red_pixels_w.fill(1 / red_pixels.shape[0])
    green_pixels_w = np.empty(green_pixels.shape)
    green_pixels_w.fill(1 / green_pixels.shape[0])
    blue_pixels_w = np.empty(blue_pixels.shape)
    blue_pixels_w.fill(1 / blue_pixels.shape[0])

    return red_pixels_w, green_pixels_w, blue_pixels_w


def plot_rgb_channel_intensities(ax: plt.Axes, red_pixels, green_pixels, blue_pixels):
    red_pixels_w, green_pixels_w, blue_pixels_w = calculate_rgb_channel_weights(red_pixels, green_pixels, blue_pixels)

    ax.hist([red_pixels, green_pixels, blue_pixels], np.linspace(0, math.log(256, log_base), 16),
            alpha=0.7,
            weights=[red_pixels_w, green_pixels_w, blue_pixels_w],
            label=['red pixel intensities', 'green pixel intensities', 'blue pixel intensities'],
            color=['lightcoral', 'lawngreen', 'cornflowerblue'])


def get_legend_handles() -> List[matplotlib.patches.Patch]:
    return [
        Patch(color='lightcoral', label='red pixel intensities'),
        Patch(color='lawngreen', label='green pixel intensities'),
        Patch(color='cornflowerblue', label='blue pixel intensities'),
    ]


def plot_aggregate_pixel_intensity_histogram(class_name, image_paths: List[str], ignore_zeros=True):
    red_pixels, green_pixels, blue_pixels = aggregate_rgb_channel_intensities(image_paths)

    if len(image_paths) == 1:
        callback = (lambda _fig, axes: __plt_callback_px_intensity_single(image_paths[0], ignore_zeros, _fig, axes))
    else:
        callback = (lambda _fig, axes: __plt_callback_px_intensity_multiple(class_name, ignore_zeros, _fig, axes))

    fig, ax = plt.subplots(figsize=(10, 10))

    plot_rgb_channel_intensities(ax, red_pixels, green_pixels, blue_pixels)
    callback(fig, ax)
    plt.show()


# Sets the values for the plot for an RGB pixel intensity histogram for a SINGLE image
def __plt_callback_px_intensity_single(image_name, ignore_zeros, fig: plt.Figure, axes: plt.Axes) -> None:
    title_text = (f"Overlapped RGB pixel intensity histogram"
                  f"{f" for the image \"{image_name}\"" if image_name else ""}"
                  f"{f" (Zero values omitted)" if ignore_zeros else ""}")
    axes.set_title(title_text)
    axes.set_xlabel(f'Pixel intensity (log w/ base {log_base})')
    axes.set_ylabel('Normalized frequency')

    axes.legend(handles=get_legend_handles(), loc='upper left')


# Sets the values for the plot for an RGB pixel intensity histogram for MULTIPLE images
def __plt_callback_px_intensity_multiple(class_name, ignore_zeros, fig: plt.Figure, axes: plt.Axes) -> None:
    axes.set_title(f'Intensity distributions of the RGB channels \"{class_name}\"'
                   f"{f" (Zero values omitted)" if ignore_zeros else ""}")
    axes.set_xlabel(f'Pixel intensity (log w/ base {log_base})')
    axes.set_ylabel('Normalized frequency')

    axes.legend(handles=get_legend_handles(), loc='upper left')


def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        df = get_metadata()

        match arg:
            case "anger":
                print("Plotting the RGB pixel intensities histogram for the class \"anger\"")
                df_grouped_by_label = df.groupby('label')
                df_anger = df_grouped_by_label.get_group('anger')
                plot_aggregate_pixel_intensity_histogram('anger', df_anger['path'].tolist())
            case "engaged":
                print("Plotting the RGB pixel intensities histogram for the class \"engaged\"")
                df_grouped_by_label = df.groupby('label')
                df_engaged = df_grouped_by_label.get_group('engaged')
                plot_aggregate_pixel_intensity_histogram('engaged', df_engaged['path'].tolist())
            case "happy":
                print("Plotting the RGB pixel intensities histogram for the class \"happy\"")
                df_grouped_by_label = df.groupby('label')
                df_happy = df_grouped_by_label.get_group('happy')
                plot_aggregate_pixel_intensity_histogram('happy', df_happy['path'].tolist())
            case "neutral":
                print("Plotting the RGB pixel intensities histogram for the class \"neutral\"")
                df_grouped_by_label = df.groupby('label')
                df_neutral = df_grouped_by_label.get_group('neutral')
                plot_aggregate_pixel_intensity_histogram('neutral', df_neutral['path'].tolist())
            case "image_dimensions":
                print("Plotting the dimensions for all  \"neutral\"")
                image_paths = df['path'].tolist()
                plot_image_dimensions_histogram(image_paths)
            case _:
                # search for a specific image then do pixel RGB channel visualization for that specific image
                pass
    else:
        print("No arguments received.")


if __name__ == "__main__":
    main()
