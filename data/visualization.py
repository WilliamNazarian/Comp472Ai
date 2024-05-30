from typing import List, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pipe import *
from PIL import Image

# Loads image and returns pixel intensities
def __get_pixel_intensity(image_path: str):
    with Image.open(image_path) as image:
        image_greyscale = image.convert('L')
        pixels = np.asarray(image_greyscale).flatten()
        return pixels


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


def plot_classes_distribution(df_anger, df_engaged, df_happy, df_neutral) -> None:
    classes = ['anger', 'engaged', 'happy', 'neutral']
    images_per_class = list([df_anger, df_engaged, df_happy, df_neutral]
                            | select(lambda df: df.shape[0]))

    plt.bar(classes, images_per_class)
    plt.title('Class distribution')
    plt.xlabel('Image classes')
    plt.ylabel('Images in class')
    plt.show()


def plot_pixel_intensity_distribution(class_name: str, image_paths: List[str]) -> None:
    all_pixels = []
    for path in image_paths:
        pixels = __get_pixel_intensity(path)
        all_pixels.extend(pixels)

    plt.hist(all_pixels, bins=256, range=(0, 256), density=True, color='gray', alpha=0.7)
    plt.title(f"Pixel intensity distribution for \"{class_name}\"")
    plt.xlabel("Pixel intensity")
    plt.ylabel("Frequency")
    plt.grid(True)

    plt.show()