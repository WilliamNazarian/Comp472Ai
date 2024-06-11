from typing import List, Callable

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math

from matplotlib.patches import Patch
from pipe import *

from scripts import *
from utils.image import *


log_base = 2


# Namespace/Class for creating/calculating the data required for plotting
class Data:
    @staticmethod
    def calculate_rgb_channel_weights(red_pixels, green_pixels, blue_pixels) -> (np.ndarray, np.ndarray, np.ndarray):
        red_pixels_w = np.empty(red_pixels.shape)
        red_pixels_w.fill(1 / red_pixels.shape[0])
        green_pixels_w = np.empty(green_pixels.shape)
        green_pixels_w.fill(1 / green_pixels.shape[0])
        blue_pixels_w = np.empty(blue_pixels.shape)
        blue_pixels_w.fill(1 / blue_pixels.shape[0])

        return red_pixels_w, green_pixels_w, blue_pixels_w

    @staticmethod
    def calculate_aggregate_rgb_channel_intensities(image_paths, ignore_zeros=True) -> (
            np.ndarray, np.ndarray, np.ndarray):
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


# Namespace/Class for returning raw unformatted plots
class PlotCreator:
    @staticmethod
    def generate_rgb_channel_intensities_plot(ax, red_pixels, green_pixels, blue_pixels):
        red_pixels_w, green_pixels_w, blue_pixels_w = Data.calculate_rgb_channel_weights(red_pixels, green_pixels,
                                                                                         blue_pixels)

        ax.hist([red_pixels, green_pixels, blue_pixels], np.linspace(0, math.log(256, log_base), 16),
                alpha=0.7,
                weights=[red_pixels_w, green_pixels_w, blue_pixels_w],
                label=['red pixel intensities', 'green pixel intensities', 'blue pixel intensities'],
                color=['lightcoral', 'lawngreen', 'cornflowerblue'])
        return ax


# Namespace/Class for formatting plots
class PlotFormatter:
    class RgbChannelsFormatter:
        @staticmethod
        def add_legend(fig=None, ax=None):
            legend_handles = [
                Patch(color='lightcoral', label='red pixel intensities'),
                Patch(color='lawngreen', label='green pixel intensities'),
                Patch(color='cornflowerblue', label='blue pixel intensities'),
            ]

            if ax:
                ax.legend(handles=legend_handles, loc='upper left')
            elif fig:
                fig.legend(handles=legend_handles, loc='upper left')

        @staticmethod
        def get_x_label(is_log):
            x_label = "Pixel intensities"
            if is_log:
                x_label += f" (using log with base {log_base})"
            return x_label

        # Formats the plot for AGGREGATE (a group of images) rgb channel intensities plot
        @staticmethod
        def format_aggregate_rgb_channel_intensities_plot(ax: plt.Axes, class_name,
                                                          ignore_zeros=True,
                                                          is_log=True,
                                                          add_title=True) -> None:
            title = f"Intensity distributions of the RGB channels for the class \"{class_name}\""
            if ignore_zeros:
                title += "\n(Zero values omitted)"

            if add_title:
                ax.set_title(title)

            ax.set_xlabel(PlotFormatter.RgbChannelsFormatter.get_x_label(is_log))
            ax.set_ylabel('Normalized frequency')
            ax.grid(True)

        # Formats the rgb channel intensities plot for a SINGLE image
        @staticmethod
        def format_rgb_channel_intensities_plot(ax, image_name,
                                                ignore_zeros=True,
                                                is_log=True,
                                                add_title=True) -> None:
            title = f"Intensity distributions of the RGB channels for the image \"{image_name}\""
            if ignore_zeros:
                title += "\n(Zero values omitted)"

            if add_title:
                ax.set_title(title)

            ax.set_xlabel(PlotFormatter.RgbChannelsFormatter.get_x_label(is_log))
            ax.set_ylabel('Normalized frequency')
            ax.grid(True)


# Defining aliases
pf_rgb = PlotFormatter.RgbChannelsFormatter


# Plots the dimensions of each image on a histogram
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

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 15))

    df['height'].plot(kind='hist', bins=5, ax=ax[0])
    ax[0].set_title('Image height distribution')
    ax[0].set_xlabel('Image Height (px)')
    ax[0].set_ylabel('Number of images')

    df['width'].plot(kind='hist', bins=5, ax=ax[1])
    ax[1].set_title('Image width distribution')
    ax[1].set_xlabel('Image Width (px)')
    ax[1].set_ylabel('Number of images')

    plt.show()


# Plots the aggregate rgb channel intensities of a group of images. All should be in the same class.
def plot_aggregate_pixel_intensity_histogram(class_name, image_paths: List[str], ignore_zeros=True):
    red_pixels, green_pixels, blue_pixels = Data.calculate_aggregate_rgb_channel_intensities(image_paths)

    fig, ax = plt.subplots()
    _ = PlotCreator.generate_rgb_channel_intensities_plot(ax, red_pixels, green_pixels, blue_pixels)

    if len(image_paths) == 1:
        pf_rgb.format_rgb_channel_intensities_plot(ax, image_paths[0])
    else:
        pf_rgb.format_aggregate_rgb_channel_intensities_plot(ax, class_name)

    pf_rgb.add_legend(ax=ax)
    plt.show()


# Plots the aggregate rgb channel intensities of all classes in a 2x2 grid.
def plot_2_by_2_aggregate_pixel_intensity_histogram(anger_image_paths, engaged_image_paths, happy_image_paths,
                                                    neutral_image_paths):
    ignore_zeros = True
    current_index = 0
    image_paths_and_classes = [
        (anger_image_paths, "anger"),
        (engaged_image_paths, "engaged"),
        (happy_image_paths, "happy"),
        (neutral_image_paths, "neutral")
    ]

    fig = plt.figure(figsize=(12, 8))
    outer_grid = gridspec.GridSpec(2, 2, wspace=0.25, hspace=0.5)

    for i in range(2):
        for j in range(2):
            image_paths = image_paths_and_classes[i][0]
            class_name = image_paths_and_classes[i][1]

            ax = fig.add_subplot(outer_grid[i, j])

            red_pixels, green_pixels, blue_pixels = Data.calculate_aggregate_rgb_channel_intensities(image_paths,ignore_zeros=ignore_zeros)
            _ = PlotCreator.generate_rgb_channel_intensities_plot(ax, red_pixels, green_pixels, blue_pixels)
            pf_rgb.format_aggregate_rgb_channel_intensities_plot(ax, class_name, ignore_zeros=ignore_zeros)

            current_index += 1

    fig.suptitle(f"Aggregate RGB Pixel Intensities for each Class:")
    pf_rgb.add_legend(fig=fig)
    plt.show()


def main():
    df = get_metadata()
    df_grouped_by_label = df.groupby('label')

    if len(sys.argv) > 1:
        arg = sys.argv[1]

        match arg:
            case "anger":
                print("Plotting the RGB pixel intensities histogram for the class \"anger\"")
                df_anger = df_grouped_by_label.get_group('anger')
                plot_aggregate_pixel_intensity_histogram('anger', df_anger['path'].tolist())
            case "engaged":
                print("Plotting the RGB pixel intensities histogram for the class \"engaged\"")
                df_engaged = df_grouped_by_label.get_group('engaged')
                plot_aggregate_pixel_intensity_histogram('engaged', df_engaged['path'].tolist())
            case "happy":
                print("Plotting the RGB pixel intensities histogram for the class \"happy\"")
                df_happy = df_grouped_by_label.get_group('happy')
                plot_aggregate_pixel_intensity_histogram('happy', df_happy['path'].tolist())
            case "neutral":
                print("Plotting the RGB pixel intensities histogram for the class \"neutral\"")
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
        print("Plotting the RGB pixel intensities histogram for all classes. This might take time.")
        anger_image_paths = df_grouped_by_label.get_group('anger')['path'].tolist()
        engaged_image_paths = df_grouped_by_label.get_group('engaged')['path'].tolist()
        happy_image_paths = df_grouped_by_label.get_group('happy')['path'].tolist()
        neutral_image_paths = df_grouped_by_label.get_group('neutral')['path'].tolist()
        plot_2_by_2_aggregate_pixel_intensity_histogram(anger_image_paths, engaged_image_paths, happy_image_paths,
                                                        neutral_image_paths)


if __name__ == "__main__":
    main()
