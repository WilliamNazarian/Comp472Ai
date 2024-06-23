from PIL import Image as PilImage

import numpy as np


def convert_to_redscale(image):
    r, g, b = image.split()
    g = g.point(lambda p: 0)
    b = b.point(lambda p: 0)
    return PilImage.merge("RGB", (r, g, b))


def convert_to_greenscale(image):
    r, g, b = image.split()
    r = g.point(lambda p: 0)
    b = b.point(lambda p: 0)
    return PilImage.merge("RGB", (r, g, b))


def convert_to_bluescale(image):
    r, g, b = image.split()
    r = g.point(lambda p: 0)
    g = b.point(lambda p: 0)
    return PilImage.merge("RGB", (r, g, b))


def get_image_pixel_intensity(image, transform=None):
    transformed_image = transform(image) if transform else image
    pixels = np.asarray(transformed_image).flatten()
    return pixels
