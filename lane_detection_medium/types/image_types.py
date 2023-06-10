from typing import Any, NewType, Union

import numpy as np

Image = NewType("Image", np.ndarray)
ImageGray = NewType("ImageGray", Image)
ImageBinary = NewType("ImageBinary", Image)
ImageBinaryInverted = NewType("ImageBinaryInverted", Image)
ImageRGB = NewType("ImageRGB", Image)


def is_image(img: Union[Image, Any]) -> bool:
    """Check that object is image."""

    return isinstance(img, np.ndarray) and len(img.shape) in {2, 3}


def is_gray(img: Image) -> bool:
    """Check is image is single-channeled."""

    return get_channels(img) == 1


def is_ycbcr(img: Image) -> bool:
    """Check if image look like YCbCr color space."""

    if is_gray(img):
        return False

    color_means = np.mean(img, axis=(0, 1))
    return np.any(color_means > 50) and np.all(np.abs(color_means[1:] - 128) < 5)


def get_channels(img: Image) -> int:
    """Get the number of channels of image."""

    if not is_image(img):
        raise ValueError("Input should be image")

    if len(img.shape) > 2:
        return img.shape[-1]
    return 1
