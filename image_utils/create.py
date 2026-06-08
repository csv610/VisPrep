"""Image creation: blank, random noise, and gradient images."""
from __future__ import annotations

import random
from typing import Literal

import numpy as np
from PIL import Image

from image_utils._utils import logger


def create_blank_image(
    width: int,
    height: int,
    color: tuple[int, ...] | Literal["random"] = (255, 255, 255),
    image_mode: Literal["RGB", "RGBA", "L"] = "RGB",
) -> Image.Image:
    """Create a solid-colour (or random-colour) image.

    Example::

        >>> from image_utils import create_blank_image
        >>> img = create_blank_image(64, 64, color=(255, 0, 0))
        >>> img.size
        (64, 64)
        >>> img.getpixel((0, 0))
        (255, 0, 0)

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        color: RGB/RGBA tuple or ``"random"`` for a random solid colour.
        image_mode: ``"RGB"``, ``"RGBA"``, or ``"L"`` (grayscale).

    Returns:
        New PIL Image filled with the requested colour.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Width and height must be positive, got {width}x{height}")
    if image_mode not in ("RGB", "RGBA", "L"):
        raise ValueError(f"image_mode must be 'RGB', 'RGBA', or 'L', got '{image_mode}'")

    try:
        if color == "random":
            if image_mode == "L":
                fill_color = random.randint(0, 255)
            elif image_mode == "RGBA":
                fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255)
            else:
                fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        else:
            if not isinstance(color, tuple) or not all(0 <= c <= 255 for c in color):
                raise ValueError(f"Color values must be 0-255 tuple, got {color}")

            if image_mode == "L" and len(color) != 1:
                raise ValueError(f"Grayscale requires single value, got {color}")
            elif image_mode == "RGBA" and len(color) not in (3, 4):
                raise ValueError(f"RGBA requires 3 or 4 values, got {color}")
            elif image_mode == "RGB" and len(color) != 3:
                raise ValueError(f"RGB requires 3 values, got {color}")

            fill_color = color

        image = Image.new(image_mode, (width, height), fill_color)
        logger.info(f"Created blank {image_mode} image {width}x{height}")
        return image
    except Exception:
        raise


def create_random_image(width: int, height: int, image_mode: Literal["RGB", "RGBA", "L"] = "RGB") -> Image.Image:
    """Create an image filled with random noise pixels.

    Example::

        >>> from image_utils import create_random_image
        >>> img = create_random_image(64, 48)
        >>> img.size
        (64, 48)
        >>> img.mode
        'RGB'

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        image_mode: ``"RGB"``, ``"RGBA"``, or ``"L"`` (grayscale).

    Returns:
        New PIL Image with random pixel data.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Width and height must be positive, got {width}x{height}")
    if image_mode not in ("RGB", "RGBA", "L"):
        raise ValueError(f"image_mode must be 'RGB', 'RGBA', or 'L', got '{image_mode}'")

    channels = {"L": (height, width), "RGBA": (height, width, 4), "RGB": (height, width, 3)}
    data = np.random.randint(0, 256, channels[image_mode], dtype=np.uint8)
    image = Image.fromarray(data, mode=image_mode)
    logger.info(f"Created random noise {image_mode} image {width}x{height}")
    return image


def create_gradient_image(
    width: int,
    height: int,
    start_color: tuple[int, int, int],
    end_color: tuple[int, int, int],
    direction: Literal["horizontal", "vertical", "diagonal"] = "horizontal",
) -> Image.Image:
    """Create a smooth colour gradient image.

    All directions are fully vectorised (numpy).

    Example::

        >>> from image_utils import create_gradient_image
        >>> img = create_gradient_image(64, 64, (255, 0, 0), (0, 0, 255))
        >>> img.size
        (64, 64)
        >>> img.mode
        'RGB'
        >>> img.getpixel((0, 0))
        (255, 0, 0)

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        start_color: RGB colour at the origin.
        end_color: RGB colour at the far edge / corner.
        direction: ``"horizontal"``, ``"vertical"``, or ``"diagonal"``.

    Returns:
        New PIL Image in RGB mode.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Width and height must be positive, got {width}x{height}")
    if not all(0 <= c <= 255 for c in start_color) or not all(0 <= c <= 255 for c in end_color):
        raise ValueError("Color values must be 0-255")
    if direction not in ("horizontal", "vertical", "diagonal"):
        raise ValueError(f"direction must be 'horizontal', 'vertical', or 'diagonal', got '{direction}'")

    try:
        sc, ec = np.array(start_color, dtype=np.float64).reshape(1, 1, 3), np.array(end_color, dtype=np.float64).reshape(1, 1, 3)

        if direction == "horizontal":
            t = np.linspace(0, 1, width, dtype=np.float64).reshape(1, width, 1)
            data = sc * (1 - t) + ec * t
            data = data.repeat(height, axis=0)
        elif direction == "vertical":
            t = np.linspace(0, 1, height, dtype=np.float64).reshape(height, 1, 1)
            data = sc * (1 - t) + ec * t
            data = data.repeat(width, axis=1)
        else:
            y_grid, x_grid = np.mgrid[0:height, 0:width].astype(np.float64)
            max_dist = (height**2 + width**2) ** 0.5
            t = (np.sqrt(x_grid**2 + y_grid**2) / max_dist)[:, :, np.newaxis]
            data = sc * (1 - t) + ec * t

        data = np.clip(data, 0, 255).astype(np.uint8)

        image = Image.fromarray(data, mode="RGB")
        logger.info(f"Created {direction} gradient image {width}x{height}")
        return image
    except Exception:
        raise
