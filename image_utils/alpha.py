"""Alpha channel operations: add and remove transparency."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from image_utils._utils import _convert_to_rgb, logger


def remove_alpha_channel(image: str | Image.Image, background_color: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """Flatten transparency against a solid background.

    RGBA, LA, and palette images are composited over the background colour.
    RGB and grayscale images are returned unchanged.

    Args:
        image: File path or PIL Image.
        background_color: RGB colour to use behind transparent areas.

    Returns:
        PIL Image in RGB mode.
    """
    try:
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {image}")
            try:
                with Image.open(path) as img:
                    img = img.copy()
            except Image.UnidentifiedImageError:
                raise ValueError(f"File is not a valid image: {image}")
        elif isinstance(image, Image.Image):
            img = image.copy()
        else:
            raise TypeError(f"image must be a file path (str) or PIL Image object, got {type(image)}")

        if not (isinstance(background_color, tuple) and len(background_color) == 3):
            raise ValueError(f"background_color must be an RGB tuple (R, G, B), got {background_color}")
        if not all(isinstance(c, int) and 0 <= c <= 255 for c in background_color):
            raise ValueError(f"All color values must be integers 0-255, got {background_color}")

        if img.mode in ("RGB", "L"):
            return img

        result = _convert_to_rgb(img, background_color)
        logger.info(f"Removed alpha channel from {img.mode} image (size: {img.size})")
        return result
    except (FileNotFoundError, ValueError, TypeError):
        raise
    except Exception:
        raise


def add_alpha_channel(image: str | Image.Image, alpha_value: int = 255) -> Image.Image:
    """Add an alpha channel to an image.

    RGB → RGBA, L → LA (grayscale+alpha).  RGBA and LA images are returned
    unchanged unless a custom ``alpha_value`` is given.

    Args:
        image: File path or PIL Image.
        alpha_value: Opacity 0–255 (255 = fully opaque).

    Returns:
        PIL Image with an alpha channel (RGBA or LA mode).
    """
    try:
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {image}")
            try:
                with Image.open(path) as img:
                    img = img.copy()
            except Image.UnidentifiedImageError:
                raise ValueError(f"File is not a valid image: {image}")
        elif isinstance(image, Image.Image):
            img = image.copy()
        else:
            raise TypeError(f"image must be a file path (str) or PIL Image object, got {type(image)}")

        if not isinstance(alpha_value, int) or not (0 <= alpha_value <= 255):
            raise ValueError(f"alpha_value must be an integer 0-255, got {alpha_value}")

        if img.mode == "RGBA":
            return img
        elif img.mode == "LA":
            return img
        elif img.mode == "RGB":
            rgba = img.convert("RGBA")
        elif img.mode == "L":
            la = img.convert("LA")
            if alpha_value < 255:
                la.putalpha(Image.new("L", img.size, alpha_value))
            logger.info(f"Added alpha channel to grayscale image (alpha={alpha_value})")
            return la
        elif img.mode == "P":
            rgba = img.convert("RGBA")
        else:
            rgba = img.convert("RGB").convert("RGBA")

        if alpha_value < 255:
            rgba.putalpha(Image.new("L", img.size, alpha_value))
        logger.info(f"Added alpha channel to {img.mode} image (alpha={alpha_value})")
        return rgba
    except (FileNotFoundError, ValueError, TypeError):
        raise
    except Exception:
        raise
