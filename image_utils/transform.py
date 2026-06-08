"""Image transformations: resize, crop, square, batch resize-to-fit."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from PIL import Image

from image_utils._utils import (
    _convert_to_rgb, _estimate_base64_size, _resize_image_to_quality,
    MAX_TOTAL_IMAGE_PAYLOAD_BYTES, MAX_TOTAL_IMAGE_PAYLOAD_MB, logger,
)


def resize_images_to_fit(image_paths: list[str]) -> list[str]:
    """Compress images as a batch to stay under ``MAX_TOTAL_IMAGE_PAYLOAD_BYTES``.

    Iteratively reduces quality then scale until the total payload fits the limit.
    Temp files named ``.{stem}_resized_{q}_{s}.jpg`` are written
    alongside the originals.

    Args:
        image_paths: List of paths to image files.

    Returns:
        Original paths if already within limit, otherwise paths to resized temp files.

    Raises:
        ValueError: Cannot resize images enough to meet the limit.
    """
    if not image_paths:
        return image_paths

    total_size = 0
    for path in image_paths:
        p = Path(path)
        with open(p, "rb") as f:
            data = f.read()
            total_size += _estimate_base64_size(data)

    if total_size <= MAX_TOTAL_IMAGE_PAYLOAD_BYTES:
        logger.info(f"Total image payload {total_size / 1024 / 1024:.2f}MB is within {MAX_TOTAL_IMAGE_PAYLOAD_MB}MB limit")
        return image_paths

    logger.warning(f"Total image payload {total_size / 1024 / 1024:.2f}MB exceeds limit. Auto-resizing...")

    quality = 85
    scale_factor = 1.0
    total_resized_size = 0

    while quality > 10 or scale_factor < 1.0:
        resized_data = []
        total_resized_size = 0

        for path in image_paths:
            try:
                compressed = _resize_image_to_quality(path, quality, scale_factor)
                resized_data.append(compressed)
                total_resized_size += _estimate_base64_size(compressed)
            except Exception:
                raise

        if total_resized_size <= MAX_TOTAL_IMAGE_PAYLOAD_BYTES:
            temp_paths = []
            for i, data in enumerate(resized_data):
                p = Path(image_paths[i])
                temp = p.parent / f".{p.stem}_resized_{quality}_{int(scale_factor*100)}.jpg"
                temp.write_bytes(data)
                temp_paths.append(str(temp))
                logger.info(f"Resized {p.name} to {temp.stat().st_size / 1024 / 1024:.2f}MB (quality {quality}, scale {scale_factor:.2f})")
            return temp_paths

        if quality > 10:
            quality -= 5
        else:
            scale_factor -= 0.1

    raise ValueError(
        f"Cannot resize images to fit within {MAX_TOTAL_IMAGE_PAYLOAD_MB}MB limit. "
        f"Total size at minimum settings: {total_resized_size / 1024 / 1024:.2f}MB"
    )


def square_image(image_path: str, max_size: int, background_color: tuple[int, int, int], position: Literal["top-left", "center"] = "center") -> Image.Image:
    """Create a square canvas and embed the image, scaling down if needed.

    Args:
        image_path: Path to the source image.
        max_size: Side length of the square canvas in pixels.
        background_color: RGB fill color for canvas (``(R, G, B)``, 0–255 each).
        position: ``"center"`` (default) or ``"top-left"`` placement.

    Returns:
        New PIL Image with the image pasted onto a square background.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if max_size <= 0:
        raise ValueError(f"max_size must be positive, got {max_size}")
    if not all(0 <= c <= 255 for c in background_color):
        raise ValueError(f"background_color values must be 0-255, got {background_color}")
    if position not in ("top-left", "center"):
        raise ValueError(f"position must be 'top-left' or 'center', got '{position}'")

    try:
        with Image.open(path) as img:
            img = _convert_to_rgb(img)
            w, h = img.size

            if w > max_size or h > max_size:
                scale = min(max_size / w, max_size / h)
                img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
                w, h = img.size

            canvas = Image.new("RGB", (max_size, max_size), background_color)
            x = 0 if position == "top-left" else (max_size - w) // 2
            y = 0 if position == "top-left" else (max_size - h) // 2
            canvas.paste(img, (x, y))
            return canvas
    except Image.UnidentifiedImageError:
        raise ValueError(f"File is not a valid image: {image_path}")
    except Exception:
        raise


def resize_to_dimensions(image_path: str, width: int, height: int, background_color: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """Resize an image to fit within exact dimensions, adding padding as needed.

    Scales uniformly (maintaining aspect ratio) and centres the result on a
    coloured canvas.

    Args:
        image_path: Path to the source image.
        width: Target canvas width in pixels.
        height: Target canvas height in pixels.
        background_color: RGB fill for the padded area.

    Returns:
        New PIL Image at the requested dimensions.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if width <= 0 or height <= 0:
        raise ValueError(f"Dimensions must be positive, got {width}x{height}")
    if not all(0 <= c <= 255 for c in background_color):
        raise ValueError(f"background_color values must be 0-255, got {background_color}")

    try:
        with Image.open(path) as img:
            img = _convert_to_rgb(img)
            w, h = img.size
            scale = min(width / w, height / h)
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

            canvas = Image.new("RGB", (width, height), background_color)
            canvas.paste(img, ((width - img.width) // 2, (height - img.height) // 2))
            return canvas
    except Image.UnidentifiedImageError:
        raise ValueError(f"File is not a valid image: {image_path}")
    except Exception:
        raise


def crop(image: str | Image.Image, left: int, top: int, right: int, bottom: int) -> Image.Image:
    """Crop a rectangular region from an image.

    Args:
        image: File path or PIL Image.
        left: X coordinate of the left edge.
        top: Y coordinate of the top edge.
        right: X coordinate of the right edge (exclusive).
        bottom: Y coordinate of the bottom edge (exclusive).

    Returns:
        Cropped PIL Image.

    Raises:
        ValueError: Invalid or out-of-bounds crop coordinates.
    """
    if left >= right or top >= bottom:
        raise ValueError(f"Invalid crop coordinates: left={left}, top={top}, right={right}, bottom={bottom}")

    try:
        if isinstance(image, Image.Image):
            img = image
        else:
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {image}")
            img = Image.open(path)

        w, h = img.size
        if left < 0 or top < 0 or right > w or bottom > h:
            raise ValueError(f"Crop coordinates out of bounds. Image size: {w}x{h}, requested: left={left}, top={top}, right={right}, bottom={bottom}")
        return img.crop((left, top, right, bottom))
    except Image.UnidentifiedImageError:
        raise ValueError(f"File is not a valid image: {image}")
    except Exception:
        raise
