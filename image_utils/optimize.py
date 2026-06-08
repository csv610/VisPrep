"""Optimization: resize-to-max-size, save-to-max-size, estimate size."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Literal

from PIL import Image

from image_utils._utils import _convert_to_rgb, _validate_file_exists, MIN_IMAGE_DIMENSION, logger


def resize_to_max_size(
    image_path: str,
    max_size: float,
    size_unit: Literal["MB", "GB"] = "MB",
    target_format: Literal["JPEG", "PNG", "WEBP"] = "JPEG",
    min_quality: int = 10,
) -> Image.Image:
    """Reduce quality and/or scale until the image fits within a byte limit.

    Two-phase approach: first reduce JPEG/WEBP quality, then scale down
    dimensions.  The original file is never modified.

    Args:
        image_path: Path to the source image.
        max_size: Target size limit.
        size_unit: ``"MB"`` or ``"GB"``.
        target_format: Output format for size estimation.
        min_quality: Lowest acceptable quality before scaling kicks in.

    Returns:
        Resized PIL Image (always RGB mode).
    """
    path = _validate_file_exists(image_path)
    if max_size <= 0:
        raise ValueError(f"max_size must be positive, got {max_size}")
    if size_unit not in ("MB", "GB"):
        raise ValueError(f"size_unit must be 'MB' or 'GB', got '{size_unit}'")
    if not (1 <= min_quality <= 100):
        raise ValueError(f"min_quality must be 1-100, got {min_quality}")

    max_bytes = max_size * (1024 * 1024 * (1024 if size_unit == "GB" else 1))

    try:
        with Image.open(path) as img:
            img = _convert_to_rgb(img).copy()
            original_size = path.stat().st_size

            if original_size <= max_bytes:
                logger.info(f"Image {path.name} already within {max_size}{size_unit} limit ({original_size / 1024 / 1024:.2f}MB)")
                return img

            logger.warning(f"Image {path.name} is {original_size / 1024 / 1024:.2f}MB, exceeds limit of {max_size}{size_unit}. Compressing...")

            quality = 85
            scale_factor = 1.0
            best_img = img

            while quality >= min_quality:
                output = io.BytesIO()
                best_img.save(output, format=target_format, quality=quality, optimize=True)
                if output.tell() <= max_bytes:
                    logger.info(f"Compressed to {output.tell() / 1024 / 1024:.2f}MB at quality {quality}")
                    return best_img
                quality -= 5

            while scale_factor > 0.1:
                scale_factor -= 0.1
                new_w = max(MIN_IMAGE_DIMENSION, int(img.width * scale_factor))
                new_h = max(MIN_IMAGE_DIMENSION, int(img.height * scale_factor))
                scaled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

                output = io.BytesIO()
                scaled.save(output, format=target_format, quality=min_quality, optimize=True)
                if output.tell() <= max_bytes:
                    logger.info(f"Compressed to {output.tell() / 1024 / 1024:.2f}MB at {int(scale_factor * 100)}% scale, quality {min_quality}")
                    return scaled

            raise ValueError(
                f"Cannot compress image to fit within {max_size}{size_unit} limit. "
                f"Minimum size: {output.tell() / 1024 / 1024:.2f}MB at {int(scale_factor * 100)}% scale, quality {min_quality}"
            )
    except Image.UnidentifiedImageError:
        raise ValueError(f"File is not a valid image: {image_path}")
    except Exception:
        raise


def save_image_to_max_size(
    image_path: str,
    output_path: str,
    max_size: float,
    size_unit: Literal["MB", "GB"] = "MB",
    target_format: Literal["PNG", "JPG", "JPEG", "WEBP"] = "JPEG",
    min_quality: int = 10,
) -> str:
    """Convenience: resize then save directly to a file.

    Args:
        image_path: Path to the source image.
        output_path: Destination file path.
        max_size: Target size limit.
        size_unit: ``"MB"`` or ``"GB"``.
        target_format: Output format.
        min_quality: Lowest acceptable quality.

    Returns:
        The ``output_path`` string.
    """
    fmt = "JPEG" if target_format.upper() in ("JPG", "JPEG") else target_format.upper()
    from image_utils.save import save_pil_image
    return save_pil_image(resize_to_max_size(image_path, max_size, size_unit, fmt, min_quality), output_path, format=fmt)


def estimate_compressed_size(image_path: str, target_format: Literal["JPEG", "PNG", "WEBP"] = "JPEG", quality: int = 85) -> float:
    """Estimate the file size if the image were re-encoded at the given settings.

    Useful for checking whether compression is worthwhile before running it.

    Args:
        image_path: Path to the source image.
        target_format: ``"JPEG"``, ``"PNG"``, or ``"WEBP"``.
        quality: Quality 1–100 (JPEG/WEBP only).

    Returns:
        Estimated size in megabytes.
    """
    path = _validate_file_exists(image_path)
    if target_format not in ("JPEG", "PNG", "WEBP"):
        raise ValueError(f"target_format must be 'JPEG', 'PNG', or 'WEBP', got '{target_format}'")
    if target_format in ("JPEG", "WEBP") and not (1 <= quality <= 100):
        raise ValueError(f"quality must be 1-100, got {quality}")

    try:
        with Image.open(path) as img:
            if target_format in ("JPEG", "WEBP"):
                img = _convert_to_rgb(img)
            elif img.mode not in ("RGB", "RGBA", "L"):
                img = img.convert("RGB")

            output = io.BytesIO()
            kwargs = {"format": target_format}
            if target_format in ("JPEG", "WEBP"):
                kwargs["quality"] = quality
            if target_format == "JPEG":
                kwargs["optimize"] = True
            img.save(output, **kwargs)
            return output.tell() / (1024 * 1024)
    except Image.UnidentifiedImageError:
        raise ValueError(f"File is not a valid image: {image_path}")
    except Exception:
        raise
