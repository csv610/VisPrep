"""Image saving functions: single, batch, from path/PIL/cv2/base64."""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from image_utils._utils import _convert_to_rgb, logger
from image_utils.convert import cv2_to_pil, b64_to_pil


_FORMAT_ALIASES = {"JPG": "JPEG"}
_SAVE_FORMATS = frozenset({"PNG", "JPEG", "WEBP"})


def save_image(
    image_data: str | Image.Image | np.ndarray,
    output_path: str,
    format: Literal["PNG", "JPG", "JPEG", "WEBP"] = "PNG",
    quality: int = 85,
    input_type: Literal["path", "pil", "cv2", "base64"] | None = None,
) -> str:
    """Save an image to disk with automatic input-type detection.

    Accepts a file path, PIL Image, OpenCV numpy array, or base64 string.
    The input type is auto-detected when ``input_type`` is ``None``.

    Args:
        image_data: File path, PIL Image, OpenCV array, or base64 string.
        output_path: Destination file path (parent dirs created if missing).
        format: Output format — ``"PNG"`` (default), ``"JPEG"``/``"JPG"``,
            or ``"WEBP"``.
        quality: JPEG/WEBP quality 1–100 (default 85).
        input_type: Explicit type hint — ``"path"``, ``"pil"``, ``"cv2"``,
            ``"base64"``, or ``None`` for auto-detection.

    Returns:
        Absolute path to the saved file.

    Raises:
        ValueError: Unsupported format, quality out of range, or
            unrecognized input type.
    """
    fmt = _FORMAT_ALIASES.get(format.upper(), format.upper())
    if fmt not in _SAVE_FORMATS:
        raise ValueError(f"format must be one of {sorted(_SAVE_FORMATS)}, got '{format}'")
    if not (1 <= quality <= 100):
        raise ValueError(f"quality must be 1-100, got {quality}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        if input_type is None:
            if isinstance(image_data, str):
                p = Path(image_data)
                if p.exists():
                    input_type = "path"
                elif image_data.startswith("data:"):
                    input_type = "base64"
                elif image_data.endswith((".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff")):
                    input_type = "path"
                else:
                    try:
                        base64.b64decode(image_data)
                        input_type = "base64"
                    except Exception:
                        input_type = "path"
            elif isinstance(image_data, Image.Image):
                input_type = "pil"
            elif isinstance(image_data, np.ndarray):
                input_type = "cv2"
            else:
                raise ValueError(f"Unsupported image_data type: {type(image_data)}. Expected str, PIL Image, or numpy array")

        pil_image = {
            "path": lambda: Image.open(image_data),
            "pil": lambda: image_data,
            "cv2": lambda: cv2_to_pil(image_data),
            "base64": lambda: b64_to_pil(image_data),
        }[input_type]()

        if fmt == "JPEG":
            pil_image = _convert_to_rgb(pil_image)
        elif pil_image.mode not in ("RGB", "RGBA", "L"):
            pil_image = pil_image.convert("RGB")

        kwargs = {}
        if fmt in ("JPEG", "WEBP"):
            kwargs["quality"] = quality
        if fmt == "JPEG":
            kwargs["optimize"] = True

        pil_image.save(str(out), format=fmt, **kwargs)
        logger.info(f"Saved image to {output_path} ({fmt}, {quality}% quality)")
        return str(out)
    except Exception:
        raise


def save_images_batch(
    image_data_list: list[str | Image.Image | np.ndarray],
    output_directory: str,
    format: Literal["PNG", "JPG", "JPEG", "WEBP"] = "PNG",
    quality: int = 85,
    filename_prefix: str = "image",
    input_type: Literal["path", "pil", "cv2", "base64"] | None = None,
) -> list[str]:
    """Save multiple images to a directory with auto-generated filenames.

    Args:
        image_data_list: List of file paths, PIL Images, numpy arrays, or
            base64 strings.
        output_directory: Destination directory (created if missing).
        format: ``"PNG"``, ``"JPG"``/``"JPEG"``, or ``"WEBP"``.
        quality: JPEG/WEBP quality 1–100.
        filename_prefix: Prefix for auto-generated filenames.
        input_type: Explicit type hint (bypasses auto-detection).

    Returns:
        List of saved file paths.
    """
    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt_upper = format.upper()
    ext_map = {"PNG": ".png", "WEBP": ".webp"}
    ext = ext_map.get(fmt_upper, ".jpg")

    saved = []
    for idx, data in enumerate(image_data_list):
        saved.append(save_image(data, str(out_dir / f"{filename_prefix}_{idx:04d}{ext}"), format=format, quality=quality, input_type=input_type))

    logger.info(f"Saved {len(saved)} images to {output_directory}")
    return saved


def save_from_path(input_path: str, output_path: str, format: Literal["PNG", "JPG", "JPEG", "WEBP"] = "PNG", quality: int = 85) -> str:
    """Load an image from disk and save it, optionally converting format."""
    return save_image(input_path, output_path, format=format, quality=quality, input_type="path")


def save_pil_image(pil_image: Image.Image, output_path: str, format: Literal["PNG", "JPG", "JPEG", "WEBP"] = "PNG", quality: int = 85) -> str:
    """Save a PIL Image object to disk."""
    return save_image(pil_image, output_path, format=format, quality=quality, input_type="pil")


def save_cv2_image(cv_image: np.ndarray, output_path: str, format: Literal["PNG", "JPG", "JPEG", "WEBP"] = "PNG", quality: int = 85) -> str:
    """Save an OpenCV image (numpy array) to disk. Requires opencv-python."""
    return save_image(cv_image, output_path, format=format, quality=quality, input_type="cv2")


def save_from_b64(b64_string: str, output_path: str, format: Literal["PNG", "JPG", "JPEG", "WEBP"] = "PNG", quality: int = 85) -> str:
    """Decode a base64 string and save the resulting image to disk."""
    return save_image(b64_string, output_path, format=format, quality=quality, input_type="base64")
