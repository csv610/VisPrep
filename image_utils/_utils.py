"""Private helpers, constants, and optional cv2 import."""
from __future__ import annotations

import io
import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False

MAX_IMAGE_SIZE_MB = 20
MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024
MAX_TOTAL_IMAGE_PAYLOAD_MB = 50
MAX_TOTAL_IMAGE_PAYLOAD_BYTES = MAX_TOTAL_IMAGE_PAYLOAD_MB * 1024 * 1024
MIN_IMAGE_DIMENSION = 32

VALID_IMAGE_EXTENSIONS: frozenset[str] = frozenset({
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif",
})


_CMYK_YCBCR_MODES = {"CMYK", "YCbCr", "LAB", "HSV", "I", "F"}


def _convert_to_rgb(img: Image.Image, background_color: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    if img.mode in ("RGBA", "LA", "P"):
        rgb_img = Image.new("RGB", img.size, background_color)
        if img.mode == "RGBA":
            rgb_img.paste(img, mask=img.split()[3])
        elif img.mode == "LA":
            rgb_converted = img.convert("RGB")
            rgb_img.paste(rgb_converted, mask=img.split()[1])
        else:
            rgba = img.convert("RGBA")
            rgb_img.paste(rgba.convert("RGB"), mask=rgba.split()[3])
        return rgb_img
    elif img.mode != "RGB":
        if img.mode in _CMYK_YCBCR_MODES:
            logger.info(f"Converting image from {img.mode} to RGB")
        return img.convert("RGB")
    return img


def _validate_file_exists(file_path: str) -> Path:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return path


def _validate_directory_exists(directory_path: str) -> Path:
    directory = Path(directory_path)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory_path}")
    return directory


def _estimate_base64_size(data: bytes) -> int:
    return int(len(data) * 4 / 3) + 50


def _resize_image_to_quality(image_path: str, quality: int = 85, scale_factor: float = 1.0) -> bytes:
    try:
        with Image.open(image_path) as img:
            img = _convert_to_rgb(img).copy()
            if scale_factor < 1.0:
                new_width = max(MIN_IMAGE_DIMENSION, int(img.width * scale_factor))
                new_height = max(MIN_IMAGE_DIMENSION, int(img.height * scale_factor))
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            img.save(output, format="JPEG", quality=quality, optimize=True)
            return output.getvalue()
    except Exception:
        raise
