"""Format conversions between PIL, OpenCV, and base64."""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from visprep._utils import (
    _convert_to_rgb, CV2_AVAILABLE, cv2,
    MAX_IMAGE_SIZE_BYTES, MIN_IMAGE_DIMENSION, VALID_IMAGE_EXTENSIONS, logger,
)


_MIME_MAP: dict[str, str] = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff", ".tif": "image/tiff",
}


def encode_to_base64(image_path: str) -> str:
    """Read an image file and return a data-URI base64 string.

    Args:
        image_path: Path to the image file.

    Returns:
        Base64 string with data URI prefix (e.g. ``data:image/jpeg;base64,...``).

    Raises:
        FileNotFoundError: File does not exist.
        ValueError: Unsupported format, undersized, or exceeds size limit.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    ext = path.suffix.lower()
    mime = _MIME_MAP.get(ext)
    if mime is None:
        raise ValueError(f"File is not a valid image: {image_path}. Supported formats: PNG, JPEG, WEBP, GIF")

    try:
        with open(path, "rb") as f:
            file_data = f.read()
    except Exception:
        raise

    if len(file_data) > MAX_IMAGE_SIZE_BYTES:
        file_size_mb = len(file_data) / (1024 * 1024)
        raise ValueError(f"Image file exceeds {MAX_IMAGE_SIZE_BYTES / 1024 / 1024:.0f}MB limit: {image_path} ({file_size_mb:.2f}MB)")

    try:
        with Image.open(io.BytesIO(file_data)) as img:
            if img.width < MIN_IMAGE_DIMENSION or img.height < MIN_IMAGE_DIMENSION:
                raise ValueError(f"Image must be at least {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION} pixels: {image_path}")
    except ValueError:
        raise
    except Image.UnidentifiedImageError:
        raise ValueError(f"File is not a valid image or is corrupt: {image_path}")
    except Exception:
        raise

    try:
        encoded_file = base64.b64encode(file_data).decode("utf-8")
        return f"data:{mime};base64,{encoded_file}"
    except Exception:
        raise


def b64_to_pil(b64_string: str) -> Image.Image:
    """Decode a base64 string to a PIL Image.

    Example::

        >>> from visprep import b64_to_pil
        >>> b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
        >>> img = b64_to_pil(b64)
        >>> img.size
        (1, 1)

    Args:
        b64_string: Raw base64 or data-URI prefixed string.

    Returns:
        PIL Image object.

    Raises:
        ValueError: Invalid base64 data or corrupt image.
    """
    try:
        if "," in b64_string:
            b64_string = b64_string.split(",", 1)[1]
        image_data = base64.b64decode(b64_string)
        return Image.open(io.BytesIO(image_data))
    except Exception as e:
        raise ValueError(f"Invalid base64 image data: {e}")


def pil_to_b64(image: Image.Image, image_format: Literal["JPEG", "PNG", "WEBP"] = "JPEG", quality: int = 85, include_data_uri: bool = True) -> str:
    """Encode a PIL Image to a base64 string.

    Example::

        >>> from PIL import Image
        >>> from visprep import pil_to_b64
        >>> img = Image.new("RGB", (10, 10), color=(0, 128, 0))
        >>> b64 = pil_to_b64(img, image_format="PNG", include_data_uri=False)
        >>> isinstance(b64, str)
        True
        >>> b64[:6]
        'iVBORw'

    Args:
        image: PIL Image object.
        image_format: Output format — ``"JPEG"``, ``"PNG"``, or ``"WEBP"``.
        quality: JPEG/WEBP quality 1–100.
        include_data_uri: Prepend ``data:image/...;base64,`` prefix.

    Returns:
        Base64-encoded string.

    Raises:
        ValueError: Invalid image, format, or quality.
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image object")
    if image_format not in ("JPEG", "PNG", "WEBP"):
        raise ValueError(f"image_format must be 'JPEG', 'PNG', or 'WEBP', got '{image_format}'")
    if not (1 <= quality <= 100):
        raise ValueError(f"quality must be 1-100, got {quality}")

    try:
        if image_format == "JPEG":
            image = _convert_to_rgb(image)
        elif image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        output = io.BytesIO()
        save_kwargs = {"format": image_format}
        if image_format in ("JPEG", "WEBP"):
            save_kwargs["quality"] = quality
            if image_format == "JPEG":
                save_kwargs["optimize"] = True

        image.save(output, **save_kwargs)
        b64_string = base64.b64encode(output.getvalue()).decode("utf-8")

        if include_data_uri:
            return f"data:image/{image_format.lower()};base64,{b64_string}"
        return b64_string
    except Exception:
        raise


def cv2_to_pil(cv_image: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR/BGRA array to a PIL RGB/RGBA Image.

    Requires opencv-python. Raises ``ImportError`` if not installed.

    Args:
        cv_image: OpenCV image as a numpy array (H×W, H×W×3, or H×W×4).

    Returns:
        PIL Image in RGB or RGBA mode.

    Raises:
        ImportError: opencv-python not installed.
        ValueError: Unsupported array shape or channel count.
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV (cv2) is required. Install with: pip install opencv-python")
    if not isinstance(cv_image, np.ndarray):
        raise ValueError(f"Input must be a numpy array, got {type(cv_image)}")

    try:
        if len(cv_image.shape) == 2:
            return Image.fromarray(cv_image, mode="L")
        if len(cv_image.shape) == 3:
            if cv_image.shape[2] == 3:
                return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            elif cv_image.shape[2] == 4:
                return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGBA), mode="RGBA")
            raise ValueError(f"Unsupported number of channels: {cv_image.shape[2]}")
        raise ValueError(f"Unsupported image shape: {cv_image.shape}")
    except Exception:
        raise


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Convert a PIL Image to an OpenCV BGR/BGRA array.

    Requires opencv-python. Raises ``ImportError`` if not installed.

    Args:
        image: PIL Image object.

    Returns:
        OpenCV image as numpy array (BGR or BGRA ordering).

    Raises:
        ImportError: opencv-python not installed.
        ValueError: Invalid input type.
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV (cv2) is required. Install with: pip install opencv-python")
    if not isinstance(image, Image.Image):
        raise ValueError(f"Input must be a PIL Image object, got {type(image)}")

    try:
        if image.mode == "L":
            return np.array(image)
        if image.mode == "RGBA":
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)
        return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception:
        raise


def cv2_to_b64(cv_image: np.ndarray, image_format: Literal["JPEG", "PNG", "WEBP"] = "JPEG", quality: int = 85, include_data_uri: bool = True) -> str:
    """Convert an OpenCV BGR array to a base64 string.

    Requires opencv-python.  Delegates to ``pil_to_b64()``.

    Args:
        cv_image: OpenCV BGR/BGRA numpy array.
        image_format: ``"JPEG"``, ``"PNG"``, or ``"WEBP"``.
        quality: JPEG/WEBP quality 1–100.
        include_data_uri: Prepend ``data:image/...;base64,`` prefix.

    Returns:
        Base64-encoded string.

    Raises:
        ImportError: opencv-python not installed.
    """
    return pil_to_b64(cv2_to_pil(cv_image), image_format, quality, include_data_uri)


def b64_to_cv2(b64_string: str) -> np.ndarray:
    """Decode a base64 string to an OpenCV BGR array.

    Requires opencv-python.  Delegates to ``b64_to_pil()``.

    Args:
        b64_string: Raw base64 or data-URI prefixed string.

    Returns:
        OpenCV BGR numpy array.

    Raises:
        ImportError: opencv-python not installed.
        ValueError: Invalid base64 image data.
    """
    return pil_to_cv2(b64_to_pil(b64_string))


def convert_format(image_path: str, target_format: Literal["JPEG", "PNG", "WEBP"], quality: int = 85) -> bytes:
    """Re-encode an image file to a different format and return the bytes.

    Args:
        image_path: Path to the source image.
        target_format: Output format — ``"JPEG"``, ``"PNG"``, or ``"WEBP"``.
        quality: JPEG/WEBP quality 1–100.

    Returns:
        Raw image bytes in the target format.

    Raises:
        FileNotFoundError: File does not exist.
        ValueError: Unsupported format, quality, or invalid image.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if target_format not in ("JPEG", "PNG", "WEBP"):
        raise ValueError(f"target_format must be 'JPEG', 'PNG', or 'WEBP', got '{target_format}'")
    if not (1 <= quality <= 100):
        raise ValueError(f"quality must be 1-100, got {quality}")

    try:
        with Image.open(path) as img:
            if target_format == "JPEG":
                img = _convert_to_rgb(img)
            elif img.mode not in ("RGB", "L", "P"):
                img = img.convert("RGB")

            output = io.BytesIO()
            save_kwargs = {"format": target_format}
            if target_format in ("JPEG", "WEBP"):
                save_kwargs["quality"] = quality
                if target_format == "JPEG":
                    save_kwargs["optimize"] = True
            img.save(output, **save_kwargs)
            return output.getvalue()
    except Image.UnidentifiedImageError:
        raise ValueError(f"File is not a valid image: {image_path}")
    except Exception:
        raise


def is_valid_image(path: Path) -> bool:
    """Check whether the file extension is a supported image format.

    Example::

        >>> from pathlib import Path
        >>> from visprep import is_valid_image
        >>> is_valid_image(Path("photo.jpg"))
        True
        >>> is_valid_image(Path("document.pdf"))
        False
    """
    return path.suffix.lower() in VALID_IMAGE_EXTENSIONS


def is_valid_size(path: Path) -> bool:
    """Check whether the file size is within ``MAX_IMAGE_SIZE_BYTES``."""
    return path.stat().st_size <= MAX_IMAGE_SIZE_BYTES


def is_valid_dimensions(path: Path) -> bool:
    """Check whether the image meets the minimum dimension requirement (``MIN_IMAGE_DIMENSION``)."""
    try:
        with Image.open(path) as img:
            w, h = img.size
            return w >= MIN_IMAGE_DIMENSION and h >= MIN_IMAGE_DIMENSION
    except Exception:
        return False
