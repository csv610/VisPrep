"""Image metadata: info, auto-orient, EXIF removal, size queries."""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from visprep._utils import _convert_to_rgb, _validate_file_exists, logger


def get_image_info(image_path: str) -> dict:
    """Read image metadata — dimensions, format, colour mode, file size, EXIF.

    Args:
        image_path: Path to the image file.

    Returns:
        Dictionary with keys ``width``, ``height``, ``format``, ``color_mode``,
        ``file_size_bytes``, ``file_size_mb``, ``has_exif``, ``created_date``.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        stat = path.stat()
        file_size_mb = stat.st_size / (1024 * 1024)

        with Image.open(path) as img:
            has_exif = False
            created_date = None
            try:
                exif_data = img.getexif()
                has_exif = len(exif_data) > 0
                if 306 in exif_data:
                    created_date = str(exif_data[306])
            except Exception:
                pass

            return {
                "width": img.width,
                "height": img.height,
                "format": img.format or "Unknown",
                "color_mode": img.mode,
                "file_size_bytes": stat.st_size,
                "file_size_mb": round(file_size_mb, 4),
                "has_exif": has_exif,
                "created_date": created_date,
            }
    except Image.UnidentifiedImageError:
        raise ValueError(f"File is not a valid image: {image_path}")
    except Exception:
        raise


def auto_orient(image_path: str) -> Image.Image:
    """Rotate/flip an image according to its EXIF orientation tag.

    Handles all 8 EXIF orientation values.  The result is always RGB mode.

    Args:
        image_path: Path to the image file.

    Returns:
        Correctly oriented PIL Image.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        with Image.open(path) as img:
            try:
                orientation = img.getexif().get(274)
            except Exception:
                orientation = None

            if orientation == 2:
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            elif orientation == 3:
                img = img.rotate(180, expand=False)
            elif orientation == 4:
                img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            elif orientation == 5:
                img = img.transpose(Image.Transpose.TRANSPOSE).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            elif orientation == 6:
                img = img.rotate(270, expand=False)
            elif orientation == 7:
                img = img.transpose(Image.Transpose.TRANSVERSE).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            elif orientation == 8:
                img = img.rotate(90, expand=False)

            img = _convert_to_rgb(img)
            logger.info(f"Auto-oriented image {path.name} (EXIF orientation: {orientation or 'None'})")
            return img
    except Image.UnidentifiedImageError:
        raise ValueError(f"File is not a valid image: {image_path}")
    except Exception:
        raise


def remove_exif(image_path: str) -> Image.Image:
    """Strip all EXIF and metadata from an image.

    Re-encodes the image to a clean buffer — JPEG sources use a lightweight
    JPEG re-encode with ``exif=b""``; other formats are re-encoded as PNG.

    Args:
        image_path: Path to the image file.

    Returns:
        Clean PIL Image (RGB mode) with no embedded metadata.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        with Image.open(path) as img:
            img = _convert_to_rgb(img)
            buf = io.BytesIO()
            ext = path.suffix.lower()
            if ext in (".jpg", ".jpeg"):
                img.save(buf, format="JPEG", quality=95, optimize=True, exif=b"")
            else:
                img.save(buf, format="PNG")
            buf.seek(0)
            clean = Image.open(buf)
            clean.load()
            logger.info(f"Removed EXIF/metadata from {path.name}")
            return clean
    except Image.UnidentifiedImageError:
        raise ValueError(f"File is not a valid image: {image_path}")
    except Exception:
        raise


def get_image_size_mb(image_path: str) -> float:
    """Return the on-disk file size in megabytes."""
    return _validate_file_exists(image_path).stat().st_size / (1024 * 1024)
