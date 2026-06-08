"""Directory scanning: collect images with optional validation and metadata."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from PIL import Image

from image_utils._utils import _validate_directory_exists, VALID_IMAGE_EXTENSIONS, logger
from image_utils.metadata import get_image_info


def collect_images(
    directory_path: str,
    recursive: bool = False,
    formats: list[str] | None = None,
    validate: bool = True,
    sort_by: Literal["name", "size", "date"] = "name",
) -> list[str]:
    """Scan a directory and return paths of discovered images.

    When ``validate=True``, each file is opened with PIL to confirm it is a
    valid image.  When ``validate=False``, only the file extension is checked.

    Args:
        directory_path: Root directory to scan.
        recursive: Scan subdirectories recursively.
        formats: Filter by image format (e.g. ``["JPEG", "PNG"]``).
        validate: Open each file to confirm it is a valid image.
        sort_by: Sort results by ``"name"``, ``"size"``, or ``"date"``.

    Returns:
        Sorted list of image file paths.
    """
    directory = _validate_directory_exists(directory_path)
    if formats:
        formats = [f.upper() for f in formats]

    ext_to_fmt = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".gif": "GIF",
                  ".webp": "WEBP", ".bmp": "BMP", ".tiff": "TIFF", ".tif": "TIFF"}
    images = []

    try:
        file_paths = [
            fp for fp in (directory.rglob("*") if recursive else directory.glob("*"))
            if not fp.is_dir() and fp.suffix.lower() in VALID_IMAGE_EXTENSIONS
        ]

        if validate:
            def _check(fp: Path) -> str | None:
                try:
                    with Image.open(fp) as img:
                        if formats and (img.format or "Unknown").upper() not in formats:
                            return None
                    return str(fp)
                except Exception:
                    return None

            with ThreadPoolExecutor() as executor:
                for result in executor.map(_check, file_paths):
                    if result is not None:
                        images.append(result)
        else:
            for fp in file_paths:
                if formats and ext_to_fmt.get(fp.suffix.lower()) not in formats:
                    continue
                images.append(str(fp))

        key = {"size": lambda p: Path(p).stat().st_size, "date": lambda p: Path(p).stat().st_mtime}.get(sort_by)
        if key:
            images.sort(key=key)
        else:
            images.sort()

        logger.info(f"Found {len(images)} images in {directory_path} ({'recursive' if recursive else 'non-recursive'})")
        return images
    except Exception:
        raise


def collect_images_with_info(
    directory_path: str,
    recursive: bool = False,
    formats: list[str] | None = None,
    sort_by: Literal["name", "size", "date"] = "name",
) -> list[dict]:
    """Scan a directory and return image metadata dicts.

    Each dict includes the keys from ``get_image_info()`` plus ``"path"``.

    Args:
        directory_path: Root directory to scan.
        recursive: Scan subdirectories recursively.
        formats: Filter by image format.
        sort_by: Sort results by ``"name"``, ``"size"``, or ``"date"``.

    Returns:
        List of metadata dicts sorted by the chosen key.
    """
    directory = _validate_directory_exists(directory_path)
    if formats:
        formats = [f.upper() for f in formats]

    results = []

    try:
        for fp in (directory.rglob("*") if recursive else directory.glob("*")):
            if fp.is_dir() or fp.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
                continue
            try:
                info = get_image_info(str(fp))
                if formats and info["format"].upper() not in formats:
                    continue
                info["path"] = str(fp)
                results.append(info)
            except Exception:
                continue

        if sort_by == "size":
            results.sort(key=lambda x: x["file_size_bytes"])
        elif sort_by == "date":
            results.sort(key=lambda p: Path(p["path"]).stat().st_mtime)
        else:
            results.sort(key=lambda x: x["path"])

        logger.info(f"Found {len(results)} images in {directory_path} ({'recursive' if recursive else 'non-recursive'})")
        return results
    except Exception:
        raise
