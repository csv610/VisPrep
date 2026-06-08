from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Literal

from PIL import Image

from visprep._utils import _convert_to_rgb, MIN_IMAGE_DIMENSION, logger


_ORIENT_MAP: dict[int, tuple] = {
    2: (Image.Transpose.FLIP_LEFT_RIGHT,),
    3: (Image.Transpose.ROTATE_180,),
    4: (Image.Transpose.FLIP_TOP_BOTTOM,),
    5: (Image.Transpose.TRANSPOSE, Image.Transpose.FLIP_LEFT_RIGHT),
    6: (Image.Transpose.ROTATE_270,),
    7: (Image.Transpose.TRANSVERSE, Image.Transpose.FLIP_LEFT_RIGHT),
    8: (Image.Transpose.ROTATE_90,),
}


def _orient_pil_in_memory(img: Image.Image) -> Image.Image:
    exif = img.getexif()
    orientation = exif.get(274)
    if orientation and orientation > 1:
        transforms = _ORIENT_MAP.get(orientation)
        if transforms:
            for t in transforms:
                img = img.transpose(t)
    return img


def prepare_for_api(
    image_input: str | Image.Image,
    max_size_mb: float = 20.0,
    auto_rotate: bool = True,
    strip_exif: bool = True,
    target_format: Literal["JPEG", "PNG", "WEBP"] = "JPEG",
    quality: int = 85,
    max_dimension: int | None = None,
    background_color: tuple[int, int, int] = (255, 255, 255),
) -> str:
    """Validate → auto-orient → strip EXIF → resize → encode as base64.

    This is the primary entry point for preparing images for LLM vision APIs
    (Claude, GPT-4V, Gemini, etc.).  The pipeline runs:
    orientation correction → metadata removal → RGB conversion → optional
    dimension cap → base64 encoding.

    To meet the ``max_size_mb`` budget, dimensions are scaled down first
    (preserving quality), then quality is reduced only as a last resort.

    Args:
        image_input: File path or PIL Image.
        max_size_mb: Maximum output file size in MB (default 20).
            Dimensions are scaled down (by 20% steps) before reducing quality.
        auto_rotate: Apply EXIF-based orientation correction.
        strip_exif: Remove all metadata from the image.
        target_format: ``"JPEG"``, ``"PNG"``, or ``"WEBP"``.
        quality: JPEG/WEBP quality 1–100.
        max_dimension: Hard cap on the longest side in pixels.  If the image
            already fits ``max_size_mb`` at this size, quality is preserved.
        background_color: RGB fill for alpha compositing.

    Returns:
        Data-URI base64 string ready for API submission.
    """
    if max_size_mb <= 0:
        raise ValueError(f"max_size_mb must be positive, got {max_size_mb}")
    max_bytes = int(max_size_mb * 1024 * 1024)

    if isinstance(image_input, str):
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_input}")
        img = Image.open(path)
    elif isinstance(image_input, Image.Image):
        img = image_input.copy()
    else:
        raise TypeError(f"image_input must be a file path (str) or PIL Image, got {type(image_input)}")

    if auto_rotate:
        try:
            from visprep.metadata import auto_orient as _orient
            if isinstance(image_input, str):
                img = _orient(image_input)
            else:
                img = _orient_pil_in_memory(img)
        except Exception:
            pass

    if strip_exif:
        if isinstance(image_input, str):
            from visprep.metadata import remove_exif as _strip
            img = _strip(image_input)
        else:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            img = Image.open(buf)
            img.load()

    img = _convert_to_rgb(img, background_color)

    # Apply user's explicit pixel cap
    if max_dimension and (img.width > max_dimension or img.height > max_dimension):
        s = min(max_dimension / img.width, max_dimension / img.height)
        img = img.resize((int(img.width * s), int(img.height * s)), Image.Resampling.LANCZOS)

    # Scale down dimensions first, then reduce quality
    w, h = img.width, img.height
    quality_to_try = quality
    buf = io.BytesIO()

    while True:
        buf.seek(0)
        buf.truncate(0)
        save_kwargs = {"format": target_format}
        if target_format in ("JPEG", "WEBP"):
            save_kwargs["quality"] = quality_to_try
            if target_format == "JPEG":
                save_kwargs["optimize"] = True
        current = img.resize((w, h), Image.Resampling.LANCZOS) if (w, h) != (img.width, img.height) else img
        current.save(buf, **save_kwargs)

        if buf.tell() <= max_bytes:
            break

        # Dimension-first: scale down by 20%
        if w > MIN_IMAGE_DIMENSION and h > MIN_IMAGE_DIMENSION:
            s = 0.8
            w = max(MIN_IMAGE_DIMENSION, int(w * s))
            h = max(MIN_IMAGE_DIMENSION, int(h * s))
        elif quality_to_try > 10:
            quality_to_try -= 5
        else:
            raise ValueError(
                f"Cannot fit image within {max_size_mb}MB limit "
                f"(minimum size: {buf.tell() / 1024 / 1024:.2f}MB at {w}x{h}, quality {quality_to_try})"
            )

    b64_string = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{target_format.lower()};base64,{b64_string}"


