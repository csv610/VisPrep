from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Literal

from PIL import Image

from image_utils._utils import _convert_to_rgb, MAX_IMAGE_SIZE_BYTES, logger


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

    Args:
        image_input: File path or PIL Image.
        max_size_mb: Maximum output file size in MB.  Quality is reduced
            iteratively (down to 10) to meet this limit.
        auto_rotate: Apply EXIF-based orientation correction.
        strip_exif: Remove all metadata from the image.
        target_format: ``"JPEG"``, ``"PNG"``, or ``"WEBP"``.
        quality: JPEG/WEBP quality 1–100.
        max_dimension: Cap the longest side (preserving aspect ratio).
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
            from image_utils.metadata import auto_orient as _orient
            if isinstance(image_input, str):
                img = _orient(image_input)
            else:
                img = _orient_pil_in_memory(img)
        except Exception:
            pass

    if strip_exif:
        if isinstance(image_input, str):
            from image_utils.metadata import remove_exif as _strip
            img = _strip(image_input)
        else:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            img = Image.open(buf)
            img.load()

    img = _convert_to_rgb(img, background_color)

    if max_dimension and (img.width > max_dimension or img.height > max_dimension):
        s = min(max_dimension / img.width, max_dimension / img.height)
        img = img.resize((int(img.width * s), int(img.height * s)), Image.Resampling.LANCZOS)

    quality_to_try = quality
    buf = io.BytesIO()
    while quality_to_try >= 10:
        buf.seek(0)
        buf.truncate(0)
        save_kwargs = {"format": target_format}
        if target_format in ("JPEG", "WEBP"):
            save_kwargs["quality"] = quality_to_try
            if target_format == "JPEG":
                save_kwargs["optimize"] = True
        img.save(buf, **save_kwargs)
        if buf.tell() <= max_bytes:
            break
        quality_to_try -= 5

    if buf.tell() > max_bytes:
        raise ValueError(
            f"Cannot fit image within {max_size_mb}MB limit "
            f"(minimum size: {buf.tell() / 1024 / 1024:.2f}MB at quality {quality_to_try})"
        )

    b64_string = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{target_format.lower()};base64,{b64_string}"


def anthropic_prepare(image_input: str | Image.Image, max_size_mb: float = 20.0) -> str:
    """Prepare image for Anthropic Claude API (JPEG, auto-oriented, no EXIF)."""
    return prepare_for_api(image_input, max_size_mb=max_size_mb, auto_rotate=True, strip_exif=True, target_format="JPEG", quality=85)


def openai_prepare(image_input: str | Image.Image, max_size_mb: float = 20.0, max_dimension: int = 2048) -> str:
    """Prepare image for OpenAI GPT-4V / DALL-E API (PNG, auto-oriented, no EXIF)."""
    return prepare_for_api(image_input, max_size_mb=max_size_mb, auto_rotate=True, strip_exif=True, target_format="PNG", quality=95, max_dimension=max_dimension)
