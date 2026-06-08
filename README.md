# VisPrep

Image preparation for vision-language model APIs. Auto-orient, strip EXIF, resize, compress to size limit, and base64 encode — all in one call.

## Install

```bash
pip install visprep             # from PyPI
pip install opencv-python       # optional — for cv2 functions
```

Or from source:
```bash
pip install -e .            # adds visprep CLI command
pip install -e ".[cv2]"     # include opencv-python
```

## Quick Start

```python
from visprep import prepare_for_api

# One-shot: validate → orient → strip EXIF → resize → base64
b64 = prepare_for_api("photo.jpg")
# Send b64 to Claude/GPT-4V…
```

## CLI

```bash
# Prepare an image for API submission
visprep prepare photo.jpg

# Get image info
visprep info photo.jpg

# Resize and save
visprep resize photo.jpg --max-size 5 -o small.jpg

# Square crop with custom background
visprep square photo.jpg --size 512 --bg 0,0,0 -o square.jpg

# Collect images in a directory
visprep collect . --recursive
```

## Key Functions

| Category | Functions |
|----------|-----------|
| **API Prep** | `prepare_for_api()`, `anthropic_prepare()`, `openai_prepare()` |
| **Convert** | `encode_to_base64()`, `pil_to_b64()`, `b64_to_pil()`, `cv2_to_pil()`, `pil_to_cv2()` |
| **Transform** | `resize_to_dimensions()`, `square_image()`, `crop()`, `resize_images_to_fit()` |
| **Metadata** | `get_image_info()`, `auto_orient()`, `remove_exif()`, `get_image_size_mb()` |
| **Create** | `create_blank_image()`, `create_random_image()`, `create_gradient_image()` |
| **Optimize** | `resize_to_max_size()`, `save_image_to_max_size()`, `estimate_compressed_size()` |
| **Save** | `save_image()`, `save_pil_image()`, `save_from_b64()`, `save_images_batch()` |
| **Collect** | `collect_images()`, `collect_images_with_info()` |

## API Adapters

```python
# Claude: JPEG, auto-oriented, stripped, quality 85
b64 = anthropic_prepare("photo.jpg")

# GPT-4V: PNG, auto-oriented, stripped, max 2048px
b64 = openai_prepare("photo.jpg")

# Full control
b64 = prepare_for_api(
    "photo.jpg",
    auto_rotate=True,
    strip_exif=True,
    target_format="WEBP",
    quality=80,
    max_dimension=1024,
)
```

## OpenCV Optional

```python
from visprep import CV2_AVAILABLE
if CV2_AVAILABLE:
    from visprep import cv2_to_pil, pil_to_cv2
```

## Requirements

- Python ≥3.10
- Pillow, numpy (required)
- opencv-python (optional)

## Tests

```bash
pytest tests/ -v
```
