# Image Utils

Image processing utilities for vision analysis pipelines. Prepare images for LLM vision APIs (Claude, GPT-4V, etc.) with auto-orient, EXIF stripping, resize, and base64 encoding — all in one call.

## Install

```bash
pip install pillow numpy       # core
pip install opencv-python      # optional — for cv2 functions
```

Or from source:
```bash
pip install -e .            # adds image-utils CLI command
pip install -e ".[cv2]"     # include opencv-python
```

## Quick Start

```python
from image_utils import prepare_for_api

# One-shot: validate → orient → strip EXIF → resize → base64
b64 = prepare_for_api("photo.jpg")
# Send b64 to Claude/GPT-4V…
```

## CLI

```bash
# Prepare an image for API submission
image-utils prepare photo.jpg

# Get image info
image-utils info photo.jpg

# Resize and save
image-utils resize photo.jpg --max-size 5 -o small.jpg

# Square crop with custom background
image-utils square photo.jpg --size 512 --bg 0,0,0 -o square.jpg

# Collect images in a directory
image-utils collect . --recursive
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
from image_utils import CV2_AVAILABLE
if CV2_AVAILABLE:
    from image_utils import cv2_to_pil, pil_to_cv2
```

## Requirements

- Python ≥3.10
- Pillow, numpy (required)
- opencv-python (optional)

## Tests

```bash
pytest tests/ -v
```
