# ImageUtils - Vision Language Model Image Processing Library

A comprehensive Python library for image processing, transformation, and format conversion. Designed for vision language model (VLM) workflows with support for multiple input/output formats.

## Features

### 📷 **Format Conversion**
- Convert between PIL Images, OpenCV arrays, and base64 strings
- Automatic color space handling (RGB, BGR, RGBA, Grayscale)
- Seamless bidirectional conversion

### 🎨 **Image Transformation**
- **Resize** with padding to maintain aspect ratio
- **Square Image** creation with customizable background color and positioning
- **Crop** rectangular regions
- **Auto-orient** based on EXIF data
- **Format conversion** (JPEG, PNG, WEBP)

### 📁 **Batch Operations**
- Collect images from directories (recursive support)
- Get detailed image metadata (dimensions, format, file size, EXIF)
- Batch save multiple images with auto-naming
- Remove EXIF/metadata for privacy

### 💾 **Universal Save**
- Save from any input type (file path, PIL Image, OpenCV array, base64) to PNG/JPG
- Quality control for JPEG compression
- Automatic format normalization
- Create output directories automatically

### 🖼️ **Image Creation**
- Create blank images with solid colors
- Generate random color and noise images
- Create smooth color gradients (horizontal, vertical, diagonal)
- Perfect for testing, placeholders, and composition

### 📏 **Size Optimization**
- Resize images to fit maximum file size in MB/GB
- Estimate compressed size without saving
- Two-phase compression (quality first, then scaling)
- Perfect for API upload limits and bandwidth constraints

### ✅ **Validation**
- File and image format validation
- Size and dimension checking
- EXIF data detection and extraction

## Quick Summary

**40+ Functions** organized into 7 categories:
- **Image Creation** - 3 functions (blank, random, gradient)
- **Image Transformation** - 7 functions (resize, crop, square, etc.)
- **Format Conversion** - 8 functions (PIL ↔ OpenCV ↔ base64)
- **Size Optimization** - 4 functions (compress to fit max size)
- **Metadata** - 3 functions (info, orientation, EXIF)
- **Batch Operations** - 3 functions (collect, save, fit)
- **Save/Export** - 6 functions (universal save, type-specific)
- **Validation** - 3 functions (validate image/size/dimensions)

## Installation

### Requirements
- Python 3.8+
- Pillow
- OpenCV (cv2)
- NumPy

### Setup

1. Clone or download the project:
```bash
cd ImageUtils
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Update `config.py` with your image MIME type preference:
```python
# config.py
IMAGE_MIME_TYPE = "image/jpeg"  # or "image/png"
```

## Quick Start

### Basic Image Processing

```python
from image_utils import ImageUtils
from PIL import Image

# 1. Load and process image
img = ImageUtils.auto_orient("photo.jpg")
img = ImageUtils.remove_exif("photo.jpg")

# 2. Resize to exact dimensions
img = ImageUtils.resize_to_dimensions("photo.jpg", 1024, 1024)

# 3. Save to file
ImageUtils.save_image(img, "output.jpg", format="JPG", quality=90)
```

### Convert Between Formats

```python
# File path → PIL Image
pil_img = Image.open("photo.jpg")

# PIL Image → base64
b64 = ImageUtils.pil_to_b64(pil_img, include_data_uri=True)

# base64 → OpenCV
cv_img = ImageUtils.b64_to_cv2(b64)

# OpenCV → PNG file
ImageUtils.save_cv2_image(cv_img, "output.png")
```

### Batch Processing

```python
# Collect images from directory
image_paths = ImageUtils.collect_images(
    "dataset/",
    recursive=True,
    formats=["JPEG", "PNG"],
    sort_by="size"
)

# Get metadata for all images
images_with_info = ImageUtils.collect_images_with_info(
    "dataset/",
    recursive=True
)

# Save all images in different format
saved_paths = ImageUtils.save_images_batch(
    image_paths,
    output_directory="processed/",
    format="JPG",
    quality=85,
    filename_prefix="image"
)
```

### Get Image Information

```python
# Get detailed image info
info = ImageUtils.get_image_info("photo.jpg")
print(f"Size: {info['width']}x{info['height']}")
print(f"Format: {info['format']}")
print(f"File size: {info['file_size_mb']}MB")
print(f"Has EXIF: {info['has_exif']}")
if info['created_date']:
    print(f"Created: {info['created_date']}")
```

### Create Square Images

```python
# Create square image with centered content
square = ImageUtils.square_image(
    "photo.jpg",
    max_size=512,
    background_color=(255, 255, 255),
    position="center"
)
ImageUtils.save_pil_image(square, "square.png")

# Top-left positioning with black background
square = ImageUtils.square_image(
    "photo.jpg",
    max_size=512,
    background_color=(0, 0, 0),
    position="top-left"
)
ImageUtils.save_pil_image(square, "square_topleft.jpg")
```

### Privacy-Aware Processing

```python
# Remove sensitive metadata before sharing
clean_img = ImageUtils.remove_exif("user_photo.jpg")
b64 = ImageUtils.pil_to_b64(clean_img)
# Safe to send to API without location/camera info
```

### Create Images (Blank, Noise, Gradients)

```python
# Create solid color images
white_bg = ImageUtils.create_blank_image(512, 512, color=(255, 255, 255))
red_bg = ImageUtils.create_blank_image(512, 512, color=(255, 0, 0))

# Random colors and noise
random_color = ImageUtils.create_blank_image(512, 512, color="random")
noise_image = ImageUtils.create_random_image(512, 512)

# Color gradients
gradient = ImageUtils.create_gradient_image(
    512, 512,
    start_color=(255, 0, 0),      # Red
    end_color=(0, 0, 255),        # Blue
    direction="horizontal"
)

# Save them
ImageUtils.save_pil_image(white_bg, "white.png")
ImageUtils.save_pil_image(gradient, "gradient.png")
```

### Optimize Images for Size Constraints

```python
# Check if image needs compression
size_mb = ImageUtils.get_image_size_mb("large_photo.jpg")
print(f"Current size: {size_mb}MB")

# Estimate compressed size
estimated = ImageUtils.estimate_compressed_size("large_photo.jpg", quality=85)
print(f"Estimated after compression: {estimated}MB")

# Resize to fit within 5MB limit
img = ImageUtils.resize_to_max_size("large_photo.jpg", max_size=5, size_unit="MB")
ImageUtils.save_pil_image(img, "optimized.jpg")

# Or do it all in one call
ImageUtils.save_image_to_max_size(
    "large_photo.jpg",
    output_path="upload.jpg",
    max_size=20,  # 20MB limit
    size_unit="MB"
)
```

## API Reference

### Image Creation Functions

```python
ImageUtils.create_blank_image(width: int, height: int, color: Tuple,
                             image_mode: str) -> Image.Image

ImageUtils.create_random_image(width: int, height: int,
                              image_mode: str) -> Image.Image

ImageUtils.create_gradient_image(width: int, height: int,
                               start_color: Tuple, end_color: Tuple,
                               direction: str) -> Image.Image
```

### Size Optimization Functions

```python
ImageUtils.resize_to_max_size(image_path: str, max_size: float,
                             size_unit: str, target_format: str,
                             min_quality: int) -> Image.Image

ImageUtils.save_image_to_max_size(image_path: str, output_path: str,
                                 max_size: float, size_unit: str,
                                 target_format: str,
                                 min_quality: int) -> str

ImageUtils.get_image_size_mb(image_path: str) -> float

ImageUtils.estimate_compressed_size(image_path: str, target_format: str,
                                   quality: int) -> float
```

### Validation Functions

```python
ImageUtils.is_valid_image(path: Path) -> bool
ImageUtils.is_valid_size(path: Path) -> bool
ImageUtils.is_valid_dimensions(path: Path) -> bool
```

### Transformation Functions

```python
ImageUtils.square_image(image_path: str, max_size: int,
                       background_color: Tuple, position: str) -> Image.Image

ImageUtils.resize_to_dimensions(image_path: str, width: int, height: int,
                               background_color: Tuple) -> Image.Image

ImageUtils.crop(image_path: str, left: int, top: int,
               right: int, bottom: int) -> Image.Image

ImageUtils.convert_format(image_path: str, target_format: str,
                         quality: int) -> bytes
```

### Format Conversion Functions

```python
ImageUtils.b64_to_pil(b64_string: str) -> Image.Image
ImageUtils.pil_to_b64(image: Image.Image, image_format: str,
                     quality: int, include_data_uri: bool) -> str

ImageUtils.cv2_to_pil(cv_image: np.ndarray) -> Image.Image
ImageUtils.pil_to_cv2(image: Image.Image) -> np.ndarray

ImageUtils.cv2_to_b64(cv_image: np.ndarray, image_format: str,
                     quality: int, include_data_uri: bool) -> str
ImageUtils.b64_to_cv2(b64_string: str) -> np.ndarray
```

### Metadata Functions

```python
ImageUtils.get_image_info(image_path: str) -> Dict
ImageUtils.auto_orient(image_path: str) -> Image.Image
ImageUtils.remove_exif(image_path: str) -> Image.Image
```

### Batch Operations

```python
ImageUtils.collect_images(directory_path: str, recursive: bool,
                         formats: List[str], validate: bool,
                         sort_by: str) -> List[str]

ImageUtils.collect_images_with_info(directory_path: str, recursive: bool,
                                   formats: List[str],
                                   sort_by: str) -> List[Dict]

ImageUtils.resize_images_to_fit(image_paths: List[str]) -> List[str]
```

### Save/Export Functions

```python
ImageUtils.save_image(image_data: Union[str, Image.Image, np.ndarray],
                     output_path: str, format: str, quality: int,
                     input_type: Optional[str]) -> str

ImageUtils.save_images_batch(image_data_list: List, output_directory: str,
                            format: str, quality: int,
                            filename_prefix: str) -> List[str]

ImageUtils.save_from_path(input_path: str, output_path: str,
                         format: str, quality: int) -> str

ImageUtils.save_pil_image(pil_image: Image.Image, output_path: str,
                         format: str, quality: int) -> str

ImageUtils.save_cv2_image(cv_image: np.ndarray, output_path: str,
                         format: str, quality: int) -> str

ImageUtils.save_from_b64(b64_string: str, output_path: str,
                        format: str, quality: int) -> str
```

## Supported Formats

**Input Formats:**
- JPEG, PNG, GIF, WebP, BMP, TIFF

**Output Formats:**
- PNG (lossless)
- JPEG/JPG (lossy, with quality control)

**Color Spaces:**
- RGB, RGBA, BGR, BGRA, Grayscale (L), Palette (P)

## Configuration

Edit `config.py` to set your preferences:

```python
# config.py
IMAGE_MIME_TYPE = "image/jpeg"  # Default MIME type for encoding
```

## Examples

### Example 1: VLM Image Preparation

```python
from image_utils import ImageUtils

# Process image for Claude Vision API
image_path = "user_upload.jpg"

# Validate
info = ImageUtils.get_image_info(image_path)
if info['file_size_mb'] > 5:
    print(f"Image too large: {info['file_size_mb']}MB")

# Prepare
img = ImageUtils.auto_orient(image_path)
img = ImageUtils.remove_exif(image_path)
img = ImageUtils.resize_to_dimensions(image_path, 1024, 1024)

# Export
b64 = ImageUtils.pil_to_b64(img)
# Send b64 to Claude Vision API
```

### Example 2: Batch Dataset Preparation

```python
from pathlib import Path

# Collect and process dataset
image_files = ImageUtils.collect_images(
    "raw_dataset/",
    recursive=True,
    formats=["JPEG", "PNG"]
)

for img_path in image_files:
    img = ImageUtils.auto_orient(img_path)
    img = ImageUtils.resize_to_dimensions(img_path, 512, 512)

    output_file = f"processed/{Path(img_path).stem}.jpg"
    ImageUtils.save_image(img, output_file, format="JPG", quality=85)

print(f"Processed {len(image_files)} images")
```

### Example 3: Format Conversion with Metadata Removal

```python
# Convert PNG to optimized JPG, removing metadata
images = ImageUtils.collect_images("images/", formats=["PNG"])

saved = ImageUtils.save_images_batch(
    images,
    output_directory="converted/",
    format="JPG",
    quality=90,
    filename_prefix="optimized"
)

print(f"Converted {len(saved)} images with quality control")
```

### Example 4: Create Test Images with Placeholders

```python
from image_utils import ImageUtils

# Create test dataset with various image types
test_images = [
    ImageUtils.create_blank_image(256, 256, color=(200, 100, 50)),
    ImageUtils.create_random_image(512, 512),
    ImageUtils.create_gradient_image(
        256, 256,
        start_color=(255, 0, 0),
        end_color=(0, 255, 0),
        direction="horizontal"
    ),
    ImageUtils.create_blank_image(512, 512, color="random"),
]

# Save all test images
saved = ImageUtils.save_images_batch(
    test_images,
    output_directory="test_data/",
    format="PNG"
)

print(f"Created {len(saved)} test images")
```

### Example 5: Handle Upload Size Limits

```python
from image_utils import ImageUtils
from pathlib import Path

# User uploads large files - must be under 20MB for API
upload_dir = "user_uploads/"
processed_dir = "processed/"

# Check and optimize all uploads
for image_file in Path(upload_dir).glob("*.jpg"):
    size = ImageUtils.get_image_size_mb(str(image_file))

    if size > 20:
        print(f"Compressing {image_file.name} ({size}MB)...")
        # Resize to fit 20MB limit
        ImageUtils.save_image_to_max_size(
            str(image_file),
            output_path=f"{processed_dir}{image_file.name}",
            max_size=20,
            size_unit="MB"
        )
    else:
        # Already within limit, just copy
        import shutil
        shutil.copy(str(image_file), f"{processed_dir}{image_file.name}")

print("All images ready for upload")
```

### Example 6: Estimate Compression Before Processing

```python
from image_utils import ImageUtils

# Check if compression is necessary
image = "high_res_photo.jpg"
original_size = ImageUtils.get_image_size_mb(image)
estimated_size = ImageUtils.estimate_compressed_size(
    image,
    target_format="JPEG",
    quality=85
)

print(f"Original: {original_size}MB → Estimated: {estimated_size}MB")
print(f"Savings: {(1 - estimated_size/original_size)*100:.1f}%")

# Only process if it will help significantly
if estimated_size < original_size * 0.8:  # 20% savings
    img = ImageUtils.resize_to_max_size(image, max_size=5)
    ImageUtils.save_pil_image(img, "compressed.jpg")
else:
    print("Compression won't help much, skipping")
```

### Example 7: Create Gradient Backgrounds for Composition

```python
from image_utils import ImageUtils

# Create gradient backgrounds for different purposes
backgrounds = {
    "professional": ImageUtils.create_gradient_image(
        1920, 1080,
        start_color=(40, 40, 40),      # Dark gray
        end_color=(100, 100, 100),     # Light gray
        direction="vertical"
    ),
    "vibrant": ImageUtils.create_gradient_image(
        1920, 1080,
        start_color=(255, 100, 100),   # Red
        end_color=(255, 200, 100),     # Orange
        direction="diagonal"
    ),
    "ocean": ImageUtils.create_gradient_image(
        1920, 1080,
        start_color=(0, 50, 150),      # Dark blue
        end_color=(100, 200, 255),     # Light blue
        direction="horizontal"
    ),
}

# Save all backgrounds
for name, bg_image in backgrounds.items():
    ImageUtils.save_pil_image(bg_image, f"backgrounds/{name}.png")
```

## Logging

The library uses Python's `logging` module. Configure logging in your application:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Or configure just ImageUtils
logger = logging.getLogger('image_utils')
logger.setLevel(logging.DEBUG)
```

## Performance Tips

1. **Use `validate=False` for fast collection** - Skip image validation for speed
2. **Explicit `input_type`** - Avoid auto-detection overhead in `save_image()`
3. **Batch operations** - Use `save_images_batch()` instead of loop
4. **Cache metadata** - Use `collect_images_with_info()` once, reuse results
5. **Format choice** - Use JPG for photos (smaller), PNG for graphics (lossless)
6. **Estimate before compressing** - Use `estimate_compressed_size()` to check if compression is needed
7. **Image creation is fast** - Create test/placeholder images in-memory (no disk I/O)
8. **Gradient rendering** - Diagonal gradients are slower (nested loops), prefer horizontal/vertical

## Error Handling

```python
from PIL import Image
from image_utils import ImageUtils

try:
    img = ImageUtils.auto_orient("photo.jpg")
except FileNotFoundError:
    print("Image file not found")
except ValueError as e:
    print(f"Invalid image: {e}")
```

## Contributing

Contributions welcome! Please ensure:
- Code follows existing style
- All functions have docstrings
- Error handling is comprehensive
- Logging is used appropriately

## License

MIT License - See LICENSE file for details

## Support

For issues, feature requests, or questions:
1. Check existing code documentation
2. Review examples in README
3. Check logs for error details
4. Ensure all dependencies are installed correctly

## Changelog

### v1.1.0
- Added image creation functions (blank, random, gradient)
- Added size optimization (resize to max MB/GB)
- Added size estimation and checking functions
- Added 6 comprehensive new examples
- Updated documentation with all new features

### v1.0.0
- Initial release
- Complete format conversion (PIL ↔ OpenCV ↔ base64)
- Image transformation (resize, crop, square, rotate)
- Metadata handling (EXIF, orientation)
- Batch operations with directory collection
- Universal save function supporting all input types
- Comprehensive validation and error handling
