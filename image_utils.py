"""Image processing utilities for vision analysis."""

import base64
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Literal, Dict, Union
from PIL import Image
from PIL.Image import Exif
import io
import cv2
import numpy as np
from datetime import datetime
import random

from config import IMAGE_MIME_TYPE

logger = logging.getLogger(__name__)

# ============================================================================
# HELPER FUNCTIONS (Private utilities)
# ============================================================================


def _convert_to_rgb(img: Image.Image) -> Image.Image:
    """
    Convert PIL image to RGB mode, handling transparency and palettes.

    Private helper function used throughout the module to normalize
    image color modes before processing.
    """
    if img.mode in ("RGBA", "LA", "P"):
        rgb_img = Image.new("RGB", img.size, (255, 255, 255))
        rgb_img.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
        return rgb_img
    elif img.mode != "RGB":
        return img.convert("RGB")
    return img


def _validate_file_exists(file_path: str) -> Path:
    """
    Validate that a file exists and return Path object.

    Private helper for consistent file validation across functions.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return path


def _validate_directory_exists(directory_path: str) -> Path:
    """
    Validate that a directory exists and is a directory.

    Private helper for consistent directory validation.
    """
    directory = Path(directory_path)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory_path}")
    return directory


# API constraints
MAX_IMAGE_SIZE_MB = 50
MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024
MAX_TOTAL_IMAGE_PAYLOAD_MB = 50
MAX_TOTAL_IMAGE_PAYLOAD_BYTES = MAX_TOTAL_IMAGE_PAYLOAD_MB * 1024 * 1024
MIN_IMAGE_DIMENSION = 32  # Minimum 32x32 pixels

class ImageUtils:
    """Handles image encoding, transformation, and validation for vision models."""

    # ========================================================================
    # IMAGE CREATION FUNCTIONS (Create blank/empty images)
    # ========================================================================

    @staticmethod
    def create_blank_image(
        width: int,
        height: int,
        color: Union[Tuple[int, int, int], Literal["random"]] = (255, 255, 255),
        image_mode: Literal["RGB", "RGBA", "L"] = "RGB",
    ) -> Image.Image:
        """
        Create a blank image with uniform or random color.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            color: Fill color as RGB tuple (R, G, B) with values 0-255,
                   or "random" for random color (default white (255, 255, 255))
            image_mode: Image color mode - "RGB", "RGBA", or "L" (grayscale) (default "RGB")

        Returns:
            PIL Image object with specified dimensions and color

        Raises:
            ValueError: If dimensions are invalid or color values out of range
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"Width and height must be positive, got {width}x{height}")

        if image_mode not in ("RGB", "RGBA", "L"):
            raise ValueError(f"image_mode must be 'RGB', 'RGBA', or 'L', got '{image_mode}'")

        try:
            # Determine fill color
            if color == "random":
                if image_mode == "L":
                    fill_color = random.randint(0, 255)
                elif image_mode == "RGBA":
                    fill_color = (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255),
                        255,
                    )
                else:  # RGB
                    fill_color = (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255),
                    )
            else:
                # Validate color tuple
                if not isinstance(color, tuple) or not all(0 <= c <= 255 for c in color):
                    raise ValueError(f"Color values must be 0-255 tuple, got {color}")

                if image_mode == "L" and len(color) != 1:
                    raise ValueError(f"Grayscale requires single value, got {color}")
                elif image_mode == "RGBA" and len(color) not in (3, 4):
                    raise ValueError(f"RGBA requires 3 or 4 values, got {color}")
                elif image_mode == "RGB" and len(color) != 3:
                    raise ValueError(f"RGB requires 3 values, got {color}")

                fill_color = color

            # Create blank image
            image = Image.new(image_mode, (width, height), fill_color)
            logger.info(f"Created blank {image_mode} image {width}x{height} with color {fill_color}")
            return image

        except Exception as e:
            logger.error(f"Error creating blank image: {e}")
            raise

    @staticmethod
    def create_random_image(
        width: int,
        height: int,
        image_mode: Literal["RGB", "RGBA", "L"] = "RGB",
    ) -> Image.Image:
        """
        Create image with random pixel values (noise).

        Creates an image filled with random noise, useful for testing or as placeholder.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            image_mode: Image color mode - "RGB", "RGBA", or "L" (default "RGB")

        Returns:
            PIL Image object with random noise

        Raises:
            ValueError: If dimensions are invalid
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"Width and height must be positive, got {width}x{height}")

        if image_mode not in ("RGB", "RGBA", "L"):
            raise ValueError(f"image_mode must be 'RGB', 'RGBA', or 'L', got '{image_mode}'")

        try:
            if image_mode == "L":
                # Grayscale noise
                data = np.random.randint(0, 256, (height, width), dtype=np.uint8)
                image = Image.fromarray(data, mode="L")
            elif image_mode == "RGBA":
                # RGBA noise
                data = np.random.randint(0, 256, (height, width, 4), dtype=np.uint8)
                image = Image.fromarray(data, mode="RGBA")
            else:  # RGB
                # RGB noise
                data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
                image = Image.fromarray(data, mode="RGB")

            logger.info(f"Created random noise {image_mode} image {width}x{height}")
            return image

        except Exception as e:
            logger.error(f"Error creating random image: {e}")
            raise

    @staticmethod
    def create_gradient_image(
        width: int,
        height: int,
        start_color: Tuple[int, int, int],
        end_color: Tuple[int, int, int],
        direction: Literal["horizontal", "vertical", "diagonal"] = "horizontal",
    ) -> Image.Image:
        """
        Create image with color gradient.

        Creates a smooth color gradient between two colors.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            start_color: Starting color as RGB tuple (R, G, B)
            end_color: Ending color as RGB tuple (R, G, B)
            direction: Gradient direction - "horizontal", "vertical", or "diagonal" (default "horizontal")

        Returns:
            PIL Image object with gradient

        Raises:
            ValueError: If dimensions or colors are invalid
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"Width and height must be positive, got {width}x{height}")

        if not all(0 <= c <= 255 for c in start_color) or not all(0 <= c <= 255 for c in end_color):
            raise ValueError("Color values must be 0-255")

        if direction not in ("horizontal", "vertical", "diagonal"):
            raise ValueError(f"direction must be 'horizontal', 'vertical', or 'diagonal', got '{direction}'")

        try:
            # Create gradient array
            if direction == "horizontal":
                # Left to right
                gradient = np.linspace(0, 1, width)
                data = np.zeros((height, width, 3), dtype=np.uint8)
                for x in range(width):
                    t = gradient[x]
                    data[:, x] = [
                        int(start_color[0] * (1 - t) + end_color[0] * t),
                        int(start_color[1] * (1 - t) + end_color[1] * t),
                        int(start_color[2] * (1 - t) + end_color[2] * t),
                    ]
            elif direction == "vertical":
                # Top to bottom
                gradient = np.linspace(0, 1, height)
                data = np.zeros((height, width, 3), dtype=np.uint8)
                for y in range(height):
                    t = gradient[y]
                    data[y, :] = [
                        int(start_color[0] * (1 - t) + end_color[0] * t),
                        int(start_color[1] * (1 - t) + end_color[1] * t),
                        int(start_color[2] * (1 - t) + end_color[2] * t),
                    ]
            else:  # diagonal
                # Top-left to bottom-right
                data = np.zeros((height, width, 3), dtype=np.uint8)
                max_dist = np.sqrt(height**2 + width**2)
                for y in range(height):
                    for x in range(width):
                        t = np.sqrt(x**2 + y**2) / max_dist
                        data[y, x] = [
                            int(start_color[0] * (1 - t) + end_color[0] * t),
                            int(start_color[1] * (1 - t) + end_color[1] * t),
                            int(start_color[2] * (1 - t) + end_color[2] * t),
                        ]

            image = Image.fromarray(data, mode="RGB")
            logger.info(f"Created {direction} gradient image {width}x{height}")
            return image

        except Exception as e:
            logger.error(f"Error creating gradient image: {e}")
            raise

    # ========================================================================
    # VALIDATION FUNCTIONS
    # ========================================================================

    @staticmethod
    def encode_to_base64(image_path: str) -> str:
        """
        Convert an image file to base64 encoding.

        Args:
            image_path: Path to the image file

        Returns:
            Base64 encoded image URL in data URI format

        Raises:
            FileNotFoundError: If the image file doesn't exist
            ValueError: If file is not a valid image or exceeds size limit
        """
        path = Path(image_path)

        if not path.exists():
            logger.error(f"Image file not found: {image_path}")
            raise FileNotFoundError(f"Image file not found: {image_path}")

        if not ImageUtils.is_valid_image(path):
            logger.error(f"Invalid image file: {image_path}")
            raise ValueError(f"File is not a valid image: {image_path}. Supported formats: PNG, JPEG, WEBP, GIF")

        if not ImageUtils.is_valid_dimensions(path):
            try:
                with Image.open(path) as img:
                    width, height = img.size
                logger.error(f"Image dimensions too small: {image_path} ({width}x{height})")
            except:
                pass
            raise ValueError(f"Image must be at least {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION} pixels: {image_path}")

        if not ImageUtils.is_valid_size(path):
            file_size_mb = path.stat().st_size / (1024 * 1024)
            logger.error(f"Image file too large: {image_path} ({file_size_mb:.2f}MB)")
            raise ValueError(f"Image file exceeds {MAX_IMAGE_SIZE_MB}MB limit: {image_path} ({file_size_mb:.2f}MB)")

        try:
            with open(path, "rb") as file:
                file_data = file.read()
                encoded_file = base64.b64encode(file_data).decode("utf-8")
                base64_url = f"data:{IMAGE_MIME_TYPE};base64,{encoded_file}"
            return base64_url
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            raise

    @staticmethod
    def is_valid_image(path: Path) -> bool:
        """
        Check if a file is a valid image based on extension.

        Args:
            path: Path object to the file

        Returns:
            True if file has a valid image extension, False otherwise
        """
        valid_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
        return path.suffix.lower() in valid_extensions

    @staticmethod
    def is_valid_size(path: Path) -> bool:
        """
        Check if image file size is within the 50MB limit.

        Args:
            path: Path object to the file

        Returns:
            True if file size is <= 50MB, False otherwise
        """
        return path.stat().st_size <= MAX_IMAGE_SIZE_BYTES

    @staticmethod
    def is_valid_dimensions(path: Path) -> bool:
        """
        Check if image dimensions meet minimum requirement (32x32).

        Args:
            path: Path object to the file

        Returns:
            True if image is at least 32x32 pixels, False otherwise
        """
        try:
            with Image.open(path) as img:
                width, height = img.size
                return width >= MIN_IMAGE_DIMENSION and height >= MIN_IMAGE_DIMENSION
        except Exception as e:
            logger.error(f"Error reading image dimensions: {e}")
            return False

    @staticmethod
    def _estimate_base64_size(data: bytes) -> int:
        """
        Estimate the base64 encoded size of binary data.

        Args:
            data: Binary data

        Returns:
            Estimated size in bytes after base64 encoding
        """
        # Base64 encoding increases size by ~33%
        return int(len(data) * 4 / 3) + 50  # +50 for data URI prefix overhead

    @staticmethod
    def _resize_image_to_quality(image_path: str, quality: int = 85, scale_factor: float = 1.0) -> bytes:
        """
        Resize and compress an image to specified quality and dimensions.

        Args:
            image_path: Path to the image file
            quality: JPEG quality (1-100), default 85
            scale_factor: Scale factor for dimensions (0.0-1.0), default 1.0 (no scaling)

        Returns:
            Compressed image data as bytes

        Raises:
            ValueError: If scaled dimensions would be below minimum
        """
        try:
            with Image.open(image_path) as img:
                # Calculate scaled dimensions
                if scale_factor < 1.0:
                    new_width = max(MIN_IMAGE_DIMENSION, int(img.width * scale_factor))
                    new_height = max(MIN_IMAGE_DIMENSION, int(img.height * scale_factor))
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Convert RGBA to RGB if necessary (for JPEG compatibility)
                if img.mode in ("RGBA", "LA", "P"):
                    rgb_img = Image.new("RGB", img.size, (255, 255, 255))
                    rgb_img.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
                    img = rgb_img

                # Save with compression
                output = io.BytesIO()
                img.save(output, format="JPEG", quality=quality, optimize=True)
                return output.getvalue()
        except Exception as e:
            logger.error(f"Error resizing image {image_path}: {e}")
            raise

    # ========================================================================
    # TRANSFORMATION FUNCTIONS (Resize, Square, Crop, etc.)
    # ========================================================================

    @staticmethod
    def resize_images_to_fit(image_paths: List[str]) -> List[str]:
        """
        Automatically resize images so total payload doesn't exceed limit.

        Resizes images proportionally if their combined base64 size exceeds the limit.
        Returns paths to resized temporary images or original paths if no resizing needed.

        Args:
            image_paths: List of image file paths

        Returns:
            List of image paths (original or resized temporary files)

        Raises:
            ValueError: If images cannot be resized to fit within limit
        """
        if not image_paths:
            return image_paths

        # Calculate total size
        total_size = 0
        image_data = []
        for path in image_paths:
            p = Path(path)
            with open(p, "rb") as f:
                data = f.read()
                image_data.append(data)
                total_size += ImageUtils._estimate_base64_size(data)

        # If within limit, return original paths
        if total_size <= MAX_TOTAL_IMAGE_PAYLOAD_BYTES:
            logger.info(f"Total image payload {total_size / 1024 / 1024:.2f}MB is within {MAX_TOTAL_IMAGE_PAYLOAD_MB}MB limit")
            return image_paths

        # Need to resize - start with quality 85 and reduce if needed
        logger.warning(f"Total image payload {total_size / 1024 / 1024:.2f}MB exceeds limit. Auto-resizing...")

        quality = 85
        scale_factor = 1.0

        while quality > 10 or scale_factor < 1.0:
            resized_data = []
            total_resized_size = 0

            for i, path in enumerate(image_paths):
                try:
                    compressed = ImageUtils._resize_image_to_quality(path, quality, scale_factor)
                    resized_data.append(compressed)
                    total_resized_size += ImageUtils._estimate_base64_size(compressed)
                except Exception as e:
                    logger.error(f"Failed to resize {path} at quality {quality}, scale {scale_factor:.2f}: {e}")
                    raise

            if total_resized_size <= MAX_TOTAL_IMAGE_PAYLOAD_BYTES:
                # Write resized images to temporary files
                temp_paths = []
                for i, data in enumerate(resized_data):
                    temp_path = Path(image_paths[i]).parent / f".{Path(image_paths[i]).stem}_resized_{quality}_{int(scale_factor*100)}.jpg"
                    with open(temp_path, "wb") as f:
                        f.write(data)
                    temp_paths.append(str(temp_path))
                    logger.info(f"Resized {Path(image_paths[i]).name} to {temp_path.stat().st_size / 1024 / 1024:.2f}MB (quality {quality}, scale {scale_factor:.2f})")

                return temp_paths

            # Reduce quality first, then scale if quality hits minimum
            if quality > 10:
                quality -= 5
            else:
                scale_factor -= 0.1

        # If we get here, even at quality 10 and heavily scaled it's too large
        raise ValueError(
            f"Cannot resize images to fit within {MAX_TOTAL_IMAGE_PAYLOAD_MB}MB limit. "
            f"Total size at minimum settings: {total_resized_size / 1024 / 1024:.2f}MB"
        )

    @staticmethod
    def square_image(image_path: str, max_size: int, background_color: Tuple[int, int, int], position: Literal["top-left", "center"] = "center") -> Image.Image:
        """
        Create a square image with specified background color and positioned image.

        If the input image is larger than max_size, it will be resized proportionally
        to fit. The image is then placed on a square canvas of max_size x max_size
        with the specified background color.

        Args:
            image_path: Path to the image file
            max_size: Size of the square canvas (width and height in pixels)
            background_color: RGB tuple (R, G, B) with values 0-255
            position: Where to place the image - "top-left" or "center" (default)

        Returns:
            PIL Image object with the square canvas and positioned image

        Raises:
            FileNotFoundError: If the image file doesn't exist
            ValueError: If max_size is invalid or background_color values out of range
        """
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")

        if not all(0 <= c <= 255 for c in background_color):
            raise ValueError(f"background_color values must be 0-255, got {background_color}")

        if position not in ("top-left", "center"):
            raise ValueError(f"position must be 'top-left' or 'center', got '{position}'")

        try:
            with Image.open(path) as img:
                img = _convert_to_rgb(img)

                # Resize if image is larger than max_size
                img_width, img_height = img.size
                if img_width > max_size or img_height > max_size:
                    scale_factor = min(max_size / img_width, max_size / img_height)
                    new_width = int(img_width * scale_factor)
                    new_height = int(img_height * scale_factor)
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    img_width, img_height = img.size

                # Create square canvas
                canvas = Image.new("RGB", (max_size, max_size), background_color)

                # Calculate position
                if position == "center":
                    x = (max_size - img_width) // 2
                    y = (max_size - img_height) // 2
                else:  # top-left
                    x = 0
                    y = 0

                # Paste image onto canvas
                canvas.paste(img, (x, y))

                return canvas

        except Image.UnidentifiedImageError:
            raise ValueError(f"File is not a valid image: {image_path}")
        except Exception as e:
            logger.error(f"Error creating square image: {e}")
            raise

    @staticmethod
    def resize_to_dimensions(image_path: str, width: int, height: int, background_color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
        """
        Resize image to exact dimensions while maintaining aspect ratio with padding.

        The image is resized to fit within the specified dimensions and centered on
        a canvas of the exact size with the background color. Useful for VLM input
        constraints that require specific sizes.

        Args:
            image_path: Path to the image file
            width: Target width in pixels
            height: Target height in pixels
            background_color: RGB tuple for padding background (default white)

        Returns:
            PIL Image object resized to exact dimensions with padding

        Raises:
            FileNotFoundError: If the image file doesn't exist
            ValueError: If dimensions are invalid
        """
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        if width <= 0 or height <= 0:
            raise ValueError(f"Dimensions must be positive, got {width}x{height}")

        if not all(0 <= c <= 255 for c in background_color):
            raise ValueError(f"background_color values must be 0-255, got {background_color}")

        try:
            with Image.open(path) as img:
                img = _convert_to_rgb(img)

                # Calculate scale to fit within target dimensions
                img_width, img_height = img.size
                scale = min(width / img_width, height / img_height)
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)

                # Resize image
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Create canvas and center image
                canvas = Image.new("RGB", (width, height), background_color)
                x = (width - new_width) // 2
                y = (height - new_height) // 2
                canvas.paste(img, (x, y))

                return canvas

        except Image.UnidentifiedImageError:
            raise ValueError(f"File is not a valid image: {image_path}")
        except Exception as e:
            logger.error(f"Error resizing to dimensions: {e}")
            raise

    @staticmethod
    def convert_format(image_path: str, target_format: Literal["JPEG", "PNG", "WEBP"], quality: int = 85) -> bytes:
        """
        Convert image to a specific format with optional quality control.

        Args:
            image_path: Path to the image file
            target_format: Target format - "JPEG", "PNG", or "WEBP"
            quality: Quality level for JPEG/WEBP (1-100, default 85). Ignored for PNG.

        Returns:
            Image data as bytes in the target format

        Raises:
            FileNotFoundError: If the image file doesn't exist
            ValueError: If format is invalid or quality is out of range
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
        except Exception as e:
            logger.error(f"Error converting image format: {e}")
            raise

    @staticmethod
    def crop(image_path: str, left: int, top: int, right: int, bottom: int) -> Image.Image:
        """
        Crop a rectangular region from an image.

        Args:
            image_path: Path to the image file
            left: Left edge coordinate in pixels
            top: Top edge coordinate in pixels
            right: Right edge coordinate in pixels
            bottom: Bottom edge coordinate in pixels

        Returns:
            PIL Image object containing the cropped region

        Raises:
            FileNotFoundError: If the image file doesn't exist
            ValueError: If coordinates are invalid or out of bounds
        """
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        if left >= right or top >= bottom:
            raise ValueError(f"Invalid crop coordinates: left={left}, top={top}, right={right}, bottom={bottom}")

        try:
            with Image.open(path) as img:
                width, height = img.size

                # Validate coordinates are within bounds
                if left < 0 or top < 0 or right > width or bottom > height:
                    raise ValueError(
                        f"Crop coordinates out of bounds. Image size: {width}x{height}, "
                        f"requested: left={left}, top={top}, right={right}, bottom={bottom}"
                    )

                # Crop the image
                cropped = img.crop((left, top, right, bottom))
                return cropped

        except Image.UnidentifiedImageError:
            raise ValueError(f"File is not a valid image: {image_path}")
        except Exception as e:
            logger.error(f"Error cropping image: {e}")
            raise

    # ========================================================================
    # FORMAT CONVERSION FUNCTIONS (PIL ↔ OpenCV ↔ Base64)
    # ========================================================================

    @staticmethod
    def b64_to_pil(b64_string: str) -> Image.Image:
        """
        Convert base64 encoded image to PIL Image object.

        Accepts base64 strings with or without data URI prefix
        (e.g., "data:image/jpeg;base64,..." or raw base64).

        Args:
            b64_string: Base64 encoded image string, optionally with data URI prefix

        Returns:
            PIL Image object

        Raises:
            ValueError: If the base64 string is invalid or not a valid image
        """
        try:
            # Remove data URI prefix if present
            if "," in b64_string:
                b64_string = b64_string.split(",", 1)[1]

            # Decode base64
            image_data = base64.b64decode(b64_string)

            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))
            return image

        except Exception as e:
            logger.error(f"Error converting base64 to PIL Image: {e}")
            raise ValueError(f"Invalid base64 image data: {e}")

    @staticmethod
    def pil_to_b64(image: Image.Image, image_format: str = "JPEG", quality: int = 85, include_data_uri: bool = True) -> str:
        """
        Convert PIL Image object to base64 encoded string.

        Args:
            image: PIL Image object
            image_format: Image format - "JPEG", "PNG", or "WEBP" (default "JPEG")
            quality: Quality level for JPEG/WEBP (1-100, default 85). Ignored for PNG.
            include_data_uri: If True, includes "data:image/...;base64," prefix (default True)

        Returns:
            Base64 encoded image string, optionally with data URI prefix

        Raises:
            ValueError: If image format is invalid or quality is out of range
        """
        if not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL Image object")

        if image_format not in ("JPEG", "PNG", "WEBP"):
            raise ValueError(f"image_format must be 'JPEG', 'PNG', or 'WEBP', got '{image_format}'")

        if not (1 <= quality <= 100):
            raise ValueError(f"quality must be 1-100, got {quality}")

        try:
            # Convert to RGB if necessary
            if image_format == "JPEG":
                image = _convert_to_rgb(image)
            elif image.mode not in ("RGB", "L"):
                image = image.convert("RGB")

            # Encode to bytes
            output = io.BytesIO()
            save_kwargs = {"format": image_format}

            if image_format in ("JPEG", "WEBP"):
                save_kwargs["quality"] = quality
                if image_format == "JPEG":
                    save_kwargs["optimize"] = True

            image.save(output, **save_kwargs)
            image_bytes = output.getvalue()

            # Encode to base64
            b64_string = base64.b64encode(image_bytes).decode("utf-8")

            # Add data URI prefix if requested
            if include_data_uri:
                mime_type = f"image/{image_format.lower()}"
                return f"data:{mime_type};base64,{b64_string}"

            return b64_string

        except Exception as e:
            logger.error(f"Error converting PIL Image to base64: {e}")
            raise

    @staticmethod
    def cv2_to_pil(cv_image: np.ndarray) -> Image.Image:
        """
        Convert OpenCV (cv2) image to PIL Image object.

        OpenCV images are numpy arrays in BGR format. This function converts
        them to RGB format and returns a PIL Image.

        Args:
            cv_image: OpenCV image as numpy array (BGR format)

        Returns:
            PIL Image object in RGB format

        Raises:
            ValueError: If input is not a valid OpenCV image array
        """
        if not isinstance(cv_image, np.ndarray):
            raise ValueError(f"Input must be a numpy array (OpenCV image), got {type(cv_image)}")

        try:
            # Handle grayscale images (2D arrays)
            if len(cv_image.shape) == 2:
                return Image.fromarray(cv_image, mode="L")

            # Handle color images (3D arrays) - convert BGR to RGB
            if len(cv_image.shape) == 3:
                if cv_image.shape[2] == 3:
                    # BGR to RGB conversion
                    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                    return Image.fromarray(rgb_image)
                elif cv_image.shape[2] == 4:
                    # BGRA to RGBA conversion
                    rgba_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGBA)
                    return Image.fromarray(rgba_image, mode="RGBA")
                else:
                    raise ValueError(f"Unsupported number of channels: {cv_image.shape[2]}")

            raise ValueError(f"Unsupported image shape: {cv_image.shape}")

        except Exception as e:
            logger.error(f"Error converting OpenCV image to PIL: {e}")
            raise

    @staticmethod
    def pil_to_cv2(image: Image.Image) -> np.ndarray:
        """
        Convert PIL Image object to OpenCV (cv2) image.

        PIL images are converted to OpenCV numpy arrays in BGR format.

        Args:
            image: PIL Image object

        Returns:
            OpenCV image as numpy array in BGR format

        Raises:
            ValueError: If input is not a PIL Image object
        """
        if not isinstance(image, Image.Image):
            raise ValueError(f"Input must be a PIL Image object, got {type(image)}")

        try:
            # Handle grayscale images
            if image.mode == "L":
                return np.array(image)

            # Handle RGBA images - convert to BGRA
            if image.mode == "RGBA":
                rgb_array = np.array(image)
                bgra_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGBA2BGRA)
                return bgra_array

            # Handle RGB and other color modes - convert to BGR
            rgb_array = np.array(image.convert("RGB"))
            bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            return bgr_array

        except Exception as e:
            logger.error(f"Error converting PIL Image to OpenCV: {e}")
            raise

    @staticmethod
    def cv2_to_b64(cv_image: np.ndarray, image_format: str = "JPEG", quality: int = 85, include_data_uri: bool = True) -> str:
        """
        Convert OpenCV (cv2) image to base64 encoded string.

        Args:
            cv_image: OpenCV image as numpy array (BGR format)
            image_format: Image format - "JPEG", "PNG", or "WEBP" (default "JPEG")
            quality: Quality level for JPEG/WEBP (1-100, default 85). Ignored for PNG.
            include_data_uri: If True, includes "data:image/...;base64," prefix (default True)

        Returns:
            Base64 encoded image string, optionally with data URI prefix

        Raises:
            ValueError: If input is invalid or image format is unsupported
        """
        try:
            # Convert cv2 to PIL
            pil_image = ImageUtils.cv2_to_pil(cv_image)
            # Convert PIL to base64
            return ImageUtils.pil_to_b64(pil_image, image_format, quality, include_data_uri)
        except Exception as e:
            logger.error(f"Error converting OpenCV image to base64: {e}")
            raise

    @staticmethod
    def b64_to_cv2(b64_string: str) -> np.ndarray:
        """
        Convert base64 encoded image to OpenCV (cv2) image.

        Accepts base64 strings with or without data URI prefix
        (e.g., "data:image/jpeg;base64,..." or raw base64).

        Args:
            b64_string: Base64 encoded image string, optionally with data URI prefix

        Returns:
            OpenCV image as numpy array in BGR format

        Raises:
            ValueError: If the base64 string is invalid or not a valid image
        """
        try:
            # Convert base64 to PIL
            pil_image = ImageUtils.b64_to_pil(b64_string)
            # Convert PIL to cv2
            return ImageUtils.pil_to_cv2(pil_image)
        except Exception as e:
            logger.error(f"Error converting base64 to OpenCV image: {e}")
            raise

    # ========================================================================
    # METADATA & ORIENTATION FUNCTIONS
    # ========================================================================

    @staticmethod
    def get_image_info(image_path: str) -> Dict:
        """
        Get detailed information about an image file.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary containing:
                - width: Image width in pixels
                - height: Image height in pixels
                - format: Image format (JPEG, PNG, WEBP, GIF, etc.)
                - color_mode: Color mode (RGB, RGBA, L, etc.)
                - file_size_mb: File size in megabytes
                - file_size_bytes: File size in bytes
                - has_exif: Whether image has EXIF data
                - created_date: Image creation date if available in EXIF

        Raises:
            FileNotFoundError: If the image file doesn't exist
            ValueError: If file is not a valid image
        """
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            file_stat = path.stat()
            file_size_bytes = file_stat.st_size
            file_size_mb = file_size_bytes / (1024 * 1024)

            with Image.open(path) as img:
                width, height = img.size
                img_format = img.format or "Unknown"
                color_mode = img.mode

                # Check for EXIF data
                has_exif = False
                created_date = None
                try:
                    exif_data = img.getexif()
                    has_exif = len(exif_data) > 0
                    # EXIF tag 306 is DateTime (in format YYYY:MM:DD HH:MM:SS)
                    if 306 in exif_data:
                        created_date = str(exif_data[306])
                except Exception:
                    pass

                return {
                    "width": width,
                    "height": height,
                    "format": img_format,
                    "color_mode": color_mode,
                    "file_size_bytes": file_size_bytes,
                    "file_size_mb": round(file_size_mb, 2),
                    "has_exif": has_exif,
                    "created_date": created_date,
                }

        except Image.UnidentifiedImageError:
            raise ValueError(f"File is not a valid image: {image_path}")
        except Exception as e:
            logger.error(f"Error reading image info: {e}")
            raise

    @staticmethod
    def auto_orient(image_path: str) -> Image.Image:
        """
        Auto-rotate image based on EXIF orientation data.

        Many cameras and phones embed orientation info in EXIF data.
        This function reads that data and rotates the image accordingly.
        Returns the corrected image in correct orientation.

        Args:
            image_path: Path to the image file

        Returns:
            PIL Image object with correct orientation

        Raises:
            FileNotFoundError: If the image file doesn't exist
            ValueError: If file is not a valid image
        """
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            with Image.open(path) as img:
                # Try to get EXIF orientation tag (tag 274)
                try:
                    exif_data = img.getexif()
                    orientation = exif_data.get(274)  # Orientation tag
                except Exception:
                    orientation = None

                # EXIF orientation values:
                # 1 = Normal (0°)
                # 2 = Flipped horizontally
                # 3 = Rotated 180°
                # 4 = Flipped vertically
                # 5 = Transposed (rotated 90° CCW + flipped)
                # 6 = Rotated 90° clockwise
                # 7 = Transposed (rotated 90° CW + flipped)
                # 8 = Rotated 270° clockwise (or 90° CCW)

                if orientation == 2:
                    img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                elif orientation == 3:
                    img = img.rotate(180, expand=False)
                elif orientation == 4:
                    img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                elif orientation == 5:
                    img = img.transpose(Image.Transpose.TRANSPOSE)
                    img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                elif orientation == 6:
                    img = img.rotate(270, expand=False)
                elif orientation == 7:
                    img = img.transpose(Image.Transpose.TRANSVERSE)
                    img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                elif orientation == 8:
                    img = img.rotate(90, expand=False)

                img = _convert_to_rgb(img)

                logger.info(f"Auto-oriented image {path.name} (EXIF orientation: {orientation or 'None'})")
                return img

        except Image.UnidentifiedImageError:
            raise ValueError(f"File is not a valid image: {image_path}")
        except Exception as e:
            logger.error(f"Error auto-orienting image: {e}")
            raise

    @staticmethod
    def remove_exif(image_path: str) -> Image.Image:
        """
        Remove all EXIF and metadata from an image.

        Strips all EXIF data, IPTC data, and XMP data for privacy/compliance.
        Returns a clean image with no embedded metadata.

        Args:
            image_path: Path to the image file

        Returns:
            PIL Image object with all metadata removed

        Raises:
            FileNotFoundError: If the image file doesn't exist
            ValueError: If file is not a valid image
        """
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            with Image.open(path) as img:
                img = _convert_to_rgb(img)

                # Create a new image without any metadata
                # This is the most reliable way - create a new image from the data
                clean_image = Image.new(img.mode, img.size)
                clean_image.putdata(img.getdata())

                logger.info(f"Removed EXIF/metadata from {path.name}")
                return clean_image

        except Image.UnidentifiedImageError:
            raise ValueError(f"File is not a valid image: {image_path}")
        except Exception as e:
            logger.error(f"Error removing EXIF data: {e}")
            raise

    # ========================================================================
    # BATCH OPERATIONS (Collect, process multiple images)
    # ========================================================================

    @staticmethod
    def collect_images(
        directory_path: str,
        recursive: bool = False,
        formats: Optional[List[str]] = None,
        validate: bool = True,
        sort_by: Literal["name", "size", "date"] = "name",
    ) -> List[str]:
        """
        Collect image file paths from a directory.

        Scans a directory for image files and optionally recurses into subdirectories.
        Can filter by format and validate that files are actual images.

        Args:
            directory_path: Path to the directory to scan
            recursive: If True, recursively scan subdirectories (default False)
            formats: List of allowed formats in uppercase (e.g., ["JPEG", "PNG", "WEBP"]).
                    If None, allows all supported formats. (default None)
            validate: If True, validates each file is a valid image (slower but reliable).
                     If False, only checks by extension (faster). (default True)
            sort_by: Sort results by "name", "size", or "date" (default "name")

        Returns:
            List of image file paths, sorted as specified

        Raises:
            FileNotFoundError: If the directory doesn't exist
            ValueError: If directory is not a valid directory
        """
        directory = _validate_directory_exists(directory_path)

        if formats:
            formats = [fmt.upper() for fmt in formats]

        # Define valid image extensions
        valid_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"}

        image_paths = []

        try:
            # Determine search pattern
            if recursive:
                file_iter = directory.rglob("*")
            else:
                file_iter = directory.glob("*")

            for file_path in file_iter:
                # Skip directories
                if file_path.is_dir():
                    continue

                # Check extension
                if file_path.suffix.lower() not in valid_extensions:
                    continue

                # Validate if requested
                if validate:
                    try:
                        with Image.open(file_path) as img:
                            img_format = img.format or "Unknown"
                            # Filter by format if specified
                            if formats and img_format.upper() not in formats:
                                continue
                        image_paths.append(str(file_path))
                    except Exception:
                        # Skip invalid image files
                        logger.debug(f"Skipped invalid image file: {file_path}")
                        continue
                else:
                    # Just check format by extension
                    if formats:
                        try:
                            with Image.open(file_path) as img:
                                img_format = img.format or "Unknown"
                                if img_format.upper() not in formats:
                                    continue
                        except Exception:
                            continue
                    image_paths.append(str(file_path))

            # Sort results
            if sort_by == "size":
                image_paths.sort(key=lambda p: Path(p).stat().st_size)
            elif sort_by == "date":
                image_paths.sort(key=lambda p: Path(p).stat().st_mtime)
            else:  # name (default)
                image_paths.sort()

            logger.info(
                f"Found {len(image_paths)} images in {directory_path} "
                f"({'recursive' if recursive else 'non-recursive'})"
            )
            return image_paths

        except Exception as e:
            logger.error(f"Error collecting images from directory: {e}")
            raise

    @staticmethod
    def collect_images_with_info(
        directory_path: str,
        recursive: bool = False,
        formats: Optional[List[str]] = None,
        sort_by: Literal["name", "size", "date"] = "name",
    ) -> List[Dict]:
        """
        Collect image file paths and metadata from a directory.

        Similar to collect_images() but also returns image information for each file.
        More expensive operation as it reads metadata, but useful for filtering or displaying.

        Args:
            directory_path: Path to the directory to scan
            recursive: If True, recursively scan subdirectories (default False)
            formats: List of allowed formats in uppercase (e.g., ["JPEG", "PNG", "WEBP"]).
                    If None, allows all supported formats. (default None)
            sort_by: Sort results by "name", "size", or "date" (default "name")

        Returns:
            List of dictionaries containing:
                - path: File path
                - width: Image width
                - height: Image height
                - format: Image format
                - file_size_mb: File size in MB
                - has_exif: Whether image has EXIF data

        Raises:
            FileNotFoundError: If the directory doesn't exist
            ValueError: If directory is not a valid directory
        """
        directory = _validate_directory_exists(directory_path)

        if formats:
            formats = [fmt.upper() for fmt in formats]

        valid_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"}
        image_info_list = []

        try:
            if recursive:
                file_iter = directory.rglob("*")
            else:
                file_iter = directory.glob("*")

            for file_path in file_iter:
                if file_path.is_dir():
                    continue

                if file_path.suffix.lower() not in valid_extensions:
                    continue

                try:
                    info = ImageUtils.get_image_info(str(file_path))

                    # Filter by format if specified
                    if formats and info["format"].upper() not in formats:
                        continue

                    info["path"] = str(file_path)
                    image_info_list.append(info)

                except Exception:
                    logger.debug(f"Skipped invalid image file: {file_path}")
                    continue

            # Sort results
            if sort_by == "size":
                image_info_list.sort(key=lambda x: x["file_size_bytes"])
            elif sort_by == "date":
                image_info_list.sort(
                    key=lambda p: Path(p["path"]).stat().st_mtime
                )
            else:  # name (default)
                image_info_list.sort(key=lambda x: x["path"])

            logger.info(
                f"Found {len(image_info_list)} images in {directory_path} "
                f"({'recursive' if recursive else 'non-recursive'})"
            )
            return image_info_list

        except Exception as e:
            logger.error(f"Error collecting images with info: {e}")
            raise

    # ========================================================================
    # SIZE OPTIMIZATION FUNCTIONS (Resize to fit max file size)
    # ========================================================================

    @staticmethod
    def resize_to_max_size(
        image_path: str,
        max_size: float,
        size_unit: Literal["MB", "GB"] = "MB",
        target_format: Literal["JPEG", "PNG"] = "JPEG",
        min_quality: int = 10,
    ) -> Image.Image:
        """
        Resize image to fit within a maximum file size in MB or GB.

        Progressively reduces quality and dimensions until image fits
        within the specified file size limit. Useful for API constraints
        or bandwidth-limited scenarios.

        Args:
            image_path: Path to the image file
            max_size: Maximum file size (in MB or GB)
            size_unit: Unit of max_size - "MB" or "GB" (default "MB")
            target_format: Output format - "JPEG" or "PNG" (default "JPEG")
            min_quality: Minimum JPEG quality before scaling (default 10, range 1-100)

        Returns:
            PIL Image object sized to fit within max_size

        Raises:
            FileNotFoundError: If the image file doesn't exist
            ValueError: If max_size is invalid or image cannot fit
        """
        path = _validate_file_exists(image_path)

        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")

        if size_unit not in ("MB", "GB"):
            raise ValueError(f"size_unit must be 'MB' or 'GB', got '{size_unit}'")

        if not (1 <= min_quality <= 100):
            raise ValueError(f"min_quality must be 1-100, got {min_quality}")

        # Convert to bytes
        max_bytes = max_size * (1024 * 1024) if size_unit == "MB" else max_size * (1024 * 1024 * 1024)

        try:
            with Image.open(path) as img:
                img = _convert_to_rgb(img)
                original_size = path.stat().st_size

                # If already within size, return as-is
                if original_size <= max_bytes:
                    logger.info(
                        f"Image {path.name} already within {max_size}{size_unit} limit "
                        f"({original_size / 1024 / 1024:.2f}MB)"
                    )
                    return img

                logger.warning(
                    f"Image {path.name} is {original_size / 1024 / 1024:.2f}MB, "
                    f"exceeds limit of {max_size}{size_unit}. Compressing..."
                )

                # Strategy: First reduce quality, then scale dimensions
                quality = 85
                scale_factor = 1.0
                best_img = img

                # Phase 1: Reduce quality
                while quality >= min_quality:
                    output = io.BytesIO()
                    best_img.save(output, format=target_format, quality=quality, optimize=True)
                    current_size = output.tell()

                    if current_size <= max_bytes:
                        logger.info(
                            f"Successfully compressed to {current_size / 1024 / 1024:.2f}MB "
                            f"at quality {quality}"
                        )
                        return best_img

                    quality -= 5

                # Phase 2: Scale dimensions
                while scale_factor > 0.1:
                    scale_factor -= 0.1
                    new_width = max(MIN_IMAGE_DIMENSION, int(img.width * scale_factor))
                    new_height = max(MIN_IMAGE_DIMENSION, int(img.height * scale_factor))

                    scaled_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                    output = io.BytesIO()
                    scaled_img.save(output, format=target_format, quality=min_quality, optimize=True)
                    current_size = output.tell()

                    if current_size <= max_bytes:
                        logger.info(
                            f"Successfully compressed to {current_size / 1024 / 1024:.2f}MB "
                            f"at {int(scale_factor * 100)}% scale and quality {min_quality}"
                        )
                        return scaled_img

                # If we get here, even at minimum settings it's too large
                raise ValueError(
                    f"Cannot compress image to fit within {max_size}{size_unit} limit. "
                    f"Minimum possible size: {current_size / 1024 / 1024:.2f}MB at "
                    f"{int(scale_factor * 100)}% scale, quality {min_quality}"
                )

        except Image.UnidentifiedImageError:
            raise ValueError(f"File is not a valid image: {image_path}")
        except Exception as e:
            logger.error(f"Error resizing to max size: {e}")
            raise

    @staticmethod
    def save_image_to_max_size(
        image_path: str,
        output_path: str,
        max_size: float,
        size_unit: Literal["MB", "GB"] = "MB",
        target_format: Literal["PNG", "JPG", "JPEG"] = "JPEG",
        min_quality: int = 10,
    ) -> str:
        """
        Resize and save image to fit within a maximum file size.

        Convenience function that combines resizing and saving.

        Args:
            image_path: Path to input image
            output_path: Path where to save resized image
            max_size: Maximum file size
            size_unit: Unit of max_size - "MB" or "GB" (default "MB")
            target_format: Output format - "PNG" or "JPG"/"JPEG" (default "JPEG")
            min_quality: Minimum JPEG quality before scaling (default 10)

        Returns:
            Path to the saved image file
        """
        format_upper = "JPEG" if target_format.upper() in ("JPG", "JPEG") else target_format.upper()
        img = ImageUtils.resize_to_max_size(
            image_path,
            max_size,
            size_unit,
            format_upper,
            min_quality,
        )
        return ImageUtils.save_pil_image(img, output_path, format=format_upper)

    @staticmethod
    def get_image_size_mb(image_path: str) -> float:
        """
        Get image file size in MB.

        Args:
            image_path: Path to the image file

        Returns:
            File size in megabytes

        Raises:
            FileNotFoundError: If the image file doesn't exist
        """
        path = _validate_file_exists(image_path)
        return path.stat().st_size / (1024 * 1024)

    @staticmethod
    def estimate_compressed_size(
        image_path: str,
        target_format: Literal["JPEG", "PNG"] = "JPEG",
        quality: int = 85,
    ) -> float:
        """
        Estimate the file size after compression without actually saving.

        Useful for checking if image will fit within size limits.

        Args:
            image_path: Path to the image file
            target_format: Output format - "JPEG" or "PNG"
            quality: JPEG quality (1-100, default 85)

        Returns:
            Estimated file size in MB after compression

        Raises:
            FileNotFoundError: If the image file doesn't exist
            ValueError: If file is not a valid image
        """
        path = _validate_file_exists(image_path)

        try:
            with Image.open(path) as img:
                if target_format == "JPEG":
                    img = _convert_to_rgb(img)
                else:
                    if img.mode not in ("RGB", "RGBA", "L"):
                        img = img.convert("RGB")

                output = io.BytesIO()
                save_kwargs = {"format": target_format}
                if target_format == "JPEG":
                    save_kwargs["quality"] = quality
                    save_kwargs["optimize"] = True

                img.save(output, **save_kwargs)
                size_mb = output.tell() / (1024 * 1024)
                return round(size_mb, 2)

        except Image.UnidentifiedImageError:
            raise ValueError(f"File is not a valid image: {image_path}")
        except Exception as e:
            logger.error(f"Error estimating compressed size: {e}")
            raise

    # ========================================================================
    # SAVE/EXPORT FUNCTIONS (Save any input type to PNG/JPG)
    # ========================================================================

    @staticmethod
    def save_image(
        image_data: Union[str, Image.Image, np.ndarray],
        output_path: str,
        format: Literal["PNG", "JPG", "JPEG"] = "PNG",
        quality: int = 85,
        input_type: Optional[Literal["path", "pil", "cv2", "base64"]] = None,
    ) -> str:
        """
        Save image from any input type to PNG or JPG format.

        Accepts file paths, PIL Images, OpenCV arrays, or base64 strings
        and saves them to the specified output location in PNG or JPG format.

        Args:
            image_data: Image to save - can be:
                - str: File path, base64 string, or will auto-detect
                - PIL Image object
                - OpenCV numpy array (BGR format)
            output_path: Path where to save the image
            format: Output format - "PNG" or "JPG"/"JPEG" (default "PNG")
            quality: Quality for JPG (1-100, default 85). Ignored for PNG.
            input_type: Explicitly specify input type: "path", "pil", "cv2", or "base64".
                       If None, auto-detects based on image_data type (default None)

        Returns:
            Path to the saved image file

        Raises:
            ValueError: If input is invalid or format is unsupported
            FileNotFoundError: If input file path doesn't exist
        """
        if format.upper() not in ("PNG", "JPG", "JPEG"):
            raise ValueError(f"format must be 'PNG' or 'JPG'/'JPEG', got '{format}'")

        if not (1 <= quality <= 100):
            raise ValueError(f"quality must be 1-100, got {quality}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Normalize format
        format_upper = "JPEG" if format.upper() in ("JPG", "JPEG") else format.upper()

        try:
            # Auto-detect input type if not specified
            if input_type is None:
                if isinstance(image_data, str):
                    # Could be path or base64
                    if image_data.startswith(("data:", "iVBORw", "/9j/", "image/")):
                        input_type = "base64"
                    elif (image_data.endswith((".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff")) or
                          Path(image_data).exists()):
                        input_type = "path"
                    else:
                        # Try as base64 if it looks like it
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
                    raise ValueError(
                        f"Unsupported image_data type: {type(image_data)}. "
                        "Expected str, PIL Image, or numpy array"
                    )

            # Convert to PIL Image based on input type
            if input_type == "path":
                pil_image = Image.open(image_data)
            elif input_type == "pil":
                pil_image = image_data
            elif input_type == "cv2":
                pil_image = ImageUtils.cv2_to_pil(image_data)
            elif input_type == "base64":
                pil_image = ImageUtils.b64_to_pil(image_data)
            else:
                raise ValueError(f"Unknown input_type: {input_type}")

            # Convert to appropriate color mode for saving
            if format_upper == "JPEG":
                pil_image = _convert_to_rgb(pil_image)
            else:  # PNG
                if pil_image.mode not in ("RGB", "RGBA", "L"):
                    pil_image = pil_image.convert("RGB")

            # Save image
            save_kwargs = {}
            if format_upper == "JPEG":
                save_kwargs["quality"] = quality
                save_kwargs["optimize"] = True

            pil_image.save(str(output_path), format=format_upper, **save_kwargs)

            logger.info(f"Saved image to {output_path} ({format_upper}, {quality}% quality)")
            return str(output_path)

        except Exception as e:
            logger.error(f"Error saving image: {e}")
            raise

    @staticmethod
    def save_images_batch(
        image_data_list: List[Union[str, Image.Image, np.ndarray]],
        output_directory: str,
        format: Literal["PNG", "JPG", "JPEG"] = "PNG",
        quality: int = 85,
        filename_prefix: str = "image",
        input_type: Optional[Literal["path", "pil", "cv2", "base64"]] = None,
    ) -> List[str]:
        """
        Save multiple images from any input type to PNG or JPG format.

        Batch operation to save multiple images with automatic naming.

        Args:
            image_data_list: List of images (file paths, PIL Images, CV2 arrays, or base64 strings)
            output_directory: Directory where to save images
            format: Output format - "PNG" or "JPG"/"JPEG" (default "PNG")
            quality: Quality for JPG (1-100, default 85). Ignored for PNG.
            filename_prefix: Prefix for generated filenames (default "image")
            input_type: Explicitly specify input type if all items are same type

        Returns:
            List of paths to saved image files

        Raises:
            ValueError: If input is invalid
            FileNotFoundError: If output directory or input files don't exist
        """
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []

        try:
            for idx, image_data in enumerate(image_data_list):
                file_extension = ".png" if format.upper() == "PNG" else ".jpg"
                output_path = output_dir / f"{filename_prefix}_{idx:04d}{file_extension}"

                saved_path = ImageUtils.save_image(
                    image_data,
                    str(output_path),
                    format=format,
                    quality=quality,
                    input_type=input_type,
                )
                saved_paths.append(saved_path)

            logger.info(f"Saved {len(saved_paths)} images to {output_directory}")
            return saved_paths

        except Exception as e:
            logger.error(f"Error in batch save: {e}")
            raise

    @staticmethod
    def save_from_path(
        input_path: str,
        output_path: str,
        format: Literal["PNG", "JPG", "JPEG"] = "PNG",
        quality: int = 85,
    ) -> str:
        """
        Load image from file path and save in different format.

        Convenience function for converting image files between formats.

        Args:
            input_path: Path to input image file
            output_path: Path where to save converted image
            format: Output format - "PNG" or "JPG"/"JPEG" (default "PNG")
            quality: Quality for JPG (1-100, default 85). Ignored for PNG.

        Returns:
            Path to the saved image file
        """
        return ImageUtils.save_image(
            input_path,
            output_path,
            format=format,
            quality=quality,
            input_type="path",
        )

    @staticmethod
    def save_pil_image(
        pil_image: Image.Image,
        output_path: str,
        format: Literal["PNG", "JPG", "JPEG"] = "PNG",
        quality: int = 85,
    ) -> str:
        """
        Save PIL Image to PNG or JPG format.

        Convenience function for saving PIL Images directly.

        Args:
            pil_image: PIL Image object
            output_path: Path where to save image
            format: Output format - "PNG" or "JPG"/"JPEG" (default "PNG")
            quality: Quality for JPG (1-100, default 85). Ignored for PNG.

        Returns:
            Path to the saved image file
        """
        return ImageUtils.save_image(
            pil_image,
            output_path,
            format=format,
            quality=quality,
            input_type="pil",
        )

    @staticmethod
    def save_cv2_image(
        cv_image: np.ndarray,
        output_path: str,
        format: Literal["PNG", "JPG", "JPEG"] = "PNG",
        quality: int = 85,
    ) -> str:
        """
        Save OpenCV image to PNG or JPG format.

        Converts OpenCV array (BGR) to PNG/JPG and saves.

        Args:
            cv_image: OpenCV image as numpy array (BGR format)
            output_path: Path where to save image
            format: Output format - "PNG" or "JPG"/"JPEG" (default "PNG")
            quality: Quality for JPG (1-100, default 85). Ignored for PNG.

        Returns:
            Path to the saved image file
        """
        return ImageUtils.save_image(
            cv_image,
            output_path,
            format=format,
            quality=quality,
            input_type="cv2",
        )

    @staticmethod
    def save_from_b64(
        b64_string: str,
        output_path: str,
        format: Literal["PNG", "JPG", "JPEG"] = "PNG",
        quality: int = 85,
    ) -> str:
        """
        Save base64 encoded image to PNG or JPG format.

        Decodes base64 string and saves as PNG/JPG file.

        Args:
            b64_string: Base64 encoded image (with or without data URI prefix)
            output_path: Path where to save image
            format: Output format - "PNG" or "JPG"/"JPEG" (default "PNG")
            quality: Quality for JPG (1-100, default 85). Ignored for PNG.

        Returns:
            Path to the saved image file
        """
        return ImageUtils.save_image(
            b64_string,
            output_path,
            format=format,
            quality=quality,
            input_type="base64",
        )
