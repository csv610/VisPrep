"""image_utils: prepare, convert, transform, create, optimize, save, and collect images.

Provides a comprehensive toolkit for image processing in vision-analysis
pipelines — from format conversion and metadata stripping to LLM vision API
preparation (``prepare_for_api``).

Typical usage::

    from image_utils import prepare_for_api
    b64 = prepare_for_api("photo.jpg", target_format="JPEG", max_dimension=2048)
"""

__version__ = "0.1.0"

from image_utils._utils import (
    CV2_AVAILABLE,
    MAX_IMAGE_SIZE_MB, MAX_IMAGE_SIZE_BYTES,
    MAX_TOTAL_IMAGE_PAYLOAD_MB, MAX_TOTAL_IMAGE_PAYLOAD_BYTES,
    MIN_IMAGE_DIMENSION,
    VALID_IMAGE_EXTENSIONS,
)

from image_utils.convert import (
    encode_to_base64, b64_to_pil, pil_to_b64,
    cv2_to_pil, pil_to_cv2, cv2_to_b64, b64_to_cv2,
    convert_format, is_valid_image, is_valid_size, is_valid_dimensions,
)

from image_utils.transform import (
    resize_images_to_fit, square_image, resize_to_dimensions, crop,
)

from image_utils.alpha import remove_alpha_channel, add_alpha_channel

from image_utils.metadata import get_image_info, auto_orient, remove_exif, get_image_size_mb

from image_utils.create import create_blank_image, create_random_image, create_gradient_image

from image_utils.optimize import resize_to_max_size, save_image_to_max_size, estimate_compressed_size

from image_utils.save import (
    save_image, save_images_batch,
    save_from_path, save_pil_image, save_cv2_image, save_from_b64,
)

from image_utils.collect import collect_images, collect_images_with_info

from image_utils.adapters import prepare_for_api, anthropic_prepare, openai_prepare

from image_utils.cli import main as cli_main

__all__ = [
    "prepare_for_api", "anthropic_prepare", "openai_prepare",
    "encode_to_base64", "convert_format",
    "b64_to_pil", "pil_to_b64", "cv2_to_pil", "pil_to_cv2", "cv2_to_b64", "b64_to_cv2",
    "square_image", "resize_to_dimensions", "crop", "resize_images_to_fit",
    "remove_alpha_channel", "add_alpha_channel",
    "get_image_info", "auto_orient", "remove_exif", "get_image_size_mb",
    "is_valid_image", "is_valid_size", "is_valid_dimensions",
    "create_blank_image", "create_random_image", "create_gradient_image",
    "resize_to_max_size", "save_image_to_max_size", "estimate_compressed_size",
    "save_image", "save_images_batch", "save_from_path", "save_pil_image",
    "save_cv2_image", "save_from_b64",
    "collect_images", "collect_images_with_info",
    "MAX_IMAGE_SIZE_MB", "MAX_IMAGE_SIZE_BYTES",
    "MAX_TOTAL_IMAGE_PAYLOAD_MB", "MAX_TOTAL_IMAGE_PAYLOAD_BYTES",
    "MIN_IMAGE_DIMENSION", "VALID_IMAGE_EXTENSIONS",
    "cli_main",
    "__version__",
]
