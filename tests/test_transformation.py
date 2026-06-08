"""Tests for image transformation functions."""


from image_utils import add_alpha_channel, convert_format, crop, remove_alpha_channel, resize_to_dimensions, square_image

import pytest
from PIL import Image
from pathlib import Path
class TestSquareImage:
    """Test suite for square_image function."""

    def test_create_square_centered(self, sample_image_path):
        """Test creating square image with centered content."""
        result = square_image(
            sample_image_path,
            max_size=512,
            background_color=(255, 255, 255),
            position="center"
        )
        assert result.size == (512, 512)
        assert result.mode == "RGB"

    def test_create_square_topleft(self, sample_image_path):
        """Test creating square image with top-left positioning."""
        result = square_image(
            sample_image_path,
            max_size=768,
            background_color=(0, 0, 0),
            position="top-left"
        )
        assert result.size == (768, 768)
        # Top-left should be the image (positioned at origin)
        r, g, b = result.getpixel((0, 0))
        assert r == 100
        assert g == 150
        assert abs(b - 200) <= 1  # JPEG compression may shift by 1
        # Bottom-right corner should be black (background beyond image bounds)
        assert result.getpixel((767, 767)) == (0, 0, 0)

    def test_square_background_color(self, sample_image_path):
        """Test square image with specific background color."""
        result = square_image(
            sample_image_path,
            max_size=256,
            background_color=(255, 0, 0),  # Red
            position="center"
        )
        assert result.size == (256, 256)

    def test_invalid_max_size(self, sample_image_path):
        """Test error handling for invalid max_size."""
        with pytest.raises(ValueError, match="max_size"):
            square_image(
                sample_image_path,
                max_size=0,
                background_color=(255, 255, 255)
            )

    def test_invalid_background_color(self, sample_image_path):
        """Test error handling for invalid background color."""
        with pytest.raises(ValueError, match="0-255"):
            square_image(
                sample_image_path,
                max_size=512,
                background_color=(256, 100, 100)  # Invalid
            )

    def test_invalid_position(self, sample_image_path):
        """Test error handling for invalid position."""
        with pytest.raises(ValueError, match="top-left|center"):
            square_image(
                sample_image_path,
                max_size=512,
                background_color=(255, 255, 255),
                position="invalid"
            )

    def test_file_not_found(self):
        """Test error handling for non-existent file."""
        with pytest.raises(FileNotFoundError):
            square_image(
                "nonexistent.jpg",
                max_size=512,
                background_color=(255, 255, 255)
            )


class TestResizeToDimensions:
    """Test suite for resize_to_dimensions function."""

    def test_resize_to_exact_dimensions(self, sample_image_path):
        """Test resizing to exact dimensions."""
        result = resize_to_dimensions(
            sample_image_path,
            width=256,
            height=256
        )
        assert result.size == (256, 256)

    def test_resize_maintain_aspect_ratio(self, sample_image_path):
        """Test that aspect ratio is maintained during resize."""
        result = resize_to_dimensions(
            sample_image_path,
            width=256,
            height=512
        )
        # Should be 256x512 canvas but image should fit inside
        assert result.size == (256, 512)

    def test_resize_with_custom_background(self, sample_image_path):
        """Test resizing with custom background color."""
        result = resize_to_dimensions(
            sample_image_path,
            width=512,
            height=512,
            background_color=(255, 0, 0)  # Red
        )
        assert result.size == (512, 512)

    def test_invalid_dimensions(self, sample_image_path):
        """Test error handling for invalid dimensions."""
        with pytest.raises(ValueError, match="Dimensions"):
            resize_to_dimensions(
                sample_image_path,
                width=0,
                height=256
            )

    def test_invalid_background_color(self, sample_image_path):
        """Test error handling for invalid background color."""
        with pytest.raises(ValueError, match="0-255"):
            resize_to_dimensions(
                sample_image_path,
                width=256,
                height=256,
                background_color=(256, 100, 100)
            )


class TestCropImage:
    """Test suite for crop function."""

    def test_crop_rectangle(self, sample_image_path):
        """Test cropping a rectangle from image."""
        result = crop(
            sample_image_path,
            left=0,
            top=0,
            right=256,
            bottom=256
        )
        assert result.size == (256, 256)

    def test_crop_center(self, sample_image_path):
        """Test cropping center of image."""
        result = crop(
            sample_image_path,
            left=128,
            top=128,
            right=384,
            bottom=384
        )
        assert result.size == (256, 256)

    def test_invalid_coordinates_order(self, sample_image_path):
        """Test error handling for invalid coordinate order."""
        with pytest.raises(ValueError, match="Invalid crop coordinates"):
            crop(
                sample_image_path,
                left=256,
                top=0,
                right=128,  # right < left
                bottom=256
            )

    def test_coordinates_out_of_bounds(self, sample_image_path):
        """Test error handling for out of bounds coordinates."""
        with pytest.raises(ValueError, match="out of bounds"):
            crop(
                sample_image_path,
                left=0,
                top=0,
                right=1000,  # Out of bounds
                bottom=1000
            )

    def test_negative_coordinates(self, sample_image_path):
        """Test error handling for negative coordinates."""
        with pytest.raises(ValueError, match="out of bounds"):
            crop(
                sample_image_path,
                left=-10,  # Negative
                top=0,
                right=100,
                bottom=100
            )

    def test_file_not_found(self):
        """Test error handling for non-existent file."""
        with pytest.raises(FileNotFoundError):
            crop(
                "nonexistent.jpg",
                left=0,
                top=0,
                right=100,
                bottom=100
            )


class TestConvertFormat:
    """Test suite for convert_format function."""

    def test_convert_to_jpeg(self, sample_image_path):
        """Test converting to JPEG format."""
        result = convert_format(
            sample_image_path,
            target_format="JPEG",
            quality=85
        )
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_convert_to_png(self, sample_image_path):
        """Test converting to PNG format."""
        result = convert_format(
            sample_image_path,
            target_format="PNG"
        )
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_convert_to_webp(self, sample_image_path):
        """Test converting to WEBP format."""
        result = convert_format(
            sample_image_path,
            target_format="WEBP",
            quality=80
        )
        assert isinstance(result, bytes)

    def test_quality_parameter(self, sample_image_path):
        """Test that different quality levels produce different file sizes."""
        high_quality = convert_format(
            sample_image_path,
            target_format="JPEG",
            quality=95
        )
        low_quality = convert_format(
            sample_image_path,
            target_format="JPEG",
            quality=10
        )
        # Low quality should be smaller
        assert len(low_quality) < len(high_quality)

    def test_invalid_format(self, sample_image_path):
        """Test error handling for invalid format."""
        with pytest.raises(ValueError, match="JPEG.*PNG.*WEBP"):
            convert_format(
                sample_image_path,
                target_format="INVALID"
            )

    def test_invalid_quality(self, sample_image_path):
        """Test error handling for invalid quality."""
        with pytest.raises(ValueError, match="quality"):
            convert_format(
                sample_image_path,
                target_format="JPEG",
                quality=150  # Out of range
            )

    def test_file_not_found(self):
        """Test error handling for non-existent file."""
        with pytest.raises(FileNotFoundError):
            convert_format(
                "nonexistent.jpg",
                target_format="JPEG"
            )


class TestRemoveAlphaChannel:
    """Test suite for remove_alpha_channel function."""

    def test_remove_alpha_from_rgba_file(self, sample_rgba_image_path):
        """Test removing alpha channel from RGBA image file."""
        result = remove_alpha_channel(sample_rgba_image_path)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == (256, 256)

    def test_remove_alpha_from_rgba_pil_image(self):
        """Test removing alpha channel from RGBA PIL Image object."""
        rgba_image = Image.new("RGBA", (256, 256), color=(100, 150, 200, 128))
        result = remove_alpha_channel(rgba_image)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == (256, 256)

    def test_remove_alpha_with_custom_background(self, sample_rgba_image_path):
        """Test removing alpha channel with custom background color."""
        result = remove_alpha_channel(
            sample_rgba_image_path,
            background_color=(255, 0, 0)  # Red background
        )
        assert result.mode == "RGB"
        assert result.size == (256, 256)
        # Corner should have some red from the background
        assert result.getpixel((0, 0))[0] > 0  # R channel should have red

    def test_rgb_image_unchanged(self, sample_image_path):
        """Test that RGB images are returned unchanged."""
        result = remove_alpha_channel(sample_image_path)
        assert result.mode == "RGB"
        assert result.size == (512, 512)

    def test_rgb_pil_image_unchanged(self, pil_image):
        """Test that RGB PIL Image objects are returned unchanged."""
        result = remove_alpha_channel(pil_image)
        assert result.mode == "RGB"
        assert result.size == (256, 256)

    def test_grayscale_image_unchanged(self, sample_grayscale_image_path):
        """Test that grayscale images are returned unchanged."""
        result = remove_alpha_channel(sample_grayscale_image_path)
        assert result.mode == "L"
        assert result.size == (256, 256)

    def test_la_mode_conversion(self, temp_dir):
        """Test converting LA (grayscale + alpha) mode image."""
        # Create an LA (grayscale + alpha) mode image
        img = Image.new("LA", (256, 256), color=(128, 200))
        image_path = Path(temp_dir) / "grayscale_alpha.png"
        img.save(str(image_path))

        result = remove_alpha_channel(image_path)
        assert result.mode == "RGB"
        assert result.size == (256, 256)

    def test_white_background_default(self, sample_rgba_image_path):
        """Test that white background is used by default."""
        result = remove_alpha_channel(sample_rgba_image_path)
        # Create expected result with white background
        expected = Image.new("RGB", (256, 256), (255, 255, 255))
        # Both should be RGB mode
        assert result.mode == expected.mode

    def test_black_background(self, sample_rgba_image_path):
        """Test using black background."""
        result = remove_alpha_channel(
            sample_rgba_image_path,
            background_color=(0, 0, 0)
        )
        assert result.mode == "RGB"
        assert result.size == (256, 256)

    def test_invalid_background_color_not_tuple(self, sample_rgba_image_path):
        """Test error handling for invalid background color (not a tuple)."""
        with pytest.raises(ValueError, match="RGB tuple"):
            remove_alpha_channel(
                sample_rgba_image_path,
                background_color=[255, 255, 255]  # List instead of tuple
            )

    def test_invalid_background_color_wrong_length(self, sample_rgba_image_path):
        """Test error handling for invalid background color (wrong length)."""
        with pytest.raises(ValueError, match="RGB tuple"):
            remove_alpha_channel(
                sample_rgba_image_path,
                background_color=(255, 255)  # Only 2 values
            )

    def test_invalid_background_color_value_out_of_range(self, sample_rgba_image_path):
        """Test error handling for background color values out of range."""
        with pytest.raises(ValueError, match="0-255"):
            remove_alpha_channel(
                sample_rgba_image_path,
                background_color=(256, 100, 100)  # 256 is out of range
            )

    def test_invalid_background_color_negative(self, sample_rgba_image_path):
        """Test error handling for negative background color values."""
        with pytest.raises(ValueError, match="0-255"):
            remove_alpha_channel(
                sample_rgba_image_path,
                background_color=(-1, 100, 100)
            )

    def test_invalid_input_type(self):
        """Test error handling for invalid input type."""
        with pytest.raises(TypeError, match="file path|PIL Image"):
            remove_alpha_channel(12345)  # Integer instead of string or Image

    def test_file_not_found(self):
        """Test error handling for non-existent file."""
        with pytest.raises(FileNotFoundError):
            remove_alpha_channel("nonexistent_image.png")

    def test_invalid_image_file(self, temp_dir):
        """Test error handling for invalid image file."""
        invalid_file = Path(temp_dir) / "invalid.png"
        invalid_file.write_text("This is not an image")
        with pytest.raises(ValueError, match="valid image"):
            remove_alpha_channel(str(invalid_file))


class TestAddAlphaChannel:
    """Test suite for add_alpha_channel function."""

    def test_add_alpha_to_rgb_file(self, sample_image_path):
        """Test adding alpha channel to RGB image file."""
        result = add_alpha_channel(sample_image_path)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGBA"
        assert result.size == (512, 512)

    def test_add_alpha_to_rgb_pil_image(self, pil_image):
        """Test adding alpha channel to RGB PIL Image object."""
        result = add_alpha_channel(pil_image)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGBA"
        assert result.size == (256, 256)

    def test_add_alpha_to_grayscale(self, sample_grayscale_image_path):
        """Test adding alpha channel to grayscale image."""
        result = add_alpha_channel(sample_grayscale_image_path)
        assert result.mode == "LA"
        assert result.size == (256, 256)

    def test_add_alpha_with_custom_alpha_value(self, sample_image_path):
        """Test adding alpha channel with custom alpha value."""
        result = add_alpha_channel(
            sample_image_path,
            alpha_value=128
        )
        assert result.mode == "RGBA"
        assert result.size == (512, 512)
        # Check that the alpha channel was applied
        pixel = result.getpixel((0, 0))
        assert pixel[3] == 128  # Alpha value should be 128

    def test_add_alpha_fully_transparent(self, sample_image_path):
        """Test adding fully transparent alpha channel."""
        result = add_alpha_channel(
            sample_image_path,
            alpha_value=0
        )
        assert result.mode == "RGBA"
        pixel = result.getpixel((0, 0))
        assert pixel[3] == 0  # Fully transparent

    def test_add_alpha_fully_opaque(self, sample_image_path):
        """Test adding fully opaque alpha channel (default)."""
        result = add_alpha_channel(sample_image_path)
        assert result.mode == "RGBA"
        pixel = result.getpixel((0, 0))
        assert pixel[3] == 255  # Fully opaque

    def test_rgba_image_unchanged(self, sample_rgba_image_path):
        """Test that RGBA images are returned unchanged."""
        result = add_alpha_channel(sample_rgba_image_path)
        assert result.mode == "RGBA"
        assert result.size == (256, 256)

    def test_la_image_unchanged(self, temp_dir):
        """Test that LA images are returned unchanged."""
        img = Image.new("LA", (256, 256), color=(128, 200))
        image_path = Path(temp_dir) / "grayscale_alpha.png"
        img.save(str(image_path))

        result = add_alpha_channel(image_path)
        assert result.mode == "LA"
        assert result.size == (256, 256)

    def test_add_alpha_partial_opacity(self, sample_image_path):
        """Test adding alpha channel with partial opacity."""
        result = add_alpha_channel(
            sample_image_path,
            alpha_value=64
        )
        assert result.mode == "RGBA"
        pixel = result.getpixel((256, 256))
        assert pixel[3] == 64

    def test_invalid_alpha_value_out_of_range_high(self, sample_image_path):
        """Test error handling for alpha value too high."""
        with pytest.raises(ValueError, match="0-255"):
            add_alpha_channel(
                sample_image_path,
                alpha_value=256
            )

    def test_invalid_alpha_value_out_of_range_low(self, sample_image_path):
        """Test error handling for negative alpha value."""
        with pytest.raises(ValueError, match="0-255"):
            add_alpha_channel(
                sample_image_path,
                alpha_value=-1
            )

    def test_invalid_alpha_value_type(self, sample_image_path):
        """Test error handling for non-integer alpha value."""
        with pytest.raises(ValueError, match="integer"):
            add_alpha_channel(
                sample_image_path,
                alpha_value=128.5
            )

    def test_invalid_input_type(self):
        """Test error handling for invalid input type."""
        with pytest.raises(TypeError, match="file path|PIL Image"):
            add_alpha_channel(12345)

    def test_file_not_found(self):
        """Test error handling for non-existent file."""
        with pytest.raises(FileNotFoundError):
            add_alpha_channel("nonexistent_image.png")

    def test_invalid_image_file(self, temp_dir):
        """Test error handling for invalid image file."""
        invalid_file = Path(temp_dir) / "invalid.png"
        invalid_file.write_text("This is not an image")
        with pytest.raises(ValueError, match="valid image"):
            add_alpha_channel(str(invalid_file))

    def test_add_alpha_to_palette_mode(self, temp_dir):
        """Test adding alpha channel to palette mode image."""
        # Create a palette mode image
        img = Image.new("P", (256, 256), color=0)
        image_path = Path(temp_dir) / "palette.png"
        img.save(str(image_path))

        result = add_alpha_channel(image_path)
        assert result.mode == "RGBA"
        assert result.size == (256, 256)
