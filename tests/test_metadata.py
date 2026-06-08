"""Tests for metadata functions."""


from visprep import auto_orient, get_image_info, remove_exif

import pytest
from PIL import Image
class TestGetImageInfo:
    """Test suite for get_image_info function."""

    def test_get_info_basic(self, sample_image_path):
        """Test getting basic image information."""
        info = get_image_info(sample_image_path)
        assert isinstance(info, dict)
        assert "width" in info
        assert "height" in info
        assert "format" in info
        assert "color_mode" in info
        assert "file_size_bytes" in info
        assert "file_size_mb" in info

    def test_get_info_dimensions(self, sample_image_path):
        """Test that reported dimensions are correct."""
        info = get_image_info(sample_image_path)
        assert info["width"] == 512
        assert info["height"] == 512

    def test_get_info_format(self, sample_image_path):
        """Test that format is correctly identified."""
        info = get_image_info(sample_image_path)
        assert info["format"] in ["JPEG", "JPG", "PNG", "GIF", "WEBP"]

    def test_get_info_color_mode(self, sample_image_path):
        """Test that color mode is correctly identified."""
        info = get_image_info(sample_image_path)
        assert info["color_mode"] == "RGB"

    def test_get_info_file_size(self, sample_image_path):
        """Test that file size is reported correctly."""
        info = get_image_info(sample_image_path)
        assert info["file_size_bytes"] > 0
        assert isinstance(info["file_size_mb"], float)

    def test_get_info_rgba_image(self, sample_rgba_image_path):
        """Test getting info for RGBA image."""
        info = get_image_info(sample_rgba_image_path)
        assert info["color_mode"] == "RGBA"

    def test_get_info_grayscale_image(self, sample_grayscale_image_path):
        """Test getting info for grayscale image."""
        info = get_image_info(sample_grayscale_image_path)
        assert info["color_mode"] == "L"

    def test_get_info_has_exif(self, sample_image_path):
        """Test that EXIF presence is reported."""
        info = get_image_info(sample_image_path)
        assert "has_exif" in info
        assert isinstance(info["has_exif"], bool)

    def test_get_info_created_date(self, sample_image_path):
        """Test that created_date field exists."""
        info = get_image_info(sample_image_path)
        assert "created_date" in info

    def test_file_not_found(self):
        """Test error handling for non-existent file."""
        with pytest.raises(FileNotFoundError):
            get_image_info("nonexistent.jpg")

    def test_invalid_image_file(self, temp_dir):
        """Test error handling for invalid image file."""
        # Create a text file
        invalid_path = f"{temp_dir}/invalid.jpg"
        with open(invalid_path, "w") as f:
            f.write("Not an image")

        with pytest.raises(ValueError, match="valid image"):
            get_image_info(invalid_path)


class TestAutoOrient:
    """Test suite for auto_orient function."""

    def test_auto_orient_normal_image(self, sample_image_path):
        """Test auto-orienting image without EXIF rotation."""
        result = auto_orient(sample_image_path)
        assert isinstance(result, Image.Image)

    def test_auto_orient_converts_to_rgb(self, sample_rgba_image_path):
        """Test that auto-orient converts to RGB."""
        result = auto_orient(sample_rgba_image_path)
        assert result.mode == "RGB"

    def test_auto_orient_preserves_dimensions(self, sample_image_path):
        """Test that orientation doesn't change dimensions (unless rotated)."""
        original = Image.open(sample_image_path)
        result = auto_orient(sample_image_path)
        # Dimensions should be same (no EXIF rotation in test image)
        assert original.size == result.size

    def test_file_not_found(self):
        """Test error handling for non-existent file."""
        with pytest.raises(FileNotFoundError):
            auto_orient("nonexistent.jpg")

    def test_invalid_image_file(self, temp_dir):
        """Test error handling for invalid image file."""
        invalid_path = f"{temp_dir}/invalid.jpg"
        with open(invalid_path, "w") as f:
            f.write("Not an image")

        with pytest.raises(ValueError, match="valid image"):
            auto_orient(invalid_path)


class TestRemoveExif:
    """Test suite for remove_exif function."""

    def test_remove_exif_basic(self, sample_image_path):
        """Test removing EXIF data from image."""
        result = remove_exif(sample_image_path)
        assert isinstance(result, Image.Image)

    def test_remove_exif_converts_to_rgb(self, sample_rgba_image_path):
        """Test that remove_exif converts to RGB."""
        result = remove_exif(sample_rgba_image_path)
        assert result.mode == "RGB"

    def test_remove_exif_no_metadata(self, sample_image_path):
        """Test that resulting image has no EXIF data."""
        result = remove_exif(sample_image_path)
        # PIL Image without explicit EXIF shouldn't have getexif return anything
        try:
            exif_data = result.getexif()
            # If exif exists, it should be empty or minimal
            assert len(exif_data) == 0 or exif_data is None
        except AttributeError:
            # Some PIL versions might not have getexif
            pass

    def test_remove_exif_preserves_dimensions(self, sample_image_path):
        """Test that dimensions are preserved."""
        original = Image.open(sample_image_path)
        result = remove_exif(sample_image_path)
        assert original.size == result.size

    def test_file_not_found(self):
        """Test error handling for non-existent file."""
        with pytest.raises(FileNotFoundError):
            remove_exif("nonexistent.jpg")

    def test_invalid_image_file(self, temp_dir):
        """Test error handling for invalid image file."""
        invalid_path = f"{temp_dir}/invalid.jpg"
        with open(invalid_path, "w") as f:
            f.write("Not an image")

        with pytest.raises(ValueError, match="valid image"):
            remove_exif(invalid_path)

    def test_remove_exif_grayscale(self, sample_grayscale_image_path):
        """Test removing EXIF from grayscale image."""
        result = remove_exif(sample_grayscale_image_path)
        assert isinstance(result, Image.Image)


class TestMetadataIntegration:
    """Integration tests for metadata functions."""

    def test_get_info_then_auto_orient(self, sample_image_path):
        """Test getting info and auto-orienting."""
        info = get_image_info(sample_image_path)
        img = auto_orient(sample_image_path)
        # Should have consistent dimensions
        assert img.size == (info["width"], info["height"])

    def test_auto_orient_then_remove_exif(self, sample_image_path):
        """Test auto-orienting and removing EXIF."""
        img = auto_orient(sample_image_path)
        clean_img = remove_exif(sample_image_path)
        # Both should be valid images
        assert isinstance(img, Image.Image)
        assert isinstance(clean_img, Image.Image)

    def test_all_metadata_operations(self, sample_image_path):
        """Test all metadata operations together."""
        # Get info
        info = get_image_info(sample_image_path)
        assert info is not None

        # Auto orient
        img = auto_orient(sample_image_path)
        assert isinstance(img, Image.Image)

        # Remove EXIF
        clean_img = remove_exif(sample_image_path)
        assert isinstance(clean_img, Image.Image)

        # All should have consistent dimensions
        assert img.size[0] == info["width"]
        assert clean_img.size[0] == info["width"]
