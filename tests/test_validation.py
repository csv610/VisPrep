"""Tests for validation functions."""


from image_utils import collect_images, is_valid_dimensions, is_valid_image, is_valid_size

import pytest
from pathlib import Path
from PIL import Image
class TestIsValidImage:
    """Test suite for is_valid_image function."""

    def test_valid_jpeg(self, sample_image_path):
        """Test validation of valid JPEG image."""
        path = Path(sample_image_path)
        assert is_valid_image(path)

    def test_valid_png(self, sample_rgba_image_path):
        """Test validation of valid PNG image."""
        path = Path(sample_rgba_image_path)
        assert is_valid_image(path)

    def test_valid_jpg_extension(self, temp_dir):
        """Test validation of .jpg extension."""
        img_path = Path(temp_dir) / "test.jpg"
        img = Image.new("RGB", (100, 100))
        img.save(str(img_path))
        assert is_valid_image(img_path)

    def test_valid_jpeg_extension(self, temp_dir):
        """Test validation of .jpeg extension."""
        img_path = Path(temp_dir) / "test.jpeg"
        img = Image.new("RGB", (100, 100))
        img.save(str(img_path), "JPEG")
        assert is_valid_image(img_path)

    def test_valid_gif(self, temp_dir):
        """Test validation of GIF image."""
        img_path = Path(temp_dir) / "test.gif"
        img = Image.new("RGB", (100, 100))
        img.save(str(img_path), "GIF")
        assert is_valid_image(img_path)

    def test_valid_webp(self, temp_dir):
        """Test validation of WEBP image."""
        img_path = Path(temp_dir) / "test.webp"
        img = Image.new("RGB", (100, 100))
        img.save(str(img_path), "WEBP")
        assert is_valid_image(img_path)

    def test_invalid_extension(self, temp_dir):
        """Test that invalid extensions return False."""
        txt_path = Path(temp_dir) / "test.txt"
        txt_path.write_text("Not an image")
        assert not is_valid_image(txt_path)

    def test_case_insensitive(self, temp_dir):
        """Test that extension check is case insensitive."""
        img_path = Path(temp_dir) / "test.JPG"  # Uppercase
        img = Image.new("RGB", (100, 100))
        img.save(str(img_path))
        assert is_valid_image(img_path)


class TestIsValidSize:
    """Test suite for is_valid_size function."""

    def test_small_image_is_valid(self, sample_image_path):
        """Test that small image is within size limit."""
        path = Path(sample_image_path)
        assert is_valid_size(path)

    def test_check_returns_boolean(self, sample_image_path):
        """Test that function returns boolean."""
        path = Path(sample_image_path)
        result = is_valid_size(path)
        assert isinstance(result, bool)

    def test_file_size_calculation(self, sample_image_path):
        """Test that function checks actual file size."""
        path = Path(sample_image_path)
        file_size = path.stat().st_size
        # Should be valid (less than 50MB limit)
        assert file_size < 50 * 1024 * 1024
        assert is_valid_size(path)


class TestIsValidDimensions:
    """Test suite for is_valid_dimensions function."""

    def test_valid_dimensions(self, sample_image_path):
        """Test validation of valid dimensions."""
        path = Path(sample_image_path)
        assert is_valid_dimensions(path)

    def test_small_image_still_valid(self, temp_dir):
        """Test that image at minimum dimensions is valid."""
        img_path = Path(temp_dir) / "small.jpg"
        # Create 32x32 image (minimum)
        img = Image.new("RGB", (32, 32))
        img.save(str(img_path))
        assert is_valid_dimensions(img_path)

    def test_check_returns_boolean(self, sample_image_path):
        """Test that function returns boolean."""
        path = Path(sample_image_path)
        result = is_valid_dimensions(path)
        assert isinstance(result, bool)

    def test_invalid_image_returns_false(self, temp_dir):
        """Test that invalid image file returns False."""
        invalid_path = Path(temp_dir) / "invalid.jpg"
        invalid_path.write_text("Not an image")
        assert not is_valid_dimensions(invalid_path)

    def test_dimension_check_values(self, sample_image_path):
        """Test that dimensions are correctly checked."""
        img = Image.open(sample_image_path)
        width, height = img.size
        # Both should be >= 32
        assert width >= 32
        assert height >= 32


class TestValidationIntegration:
    """Integration tests for validation functions."""

    def test_all_validation_checks(self, sample_image_path):
        """Test all validation checks together."""
        path = Path(sample_image_path)

        # All checks should pass for valid image
        assert is_valid_image(path)
        assert is_valid_size(path)
        assert is_valid_dimensions(path)

    def test_invalid_file_all_checks(self, temp_dir):
        """Test that invalid file fails appropriate checks."""
        invalid_path = Path(temp_dir) / "invalid.jpg"
        invalid_path.write_text("Not an image")

        # Extension check might pass (has .jpg)
        # But dimension check should fail
        assert not is_valid_dimensions(invalid_path)

    def test_validation_with_batch(self, test_images_directory):
        """Test validation in batch operation context."""
        images = collect_images(
            test_images_directory,
            validate=True
        )

        # All collected should pass validation
        for img_path in images:
            path = Path(img_path)
            assert is_valid_image(path)
            assert is_valid_size(path)
            assert is_valid_dimensions(path)

    def test_validation_different_formats(self, test_images_directory):
        """Test validation works across different formats."""
        # Create images in different formats
        from PIL import Image as PILImage

        formats = ["JPEG", "PNG", "GIF", "WEBP"]
        paths = []

        for fmt in formats:
            img = PILImage.new("RGB", (256, 256))
            path = Path(test_images_directory) / f"test.{fmt.lower()}"
            try:
                img.save(str(path), fmt)
                paths.append(str(path))
            except Exception:
                # Skip unsupported formats
                pass

        # All should be valid
        for img_path in paths:
            path = Path(img_path)
            assert is_valid_image(path)
