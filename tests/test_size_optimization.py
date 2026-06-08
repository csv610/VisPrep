"""Tests for size optimization functions."""


from image_utils import estimate_compressed_size, get_image_size_mb, resize_to_max_size, save_image_to_max_size

import pytest
from PIL import Image
class TestGetImageSize:
    """Test suite for get_image_size_mb function."""

    def test_get_size_small_image(self, sample_image_path):
        """Test getting size of small image."""
        size_mb = get_image_size_mb(sample_image_path)
        assert isinstance(size_mb, float)
        assert size_mb > 0

    def test_get_size_large_image(self, large_image_path):
        """Test getting size of large image."""
        size_mb = get_image_size_mb(large_image_path)
        assert size_mb > 0
        # Large image should be bigger
        assert size_mb > 0.01

    def test_file_not_found(self):
        """Test error handling for non-existent file."""
        with pytest.raises(FileNotFoundError):
            get_image_size_mb("nonexistent.jpg")


class TestEstimateCompressedSize:
    """Test suite for estimate_compressed_size function."""

    def test_estimate_jpeg_compression(self, sample_image_path):
        """Test estimating JPEG compressed size."""
        estimated = estimate_compressed_size(
            sample_image_path,
            target_format="JPEG",
            quality=85
        )
        assert isinstance(estimated, float)
        assert estimated > 0

    def test_estimate_png_compression(self, sample_image_path):
        """Test estimating PNG compressed size."""
        estimated = estimate_compressed_size(
            sample_image_path,
            target_format="PNG"
        )
        assert estimated > 0

    def test_estimate_quality_difference(self, sample_image_path):
        """Test that different quality levels produce different estimates."""
        high_quality = estimate_compressed_size(
            sample_image_path,
            target_format="JPEG",
            quality=95
        )
        low_quality = estimate_compressed_size(
            sample_image_path,
            target_format="JPEG",
            quality=10
        )
        # Low quality should be smaller estimate
        assert low_quality < high_quality

    def test_invalid_format(self, sample_image_path):
        """Test error handling for invalid format."""
        with pytest.raises(ValueError, match="JPEG.*PNG"):
            estimate_compressed_size(
                sample_image_path,
                target_format="INVALID"
            )

    def test_invalid_quality(self, sample_image_path):
        """Test error handling for invalid quality."""
        with pytest.raises(ValueError, match="quality"):
            estimate_compressed_size(
                sample_image_path,
                target_format="JPEG",
                quality=150
            )

    def test_file_not_found(self):
        """Test error handling for non-existent file."""
        with pytest.raises(FileNotFoundError):
            estimate_compressed_size(
                "nonexistent.jpg"
            )


class TestResizeToMaxSize:
    """Test suite for resize_to_max_size function."""

    def test_resize_to_max_size_mb(self, large_image_path):
        """Test resizing image to fit within MB limit."""
        # Assuming large image is bigger than 1MB
        result = resize_to_max_size(
            large_image_path,
            max_size=5,  # 5MB limit
            size_unit="MB"
        )
        assert isinstance(result, Image.Image)

    def test_resize_already_within_limit(self, sample_image_path):
        """Test image already within size limit."""
        # Small image should be within 100MB limit
        result = resize_to_max_size(
            sample_image_path,
            max_size=100,
            size_unit="MB"
        )
        assert isinstance(result, Image.Image)

    def test_resize_with_quality_parameter(self, large_image_path):
        """Test resizing with custom minimum quality."""
        result = resize_to_max_size(
            large_image_path,
            max_size=1,
            size_unit="MB",
            min_quality=20
        )
        assert isinstance(result, Image.Image)

    def test_resize_with_custom_format(self, large_image_path):
        """Test resizing with different target format."""
        result = resize_to_max_size(
            large_image_path,
            max_size=5,
            size_unit="MB",
            target_format="PNG"
        )
        assert isinstance(result, Image.Image)

    def test_invalid_max_size(self, sample_image_path):
        """Test error handling for invalid max_size."""
        with pytest.raises(ValueError, match="max_size"):
            resize_to_max_size(
                sample_image_path,
                max_size=0
            )

    def test_invalid_size_unit(self, sample_image_path):
        """Test error handling for invalid size unit."""
        with pytest.raises(ValueError, match="size_unit"):
            resize_to_max_size(
                sample_image_path,
                max_size=5,
                size_unit="KB"  # Invalid
            )

    def test_invalid_min_quality(self, sample_image_path):
        """Test error handling for invalid min_quality."""
        with pytest.raises(ValueError, match="min_quality"):
            resize_to_max_size(
                sample_image_path,
                max_size=5,
                min_quality=150
            )

    def test_file_not_found(self):
        """Test error handling for non-existent file."""
        with pytest.raises(FileNotFoundError):
            resize_to_max_size(
                "nonexistent.jpg",
                max_size=5
            )

    def test_resize_gb_unit(self, large_image_path):
        """Test resizing with GB size unit."""
        result = resize_to_max_size(
            large_image_path,
            max_size=1,
            size_unit="GB"
        )
        assert isinstance(result, Image.Image)


class TestSaveImageToMaxSize:
    """Test suite for save_image_to_max_size function."""

    def test_save_to_max_size(self, large_image_path, temp_dir):
        """Test saving and resizing image to max size."""
        output_path = f"{temp_dir}/resized.jpg"
        result = save_image_to_max_size(
            large_image_path,
            output_path=output_path,
            max_size=5,
            size_unit="MB"
        )
        assert isinstance(result, str)
        # Check file exists and is smaller than original
        from pathlib import Path
        assert Path(result).exists()

    def test_save_with_format(self, large_image_path, temp_dir):
        """Test saving with specific format."""
        output_path = f"{temp_dir}/resized.png"
        result = save_image_to_max_size(
            large_image_path,
            output_path=output_path,
            max_size=10,
            size_unit="MB",
            target_format="PNG"
        )
        assert isinstance(result, str)
        assert result.endswith(".png")

    def test_save_invalid_format(self, sample_image_path, temp_dir):
        """Test error handling for invalid format."""
        with pytest.raises(ValueError):
            save_image_to_max_size(
                sample_image_path,
                output_path=f"{temp_dir}/output.jpg",
                max_size=5,
                target_format="INVALID"
            )

    def test_file_not_found(self, temp_dir):
        """Test error handling for non-existent input file."""
        with pytest.raises(FileNotFoundError):
            save_image_to_max_size(
                "nonexistent.jpg",
                output_path=f"{temp_dir}/output.jpg",
                max_size=5
            )
