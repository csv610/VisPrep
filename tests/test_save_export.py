"""Tests for save and export functions."""


from image_utils import collect_images, convert_format, create_gradient_image, resize_to_dimensions, save_cv2_image, save_from_b64, save_from_path, save_image, save_images_batch, save_pil_image

import io

import pytest
from pathlib import Path
from PIL import Image
class TestSaveImage:
    """Test suite for save_image function."""

    def test_save_pil_image(self, pil_image, temp_dir):
        """Test saving PIL Image object."""
        output_path = f"{temp_dir}/output.jpg"
        result = save_image(pil_image, output_path)
        assert Path(result).exists()
        assert result.endswith(".jpg")

    def test_save_from_file_path(self, sample_image_path, temp_dir):
        """Test saving from file path."""
        output_path = f"{temp_dir}/output.png"
        result = save_image(sample_image_path, output_path, format="PNG")
        assert Path(result).exists()
        assert result.endswith(".png")

    def test_save_from_cv2_array(self, cv2_image, temp_dir):
        """Test saving from OpenCV array."""
        output_path = f"{temp_dir}/output.jpg"
        result = save_image(cv2_image, output_path)
        assert Path(result).exists()

    def test_save_from_base64(self, base64_data_uri, temp_dir):
        """Test saving from base64 string."""
        output_path = f"{temp_dir}/output.jpg"
        result = save_image(base64_data_uri, output_path)
        assert Path(result).exists()

    def test_save_with_quality_parameter(self, temp_dir):
        """Test saving with quality parameter."""
        gradient = create_gradient_image(
            512, 512, (0, 0, 0), (255, 255, 255), direction="horizontal"
        )
        high_quality = f"{temp_dir}/high.jpg"
        low_quality = f"{temp_dir}/low.jpg"

        save_image(gradient, high_quality, format="JPG", quality=80)
        save_image(gradient, low_quality, format="JPG", quality=5)

        assert Path(low_quality).stat().st_size < Path(high_quality).stat().st_size

    def test_save_creates_output_directory(self, pil_image, temp_dir):
        """Test that save creates output directory if needed."""
        output_path = f"{temp_dir}/subdir/subdir2/output.jpg"
        result = save_image(pil_image, output_path)
        assert Path(result).exists()
        assert Path(output_path).exists()

    def test_save_invalid_format(self, pil_image, temp_dir):
        """Test error handling for invalid format."""
        with pytest.raises(ValueError, match="must be one of"):
            save_image(
                pil_image,
                f"{temp_dir}/output.jpg",
                format="INVALID"
            )

    def test_save_invalid_quality(self, pil_image, temp_dir):
        """Test error handling for invalid quality."""
        with pytest.raises(ValueError, match="quality"):
            save_image(
                pil_image,
                f"{temp_dir}/output.jpg",
                quality=150
            )

    def test_save_explicit_input_type(self, pil_image, temp_dir):
        """Test saving with explicit input type."""
        output_path = f"{temp_dir}/output.jpg"
        result = save_image(
            pil_image,
            output_path,
            input_type="pil"
        )
        assert Path(result).exists()


class TestSaveImageBatch:
    """Test suite for save_images_batch function."""

    def test_save_batch_pil_images(self, pil_image, temp_dir):
        """Test saving batch of PIL Images."""
        images = [pil_image for _ in range(3)]
        output_dir = f"{temp_dir}/batch"
        results = save_images_batch(images, output_dir)

        assert len(results) == 3
        assert all(Path(r).exists() for r in results)

    def test_save_batch_with_custom_prefix(self, pil_image, temp_dir):
        """Test saving batch with custom filename prefix."""
        images = [pil_image for _ in range(3)]
        output_dir = f"{temp_dir}/batch"
        results = save_images_batch(
            images,
            output_dir,
            filename_prefix="custom"
        )

        assert all("custom" in Path(r).name for r in results)

    def test_save_batch_different_formats(self, sample_image_path, temp_dir):
        """Test saving batch in different formats."""
        # Collect multiple images or create them
        images = [sample_image_path] * 2
        output_dir = f"{temp_dir}/batch"

        results_png = save_images_batch(
            images,
            output_dir,
            format="PNG"
        )
        assert all(r.endswith(".png") for r in results_png)

    def test_save_batch_empty_list(self, temp_dir):
        """Test saving empty batch."""
        results = save_images_batch([], f"{temp_dir}/batch")
        assert results == []

    def test_save_batch_creates_directory(self, pil_image, temp_dir):
        """Test that batch save creates output directory."""
        images = [pil_image]
        output_dir = f"{temp_dir}/new_dir/batch"
        results = save_images_batch(images, output_dir)

        assert Path(output_dir).exists()
        assert all(Path(r).exists() for r in results)


class TestSaveConvenienceFunctions:
    """Test suite for convenience save functions."""

    def test_save_from_path(self, sample_image_path, temp_dir):
        """Test save_from_path convenience function."""
        output_path = f"{temp_dir}/output.jpg"
        result = save_from_path(sample_image_path, output_path)
        assert Path(result).exists()

    def test_save_pil_image_func(self, pil_image, temp_dir):
        """Test save_pil_image convenience function."""
        output_path = f"{temp_dir}/output.png"
        result = save_pil_image(pil_image, output_path, format="PNG")
        assert Path(result).exists()
        assert result.endswith(".png")

    def test_save_cv2_image_func(self, cv2_image, temp_dir):
        """Test save_cv2_image convenience function."""
        output_path = f"{temp_dir}/output.jpg"
        result = save_cv2_image(cv2_image, output_path)
        assert Path(result).exists()

    def test_save_from_b64_func(self, base64_data_uri, temp_dir):
        """Test save_from_b64 convenience function."""
        output_path = f"{temp_dir}/output.jpg"
        result = save_from_b64(base64_data_uri, output_path)
        assert Path(result).exists()

    def test_save_convenience_with_quality(self, pil_image, temp_dir):
        """Test convenience functions with quality parameter."""
        output_path = f"{temp_dir}/output.jpg"
        result = save_pil_image(pil_image, output_path, quality=70)
        assert Path(result).exists()


class TestSaveIntegration:
    """Integration tests for save operations."""

    def test_load_transform_save(self, sample_image_path, temp_dir):
        """Test complete workflow: load -> transform -> save."""
        # Load and transform
        img = resize_to_dimensions(sample_image_path, 256, 256)

        # Save
        output_path = f"{temp_dir}/transformed.jpg"
        result = save_pil_image(img, output_path)

        # Verify
        assert Path(result).exists()
        saved_img = Image.open(result)
        assert saved_img.size == (256, 256)

    def test_batch_workflow(self, test_images_directory, temp_dir):
        """Test complete batch workflow: collect -> save."""
        # Collect
        images = collect_images(test_images_directory, validate=False)

        # Save
        output_dir = f"{temp_dir}/processed"
        results = save_images_batch(images, output_dir)

        # Verify all saved
        assert len(results) == len(images)
        assert all(Path(r).exists() for r in results)

    def test_format_convert_save(self, sample_image_path, temp_dir):
        """Test conversion and save workflow."""
        png_bytes = convert_format(sample_image_path, target_format="PNG")
        pil_img = Image.open(io.BytesIO(png_bytes))
        output_path = f"{temp_dir}/converted.png"
        result = save_pil_image(pil_img, output_path, format="PNG")
        assert Path(result).exists()
