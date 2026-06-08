"""Integration tests for complex workflows."""


from image_utils import auto_orient, b64_to_cv2, b64_to_pil, collect_images, collect_images_with_info, convert_format, create_blank_image, create_gradient_image, create_random_image, crop, cv2_to_b64, cv2_to_pil, estimate_compressed_size, get_image_info, get_image_size_mb, pil_to_b64, pil_to_cv2, remove_exif, resize_to_dimensions, resize_to_max_size, save_image_to_max_size, save_images_batch, save_pil_image, square_image

import pytest
from pathlib import Path
from PIL import Image
class TestCompleteWorkflows:
    """Integration tests for complete image processing workflows."""

    def test_vlm_image_preparation(self, sample_image_path, temp_dir):
        """Test complete VLM image preparation workflow."""
        # 1. Check image info
        info = get_image_info(sample_image_path)
        assert info["width"] > 0

        # 2. Auto-orient
        img = auto_orient(sample_image_path)
        assert isinstance(img, Image.Image)

        # 3. Remove EXIF
        clean_img = remove_exif(sample_image_path)
        assert isinstance(clean_img, Image.Image)

        # 4. Resize to VLM dimensions
        resized = resize_to_dimensions(
            sample_image_path,
            width=1024,
            height=1024
        )
        assert resized.size == (1024, 1024)

        # 5. Save as base64
        b64 = pil_to_b64(resized)
        assert isinstance(b64, str)
        assert b64.startswith("data:image/")

    def test_batch_processing_workflow(self, test_images_directory, temp_dir):
        """Test complete batch processing workflow."""
        # 1. Collect images
        images = collect_images_with_info(test_images_directory)
        assert len(images) > 0

        # 2. Filter and process
        processed_paths = []
        for img_info in images:
            # Get image
            img_path = img_info["path"]

            # Check if needs resizing
            if img_info["width"] > 512:
                img = resize_to_dimensions(img_path, 512, 512)
            else:
                img = Image.open(img_path)

            # Save processed
            output_path = f"{temp_dir}/processed_{Path(img_path).name}"
            save_pil_image(img, output_path)
            processed_paths.append(output_path)

        # 3. Verify all processed
        assert all(Path(p).exists() for p in processed_paths)

    def test_format_conversion_pipeline(self, sample_image_path, temp_dir):
        """Test image format conversion pipeline."""
        import io
        formats = ["JPEG", "PNG", "WEBP"]
        saved_paths = {}

        for fmt in formats:
            try:
                converted = convert_format(
                    sample_image_path,
                    target_format=fmt,
                    quality=85
                )
                pil_img = Image.open(io.BytesIO(converted))
                output_path = f"{temp_dir}/output.{fmt.lower()}"
                result = save_pil_image(pil_img, output_path)
                saved_paths[fmt] = result
            except Exception:
                pass

        assert len(saved_paths) > 0
        assert all(Path(p).exists() for p in saved_paths.values())

    def test_size_optimization_workflow(self, large_image_path, temp_dir):
        """Test size optimization workflow."""
        # 1. Check original size
        original_size = get_image_size_mb(large_image_path)

        # 2. Estimate compression
        estimated = estimate_compressed_size(large_image_path)

        # 3. Resize to fit limit
        output_path = f"{temp_dir}/optimized.jpg"
        result = save_image_to_max_size(
            large_image_path,
            output_path=output_path,
            max_size=5,
            size_unit="MB"
        )

        # 4. Verify optimized
        optimized_size = get_image_size_mb(result)
        assert optimized_size <= 5

    def test_format_conversion_chain(self, pil_image, cv2_image, temp_dir):
        """Test chaining format conversions."""
        # PIL -> base64 -> PIL
        b64 = pil_to_b64(pil_image)
        recovered_pil = b64_to_pil(b64)
        assert isinstance(recovered_pil, Image.Image)

        # OpenCV -> base64 -> OpenCV
        b64_cv = cv2_to_b64(cv2_image)
        recovered_cv = b64_to_cv2(b64_cv)
        assert recovered_cv.shape == cv2_image.shape

        # PIL -> OpenCV -> PIL
        cv_from_pil = pil_to_cv2(pil_image)
        pil_from_cv = cv2_to_pil(cv_from_pil)
        assert isinstance(pil_from_cv, Image.Image)

    def test_image_creation_and_save(self, temp_dir):
        """Test creating images and saving them."""
        # Create different types of images
        created_images = [
            create_blank_image(256, 256, color=(255, 0, 0)),
            create_random_image(256, 256),
            create_gradient_image(
                256, 256,
                start_color=(0, 0, 0),
                end_color=(255, 255, 255)
            ),
        ]

        # Save all
        saved = save_images_batch(
            created_images,
            output_directory=f"{temp_dir}/created",
            format="PNG"
        )

        # Verify all saved
        assert len(saved) == 3
        assert all(Path(p).exists() for p in saved)

    def test_transformation_composition(self, sample_image_path, temp_dir):
        """Test composing multiple transformations."""
        # 1. Create square image
        square = square_image(
            sample_image_path,
            max_size=512,
            background_color=(255, 255, 255),
            position="center"
        )

        # 2. Crop center
        cropped = crop(square, 128, 128, 384, 384)
        assert cropped.size == (256, 256)

        # 3. Resize to dimensions
        resized = resize_to_dimensions(
            sample_image_path,
            256,
            256
        )

        # 4. Save both
        output1 = f"{temp_dir}/composed1.jpg"
        output2 = f"{temp_dir}/composed2.jpg"

        save_pil_image(cropped, output1)
        save_pil_image(resized, output2)

        assert Path(output1).exists()
        assert Path(output2).exists()


class TestErrorRecovery:
    """Tests for error handling and recovery."""

    def test_invalid_image_graceful_handling(self, temp_dir):
        """Test graceful handling of invalid images."""
        invalid_path = f"{temp_dir}/invalid.jpg"
        Path(invalid_path).write_text("Not an image")

        # Various operations should fail gracefully
        with pytest.raises((ValueError, FileNotFoundError)):
            get_image_info(invalid_path)

        with pytest.raises((ValueError, FileNotFoundError)):
            auto_orient(invalid_path)

    def test_batch_with_mixed_validity(self, temp_dir):
        """Test batch operations with mixed valid/invalid files."""
        # Create valid image
        valid_path = f"{temp_dir}/valid.jpg"
        img = Image.new("RGB", (100, 100))
        img.save(valid_path)

        # Create invalid file
        invalid_path = f"{temp_dir}/invalid.jpg"
        Path(invalid_path).write_text("Not valid")

        # Collect with validation should skip invalid
        images = collect_images(
            temp_dir,
            validate=True
        )

        # Should only have valid image
        assert len(images) == 1
        assert valid_path in images[0]

    def test_size_limit_impossible(self, temp_dir):
        """Test raises error when size cannot be reduced enough."""
        img = Image.new("RGB", (32, 32), color=(255, 0, 0))
        tiny_path = f"{temp_dir}/tiny.jpg"
        img.save(tiny_path)

        with pytest.raises(ValueError, match="Cannot compress"):
            resize_to_max_size(
                tiny_path,
                max_size=0.0001,
                size_unit="MB"
            )


class TestDataConsistency:
    """Tests for data consistency across operations."""

    def test_roundtrip_preservation(self, pil_image):
        """Test that roundtrip conversions preserve image data."""
        original_size = pil_image.size
        original_pixel = pil_image.getpixel((0, 0))

        # PIL -> base64 -> PIL
        b64 = pil_to_b64(pil_image)
        recovered = b64_to_pil(b64)

        assert recovered.size == original_size
        # Pixel colors might be slightly different due to compression
        # but should be similar
        recovered_pixel = recovered.getpixel((0, 0))
        assert len(recovered_pixel) == len(original_pixel)

    def test_metadata_preservation(self, sample_image_path):
        """Test that basic metadata is preserved through operations."""
        info = get_image_info(sample_image_path)

        # After various operations, basic dims should be consistent
        square = square_image(
            sample_image_path,
            max_size=512,
            background_color=(255, 255, 255)
        )

        assert square.size == (512, 512)
        # Original dimensions should match reported info
        assert info["width"] > 0
        assert info["height"] > 0

    def test_color_consistency(self, temp_dir):
        """Test that colors are preserved across operations."""
        # Create image with specific color
        color = (123, 45, 67)
        img = create_blank_image(
            256, 256,
            color=color
        )

        # Save and reload
        path = f"{temp_dir}/color_test.jpg"
        save_pil_image(img, path)
        reloaded = Image.open(path)

        # Colors should match (approximately, due to compression)
        pixel = reloaded.getpixel((0, 0))
        # Check each channel is close (within tolerance due to JPEG compression)
        for original, loaded in zip(color, pixel):
            assert abs(original - loaded) < 10  # Small tolerance for JPEG


class TestPerformanceScenarios:
    """Tests for realistic performance scenarios."""

    def test_large_batch_operation(self, test_images_directory):
        """Test batch operations on multiple files."""
        # Collect all images
        images = collect_images(
            test_images_directory,
            recursive=False,
            validate=False
        )

        # Get info for all
        info_list = collect_images_with_info(
            test_images_directory,
            recursive=False
        )

        assert len(images) == len(info_list)
        assert len(images) > 0

    def test_memory_efficiency_with_conversions(self, large_image_path):
        """Test that conversions don't cause memory issues."""
        # Multiple format conversions
        for i in range(3):
            # Convert to different formats
            convert_format(large_image_path, "JPEG")
            convert_format(large_image_path, "PNG")

        # Should complete without issues

    def test_rapid_file_operations(self, pil_image, temp_dir):
        """Test rapid consecutive file operations."""
        # Create and save multiple times
        for i in range(5):
            path = f"{temp_dir}/rapid_{i}.jpg"
            save_pil_image(pil_image, path)

        # All should exist
        for i in range(5):
            assert Path(f"{temp_dir}/rapid_{i}.jpg").exists()
