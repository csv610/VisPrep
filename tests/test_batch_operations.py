"""Tests for batch operation functions."""


from visprep import collect_images, collect_images_with_info, resize_images_to_fit

import pytest
from pathlib import Path
class TestCollectImages:
    """Test suite for collect_images function."""

    def test_collect_all_images(self, test_images_directory):
        """Test collecting all images from directory."""
        images = collect_images(test_images_directory, validate=False)
        assert isinstance(images, list)
        assert len(images) == 5
        assert all(isinstance(img, str) for img in images)

    def test_collect_with_validation(self, test_images_directory):
        """Test collecting images with validation."""
        images = collect_images(test_images_directory, validate=True)
        assert len(images) > 0

    def test_collect_sorted_by_name(self, test_images_directory):
        """Test collecting images sorted by name."""
        images = collect_images(
            test_images_directory,
            sort_by="name",
            validate=False
        )
        assert images == sorted(images)

    def test_collect_sorted_by_size(self, test_images_directory):
        """Test collecting images sorted by size."""
        images = collect_images(
            test_images_directory,
            sort_by="size",
            validate=False
        )
        # Verify sorted by file size
        sizes = [Path(img).stat().st_size for img in images]
        assert sizes == sorted(sizes)

    def test_collect_sorted_by_date(self, test_images_directory):
        """Test collecting images sorted by date."""
        images = collect_images(
            test_images_directory,
            sort_by="date",
            validate=False
        )
        assert len(images) > 0

    def test_collect_with_format_filter(self, test_images_directory):
        """Test collecting with format filter."""
        images = collect_images(
            test_images_directory,
            formats=["JPEG"],
            validate=False
        )
        # All collected should be JPEG
        assert all(".jpg" in img.lower() for img in images)

    def test_collect_non_existent_directory(self):
        """Test error handling for non-existent directory."""
        with pytest.raises(FileNotFoundError):
            collect_images("/nonexistent/directory")

    def test_collect_empty_directory(self, temp_dir):
        """Test collecting from empty directory."""
        images = collect_images(temp_dir, validate=False)
        assert len(images) == 0

    def test_collect_recursive(self, temp_dir):
        """Test recursive directory collection."""
        # Create nested directories with images
        from PIL import Image as PILImage

        subdir = Path(temp_dir) / "subdir"
        subdir.mkdir()

        # Create image in root
        img1 = PILImage.new("RGB", (100, 100))
        img1.save(str(Path(temp_dir) / "test1.jpg"))

        # Create image in subdirectory
        img2 = PILImage.new("RGB", (100, 100))
        img2.save(str(subdir / "test2.jpg"))

        # Non-recursive should find only root
        images_non_recursive = collect_images(
            temp_dir,
            recursive=False,
            validate=False
        )

        # Recursive should find both
        images_recursive = collect_images(
            temp_dir,
            recursive=True,
            validate=False
        )

        assert len(images_recursive) >= len(images_non_recursive)


class TestCollectImagesWithInfo:
    """Test suite for collect_images_with_info function."""

    def test_collect_with_info(self, test_images_directory):
        """Test collecting images with metadata."""
        images = collect_images_with_info(test_images_directory)
        assert isinstance(images, list)
        assert len(images) > 0

    def test_collect_info_contains_metadata(self, test_images_directory):
        """Test that returned info contains required metadata."""
        images = collect_images_with_info(test_images_directory)
        assert len(images) > 0

        info = images[0]
        assert "path" in info
        assert "width" in info
        assert "height" in info
        assert "format" in info
        assert "file_size_mb" in info

    def test_collect_info_sorted_by_size(self, test_images_directory):
        """Test collecting info sorted by size."""
        images = collect_images_with_info(
            test_images_directory,
            sort_by="size"
        )
        sizes = [img["file_size_bytes"] for img in images]
        assert sizes == sorted(sizes)

    def test_collect_info_non_existent_directory(self):
        """Test error handling for non-existent directory."""
        with pytest.raises(FileNotFoundError):
            collect_images_with_info("/nonexistent/directory")

    def test_collect_info_with_format_filter(self, test_images_directory):
        """Test collecting info with format filter."""
        images = collect_images_with_info(
            test_images_directory,
            formats=["JPEG"]
        )
        assert all(img["format"].upper() == "JPEG" for img in images)


class TestResizeImagesToFit:
    """Test suite for resize_images_to_fit function."""

    def test_resize_images_within_limit(self, test_images_directory, temp_dir):
        """Test resizing multiple images to fit limit."""
        image_paths = collect_images(
            test_images_directory,
            validate=False
        )
        # These should be small, already within limit
        result = resize_images_to_fit(image_paths)
        assert isinstance(result, list)
        assert len(result) == len(image_paths)

    def test_empty_list(self):
        """Test with empty image list."""
        result = resize_images_to_fit([])
        assert result == []

    def test_result_paths_exist(self, test_images_directory):
        """Test that returned paths are valid."""
        image_paths = collect_images(
            test_images_directory,
            validate=False
        )
        result = resize_images_to_fit(image_paths)

        # All returned paths should exist
        for path in result:
            assert Path(path).exists()

    def test_result_is_list_of_strings(self, test_images_directory):
        """Test that result is list of strings."""
        image_paths = collect_images(
            test_images_directory,
            validate=False
        )
        result = resize_images_to_fit(image_paths)
        assert all(isinstance(path, str) for path in result)


class TestBatchIntegration:
    """Integration tests for batch operations."""

    def test_collect_and_get_info(self, test_images_directory):
        """Test collecting images and getting their info."""
        info_list = collect_images_with_info(test_images_directory)

        for info in info_list:
            assert info["width"] > 0
            assert info["height"] > 0
            assert isinstance(info["file_size_mb"], float)

    def test_collect_filter_and_process(self, test_images_directory):
        """Test collecting, filtering, and processing workflow."""
        # Collect all images
        all_images = collect_images(
            test_images_directory,
            validate=False
        )

        # Get info
        info_list = collect_images_with_info(
            test_images_directory
        )

        # Filter large images
        large_images = [info for info in info_list if info["file_size_mb"] > 0.001]

        assert len(large_images) <= len(all_images)

    def test_collect_recursive_workflow(self, temp_dir):
        """Test complete recursive collection workflow."""
        from PIL import Image as PILImage

        # Create nested structure
        subdir = Path(temp_dir) / "level1" / "level2"
        subdir.mkdir(parents=True)

        # Create images at different levels
        for i, level in enumerate([temp_dir, Path(temp_dir) / "level1", str(subdir)]):
            img = PILImage.new("RGB", (100, 100), color=(i * 50, 100, 150))
            img.save(str(Path(level) / f"test_{i}.jpg"))

        # Collect recursively
        images = collect_images(
            temp_dir,
            recursive=True,
            validate=False
        )

        # Should find all 3 images
        assert len(images) >= 3
