"""Pytest configuration and shared fixtures for VisPrep tests."""

import pytest
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
import os


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_image_path(temp_dir):
    """Create a sample image file for testing."""
    img = Image.new("RGB", (512, 512), color=(100, 150, 200))
    image_path = Path(temp_dir) / "sample.jpg"
    img.save(str(image_path))
    return str(image_path)


@pytest.fixture
def sample_rgba_image_path(temp_dir):
    """Create a sample RGBA image file for testing."""
    img = Image.new("RGBA", (256, 256), color=(100, 150, 200, 128))
    image_path = Path(temp_dir) / "sample_rgba.png"
    img.save(str(image_path))
    return str(image_path)


@pytest.fixture
def sample_grayscale_image_path(temp_dir):
    """Create a sample grayscale image file for testing."""
    img = Image.new("L", (256, 256), color=128)
    image_path = Path(temp_dir) / "sample_gray.png"
    img.save(str(image_path))
    return str(image_path)


@pytest.fixture
def pil_image():
    """Create a PIL Image object for testing."""
    return Image.new("RGB", (256, 256), color=(200, 100, 50))


@pytest.fixture
def cv2_image():
    """Create an OpenCV numpy array image for testing."""
    # BGR format: Blue, Green, Red
    return np.ones((256, 256, 3), dtype=np.uint8) * np.array([50, 100, 200], dtype=np.uint8)


@pytest.fixture
def large_image_path(temp_dir):
    """Create a large test image (simulating high resolution)."""
    img = Image.new("RGB", (4000, 3000), color=(75, 125, 175))
    image_path = Path(temp_dir) / "large.jpg"
    img.save(str(image_path), quality=95)
    return str(image_path)


@pytest.fixture
def test_images_directory(temp_dir):
    """Create a directory with multiple test images."""
    images_dir = Path(temp_dir) / "images"
    images_dir.mkdir()

    # Create 5 test images
    for i in range(5):
        img = Image.new(
            "RGB",
            (256, 256),
            color=(i * 50, (5 - i) * 50, 128)
        )
        img.save(str(images_dir / f"test_{i}.jpg"))

    return str(images_dir)


@pytest.fixture
def base64_string():
    """Provide a sample base64 encoded image string."""
    # Small 1x1 red pixel PNG in base64
    return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="


@pytest.fixture
def base64_data_uri():
    """Provide a sample base64 data URI."""
    return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
