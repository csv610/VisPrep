"""Tests for image creation functions."""


from visprep import create_blank_image, create_gradient_image, create_random_image

import pytest
from PIL import Image
import numpy as np
class TestCreateBlankImage:
    """Test suite for create_blank_image function."""

    def test_create_white_background(self):
        """Test creating white background image."""
        img = create_blank_image(256, 256, color=(255, 255, 255))
        assert img.size == (256, 256)
        assert img.mode == "RGB"
        # Check pixel color
        assert img.getpixel((0, 0)) == (255, 255, 255)

    def test_create_black_background(self):
        """Test creating black background image."""
        img = create_blank_image(512, 512, color=(0, 0, 0))
        assert img.size == (512, 512)
        assert img.getpixel((256, 256)) == (0, 0, 0)

    def test_create_red_background(self):
        """Test creating red background image."""
        img = create_blank_image(100, 100, color=(255, 0, 0))
        assert img.getpixel((50, 50)) == (255, 0, 0)

    def test_create_random_color(self):
        """Test creating image with random color."""
        img = create_blank_image(256, 256, color="random")
        assert img.size == (256, 256)
        # Verify it has some color (not all same)
        pixel = img.getpixel((0, 0))
        assert isinstance(pixel, tuple)
        assert len(pixel) == 3

    def test_create_grayscale_image(self):
        """Test creating grayscale image."""
        img = create_blank_image(
            256, 256,
            color=(128,),
            image_mode="L"
        )
        assert img.mode == "L"
        assert img.getpixel((0, 0)) == 128

    def test_create_rgba_image(self):
        """Test creating RGBA image."""
        img = create_blank_image(
            256, 256,
            color=(255, 0, 0, 128),
            image_mode="RGBA"
        )
        assert img.mode == "RGBA"
        assert img.getpixel((0, 0)) == (255, 0, 0, 128)

    def test_create_rgba_random_color(self):
        """Test creating RGBA image with random color."""
        img = create_blank_image(
            128, 128,
            color="random",
            image_mode="RGBA"
        )
        assert img.mode == "RGBA"
        pixel = img.getpixel((0, 0))
        assert len(pixel) == 4

    def test_invalid_dimensions(self):
        """Test error handling for invalid dimensions."""
        with pytest.raises(ValueError, match="must be positive"):
            create_blank_image(0, 256)

        with pytest.raises(ValueError, match="must be positive"):
            create_blank_image(256, -100)

    def test_invalid_color_values(self):
        """Test error handling for invalid color values."""
        with pytest.raises(ValueError, match="0-255"):
            create_blank_image(256, 256, color=(256, 100, 100))

        with pytest.raises(ValueError, match="0-255"):
            create_blank_image(256, 256, color=(-1, 100, 100))

    def test_invalid_image_mode(self):
        """Test error handling for invalid image mode."""
        with pytest.raises(ValueError, match="image_mode"):
            create_blank_image(256, 256, image_mode="INVALID")

    def test_grayscale_color_format(self):
        """Test grayscale requires single value."""
        with pytest.raises(ValueError, match="Grayscale requires"):
            create_blank_image(
                256, 256,
                color=(128, 100),
                image_mode="L"
            )


class TestCreateRandomImage:
    """Test suite for create_random_image function."""

    def test_create_rgb_noise(self):
        """Test creating RGB noise image."""
        img = create_random_image(256, 256)
        assert img.size == (256, 256)
        assert img.mode == "RGB"

    def test_create_grayscale_noise(self):
        """Test creating grayscale noise image."""
        img = create_random_image(256, 256, image_mode="L")
        assert img.mode == "L"

    def test_create_rgba_noise(self):
        """Test creating RGBA noise image."""
        img = create_random_image(128, 128, image_mode="RGBA")
        assert img.mode == "RGBA"

    def test_noise_randomness(self):
        """Test that noise is actually random (not uniform color)."""
        img = create_random_image(256, 256)
        pixels = set()
        for y in range(0, 256, 16):
            for x in range(0, 256, 16):
                pixels.add(img.getpixel((x, y)))
        assert len(pixels) > 50  # Random noise should give many unique pixel values

    def test_invalid_dimensions(self):
        """Test error handling for invalid dimensions."""
        with pytest.raises(ValueError, match="must be positive"):
            create_random_image(0, 256)

    def test_invalid_image_mode(self):
        """Test error handling for invalid image mode."""
        with pytest.raises(ValueError, match="image_mode"):
            create_random_image(256, 256, image_mode="CMYK")


class TestCreateGradientImage:
    """Test suite for create_gradient_image function."""

    def test_horizontal_gradient(self):
        """Test creating horizontal gradient."""
        img = create_gradient_image(
            256, 256,
            start_color=(255, 0, 0),      # Red
            end_color=(0, 0, 255),        # Blue
            direction="horizontal"
        )
        assert img.size == (256, 256)
        assert img.mode == "RGB"
        # Left should be more red
        left_pixel = img.getpixel((0, 128))
        # Right should be more blue
        right_pixel = img.getpixel((255, 128))
        assert left_pixel[0] > right_pixel[0]  # Red component decreases
        assert left_pixel[2] < right_pixel[2]  # Blue component increases

    def test_vertical_gradient(self):
        """Test creating vertical gradient."""
        img = create_gradient_image(
            256, 256,
            start_color=(255, 255, 0),    # Yellow
            end_color=(0, 0, 0),          # Black
            direction="vertical"
        )
        assert img.size == (256, 256)
        # Top should be brighter
        top_pixel = img.getpixel((128, 0))
        # Bottom should be darker
        bottom_pixel = img.getpixel((128, 255))
        top_brightness = sum(top_pixel) / 3
        bottom_brightness = sum(bottom_pixel) / 3
        assert top_brightness > bottom_brightness

    def test_diagonal_gradient(self):
        """Test creating diagonal gradient."""
        img = create_gradient_image(
            256, 256,
            start_color=(255, 255, 255),  # White
            end_color=(0, 0, 0),          # Black
            direction="diagonal"
        )
        assert img.size == (256, 256)
        # Top-left should be brighter
        tl = img.getpixel((0, 0))
        # Bottom-right should be darker
        br = img.getpixel((255, 255))
        assert sum(tl) > sum(br)

    def test_invalid_direction(self):
        """Test error handling for invalid direction."""
        with pytest.raises(ValueError, match="direction"):
            create_gradient_image(
                256, 256,
                start_color=(255, 0, 0),
                end_color=(0, 0, 255),
                direction="invalid"
            )

    def test_invalid_color_values(self):
        """Test error handling for invalid color values."""
        with pytest.raises(ValueError, match="0-255"):
            create_gradient_image(
                256, 256,
                start_color=(256, 0, 0),   # Invalid
                end_color=(0, 0, 255)
            )

    def test_invalid_dimensions(self):
        """Test error handling for invalid dimensions."""
        with pytest.raises(ValueError, match="must be positive"):
            create_gradient_image(
                -256, 256,
                start_color=(255, 0, 0),
                end_color=(0, 0, 255)
            )

    def test_gradient_smoothness(self):
        """Test that gradient transitions are smooth."""
        img = create_gradient_image(
            256, 256,
            start_color=(0, 0, 0),
            end_color=(255, 255, 255),
            direction="horizontal"
        )
        # Sample pixels and verify gradual transition
        pixels = []
        for x in range(0, 256, 16):
            pixel = img.getpixel((x, 128))
            pixels.append(sum(pixel) / 3)  # Average brightness
        # Should be monotonically increasing
        for i in range(len(pixels) - 1):
            assert pixels[i] <= pixels[i + 1]
