"""Tests for format conversion functions."""


from visprep import b64_to_cv2, b64_to_pil, cv2_to_b64, cv2_to_pil, pil_to_b64, pil_to_cv2

import pytest
from PIL import Image
import numpy as np
class TestBase64Conversion:
    """Test suite for base64 conversion functions."""

    def test_pil_to_b64_with_data_uri(self, pil_image):
        """Test converting PIL Image to base64 with data URI prefix."""
        b64 = pil_to_b64(pil_image, include_data_uri=True)
        assert isinstance(b64, str)
        assert b64.startswith("data:image/")
        assert "base64," in b64

    def test_pil_to_b64_without_data_uri(self, pil_image):
        """Test converting PIL Image to base64 without prefix."""
        b64 = pil_to_b64(pil_image, include_data_uri=False)
        assert isinstance(b64, str)
        assert not b64.startswith("data:")

    def test_b64_to_pil_with_data_uri(self, base64_data_uri):
        """Test converting base64 with data URI to PIL Image."""
        img = b64_to_pil(base64_data_uri)
        assert isinstance(img, Image.Image)

    def test_b64_to_pil_raw_base64(self, base64_string):
        """Test converting raw base64 to PIL Image."""
        img = b64_to_pil(base64_string)
        assert isinstance(img, Image.Image)

    def test_pil_to_b64_different_formats(self, pil_image):
        """Test converting PIL to base64 with different formats."""
        b64_jpeg = pil_to_b64(pil_image, image_format="JPEG")
        b64_png = pil_to_b64(pil_image, image_format="PNG")
        # Both should be valid base64
        assert len(b64_jpeg) > 0
        assert len(b64_png) > 0
        # PNG typically larger than JPEG
        assert len(b64_png) > len(b64_jpeg)

    def test_pil_to_b64_quality_parameter(self, pil_image):
        """Test that quality parameter affects file size."""
        b64_high = pil_to_b64(
            pil_image,
            image_format="JPEG",
            quality=95
        )
        b64_low = pil_to_b64(
            pil_image,
            image_format="JPEG",
            quality=10
        )
        assert len(b64_low) < len(b64_high)

    def test_b64_roundtrip(self, pil_image):
        """Test roundtrip conversion: PIL -> base64 -> PIL."""
        b64 = pil_to_b64(pil_image)
        recovered_img = b64_to_pil(b64)
        assert isinstance(recovered_img, Image.Image)
        assert recovered_img.size == pil_image.size

    def test_invalid_base64(self):
        """Test error handling for invalid base64 string."""
        with pytest.raises(ValueError, match="Invalid base64"):
            b64_to_pil("invalid@#$%^&*()base64")

    def test_invalid_pil_object(self):
        """Test error handling for invalid PIL object."""
        with pytest.raises(ValueError, match="PIL Image"):
            pil_to_b64("not an image")

    def test_invalid_format(self, pil_image):
        """Test error handling for invalid format."""
        with pytest.raises(ValueError, match="JPEG.*PNG.*WEBP"):
            pil_to_b64(pil_image, image_format="INVALID")


class TestCv2Conversion:
    """Test suite for OpenCV conversion functions."""

    def test_pil_to_cv2_rgb(self, pil_image):
        """Test converting RGB PIL Image to OpenCV."""
        cv_img = pil_to_cv2(pil_image)
        assert isinstance(cv_img, np.ndarray)
        assert cv_img.shape == (256, 256, 3)

    def test_cv2_to_pil_rgb(self, cv2_image):
        """Test converting RGB OpenCV array to PIL."""
        pil_img = cv2_to_pil(cv2_image)
        assert isinstance(pil_img, Image.Image)
        assert pil_img.mode == "RGB"

    def test_cv2_to_pil_grayscale(self):
        """Test converting grayscale OpenCV array to PIL."""
        cv_gray = np.ones((256, 256), dtype=np.uint8) * 128
        pil_img = cv2_to_pil(cv_gray)
        assert isinstance(pil_img, Image.Image)
        assert pil_img.mode == "L"

    def test_pil_to_cv2_grayscale(self, sample_grayscale_image_path):
        """Test converting grayscale PIL to OpenCV."""
        img = Image.open(sample_grayscale_image_path)
        cv_img = pil_to_cv2(img)
        assert cv_img.ndim == 2

    def test_cv2_bgra_conversion(self):
        """Test converting BGRA OpenCV array to PIL."""
        bgra_array = np.zeros((256, 256, 4), dtype=np.uint8)
        bgra_array[:, :] = [50, 100, 200, 255]  # BGRA
        pil_img = cv2_to_pil(bgra_array)
        assert pil_img.mode == "RGBA"

    def test_cv2_pil_roundtrip(self, cv2_image):
        """Test roundtrip conversion: OpenCV -> PIL -> OpenCV."""
        pil_img = cv2_to_pil(cv2_image)
        cv_img_recovered = pil_to_cv2(pil_img)
        assert cv_img_recovered.shape == cv2_image.shape

    def test_invalid_cv2_object(self):
        """Test error handling for invalid OpenCV object."""
        with pytest.raises(ValueError, match="numpy array"):
            cv2_to_pil("not an array")

    def test_invalid_pil_to_cv2(self):
        """Test error handling for invalid PIL object in cv2 conversion."""
        with pytest.raises(ValueError, match="PIL Image"):
            pil_to_cv2("not an image")

    def test_unsupported_cv2_channels(self):
        """Test error handling for unsupported number of channels."""
        unsupported = np.zeros((256, 256, 2), dtype=np.uint8)  # 2 channels
        with pytest.raises(ValueError, match="Unsupported"):
            cv2_to_pil(unsupported)


class TestCv2Base64Conversion:
    """Test suite for OpenCV to base64 conversion."""

    def test_cv2_to_b64(self, cv2_image):
        """Test converting OpenCV to base64."""
        b64 = cv2_to_b64(cv2_image)
        assert isinstance(b64, str)
        assert b64.startswith("data:image/")

    def test_b64_to_cv2(self, base64_data_uri):
        """Test converting base64 to OpenCV."""
        cv_img = b64_to_cv2(base64_data_uri)
        assert isinstance(cv_img, np.ndarray)
        assert cv_img.ndim == 3 or cv_img.ndim == 2

    def test_cv2_to_b64_without_data_uri(self, cv2_image):
        """Test converting OpenCV to base64 without prefix."""
        b64 = cv2_to_b64(cv2_image, include_data_uri=False)
        assert not b64.startswith("data:")

    def test_cv2_b64_roundtrip(self, cv2_image):
        """Test roundtrip: OpenCV -> base64 -> OpenCV."""
        b64 = cv2_to_b64(cv2_image)
        cv_img_recovered = b64_to_cv2(b64)
        assert cv_img_recovered.shape == cv2_image.shape

    def test_cv2_to_b64_quality_parameter(self, cv2_image):
        """Test that quality affects output size."""
        b64_high = cv2_to_b64(cv2_image, quality=95)
        b64_low = cv2_to_b64(cv2_image, quality=10)
        assert len(b64_low) < len(b64_high)

    def test_invalid_b64_to_cv2(self):
        """Test error handling for invalid base64."""
        with pytest.raises(ValueError):
            b64_to_cv2("invalid@#$%^&*()")


class TestMultiFormatChain:
    """Test suite for multi-format conversion chains."""

    def test_pil_to_cv2_to_b64(self, pil_image):
        """Test conversion chain: PIL -> OpenCV -> base64."""
        cv_img = pil_to_cv2(pil_image)
        b64 = cv2_to_b64(cv_img)
        assert isinstance(b64, str)
        assert len(b64) > 0

    def test_b64_to_pil_to_cv2(self, base64_data_uri):
        """Test conversion chain: base64 -> PIL -> OpenCV."""
        pil_img = b64_to_pil(base64_data_uri)
        cv_img = pil_to_cv2(pil_img)
        assert isinstance(cv_img, np.ndarray)

    def test_cv2_to_b64_to_pil(self, cv2_image):
        """Test conversion chain: OpenCV -> base64 -> PIL."""
        b64 = cv2_to_b64(cv2_image)
        pil_img = b64_to_pil(b64)
        assert isinstance(pil_img, Image.Image)
