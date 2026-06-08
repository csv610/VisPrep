# Changelog

## 0.1.0 — 2025-06-08

Initial release.  Monolithic ``ImageUtils`` class refactored into a modular
pip-installable package.

### Added
- Modular package layout under ``image_utils/`` (12 modules).
- ``prepare_for_api()`` — single-call pipeline: orient → strip EXIF → convert
  → resize → base64 for LLM vision APIs.
- CLI via ``visprep`` / ``python -m visprep`` with ``prepare``,
  ``info``, ``resize``, ``square``, ``collect`` subcommands.
- OpenCV made fully optional (``pip install "visprep[cv2]"``).
- ``WEBP`` support in ``pil_to_b64``, ``save_image``, ``convert_format``,
  ``estimate_compressed_size``.
- Function-level docstrings (Args/Returns/Raises) on all 30+ public functions.
- ``__version__``, ``pyproject.toml``, ``LICENSE`` (MIT).

### Changed
- ``remove_exif`` re-encodes JPEG in-place (``exif=b""``) instead of PNG.
- ``auto_orient`` operates in-memory without temp files.
- ``encode_to_base64`` reads file once; derives MIME from extension.
- ``create_gradient_image`` fully vectorised (numpy).
- ``MAX_IMAGE_SIZE_BYTES`` = 20 MB, ``MAX_TOTAL_IMAGE_PAYLOAD_BYTES`` = 50 MB.
- ``_FORMAT_ALIASES`` maps ``"JPG"`` → ``"JPEG"`` throughout.

### Removed
- Backward-compatible ``ImageUtils`` class.
- ``config.py`` (constants moved to ``_utils.py``).
- ``IMAGE_MIME_TYPE`` (replaced by per-extension ``_MIME_MAP``).

### Fixed
- Pillow deprecation warnings (save-to-bytes-reload pattern).
- Double file-open in ``encode_to_base64``.
- Single-pixel edge case in ``create_gradient_image``.
- Dead variable in ``resize_images_to_fit``.
- ``save_image`` auto-detection now checks ``Path.exists()`` first.
