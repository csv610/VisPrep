"""Command-line interface for image_utils."""

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image-utils",
        description="Image processing utilities for vision analysis pipelines.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- prepare ---
    p_prep = sub.add_parser("prepare", help="Prepare image for API submission")
    p_prep.add_argument("input", help="Path to input image")
    p_prep.add_argument("--max-size", type=float, default=20.0, help="Max file size in MB (default 20)")
    p_prep.add_argument("--format", choices=["JPEG", "PNG", "WEBP"], default="JPEG", help="Output format (default JPEG)")
    p_prep.add_argument("--quality", type=int, default=85, help="JPEG/WEBP quality 1-100 (default 85)")
    p_prep.add_argument("--max-dim", type=int, help="Max dimension for longest side")
    p_prep.add_argument("--no-autorotate", action="store_true", help="Skip auto-orientation")
    p_prep.add_argument("--no-strip-exif", action="store_true", help="Keep EXIF data")
    p_prep.add_argument("-o", "--output", help="Output file path (default: print base64 to stdout)")

    # --- info ---
    p_info = sub.add_parser("info", help="Get image metadata")
    p_info.add_argument("input", help="Path to input image")

    # --- resize ---
    p_resize = sub.add_parser("resize", help="Resize image to max dimension")
    p_resize.add_argument("input", help="Path to input image")
    p_resize.add_argument("--width", type=int, help="Target width")
    p_resize.add_argument("--height", type=int, help="Target height")
    p_resize.add_argument("--max-size", type=float, help="Max file size in MB")
    p_resize.add_argument("-o", "--output", required=True, help="Output file path")

    # --- square ---
    p_square = sub.add_parser("square", help="Create square canvas with image")
    p_square.add_argument("input", help="Path to input image")
    p_square.add_argument("--size", type=int, required=True, help="Canvas size in pixels")
    p_square.add_argument("--bg", default="255,255,255", help="Background color as R,G,B (default 255,255,255)")
    p_square.add_argument("--position", choices=["center", "top-left"], default="center")
    p_square.add_argument("-o", "--output", required=True, help="Output file path")

    # --- batch collect ---
    p_collect = sub.add_parser("collect", help="List images in directory")
    p_collect.add_argument("directory", help="Directory to scan")
    p_collect.add_argument("--recursive", action="store_true", help="Scan subdirectories")
    p_collect.add_argument("--formats", nargs="+", help="Filter by format (e.g. JPEG PNG)")
    p_collect.add_argument("--sort", choices=["name", "size", "date"], default="name")

    return parser


def main(argv: list | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "prepare":
        _cmd_prepare(args)
    elif args.command == "info":
        _cmd_info(args)
    elif args.command == "resize":
        _cmd_resize(args)
    elif args.command == "square":
        _cmd_square(args)
    elif args.command == "collect":
        _cmd_collect(args)


def _cmd_prepare(args) -> None:
    from image_utils.adapters import prepare_for_api

    b64 = prepare_for_api(
        args.input,
        max_size_mb=args.max_size,
        auto_rotate=not args.no_autorotate,
        strip_exif=not args.no_strip_exif,
        target_format=args.format,
        quality=args.quality,
        max_dimension=args.max_dim,
    )

    if args.output:
        Path(args.output).write_text(b64)
        print(f"Saved base64 to {args.output}")
    else:
        print(b64)


def _cmd_info(args) -> None:
    from image_utils.metadata import get_image_info

    info = get_image_info(args.input)
    for key, val in info.items():
        print(f"{key}: {val}")


def _cmd_resize(args) -> None:
    if args.max_size:
        from image_utils.optimize import save_image_to_max_size
        result = save_image_to_max_size(args.input, args.output, args.max_size)
        print(f"Resized to {result}")
    elif args.width and args.height:
        from image_utils.transform import resize_to_dimensions
        from image_utils.save import save_pil_image
        img = resize_to_dimensions(args.input, args.width, args.height)
        save_pil_image(img, args.output)
        print(f"Resized to {args.output}")
    else:
        print("Specify --width/--height or --max-size", file=sys.stderr)
        sys.exit(1)


def _cmd_square(args) -> None:
    from image_utils.transform import square_image
    from image_utils.save import save_pil_image

    try:
        parts = [int(c) for c in args.bg.split(",")]
    except ValueError:
        print(f"Error: --bg must be three comma-separated integers (e.g. 255,255,255), got '{args.bg}'", file=sys.stderr)
        sys.exit(1)
    if len(parts) != 3:
        print(f"Error: --bg must have exactly 3 values, got {len(parts)}", file=sys.stderr)
        sys.exit(1)
    bg = tuple(parts)
    img = square_image(args.input, args.size, bg, args.position)
    save_pil_image(img, args.output)
    print(f"Saved square image to {args.output}")


def _cmd_collect(args) -> None:
    from image_utils.collect import collect_images

    images = collect_images(
        args.directory,
        recursive=args.recursive,
        formats=args.formats,
        sort_by=args.sort,
    )
    for img in images:
        print(img)
    print(f"\nTotal: {len(images)} images")


if __name__ == "__main__":
    main()
