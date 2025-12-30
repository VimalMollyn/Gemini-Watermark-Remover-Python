#!/usr/bin/env python3
"""
Gemini Watermark Remover - Python Implementation

A Python port of the reverse alpha blending algorithm for removing
Gemini AI watermarks from generated images.

Based on: https://github.com/journey-ad/gemini-watermark-remover

Usage:
    python gemini_watermark_remover.py <image_path> [output_path]

Requirements:
    pip install pillow numpy
"""

import sys
from pathlib import Path
from PIL import Image
import numpy as np
from io import BytesIO

# Assets directory (same folder as this script)
ASSETS_DIR = Path(__file__).parent


def load_alpha_map(size):
    """Load and calculate alpha map from watermark capture image."""
    bg_path = ASSETS_DIR / f"bg_{size}.png"
    if not bg_path.exists():
        raise FileNotFoundError(f"Alpha map file not found: {bg_path}")

    bg_img = Image.open(bg_path).convert('RGB')
    bg_array = np.array(bg_img, dtype=np.float32)

    # Take max of RGB channels and normalize to [0, 1]
    alpha_map = np.max(bg_array, axis=2) / 255.0
    return alpha_map


# Pre-computed alpha maps (lazy loaded)
_ALPHA_MAPS = {}


def get_alpha_map(size):
    """Get cached alpha map for given size."""
    if size not in _ALPHA_MAPS:
        _ALPHA_MAPS[size] = load_alpha_map(size)
    return _ALPHA_MAPS[size]


def detect_watermark_config(width, height):
    """
    Detect watermark configuration based on image dimensions.

    Gemini's watermark rules:
    - If both width and height > 1024: 96x96 logo, 64px margin
    - Otherwise: 48x48 logo, 32px margin
    """
    if width > 1024 and height > 1024:
        return {"logo_size": 96, "margin": 64}
    else:
        return {"logo_size": 48, "margin": 32}


def remove_watermark(image, verbose=False):
    """
    Remove Gemini watermark from image using reverse alpha blending.

    The algorithm reverses Gemini's watermark application:
        watermarked = alpha * logo + (1 - alpha) * original
    To recover:
        original = (watermarked - alpha * logo) / (1 - alpha)

    Args:
        image: PIL Image, file path, or bytes
        verbose: Print debug information

    Returns:
        PIL Image with watermark removed
    """
    # Handle different input types
    if isinstance(image, (str, Path)):
        img = Image.open(image).convert('RGB')
    elif isinstance(image, bytes):
        img = Image.open(BytesIO(image)).convert('RGB')
    elif isinstance(image, Image.Image):
        img = image.convert('RGB')
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    width, height = img.size
    config = detect_watermark_config(width, height)
    logo_size = config["logo_size"]
    margin = config["margin"]

    if verbose:
        print(f"Image size: {width}x{height}")
        print(f"Watermark config: {logo_size}x{logo_size}, margin={margin}px")

    # Calculate watermark position (bottom-right corner)
    x = width - margin - logo_size
    y = height - margin - logo_size

    if x < 0 or y < 0:
        if verbose:
            print("Image too small for watermark, returning unchanged")
        return img

    if verbose:
        print(f"Watermark position: ({x}, {y})")

    # Load alpha map
    alpha_map = get_alpha_map(logo_size)

    # Convert to numpy for processing
    img_array = np.array(img, dtype=np.float32)

    # Constants
    ALPHA_THRESHOLD = 0.002  # Ignore noise
    MAX_ALPHA = 0.99         # Avoid division by zero
    LOGO_VALUE = 255.0       # White watermark

    pixels_modified = 0

    # Process watermark region
    for row in range(logo_size):
        for col in range(logo_size):
            alpha = alpha_map[row, col]

            # Skip very small alpha values (noise)
            if alpha < ALPHA_THRESHOLD:
                continue

            # Limit alpha to avoid division by near-zero
            alpha = min(alpha, MAX_ALPHA)
            one_minus_alpha = 1.0 - alpha

            # Apply reverse alpha blending to each RGB channel
            for c in range(3):
                watermarked = img_array[y + row, x + col, c]
                original = (watermarked - alpha * LOGO_VALUE) / one_minus_alpha
                img_array[y + row, x + col, c] = max(0, min(255, round(original)))

            pixels_modified += 1

    if verbose:
        print(f"Pixels modified: {pixels_modified}")

    # Convert back to PIL Image
    return Image.fromarray(img_array.astype(np.uint8), 'RGB')


def remove_watermark_bytes(image_bytes, output_format='PNG', quality=95):
    """
    Remove watermark and return as bytes.

    Args:
        image_bytes: Input image as bytes
        output_format: Output format (PNG, JPEG, etc.)
        quality: JPEG quality (1-100)

    Returns:
        Processed image as bytes
    """
    result = remove_watermark(image_bytes)
    output = BytesIO()

    save_kwargs = {"format": output_format}
    if output_format.upper() == "JPEG":
        save_kwargs["quality"] = quality
        save_kwargs["optimize"] = True

    result.save(output, **save_kwargs)
    return output.getvalue()


def main():
    """Command-line interface."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    image_path = Path(sys.argv[1])

    if len(sys.argv) > 2:
        output_path = Path(sys.argv[2])
    else:
        output_path = image_path.parent / f"{image_path.stem}_clean{image_path.suffix}"

    if not image_path.exists():
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    print(f"Processing: {image_path}")

    result = remove_watermark(image_path, verbose=True)
    result.save(output_path)

    print(f"Saved to: {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
