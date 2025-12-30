# Gemini Watermark Remover (Python)

A Python implementation of the reverse alpha blending algorithm for removing Gemini AI watermarks from generated images.

Based on the JavaScript implementation: [journey-ad/gemini-watermark-remover](https://github.com/journey-ad/gemini-watermark-remover)

## Example

| Input (with watermark) | Output (watermark removed) |
|:----------------------:|:--------------------------:|
| ![Input](examples/input.png) | ![Output](examples/output.png) |

## How It Works

Gemini applies watermarks using standard alpha compositing:

```
watermarked = α × logo + (1 - α) × original
```

This tool reverses the equation to recover the original:

```
original = (watermarked - α × logo) / (1 - α)
```

The alpha map is pre-computed from captured watermark images, allowing mathematically precise restoration with zero loss (except for 8-bit quantization).

## Installation

```bash
pip install pillow numpy
```

Or with uv:

```bash
uv pip install pillow numpy
```

## Usage

### Command Line

```bash
python gemini_watermark_remover.py image_with_watermark.png

# Specify output path
python gemini_watermark_remover.py input.png output.png
```

### As a Library

```python
from gemini_watermark_remover import remove_watermark, remove_watermark_bytes

# From file path
result = remove_watermark("image.png")
result.save("clean.png")

# From PIL Image
from PIL import Image
img = Image.open("image.png")
result = remove_watermark(img)

# From bytes (useful for web apps)
with open("image.png", "rb") as f:
    image_bytes = f.read()
clean_bytes = remove_watermark_bytes(image_bytes, output_format="PNG")
```

## Watermark Detection

The tool automatically detects watermark size based on image dimensions:

| Image Size | Logo Size | Margin |
|------------|-----------|--------|
| ≤1024px (either dimension) | 48×48 | 32px |
| >1024px (both dimensions) | 96×96 | 64px |

## Limitations

- Only removes the visible watermark in the bottom-right corner
- Does **not** remove SynthID (invisible watermarks embedded during generation)
- Works best on uncompressed images (PNG); heavy JPEG compression may reduce quality

## License

MIT

## Credits

- Original algorithm: [GeminiWatermarkTool](https://github.com/allenk/GeminiWatermarkTool) (C++)
- JavaScript implementation: [journey-ad/gemini-watermark-remover](https://github.com/journey-ad/gemini-watermark-remover)

---

Originally made for [mollyn.org/coffee](https://mollyn.org/coffee)

Made with Claude Code
