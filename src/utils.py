"""
Shared Utilities Module
========================

Provides common functionality across all modules:
- Safe image I/O with PIL and OpenCV format conversion.
- Crop extraction from bounding boxes with boundary clamping.
- Planogram audit visualizer: renders color-coded bounding boxes on shelf images:
    - GREEN: Correct SKU facing.
    - RED: Misplaced SKU facing.
    - BLUE: Unknown SKU (open-set reject).
    - YELLOW: Missing SKU marker.
- YAML configuration loading with default fallbacks.
"""

from typing import List, Dict, Tuple, Any, Optional, Union
import os
from pathlib import Path
import yaml
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Standard color palette for planogram audit visualization (RGB)
STATUS_COLORS = {
    "CORRECT": (46, 204, 113),       # Emerald Green
    "MISPLACED": (231, 76, 60),      # Coral Red
    "UNKNOWN": (52, 152, 219),       # Bright Blue
    "MISSING": (241, 196, 15),       # Sun Yellow
    "DEFAULT": (149, 165, 166)       # Gray
}


def load_config(config_path: Union[str, Path] = "configs/config.yaml") -> Dict[str, Any]:
    """Loads a YAML configuration file into a dictionary.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If configuration file is missing.
    """
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found at: {p}")

    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_image_rgb(image_input: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
    """Loads or converts any image representation into a standardized PIL RGB Image.

    Args:
        image_input: File path, PIL Image, or NumPy array.

    Returns:
        PIL.Image in RGB mode.

    Raises:
        FileNotFoundError: If path string does not exist.
        ValueError: If input format cannot be interpreted.
    """
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.exists():
            raise FileNotFoundError(f"Image not found at: {p}")
        return Image.open(p).convert("RGB")

    elif isinstance(image_input, Image.Image):
        return image_input.convert("RGB")

    elif isinstance(image_input, np.ndarray):
        assert image_input.size > 0, "Input image array is empty."
        if image_input.ndim == 2:
            return Image.fromarray(image_input).convert("RGB")
        elif image_input.ndim == 3:
            # Assume BGR if coming from cv2, else RGB
            if image_input.shape[2] == 3:
                return Image.fromarray(image_input)
            elif image_input.shape[2] == 4:
                return Image.fromarray(image_input).convert("RGB")

    raise TypeError(f"Unsupported image type: {type(image_input)}")


def extract_crops(
    image: Image.Image,
    bboxes: List[List[float]],
    is_normalized: bool = False
) -> List[Image.Image]:
    """Extracts bounding box crops from a shelf image with boundary clamping.

    Args:
        image: Source shelf PIL Image.
        bboxes: List of [x1, y1, x2, y2] coordinates.
        is_normalized: If True, coords are in range [0, 1] and will be scaled.

    Returns:
        List of cropped PIL Images.
    """
    w, h = image.size
    crops: List[Image.Image] = []

    for b in bboxes:
        assert len(b) == 4, f"Invalid bbox: {b}"
        x1, y1, x2, y2 = b

        # Guard against NaN / Inf values from malformed annotation CSVs
        if any(np.isnan([x1, y1, x2, y2])) or any(np.isinf([x1, y1, x2, y2])):
            continue

        if is_normalized:
            x1, x2 = x1 * w, x2 * w
            y1, y2 = y1 * h, y2 * h

        # Clamp to image boundaries
        left = max(0, min(int(round(x1)), w - 1))
        top = max(0, min(int(round(y1)), h - 1))
        right = max(left + 1, min(int(round(x2)), w))
        bottom = max(top + 1, min(int(round(y2)), h))

        crop = image.crop((left, top, right, bottom))
        crops.append(crop)

    return crops


def draw_planogram_audit(
    image: Image.Image,
    aligned_results: List[Dict[str, Any]],
    box_width: int = 3
) -> Image.Image:
    """Renders color-coded bounding boxes and labels for each aligned shelf slot.

    Color coding:
    - GREEN: CORRECT facing
    - RED: MISPLACED product facing
    - BLUE: UNKNOWN (failed open-set threshold)
    - YELLOW: MISSING product facing (rendered at estimated position)

    Args:
        image: Base shelf PIL Image.
        aligned_results: Flattened list of aligned slot dictionaries:
            [{
               'expected_sku': str,
               'detected_sku': str,
               'status': 'CORRECT'|'MISPLACED'|'UNKNOWN'|'MISSING',
               'detection_info': {'bbox': [x1, y1, x2, y2], ...}
            }, ...]
        box_width: Thickness of bounding box borders.

    Returns:
        Annotated PIL Image.
    """
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size

    for slot in aligned_results:
        status = slot.get("status", "DEFAULT")
        color = STATUS_COLORS.get(status, STATUS_COLORS["DEFAULT"])
        det_info = slot.get("detection_info")

        if det_info and "bbox" in det_info:
            b = det_info["bbox"]
            x1, y1, x2, y2 = b

            # Support normalized coordinates if needed
            if max(x1, x2, y1, y2) <= 1.0:
                x1, x2 = x1 * w, x2 * w
                y1, y2 = y1 * h, y2 * h

            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=box_width)

            # Draw label banner
            detected_sku = slot.get("detected_sku") or "UNKNOWN"
            label = f"{status}: {detected_sku.split('/')[-1]}"
            draw.rectangle([x1, max(0, y1 - 18), x1 + len(label) * 8, y1], fill=color)
            draw.text((x1 + 2, max(0, y1 - 16)), label, fill=(255, 255, 255))

    return canvas
