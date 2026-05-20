import json
import os
import re
import uuid

from PIL import Image

# Raise PIL's decompression-bomb guard to 200 MP.
# Default is ~89 MP which is too low for large A0 engineering drawings.
Image.MAX_IMAGE_PIXELS = 200_000_000


# -----------------------------------------------------------------------------
# SLICE FUNCTIONS
# -----------------------------------------------------------------------------

def slice_image_horizontally(image_path, num_slices=8):
    """Split image into horizontal strips (top to bottom)."""
    img = Image.open(image_path)
    width, height = img.size
    num_slices = max(1, int(num_slices))
    slice_height = height // num_slices
    slice_paths = []
    for i in range(num_slices):
        top    = i * slice_height
        bottom = (i + 1) * slice_height if i < num_slices - 1 else height
        cropped = img.crop((0, top, width, bottom))
        temp_name = f"temp_slice_{uuid.uuid4().hex}.png"
        cropped.save(temp_name)
        slice_paths.append(temp_name)
    return slice_paths


def slice_image_vertically(image_path, num_slices=6):
    """Split image into vertical strips (left to right)."""
    img = Image.open(image_path)
    width, height = img.size
    num_slices = max(1, int(num_slices))
    slice_width = width // num_slices
    slice_paths = []
    for i in range(num_slices):
        left  = i * slice_width
        right = (i + 1) * slice_width if i < num_slices - 1 else width
        cropped = img.crop((left, 0, right, height))
        temp_name = f"temp_slice_{uuid.uuid4().hex}.png"
        cropped.save(temp_name)
        slice_paths.append(temp_name)
    return slice_paths


def delete_temp_slices(slice_paths):
    for path in slice_paths:
        if os.path.exists(path):
            os.remove(path)


# -----------------------------------------------------------------------------
# REGION DETECTION  (find the beam schedule area before slicing)
# -----------------------------------------------------------------------------

REGION_AND_SLICE_PROMPT = (
    "You are analysing a structural engineering drawing image. Your task is to\n"
    "locate the BEAM SCHEDULE (also called reinforcement schedule or beam\n"
    "reinforcement table) and decide how to slice it for data extraction.\n"
    "\n"
    "=== WHAT IS A BEAM SCHEDULE? ===\n"
    "A beam schedule is a DATA TABLE with:\n"
    "- A header row containing column names such as:\n"
    "    Beam Mark, Beam ID, Size (Width x Depth), Top Steel, Bottom Steel,\n"
    "    Stirrups, Spacing, No. of Bars, Dia, Reinforcement, etc.\n"
    "- Multiple DATA ROWS below the header, each representing one beam element\n"
    "  (e.g. B1, B2, FB1, GB1, RB-101, etc.)\n"
    "- Cells containing numbers, bar diameters (e.g. 12T16, 2-16mm),\n"
    "  spacing values (e.g. @150c/c), and size dimensions (e.g. 300x600)\n"
    "\n"
    "=== WHAT IS NOT A BEAM SCHEDULE? ===\n"
    "Do NOT select any of these -- they look like grids but are NOT beam schedules:\n"
    "- TITLE BLOCK: project name, drawing number, revision, date, architect /\n"
    "  client / consultant names, GOOD FOR CONSTRUCTION stamp, drawn/checked fields.\n"
    "  Usually in the bottom-right corner.\n"
    "- NOTES / GENERAL NOTES: numbered list of text instructions such as\n"
    "  '1) ALL DIMENSIONS ARE IN MM', '2) GRADE OF CONCRETE - M35'.\n"
    "  Paragraphs of text, not a data table.\n"
    "- LEGEND / SYMBOL TABLE: line types, hatch patterns, or symbols with descriptions.\n"
    "- FLOOR PLAN / STRUCTURAL PLAN: large drawing showing columns, beams, walls\n"
    "  laid out as a plan view of the building.\n"
    "- DETAIL DRAWINGS: cross-section sketches of individual beams or columns.\n"
    "\n"
    "=== YOUR TWO-PART TASK ===\n"
    "\n"
    "1. TABLE REGION -- Find the beam schedule bounding box as fractions of total\n"
    "   image width/height (0.0 = left/top edge, 1.0 = right/bottom edge).\n"
    "   If NO beam schedule exists on this page, return the full extent:\n"
    '   {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}\n'
    "\n"
    "2. SLICE STRATEGY -- How should ONLY that table region be sliced?\n"
    '   direction : "horizontal" -- single table, rows run top-to-bottom.\n'
    '               "vertical"   -- multiple side-by-side column groups (rare, wide A0 sheets).\n'
    "   num_slices: integer 1-10. Use 1 for small tables, 3-6 for large dense ones.\n"
    "\n"
    "Return ONLY valid JSON in this EXACT format -- no explanation, no markdown:\n"
    "{\n"
    '  "region":     {"x1": 0.65, "y1": 0.05, "x2": 1.0,  "y2": 0.55},\n'
    '  "direction":  "horizontal",\n'
    '  "num_slices": 4\n'
    "}\n"
)


def parse_region_and_slice_response(response_text):
    """
    Parse model response into (region_dict, direction, num_slices).
    Falls back to full-image region + horizontal on any parse failure.
    """
    FULL = {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}

    if not response_text:
        return FULL, "horizontal", None

    text = str(response_text).strip()

    for attempt in [text, None]:
        if attempt is None:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                break
            attempt = match.group()
        try:
            parsed = json.loads(attempt)

            raw_region = parsed.get("region", {})
            region = {
                "x1": float(raw_region.get("x1", 0.0)),
                "y1": float(raw_region.get("y1", 0.0)),
                "x2": float(raw_region.get("x2", 1.0)),
                "y2": float(raw_region.get("y2", 1.0)),
            }
            for k in region:
                region[k] = max(0.0, min(1.0, region[k]))
            if region["x2"] <= region["x1"] or region["y2"] <= region["y1"]:
                region = FULL

            direction  = str(parsed.get("direction", "horizontal")).lower()
            num_slices = parsed.get("num_slices")
            if direction not in ("horizontal", "vertical"):
                direction = "horizontal"

            return region, direction, num_slices

        except Exception:
            continue

    return FULL, "horizontal", None


def crop_to_region(image_path, region):
    """
    Crop image to the bounding box described by region dict
    (keys x1, y1, x2, y2 as fractions of image size).

    If the region is effectively the full image (within 1% margin on all sides)
    no crop is performed and the original path is returned as-is.

    Returns (cropped_path, is_temp).
    is_temp=True means caller must delete the file when done.
    """
    rx1, ry1, rx2, ry2 = region["x1"], region["y1"], region["x2"], region["y2"]
    is_full = (rx1 < 0.01 and ry1 < 0.01 and rx2 > 0.99 and ry2 > 0.99)

    if is_full:
        return image_path, False

    with Image.open(image_path) as img:
        w, h = img.size
        x1 = int(rx1 * w)
        y1 = int(ry1 * h)
        x2 = int(rx2 * w)
        y2 = int(ry2 * h)

        # 2% padding so table borders are not clipped
        pad_x = max(10, int(0.02 * w))
        pad_y = max(10, int(0.02 * h))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        cropped = img.crop((x1, y1, x2, y2))
        temp_name = f"temp_region_{uuid.uuid4().hex}.png"
        cropped.save(temp_name)

    print(f"  Region detected: ({rx1:.2f},{ry1:.2f})-({rx2:.2f},{ry2:.2f}) "
          f"-> cropped to {x2-x1}x{y2-y1}px (was {w}x{h})")
    return temp_name, True


def detect_region_and_strategy(image_path, suggest_fn,
                                fallback_direction="horizontal",
                                fallback_slices=None,
                                min_slices=1, max_slices=10):
    """
    Single model call: ask where the beam schedule table is AND how to slice it.
    Returns (region_dict, direction, num_slices).
    Falls back gracefully on any error.
    """
    fb_slices = (
        estimate_slice_count_by_size(image_path, min_slices=min_slices, max_slices=max_slices)
        if fallback_slices is None
        else clamp_slice_count(fallback_slices, min_slices, max_slices)
    )
    FULL = {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}

    try:
        response = suggest_fn(image_path, REGION_AND_SLICE_PROMPT)
        region, direction, num_slices = parse_region_and_slice_response(response)

        if num_slices is None:
            num_slices = fb_slices
        else:
            num_slices = clamp_slice_count(num_slices, min_slices, max_slices)

        print(f"  Model region: ({region['x1']:.2f},{region['y1']:.2f})-"
              f"({region['x2']:.2f},{region['y2']:.2f})  "
              f"slice: {direction} x{num_slices}")
        return region, direction, num_slices

    except Exception as exc:
        print(f"  Region detection failed ({exc}), "
              f"using full image / {fallback_direction} x{fb_slices}")
        return FULL, fallback_direction, fb_slices


# -----------------------------------------------------------------------------
# MODEL-DRIVEN SLICE STRATEGY  (used when detect_region=False)
# -----------------------------------------------------------------------------

SLICE_STRATEGY_PROMPT = (
    "You are deciding how to split a structural engineering drawing image into\n"
    "slices before sending it to a table extraction model.\n"
    "\n"
    "Look at the image and return ONLY valid JSON in this exact format:\n"
    '{"direction": "horizontal", "num_slices": 6}\n'
    "\n"
    "Rules for DIRECTION:\n"
    '- "horizontal" : one table spans the full width, rows run top-to-bottom.\n'
    '- "vertical"   : multiple independent tables sit side-by-side (wide A0/A1).\n'
    "\n"
    "Rules for NUM_SLICES: integer 1-10, prefer fewer when unsure.\n"
    "\n"
    "Return ONLY JSON. No explanation. No markdown.\n"
)


def parse_slice_strategy_response(response_text):
    """Parse model response into (direction, num_slices)."""
    if not response_text:
        return "horizontal", None

    text = str(response_text).strip()

    for attempt in [text, None]:
        if attempt is None:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                break
            attempt = match.group()
        try:
            parsed = json.loads(attempt)
            direction  = str(parsed.get("direction", "horizontal")).lower()
            num_slices = parsed.get("num_slices")
            if direction not in ("horizontal", "vertical"):
                direction = "horizontal"
            return direction, num_slices
        except Exception:
            continue

    match = re.search(r"\b([1-9]|10)\b", text)
    if match:
        return "horizontal", int(match.group(1))

    return "horizontal", None


def get_dynamic_slice_strategy(image_path, suggest_fn,
                                fallback_direction="horizontal",
                                fallback_slices=None,
                                min_slices=1, max_slices=10):
    """Ask model for direction + num_slices. Returns (direction, num_slices)."""
    if fallback_slices is None:
        fallback_slices = estimate_slice_count_by_size(
            image_path, min_slices=min_slices, max_slices=max_slices)
    else:
        fallback_slices = clamp_slice_count(fallback_slices, min_slices, max_slices)

    try:
        response = suggest_fn(image_path, SLICE_STRATEGY_PROMPT)
        direction, num_slices = parse_slice_strategy_response(response)
        if num_slices is None:
            num_slices = fallback_slices
        else:
            num_slices = clamp_slice_count(num_slices, min_slices, max_slices)
        print(f"  Model chose: {direction} x{num_slices} slices")
        return direction, num_slices
    except Exception as exc:
        print(f"  Slice strategy failed ({exc}), using {fallback_direction} x{fallback_slices}")
        return fallback_direction, fallback_slices


# -----------------------------------------------------------------------------
# SMART SLICE  (main entry point used by pipeline.py)
# -----------------------------------------------------------------------------

def smart_slice(image_path, suggest_fn, fallback_direction="horizontal",
                fallback_slices=None, min_slices=1, max_slices=10,
                max_strip_height=1800, detect_region=True):
    """
    Optionally detect the beam schedule region, crop to it, then slice.

    When detect_region=True (default):
        ONE model call returns BOTH the table bounding box (as % of image)
        AND the slice strategy. The image is cropped to that region first
        so irrelevant areas (floor plans, title blocks, notes) are excluded.

    When detect_region=False (legacy):
        Asks only for slice strategy; no region crop is performed.

    Slicing (applied to the work image after optional crop):
        Horizontal -- N strips top-to-bottom.
        Vertical   -- Stage 1: N vertical strips.
                      Stage 2: any strip taller than max_strip_height is
                               further sliced horizontally.

    Returns flat list of temporary slice paths.
    Caller must call delete_temp_slices() on the returned list.
    """
    region_temp = None

    if detect_region:
        region, direction, num_slices = detect_region_and_strategy(
            image_path, suggest_fn=suggest_fn,
            fallback_direction=fallback_direction,
            fallback_slices=fallback_slices,
            min_slices=min_slices, max_slices=max_slices,
        )
        work_image, is_temp = crop_to_region(image_path, region)
        if is_temp:
            region_temp = work_image
    else:
        direction, num_slices = get_dynamic_slice_strategy(
            image_path, suggest_fn=suggest_fn,
            fallback_direction=fallback_direction,
            fallback_slices=fallback_slices,
            min_slices=min_slices, max_slices=max_slices,
        )
        work_image = image_path

    if num_slices is None:
        num_slices = estimate_slice_count_by_size(
            work_image, min_slices=min_slices, max_slices=max_slices)

    # Horizontal: single pass
    if direction != "vertical":
        slices = slice_image_horizontally(work_image, num_slices=num_slices)
        if region_temp:
            os.remove(region_temp)
        return slices

    # Stage 1: vertical strips
    vertical_strips = slice_image_vertically(work_image, num_slices=num_slices)
    if region_temp:
        os.remove(region_temp)

    # Stage 2: horizontal sub-slicing for tall strips
    final_slices = []
    for strip_path in vertical_strips:
        with Image.open(strip_path) as strip_img:
            _, strip_height = strip_img.size

        if strip_height <= max_strip_height:
            final_slices.append(strip_path)
        else:
            h_count = estimate_slice_count_by_size(
                strip_path, target_slice_height=max_strip_height,
                min_slices=2, max_slices=10)
            print(f"    Strip {os.path.basename(strip_path)} is {strip_height}px tall "
                  f"-> sub-slicing horizontally x{h_count}")
            h_slices = slice_image_horizontally(strip_path, num_slices=h_count)
            final_slices.extend(h_slices)
            if os.path.exists(strip_path):
                os.remove(strip_path)

    return final_slices


# -----------------------------------------------------------------------------
# LEGACY HELPERS  (kept for backward-compatibility)
# -----------------------------------------------------------------------------

SLICE_COUNT_PROMPT = (
    "You are deciding how to split a structural drawing image into horizontal\n"
    "slices before table extraction.\n"
    "\n"
    "Return ONLY valid JSON in this exact format:\n"
    '{"num_slices": 6}\n'
    "\n"
    "Rules:\n"
    "- Choose an integer from 1 to 10.\n"
    "- Use 1 only if the page is small or the table text is already large.\n"
    "- Use 2 to 4 for normal readable table pages.\n"
    "- Use 5 to 8 for dense schedules, very tall pages, or tiny table text.\n"
    "- Use 9 to 10 only for extremely dense pages where text would be unreadable.\n"
    "- Prefer fewer slices when unsure.\n"
    "- Do not include explanation or markdown.\n"
)


def estimate_slice_count_by_size(image_path, target_slice_height=1400,
                                  min_slices=1, max_slices=10):
    """Deterministic fallback. Keeps each slice near target_slice_height pixels."""
    with Image.open(image_path) as img:
        _, height = img.size
    estimated = round(height / target_slice_height)
    return clamp_slice_count(estimated, min_slices, max_slices)


def clamp_slice_count(value, min_slices=1, max_slices=10):
    try:
        value = int(value)
    except Exception:
        value = min_slices
    return max(min_slices, min(max_slices, value))


def parse_slice_count_response(response_text):
    """Accepts strict JSON, JSON embedded in text, or a plain number."""
    if not response_text:
        return None
    text = str(response_text).strip()
    try:
        return json.loads(text).get("num_slices")
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group()).get("num_slices")
        except Exception:
            pass
    match = re.search(r"\b([1-9]|10)\b", text)
    if match:
        return int(match.group(1))
    return None


def get_dynamic_slice_count(image_path, suggest_fn=None,
                             fallback_slices=None, min_slices=1, max_slices=10):
    """Legacy: returns only a slice count (always horizontal)."""
    if fallback_slices is None:
        fallback_slices = estimate_slice_count_by_size(
            image_path, min_slices=min_slices, max_slices=max_slices)
    else:
        fallback_slices = clamp_slice_count(fallback_slices, min_slices, max_slices)

    if suggest_fn is None:
        return fallback_slices

    try:
        response  = suggest_fn(image_path, SLICE_COUNT_PROMPT)
        suggested = parse_slice_count_response(response)
        if suggested is None:
            return fallback_slices
        return clamp_slice_count(suggested, min_slices, max_slices)
    except Exception as exc:
        print("Dynamic slice-count suggestion failed:", exc)
        return fallback_slices
