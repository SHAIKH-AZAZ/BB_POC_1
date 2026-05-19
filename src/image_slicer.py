import json
import os
import re
import uuid

from PIL import Image

# Raise PIL's decompression-bomb guard to 200 MP.
# Default is ~89 MP which is too low for large A0 engineering drawings.
Image.MAX_IMAGE_PIXELS = 200_000_000


# ─────────────────────────────────────────────────────────────────────────────
# SLICE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def slice_image_horizontally(image_path, num_slices=8):
    """
    Split image into horizontal strips (top to bottom).
    Best for single-table layouts where full column width must be visible.
    Returns list of temporary slice file paths.
    """
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
    """
    Split image into vertical strips (left to right).
    Best for wide multi-column-group layouts where several independent
    tables sit side-by-side on one page (e.g. Pattern 10 with 6 column groups).
    Returns list of temporary slice file paths.
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# MODEL-DRIVEN SLICE STRATEGY
# ─────────────────────────────────────────────────────────────────────────────

SLICE_STRATEGY_PROMPT = """
You are deciding how to split a structural engineering drawing image into slices
before sending it to a table extraction model.

Look at the image and return ONLY valid JSON in this exact format:
{"direction": "horizontal", "num_slices": 6}

Rules for DIRECTION:
- "horizontal"  : Use when one table spans the full width and rows run top-to-bottom.
                  Horizontal slices give each slice the complete column headers + a few rows.
                  Use this for most beam schedule drawings.
- "vertical"    : Use when MULTIPLE independent tables (or column groups) sit side-by-side
                  across a wide page. Vertical slices isolate each table group so the model
                  is not overwhelmed by too many columns at once.
                  Use this for very wide A0/A1 drawings with 4+ column groups.

Rules for NUM_SLICES:
- Choose an integer from 1 to 10.
- For horizontal: 1-3 for short tables, 4-8 for tall dense tables.
- For vertical: match the number of independent column groups visible (e.g. 6 groups = 6).
- Do not over-slice — prefer fewer slices when unsure.

Return ONLY JSON. No explanation. No markdown.
"""


def parse_slice_strategy_response(response_text):
    """
    Parse model response into (direction, num_slices).
    Falls back to ("horizontal", None) on any parse failure.
    """
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

    # plain number fallback — assume horizontal
    match = re.search(r"\b([1-9]|10)\b", text)
    if match:
        return "horizontal", int(match.group(1))

    return "horizontal", None


def get_dynamic_slice_strategy(
    image_path,
    suggest_fn,
    fallback_direction="horizontal",
    fallback_slices=None,
    min_slices=1,
    max_slices=10,
):
    """
    Ask the vision model to look at the image and decide:
      - direction  : "horizontal" or "vertical"
      - num_slices : how many slices to make

    Returns (direction, num_slices) tuple.

    suggest_fn must accept (image_path, prompt_text) and return a string
    (vision_extractor.extract_from_image satisfies this).

    Falls back to (fallback_direction, fallback_slices) on any error.
    """
    if fallback_slices is None:
        fallback_slices = estimate_slice_count_by_size(
            image_path, min_slices=min_slices, max_slices=max_slices
        )
    else:
        fallback_slices = clamp_slice_count(fallback_slices, min_slices, max_slices)

    try:
        response  = suggest_fn(image_path, SLICE_STRATEGY_PROMPT)
        direction, num_slices = parse_slice_strategy_response(response)

        if num_slices is None:
            num_slices = fallback_slices
        else:
            num_slices = clamp_slice_count(num_slices, min_slices, max_slices)

        print(f"  Model chose: {direction} x{num_slices} slices")
        return direction, num_slices

    except Exception as exc:
        print(f"  Slice strategy suggestion failed ({exc}), "
              f"using {fallback_direction} x{fallback_slices}")
        return fallback_direction, fallback_slices


def smart_slice(image_path, suggest_fn, fallback_direction="horizontal",
                fallback_slices=None, min_slices=1, max_slices=10,
                max_strip_height=1800):
    """
    High-level helper: ask the model how to slice, then do it.

    Two-stage logic when direction = "vertical":
        Stage 1 — slice page into N vertical strips (isolates column groups).
        Stage 2 — for any strip whose height exceeds max_strip_height pixels,
                  further slice it horizontally so the model sees a manageable
                  number of rows at a time.
        The result is a flat list of sub-slice paths ready for extraction.

    When direction = "horizontal" the behaviour is unchanged (single pass).

    Returns flat list of temporary slice paths.
    Caller is responsible for calling delete_temp_slices() on the returned list.
    """
    direction, num_slices = get_dynamic_slice_strategy(
        image_path,
        suggest_fn=suggest_fn,
        fallback_direction=fallback_direction,
        fallback_slices=fallback_slices,
        min_slices=min_slices,
        max_slices=max_slices,
    )

    # ── Horizontal: single pass ───────────────────────────────────────────────
    if direction != "vertical":
        return slice_image_horizontally(image_path, num_slices=num_slices)

    # ── Stage 1: vertical strips (isolate column groups) ─────────────────────
    vertical_strips = slice_image_vertically(image_path, num_slices=num_slices)

    # ── Stage 2: horizontal sub-slicing for tall strips ───────────────────────
    final_slices = []
    for strip_path in vertical_strips:
        with Image.open(strip_path) as strip_img:
            _, strip_height = strip_img.size

        if strip_height <= max_strip_height:
            # Strip is short enough — send directly to the model
            final_slices.append(strip_path)
        else:
            # Strip is too tall — sub-slice horizontally
            h_count = estimate_slice_count_by_size(
                strip_path,
                target_slice_height=max_strip_height,
                min_slices=2,
                max_slices=10,
            )
            print(f"    Strip {os.path.basename(strip_path)} is {strip_height}px tall "
                  f"-> sub-slicing horizontally x{h_count}")
            h_slices = slice_image_horizontally(strip_path, num_slices=h_count)
            final_slices.extend(h_slices)
            # The vertical strip is now an intermediate temp; delete it here
            if os.path.exists(strip_path):
                os.remove(strip_path)

    return final_slices


# ─────────────────────────────────────────────────────────────────────────────
# LEGACY HELPERS  (kept for backward-compatibility with main_1.py / main_4.py)
# ─────────────────────────────────────────────────────────────────────────────

SLICE_COUNT_PROMPT = """
You are deciding how to split a structural drawing image into horizontal slices
before table extraction.

Return ONLY valid JSON in this exact format:
{"num_slices": 6}

Rules:
- Choose an integer from 1 to 10.
- Use 1 only if the page is small or the table text is already large.
- Use 2 to 4 for normal readable table pages.
- Use 5 to 8 for dense schedules, very tall pages, or tiny table text.
- Use 9 to 10 only for extremely dense pages where text would be unreadable.
- Prefer fewer slices when unsure.
- Do not include explanation or markdown.
"""


def estimate_slice_count_by_size(
    image_path,
    target_slice_height=1400,
    min_slices=1,
    max_slices=10,
):
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
        parsed = json.loads(text)
        return parsed.get("num_slices")
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            return parsed.get("num_slices")
        except Exception:
            pass
    match = re.search(r"\b([1-9]|10)\b", text)
    if match:
        return int(match.group(1))
    return None


def get_dynamic_slice_count(
    image_path,
    suggest_fn=None,
    fallback_slices=None,
    min_slices=1,
    max_slices=10,
):
    """Legacy: returns only a slice count (always horizontal)."""
    if fallback_slices is None:
        fallback_slices = estimate_slice_count_by_size(
            image_path, min_slices=min_slices, max_slices=max_slices
        )
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
