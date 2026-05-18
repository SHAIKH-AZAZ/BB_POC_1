import json
import os
import re
import uuid

from PIL import Image


def slice_image_horizontally(image_path, num_slices=8):
    """
    Splits image into horizontal strips.
    Returns list of temporary slice paths.
    """

    img = Image.open(image_path)
    width, height = img.size
    num_slices = max(1, int(num_slices))

    slice_height = height // num_slices
    slice_paths = []

    for i in range(num_slices):
        top = i * slice_height
        bottom = (i + 1) * slice_height if i < num_slices - 1 else height

        cropped = img.crop((0, top, width, bottom))

        temp_name = f"temp_slice_{uuid.uuid4().hex}.png"
        cropped.save(temp_name)

        slice_paths.append(temp_name)

    return slice_paths


def delete_temp_slices(slice_paths):
    for path in slice_paths:
        if os.path.exists(path):
            os.remove(path)


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
    """
    Deterministic fallback. Keeps each slice near target_slice_height pixels.
    """
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
    """
    Accepts strict JSON, JSON embedded in text, or a plain number.
    Returns int or None.
    """
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
    """
    Returns a slice count for an image.

    If suggest_fn is provided, it must accept (image_path, prompt_text) and
    return model text. The existing vision_extractor.extract_from_image fits
    this interface. If model output is invalid, fall back to image-size logic.
    """
    if fallback_slices is None:
        fallback_slices = estimate_slice_count_by_size(
            image_path,
            min_slices=min_slices,
            max_slices=max_slices,
        )
    else:
        fallback_slices = clamp_slice_count(
            fallback_slices,
            min_slices,
            max_slices,
        )

    if suggest_fn is None:
        return fallback_slices

    try:
        response = suggest_fn(image_path, SLICE_COUNT_PROMPT)
        suggested = parse_slice_count_response(response)
        if suggested is None:
            return fallback_slices
        return clamp_slice_count(suggested, min_slices, max_slices)
    except Exception as exc:
        print("Dynamic slice-count suggestion failed:", exc)
        return fallback_slices
