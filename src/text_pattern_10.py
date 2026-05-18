"""
text_pattern_10.py
==================
Coordinate-based extractor for Pattern 10 beam schedules.

PATTERN 10 TABLE STRUCTURE:
    BEAM NO. | SIZE            | LEVEL
    ─────────┼─────────────────┼──────
    B1       | B_400 x 550mm   | 26.850
    BB100    | BAND_B_1800 ... | 26.850
    B19      | VD_200 X 550/.. | 27.250

KEY OBSERVATION (from ChatGPT coordinate analysis):
----------------------------------------------------
The drawing has 6 side-by-side copies of this 3-column table on a single
wide page.  Each copy repeats the same header but lists different beams.

COORDINATE APPROACH:
--------------------
1. Extract all words with (x, y) positions.
2. Filter for SIZE tokens matching  B_NNN / BAND_B_NNN / VD_NNN.
3. For each SIZE token:
   - Beam ID  = word immediately to its LEFT on the same line.
   - Width    = first number inside the SIZE token itself.
   - Depth    = largest number inside the token to the right (e.g. "550mm",
                "550/950mm" → pick 950).
4. Sort results by visual reading order: column group first, then top-to-bottom.
5. Deduplicate by beam_id.

This produces the exact same result as the ChatGPT pdfplumber analysis but
is fully integrated into the existing project pipeline.
"""

import re
import json
import os
from tqdm import tqdm

from table_extractor import (
    is_digital_pdf,
    extract_words,
    get_left_neighbour,
    get_right_neighbours,
    cluster_values,
)
from pdf_to_images import convert_pdf_to_images
from vision_extractor import extract_from_image
from config import OUTPUT_DIR


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Regex that matches the SIZE column token in Pattern 10
# Examples: B_400  BAND_B_1800  VD_200
SIZE_TOKEN_RE = re.compile(r'^(?:B_|BAND_B_|VD_)\d+', re.IGNORECASE)

# Words that look like beam IDs but are actually header/noise tokens to skip
SKIP_LABELS = {
    "NO.", "BEAM", "SIZE", "LEVEL", "TYPE", "MARK",
    "WIDTH", "DEPTH", "B", "D", "W",
}


# ─────────────────────────────────────────────────────────────────────────────
# SIZE PARSING
# ─────────────────────────────────────────────────────────────────────────────

def _parse_width(size_text):
    """
    Extract width (first number) from a size token.
        B_400       → 400
        BAND_B_1800 → 1800
        VD_200      → 200
    """
    m = re.search(r'\d+', str(size_text))
    return int(m.group()) if m else None


def _parse_depth(right_tokens):
    """
    Extract depth from the tokens immediately to the RIGHT of the size token.
    The depth appears as the second major number, e.g.:
        ['x', '550mm', '26.850']      → 550
        ['X', '550/950mm', '27.250']  → 950  (slash → take larger)
        ['x', '550mm']                → 550

    We skip small numbers (< 50) and elevation values (> 2000).
    """
    for tok in right_tokens:
        nums = [int(n) for n in re.findall(r'\d+', str(tok))]
        nums = [n for n in nums if 50 <= n <= 2000]  # plausible depth range
        if nums:
            return max(nums)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# VISION FALLBACK
# ─────────────────────────────────────────────────────────────────────────────

def _load_prompt():
    with open(os.path.join(os.path.dirname(__file__), "prompt_10.txt"), "r") as f:
        return f.read()


def _safe_json_load(text):
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", str(text), re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group())
    except Exception:
        return None


def _extract_with_vision(pdf_path, file_output_folder, file_name):
    """
    Fallback for scanned/image-only Pattern 10 PDFs.

    This path is intentionally secondary. For digital PDFs, the coordinate
    extractor is faster and avoids long model responses that can be truncated.
    """
    print(f"\nConverting {file_name}.pdf to images for vision fallback...")
    image_paths = convert_pdf_to_images(pdf_path, file_output_folder)

    prompt = _load_prompt()
    all_beams = []

    for img_path in tqdm(image_paths):
        parsed = None
        for attempt in range(2):
            result = extract_from_image(img_path, prompt)
            parsed = _safe_json_load(result)
            if parsed and parsed.get("beams"):
                break
            if attempt == 0:
                print("Empty/invalid vision JSON -> retrying once...")

        if parsed and "beams" in parsed:
            all_beams.extend(parsed["beams"])
        else:
            print(f"Vision fallback failed for {img_path}")

    output_file = os.path.join(file_output_folder, f"{file_name}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"beams": all_beams}, f, indent=2)

    print(f"Saved {len(all_beams)} beams -> {output_file}")


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN GROUP SORTING
# ─────────────────────────────────────────────────────────────────────────────

def _build_column_order(records):
    """
    Build a visual reading-order map from actual size-token x coordinates.

    Pattern 10 drawings can contain repeated side-by-side mini tables. Equal
    width page bands are too coarse: x=1269 and x=2191 can land in the same
    band, and the small lower-right ramp table must not be interleaved with
    the main table. Coordinate clustering keeps true visual columns separate.
    """
    if not records:
        return {}

    x_map = cluster_values([r["_x"] for r in records], tolerance=3.0)
    for r in records:
        r["_column_center"] = x_map[r["_x"]]

    stats = {}
    for center in sorted(set(r["_column_center"] for r in records)):
        col_records = [r for r in records if r["_column_center"] == center]
        stats[center] = {
            "count": len(col_records),
            "min_top": min(r["_top"] for r in col_records),
        }

    max_top = max(r["_top"] for r in records)
    late_table_threshold = max_top * 0.75
    small_table_limit = max(5, int(len(records) * 0.03))

    normal_centers = []
    late_small_centers = []
    for center in sorted(stats):
        col = stats[center]
        if col["count"] <= small_table_limit and col["min_top"] >= late_table_threshold:
            late_small_centers.append(center)
        else:
            normal_centers.append(center)

    ordered_centers = normal_centers + late_small_centers
    return {center: i for i, center in enumerate(ordered_centers)}


# ─────────────────────────────────────────────────────────────────────────────
# CORE EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

def extract_pattern_10(pdf_path):
    """
    Extract beam data from a Pattern 10 PDF using coordinate-based text
    extraction.

    Returns:
        list of beam dicts  → success
        None                → PDF is scanned or extraction failed (use vision)
    """
    if not is_digital_pdf(pdf_path):
        print("  -> PDF has no text layer. Vision fallback required.")
        return None

    words_df = extract_words(pdf_path)
    if words_df.empty:
        print("  -> No words extracted.")
        return None

    # ── Find all SIZE tokens ──────────────────────────────────────────────
    size_mask = words_df["text"].str.match(SIZE_TOKEN_RE, na=False)
    size_words = words_df[size_mask].copy()

    if size_words.empty:
        print("  -> No size tokens (B_/BAND_B_/VD_) found.")
        return None

    print(f"  -> Found {len(size_words)} size tokens across"
          f" {size_words['page'].nunique()} page(s).")

    # ── Extract one beam per size token ──────────────────────────────────
    records = []

    for _, size_word in size_words.iterrows():
        # Beam ID
        beam_id = get_left_neighbour(words_df, size_word, top_tol=3.0)
        if beam_id is None:
            continue
        if beam_id.upper() in SKIP_LABELS:
            continue

        # Width
        width = _parse_width(size_word["text"])

        # Depth
        right_tokens = get_right_neighbours(words_df, size_word,
                                            top_tol=3.0, n=4)
        depth = _parse_depth(right_tokens)

        records.append({
            "beam_id": beam_id,
            "size": {"width": width, "depth": depth, "length": None},
            "reinforcement": [],
            "stirrups": {"dia": [], "spacing": []},
            # internal sort keys — removed before output
            "_x":    float(size_word["x0"]),
            "_top":  float(size_word["top"]),
            "_page": int(size_word["page"]),
        })

    if not records:
        print("  -> No valid beams found after filtering.")
        return None

    # ── Sort by reading order: visual column -> top-to-bottom ─────────────
    column_order = _build_column_order(records)
    records.sort(key=lambda r: (
        r["_page"],
        column_order.get(r["_column_center"], 999),
        r["_top"],
        r["_x"],
    ))

    # Pattern 10 is a schedule-row extractor. Preserve duplicate beam IDs
    # because small sub-schedules can repeat labels such as "RB".
    beams = []
    for r in records:
        beams.append({
            "beam_id":       r["beam_id"],
            "size":          r["size"],
            "reinforcement": r["reinforcement"],
            "stirrups":      r["stirrups"],
        })

    unique_ids = len({b["beam_id"] for b in beams})
    print(f"  -> {len(beams)} rows extracted ({unique_ids} unique beam IDs).")
    return beams


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def process_pdf(pdf_path):
    """
    Main entry point called by main_10.py / auto_runner.py.

    Strategy:
        1. Try coordinate-based text extraction (fast, free, accurate).
        2. If the PDF is scanned / text layer absent → fall back to vision.
    """
    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    file_output_folder = os.path.join(OUTPUT_DIR, file_name)
    os.makedirs(file_output_folder, exist_ok=True)

    print(f"\n[{file_name}] Pattern 10 - trying text extraction first...")

    beams = extract_pattern_10(pdf_path)

    # ── Vision fallback ───────────────────────────────────────────────────
    if beams is None:
        print("  -> Falling back to vision extraction.")
        _extract_with_vision(pdf_path, file_output_folder, file_name)
        return

    # ── Save JSON ─────────────────────────────────────────────────────────
    output_file = os.path.join(file_output_folder, f"{file_name}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"beams": beams}, f, indent=2)

    print(f"Saved {len(beams)} beams -> {output_file}")
