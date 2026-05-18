"""
table_extractor.py
==================
Coordinate-based table extraction from digital PDFs using pdfplumber.

HOW IT WORKS (the ChatGPT approach):
-------------------------------------
1. pdfplumber extracts every word with its bounding box (x0, x1, top, bottom).
2. Words that share nearly the same `top` value belong to the same visual row.
3. Words that share nearly the same `x0` value belong to the same visual column.
4. We cluster those coordinates to reconstruct the table grid without any OCR
   or vision model — pure geometry.

WHY THIS IS BETTER THAN VISION FOR DIGITAL PDFs:
-------------------------------------------------
- Speed : No image rendering, no API call. Runs in milliseconds.
- Cost  : Zero tokens / zero API spend.
- Accuracy: Exact text from the PDF vector layer — no hallucinations.
- Deterministic: Same input always gives same output.

USE CASE:
---------
- Digital/vector PDFs (AutoCAD exports, structural drawing PDFs).
- Falls back to vision for scanned/image-only PDFs.
"""

import re
import pdfplumber
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 1. DIGITAL-PDF DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def is_digital_pdf(pdf_path, min_words=20):
    """
    Return True if the PDF has a text layer with enough extractable words.
    False means it's a scanned image → fall back to vision extraction.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            count = 0
            for page in pdf.pages[:3]:
                words = page.extract_words()
                count += len(words)
                if count >= min_words:
                    return True
        return False
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 2. WORD EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_words(pdf_path):
    """
    Extract all words with bounding boxes from every page of a PDF.

    Returns a DataFrame with columns:
        text, x0, x1, top, bottom, page

    x0/x1/top/bottom are in PDF points (1 pt ≈ 0.353 mm).
    """
    records = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                words = page.extract_words(
                    x_tolerance=3,
                    y_tolerance=3,
                    keep_blank_chars=False,
                    use_text_flow=False,
                )
                for w in words:
                    records.append({
                        "text":   str(w["text"]),
                        "x0":     float(w["x0"]),
                        "x1":     float(w["x1"]),
                        "top":    float(w["top"]),
                        "bottom": float(w["bottom"]),
                        "page":   page_num + 1,
                    })
    except Exception as e:
        print(f"⚠ Word extraction failed: {e}")

    if not records:
        return pd.DataFrame(columns=["text", "x0", "x1", "top", "bottom", "page"])
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 3. COORDINATE CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def cluster_values(values, tolerance=5):
    """
    Group nearby float values into clusters.

    Example:
        cluster_values([100.1, 100.3, 200.0, 200.5], tolerance=1)
        → {100.1: 100.2, 100.3: 100.2, 200.0: 200.25, 200.5: 200.25}

    Returns dict: raw_value → cluster_center
    """
    if not values:
        return {}

    sorted_vals = sorted(set(float(v) for v in values))
    groups = [[sorted_vals[0]]]

    for v in sorted_vals[1:]:
        if v - groups[-1][-1] <= tolerance:
            groups[-1].append(v)
        else:
            groups.append([v])

    mapping = {}
    for group in groups:
        center = round(sum(group) / len(group), 2)
        for v in group:
            mapping[v] = center

    return mapping


def assign_grid_keys(words_df, row_tol=4, col_tol=8):
    """
    Add `row_key` and `col_key` columns to the words DataFrame.

    - row_key: clustered `top` value  → identifies which visual row a word is in
    - col_key: clustered `x0` value   → identifies which visual column a word is in

    Tolerances (in PDF points):
        row_tol=4  ≈ slightly different y for text on the same line
        col_tol=8  ≈ minor x-alignment variations within a column
    """
    df = words_df.copy()

    row_map = cluster_values(df["top"].tolist(), tolerance=row_tol)
    df["row_key"] = df["top"].map(row_map)

    col_map = cluster_values(df["x0"].tolist(), tolerance=col_tol)
    df["col_key"] = df["x0"].map(col_map)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. TABLE RECONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def get_table_rows(words_df, row_tol=4, col_tol=8):
    """
    Convert a words DataFrame into a list of row-dicts.

    Each dict looks like:
        {
          "_row_key": 319.5,
          "_page": 1,
          328.17: "B1",        ← col_key → cell text
          1269.75: "B_400",
          ...
        }

    Multi-word cells (two words with the same row_key + col_key) are
    space-joined in left-to-right order.

    Rows are sorted by (page, row_key) → top-to-bottom reading order.
    """
    df = assign_grid_keys(words_df, row_tol, col_tol)
    if df.empty:
        return []

    rows = []
    for (page, row_key), group in df.groupby(["page", "row_key"]):
        row = {"_row_key": row_key, "_page": page}
        for _, word in group.sort_values("x0").iterrows():
            col = word["col_key"]
            row[col] = (str(row[col]) + " " + str(word["text"])).strip() \
                       if col in row else str(word["text"])
        rows.append(row)

    return sorted(rows, key=lambda r: (r["_page"], r["_row_key"]))


# ─────────────────────────────────────────────────────────────────────────────
# 5. HEADER DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def find_header_row_index(rows, keywords, min_matches=2):
    """
    Find which row index contains table column headers by counting keyword hits.

    keywords    : list of uppercase strings expected in the header row
    min_matches : minimum keyword hits to accept as a header

    Returns the row index (int) or None if not found.
    """
    best_idx = None
    best_score = 0

    for i, row in enumerate(rows):
        row_text = " ".join(
            str(v).upper()
            for k, v in row.items()
            if not str(k).startswith("_")
        )
        score = sum(1 for kw in keywords if kw.upper() in row_text)
        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx if best_score >= min_matches else None


def build_column_map(header_row, col_keys):
    """
    Build a mapping of col_key → header_label from the detected header row.

    col_keys: list of col_key floats present in the data
    Returns dict: col_key (float) → header_text (str)
    """
    mapping = {}
    for ck in col_keys:
        if ck in header_row:
            mapping[ck] = str(header_row[ck]).strip().upper()
    return mapping


# ─────────────────────────────────────────────────────────────────────────────
# 6. LEFT-NEIGHBOUR LOOKUP  (used by Pattern 10 / strip-style layouts)
# ─────────────────────────────────────────────────────────────────────────────

def get_left_neighbour(words_df, target_word, top_tol=3.0):
    """
    Return the word immediately to the LEFT of `target_word` on the same line.

    target_word: a single-row Series from the words DataFrame
    top_tol    : vertical tolerance in PDF points (words within ±top_tol of
                 target_word['top'] are considered on the same line)

    Returns the text of the left neighbour, or None.
    """
    same_row = words_df[
        (words_df["page"] == target_word["page"]) &
        (abs(words_df["top"] - target_word["top"]) < top_tol)
    ]
    left = same_row[same_row["x1"] < target_word["x0"] - 1].sort_values("x1")
    if left.empty:
        return None
    return str(left.iloc[-1]["text"]).strip()


def get_right_neighbours(words_df, target_word, top_tol=3.0, n=4):
    """
    Return up to `n` words immediately to the RIGHT of `target_word` on the
    same line, sorted left-to-right.

    Returns list of text strings.
    """
    same_row = words_df[
        (words_df["page"] == target_word["page"]) &
        (abs(words_df["top"] - target_word["top"]) < top_tol)
    ]
    right = same_row[same_row["x0"] > target_word["x1"] - 1].sort_values("x0")
    return right["text"].head(n).tolist()


# ─────────────────────────────────────────────────────────────────────────────
# 7. REINFORCEMENT NORMALISATION  (shared utility)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_reinforcement(reinf_list):
    """
    Clean and normalise a list of reinforcement strings.
    Output format: "quantity-Tdiameter"  (e.g. "3-T20", "2-T16")

    Handles:
        7T25        → 7-T25
        2-25        → 2-T25
        2T16+3T20   → ["2-T16", "3-T20"]  (split on +)
        TOR         → T
    """
    cleaned = set()

    for item in reinf_list:
        if not item:
            continue
        item = str(item).strip().upper().replace(" ", "")

        # TOR shorthand → T
        item = item.replace("TOR", "T")
        # Fix double-T artefacts
        item = item.replace("TT", "T")

        # Split on "+" first
        parts = re.split(r'\+', item)

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Already in N-TM format
            if re.match(r'^\d+-T\d+$', part):
                cleaned.add(part)
                continue

            # 7T25 → 7-T25
            m = re.match(r'^(\d+)T(\d+)$', part)
            if m:
                cleaned.add(f"{m.group(1)}-T{m.group(2)}")
                continue

            # 2-25 → 2-T25
            m = re.match(r'^(\d+)-(\d+)$', part)
            if m:
                cleaned.add(f"{m.group(1)}-T{m.group(2)}")
                continue

            # 2-8 TOR (already handled TOR→T above): 2-T8
            # Keep remainder as-is if it contains T
            if "T" in part and re.search(r'\d', part):
                cleaned.add(part)

    def sort_key(x):
        try:
            qty, dia = x.split("-T")
            return (int(dia), int(qty))
        except Exception:
            return (999, 999)

    return sorted(cleaned, key=sort_key)
