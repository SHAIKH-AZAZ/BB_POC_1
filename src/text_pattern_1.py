"""
text_pattern_1.py
=================
Coordinate-based extractor for Pattern 1 beam schedules.

PATTERN 1 TABLE STRUCTURE:
    BEAM     | SIZE         | BOTTOM REINFORCEMENT        | TOP REINFORCEMENT           | SHEAR STIRRUPS
    NUMBERS  | WIDTH  DEPTH | LEFT    MID     RIGHT       | LEFT    MID     RIGHT       | LEFT   MID   RIGHT
    ─────────┼──────────────┼─────────────────────────────┼─────────────────────────────┼──────────────────
    B1       | 750    750   | 5-T20   7-T20   2-T25       | 2-T16   3-T25   2-T20       | 6L-T12@100   ...

HOW COORDINATE-BASED EXTRACTION WORKS FOR MULTI-LEVEL HEADERS:
----------------------------------------------------------------
The header can span 2-3 rows (merged cells in AutoCAD tables).
Strategy:
  1. Extract ALL rows as coordinate-keyed dicts.
  2. Look for the header region by scanning for known keywords
     (BEAM, SIZE, WIDTH, DEPTH, REINFORCEMENT, STIRRUPS).
  3. Build a column map: col_x_position → semantic label.
  4. For each data row below the header, read values by x-position
     and map them to the JSON schema.

This approach is pattern-generic — the same TableParser class is reusable
for patterns 2, 3, 6, 7, 9, etc. with only the column_map changing.
"""

import re
import json
import os

from table_extractor import (
    is_digital_pdf,
    extract_words,
    get_table_rows,
    find_header_row_index,
    build_column_map,
    normalize_reinforcement,
    cluster_values,
)
from config import OUTPUT_DIR


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN SEMANTIC LABELS  (what each column header text maps to)
# ─────────────────────────────────────────────────────────────────────────────

# Pattern 1 column keywords → schema field
# These are matched as substrings (case-insensitive) in the header text.
COLUMN_SCHEMA_MAP = [
    # (header_substring,          schema_field)
    ("BEAM",                       "beam_id"),
    ("WIDTH",                      "size.width"),
    ("DEPTH",                      "size.depth"),
    ("BOTTOM.*LEFT",               "reinf.bot.left"),
    ("BOTTOM.*MID",                "reinf.bot.mid"),
    ("BOTTOM.*RIGHT",              "reinf.bot.right"),
    ("TOP.*LEFT",                  "reinf.top.left"),
    ("TOP.*MID",                   "reinf.top.mid"),
    ("TOP.*RIGHT",                 "reinf.top.right"),
    ("STIRRUP.*LEFT|LEFT.*STIRRUP","stirrup.left"),
    ("STIRRUP.*MID|MID.*STIRRUP",  "stirrup.mid"),
    ("STIRRUP.*RIGHT|RIGHT.*STIRR","stirrup.right"),
    # Fallback single-col stirrups
    ("STIRRUP",                    "stirrup.mid"),
    ("DIA",                        "stirrup.dia"),
    ("SPAC",                       "stirrup.spacing"),
]

# Header keywords to detect the header row
HEADER_KEYWORDS = [
    "BEAM", "WIDTH", "DEPTH", "BOTTOM", "TOP",
    "REINFORCEMENT", "STIRRUP", "SIZE",
]

# Noise tokens to skip when looking at beam IDs
SKIP_BEAM_LABELS = {
    "NO.", "NUMBERS", "BEAM", "SIZE", "MARK", "S.NO",
    "SR", "S.", "MARKED", "REINF", "STIRRUPS",
}


# ─────────────────────────────────────────────────────────────────────────────
# GENERAL TABLE PARSER  (reusable for all patterns)
# ─────────────────────────────────────────────────────────────────────────────

class TableParser:
    """
    General coordinate-based table parser.

    Usage:
        parser = TableParser(words_df,
                             header_keywords=HEADER_KEYWORDS,
                             schema_map=COLUMN_SCHEMA_MAP)
        rows = parser.parse()   # list of schema-keyed dicts

    Each returned dict has keys like:
        beam_id, size.width, size.depth,
        reinf.bot.left, reinf.bot.mid, ...
        stirrup.left, stirrup.mid, stirrup.right
    """

    def __init__(self, words_df, header_keywords, schema_map,
                 row_tol=4, col_tol=10):
        self.words_df = words_df
        self.header_keywords = header_keywords
        self.schema_map = schema_map
        self.row_tol = row_tol
        self.col_tol = col_tol

    # ── Step 1: Reconstruct grid ──────────────────────────────────────────
    def _get_rows(self):
        return get_table_rows(self.words_df, self.row_tol, self.col_tol)

    # ── Step 2: Find header region ────────────────────────────────────────
    def _find_header_end(self, rows):
        """
        Returns the index of the last header row (data starts at index+1).
        For multi-row headers, scans up to 5 rows after the first keyword hit.
        """
        first_hit = find_header_row_index(
            rows, self.header_keywords, min_matches=2
        )
        if first_hit is None:
            return None

        # Header may span multiple rows (merged cells).  Scan forward until
        # we hit a row that looks like actual data (contains numbers/IDs).
        for i in range(first_hit, min(first_hit + 5, len(rows))):
            row_vals = [
                str(v) for k, v in rows[i].items()
                if not str(k).startswith("_")
            ]
            row_text = " ".join(row_vals)
            # If this row has obvious data-like patterns (beam IDs, numbers)
            # it's probably not a header row.
            if re.search(r'\b\d{2,4}\b', row_text) and i > first_hit:
                # Check if the numbers look like dimensions (200–2000 range)
                nums = [int(n) for n in re.findall(r'\b\d{3,4}\b', row_text)
                        if 100 <= int(n) <= 5000]
                if nums:
                    return i - 1  # last header row

        return first_hit

    # ── Step 3: Build col_key → schema_field mapping ─────────────────────
    def _build_schema_col_map(self, rows, header_end):
        """
        Merge text from all header rows into a single wide-header per col_key,
        then match each col_key to a schema field via COLUMN_SCHEMA_MAP.
        """
        # Collect all col_keys that appear in ANY header row
        all_col_keys = set()
        for row in rows[: header_end + 1]:
            for k in row:
                if not str(k).startswith("_"):
                    all_col_keys.add(k)

        # Concatenate header text across rows for each col_key
        col_text = {ck: [] for ck in all_col_keys}
        for row in rows[: header_end + 1]:
            for ck in all_col_keys:
                if ck in row:
                    col_text[ck].append(str(row[ck]).upper())

        # Match to schema fields
        col_schema = {}
        for ck, texts in col_text.items():
            combined = " ".join(texts)
            for pattern, field in self.schema_map:
                if re.search(pattern, combined, re.IGNORECASE):
                    col_schema[ck] = field
                    break  # first match wins

        return col_schema

    # ── Step 4: Parse data rows ───────────────────────────────────────────
    def parse(self):
        """
        Returns list of schema-keyed row dicts.
        Keys are schema field paths like 'beam_id', 'size.width', etc.
        """
        rows = self._get_rows()
        if not rows:
            return []

        header_end = self._find_header_end(rows)
        if header_end is None:
            print("  ⚠ Header row not found — trying full page as data.")
            header_end = -1

        col_schema = self._build_schema_col_map(rows, header_end)

        data_rows = []
        for row in rows[header_end + 1:]:
            mapped = {}
            for ck, field in col_schema.items():
                if ck in row:
                    mapped[field] = str(row[ck]).strip()
            if mapped:
                data_rows.append(mapped)

        return data_rows


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA ASSEMBLY  (convert parsed rows → standard beam JSON)
# ─────────────────────────────────────────────────────────────────────────────

def _safe_int(text):
    """Extract first integer from a string, or None."""
    m = re.search(r'\d+', str(text))
    return int(m.group()) if m else None


def _collect_reinf(row_dict, *field_names):
    """Collect reinforcement strings from multiple schema fields."""
    items = []
    for f in field_names:
        val = row_dict.get(f, "").strip()
        if val and val not in ("-", "---", "NIL", ""):
            # Split on "+" within the cell
            items.extend(re.split(r'\+', val))
    return normalize_reinforcement(items)


def _collect_stirrups(row_dict):
    """
    Extract stirrup dia and spacing from row dict.

    Handles formats like:
        "6L-T12 @ 100 C/C"   → dia=["6L-T12"]  spacing=["100 C/C"]
        "T10@150C/C"          → dia=["T10"]      spacing=["150 C/C"]
    """
    dia_set = set()
    spacing_set = set()

    stirrup_fields = [
        "stirrup.left", "stirrup.mid", "stirrup.right",
        "stirrup.dia", "stirrup.spacing",
    ]

    for field in stirrup_fields:
        val = row_dict.get(field, "").strip()
        if not val or val in ("-", "---"):
            continue

        # Pattern: DIA_TEXT @ SPACING
        m = re.match(r'^(.+?)\s*@\s*(.+)$', val)
        if m:
            d = m.group(1).strip().replace(" ", "")
            s = m.group(2).strip()
            # Normalise spacing to "NNN C/C"
            num = re.search(r'\d+', s)
            if num:
                s = f"{num.group()} C/C"
            dia_set.add(d)
            spacing_set.add(s)
        else:
            # Pure number → treat as spacing
            if re.match(r'^\d+$', val):
                spacing_set.add(f"{val} C/C")
            else:
                dia_set.add(val.replace(" ", ""))

    return sorted(dia_set), sorted(spacing_set)


def rows_to_beams(parsed_rows):
    """
    Convert a list of schema-keyed row dicts → list of standard beam JSON dicts.
    """
    beams = []
    for row in parsed_rows:
        bid = row.get("beam_id", "").strip()
        if not bid or bid.upper() in SKIP_BEAM_LABELS:
            continue

        # Must look like a real beam label (contains at least one letter)
        if not re.search(r'[A-Za-z]', bid):
            continue

        width = _safe_int(row.get("size.width", ""))
        depth = _safe_int(row.get("size.depth", ""))

        reinf = _collect_reinf(
            row,
            "reinf.bot.left", "reinf.bot.mid", "reinf.bot.right",
            "reinf.top.left", "reinf.top.mid", "reinf.top.right",
        )

        dia, spacing = _collect_stirrups(row)

        beams.append({
            "beam_id": bid,
            "size": {
                "width":  width,
                "depth":  depth,
                "length": None,
            },
            "reinforcement": reinf,
            "stirrups": {
                "dia":     dia,
                "spacing": spacing,
            },
        })

    return beams


# ─────────────────────────────────────────────────────────────────────────────
# CORE EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

def extract_pattern_1(pdf_path):
    """
    Extract beam data from a Pattern 1 PDF using coordinate-based extraction.

    Returns list of beam dicts, or None if extraction is not possible.
    """
    if not is_digital_pdf(pdf_path):
        return None

    words_df = extract_words(pdf_path)
    if words_df.empty:
        return None

    parser = TableParser(
        words_df,
        header_keywords=HEADER_KEYWORDS,
        schema_map=COLUMN_SCHEMA_MAP,
        row_tol=4,
        col_tol=10,
    )

    parsed_rows = parser.parse()
    if not parsed_rows:
        print("  ⚠ No data rows extracted from table.")
        return None

    beams = rows_to_beams(parsed_rows)
    print(f"  → {len(beams)} beams assembled from {len(parsed_rows)} table rows.")
    return beams if beams else None


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def process_pdf(pdf_path):
    """
    Main entry point. Try text extraction; fall back to vision if needed.
    """
    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    file_output_folder = os.path.join(OUTPUT_DIR, file_name)
    os.makedirs(file_output_folder, exist_ok=True)

    print(f"\n📄 [{file_name}] Pattern 1 — trying text extraction first...")

    beams = extract_pattern_1(pdf_path)

    if beams is None:
        print("  → Falling back to vision extraction (main_1_vision.py)...")
        import importlib
        vision = importlib.import_module("main_1_vision")
        vision.process_pdf(pdf_path)
        return

    output_file = os.path.join(file_output_folder, f"{file_name}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"beams": beams}, f, indent=2)

    print(f"✅ Saved {len(beams)} beams → {output_file}")
