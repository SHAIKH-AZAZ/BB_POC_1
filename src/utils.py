"""
utils.py
========
Shared utilities for all beam extraction patterns.
"""

import re
import json


# ─────────────────────────────────────────────────────────────────────────────
# REINFORCEMENT NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

def normalize_reinforcement(reinf_list):
    """
    Cleans and normalises a list of reinforcement strings.
    Output format: "quantity-Tdiameter"  e.g. "3-T20", "7-T25"

    Handles: 7T25 → 7-T25 | 2-25 → 2-T25 | TOR → T | TT → T
    """
    cleaned = set()

    for item in reinf_list:
        if not item:
            continue

        item = str(item).strip().upper().replace(" ", "")
        item = item.replace("TOR", "T").replace("TT", "T")

        for part in re.split(r'\+', item):
            part = part.strip()
            if not part:
                continue

            # Already correct: 3-T20
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

            # Keep anything that has a T + digit
            if "T" in part and re.search(r'\d', part):
                cleaned.add(part)

    def sort_key(x):
        try:
            qty, dia = x.split("-T")
            return (int(dia), int(qty))
        except Exception:
            return (999, 999)

    return sorted(cleaned, key=sort_key)


# ─────────────────────────────────────────────────────────────────────────────
# JSON PARSING
# ─────────────────────────────────────────────────────────────────────────────

def safe_parse_json(text):
    """
    Parse a JSON string that may have leading/trailing text or markdown.
    Returns parsed dict/list or None on failure.
    """
    if not text:
        return None

    text = str(text).strip()

    # Direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Extract first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

    return None


# ─────────────────────────────────────────────────────────────────────────────
# BEAM DEDUPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def deduplicate_beams(beams, normalize_fn=None):
    """
    Merge beams that share the same beam_id, combining reinforcement and
    stirrups lists. Optionally normalise reinforcement via normalize_fn.

    Returns a clean list of unique beam dicts.
    """
    unique = {}

    for beam in beams:
        bid = beam.get("beam_id", "").strip()
        if not bid:
            continue

        if bid not in unique:
            unique[bid] = beam
        else:
            existing = unique[bid]
            existing["reinforcement"] = (
                existing.get("reinforcement", []) + beam.get("reinforcement", [])
            )
            existing["stirrups"]["dia"] = (
                existing["stirrups"].get("dia", []) + beam["stirrups"].get("dia", [])
            )
            existing["stirrups"]["spacing"] = (
                existing["stirrups"].get("spacing", []) + beam["stirrups"].get("spacing", [])
            )

    result = []
    for beam in unique.values():
        if normalize_fn:
            beam["reinforcement"] = normalize_fn(beam["reinforcement"])
        beam["stirrups"]["dia"]     = sorted(set(beam["stirrups"].get("dia", [])))
        beam["stirrups"]["spacing"] = sorted(set(beam["stirrups"].get("spacing", [])))
        result.append(beam)

    return result
