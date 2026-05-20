"""
utils.py
========
Shared utilities for all beam extraction patterns.
"""

import re
import json


# -----------------------------------------------------------------------------
# REINFORCEMENT NORMALISATION
# -----------------------------------------------------------------------------

def normalize_reinforcement(reinf_list):
    """
    Cleans and normalises a list of reinforcement strings.
    Output format: "quantity-Tdiameter"  e.g. "3-T20", "7-T25"
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

            # 7T25 -> 7-T25
            m = re.match(r'^(\d+)T(\d+)$', part)
            if m:
                cleaned.add(f"{m.group(1)}-T{m.group(2)}")
                continue

            # 2-25 -> 2-T25
            m = re.match(r'^(\d+)-(\d+)$', part)
            if m:
                cleaned.add(f"{m.group(1)}-T{m.group(2)}")
                continue

            # Keep anything with a T + digit
            if "T" in part and re.search(r'\d', part):
                cleaned.add(part)

    return sorted(cleaned, key=sort_key)


def sort_key(x):
    """Sort key for reinforcement strings by diameter then quantity."""
    try:
        qty, dia = x.split("-T")
        return (int(dia), int(qty))
    except Exception:
        return (999, 999)


# -----------------------------------------------------------------------------
# JSON PARSING
# -----------------------------------------------------------------------------

def safe_parse_json(text):
    """
    Parse a JSON string that may have leading/trailing text or markdown.
    Returns parsed dict/list or None on failure.
    """
    if not text:
        return None

    text = str(text).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

    return None


# -----------------------------------------------------------------------------
# BEAM DEDUPLICATION
# -----------------------------------------------------------------------------

def _to_flat_list(val):
    """
    Safely convert any model-returned stirrups dia/spacing value to a flat list.

    The model occasionally returns these fields as a dict instead of a list.
    Handles every possible shape:
      list  -> returned as-is
      dict  -> values() flattened (keys ignored)
      str   -> wrapped in single-item list
      None  -> empty list
    """
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, dict):
        return [str(v) for v in val.values() if v and str(v).strip() not in ("", "-")]
    if isinstance(val, str):
        return [val] if val.strip() and val.strip() != "-" else []
    return [str(val)]


def deduplicate_beams(beams, normalize_fn=None):
    """
    Merge beams that share the same beam_id, combining reinforcement and
    stirrups lists. Optionally normalise reinforcement via normalize_fn.

    Robust to model output where:
    - beam_id may be None/null
    - stirrups.dia / stirrups.spacing may be dict, string, or list
    """
    unique = {}

    for beam in beams:
        if not isinstance(beam, dict):
            continue

        # beam_id: guard against explicit null from model
        bid = (beam.get("beam_id") or "").strip()
        if not bid:
            continue

        # Normalise stirrups fields to plain lists before any merge
        stirrups = beam.get("stirrups") or {}
        if not isinstance(stirrups, dict):
            stirrups = {}
        stirrups["dia"]     = _to_flat_list(stirrups.get("dia"))
        stirrups["spacing"] = _to_flat_list(stirrups.get("spacing"))
        beam["stirrups"] = stirrups

        if bid not in unique:
            unique[bid] = beam
        else:
            existing = unique[bid]
            existing["reinforcement"] = (
                existing.get("reinforcement") or []
            ) + (beam.get("reinforcement") or [])
            existing["stirrups"]["dia"]     += stirrups["dia"]
            existing["stirrups"]["spacing"] += stirrups["spacing"]

    result = []
    for beam in unique.values():
        if normalize_fn:
            beam["reinforcement"] = normalize_fn(beam.get("reinforcement") or [])
        beam["stirrups"]["dia"]     = sorted(set(beam["stirrups"]["dia"]))
        beam["stirrups"]["spacing"] = sorted(set(beam["stirrups"]["spacing"]))
        result.append(beam)

    return result
