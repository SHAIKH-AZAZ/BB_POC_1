import re
from pipeline import run_pipeline, save_beams, run_all
from utils import normalize_reinforcement, deduplicate_beams


def _clean_stirrups(stirrups):
    """
    Normalise stirrup dia and spacing to canonical forms.

    dia    : "T8@150C/C" → "T8"    |  "T8" → "T8"  |  "8" → "T8"
    spacing: "150C/C"    → "150 C/C"
             "T8@150C/C" → "150 C/C"   (full string passed by model)
    """
    dia_set, spacing_set = set(), set()

    for raw in stirrups.get("dia", []):
        if not raw:
            continue
        s = str(raw).strip().upper().replace(" ", "")

        # Full stirrup string e.g. "T8@150C/C" → extract dia part
        if "@" in s:
            dia_part = s.split("@")[0]          # "T8"
            spacing_part = s.split("@")[1]      # "150C/C"
            # also capture spacing from here
            sp = re.sub(r'[^0-9]', '', spacing_part)
            if sp:
                spacing_set.add(f"{sp} C/C")
        else:
            dia_part = s

        # Normalise dia: "8" → "T8",  "8T" → "T8",  "T8" stays
        dia_part = re.sub(r'[^0-9T]', '', dia_part)
        if dia_part.endswith("T") and dia_part[:-1].isdigit():
            dia_part = f"T{dia_part[:-1]}"
        elif dia_part.isdigit():
            dia_part = f"T{dia_part}"
        if dia_part:
            dia_set.add(dia_part)

    for raw in stirrups.get("spacing", []):
        if not raw:
            continue
        s = str(raw).strip().upper().replace(" ", "")
        # Accept "150C/C", "150", "T8@150C/C"
        if "@" in s:
            s = s.split("@")[1]   # take part after @
        sp = re.sub(r'[^0-9]', '', s)
        if sp:
            spacing_set.add(f"{sp} C/C")

    return {
        "dia":     sorted(dia_set),
        "spacing": sorted(spacing_set),
    }


def process_pdf(pdf_path):
    beams, folder, name = run_pipeline(pdf_path, "prompt_3.txt")
    beams = deduplicate_beams(beams, normalize_fn=normalize_reinforcement)
    for beam in beams:
        beam["stirrups"] = _clean_stirrups(beam.get("stirrups") or {})
    save_beams(beams, folder, name)


def main():
    run_all(process_pdf)


if __name__ == "__main__":
    main()
