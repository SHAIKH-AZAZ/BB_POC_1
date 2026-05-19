import re
from pipeline import run_pipeline, save_beams, run_all
from utils import normalize_reinforcement, deduplicate_beams


def _strict_filter(beams):
    """Pattern 3: beams typically have ≤4 bars, dia ≤32 — drop outliers."""
    for beam in beams:
        valid = []
        for r in beam.get("reinforcement", []):
            try:
                qty, dia = r.split("-T")
                if int(qty) <= 4 and int(dia) <= 32:
                    valid.append(r)
            except Exception:
                continue
        beam["reinforcement"] = valid
    return beams


def _clean_stirrups(stirrups):
    dia, spacing = set(), set()
    for d in stirrups.get("dia", []):
        if d:
            d = d.strip().upper().replace(" ", "")
            if d.endswith("T") and d[:-1].isdigit():
                d = f"T{d[:-1]}"
            dia.add(d)
    for s in stirrups.get("spacing", []):
        if s:
            s = s.upper().replace(" ", "").replace("C/C", "").replace("C", "")
            if s.isdigit():
                spacing.add(f"{s} C/C")
    return {"dia": sorted(dia), "spacing": sorted(spacing)}


def process_pdf(pdf_path):
    beams, folder, name = run_pipeline(pdf_path, "prompt_3.txt")
    beams = deduplicate_beams(beams, normalize_fn=normalize_reinforcement)
    beams = _strict_filter(beams)
    for beam in beams:
        beam["stirrups"] = _clean_stirrups(beam.get("stirrups", {}))
    save_beams(beams, folder, name)


def main(): run_all(process_pdf)
if __name__ == "__main__": main()
