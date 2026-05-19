import re
from pipeline import run_pipeline, save_beams, run_all
from utils import normalize_reinforcement, deduplicate_beams


def _clean_stirrups(stirrups):
    dia, spacing = set(), set()
    for d in stirrups.get("dia", []):
        if d: dia.add(d.strip().upper().replace(" ", ""))
    for s in stirrups.get("spacing", []):
        if s:
            s = s.upper().replace(" ", "").replace("C/C", "").replace("C", "")
            if s.isdigit():
                spacing.add(f"{s} C/C")
    return {"dia": sorted(dia), "spacing": sorted(spacing)}


def process_pdf(pdf_path):
    beams, folder, name = run_pipeline(pdf_path, "prompt_2.txt")
    beams = deduplicate_beams(beams, normalize_fn=normalize_reinforcement)
    for beam in beams:
        beam["stirrups"] = _clean_stirrups(beam.get("stirrups", {}))
    save_beams(beams, folder, name)


def main(): run_all(process_pdf)
if __name__ == "__main__": main()
