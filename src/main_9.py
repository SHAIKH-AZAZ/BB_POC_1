import re
from pipeline import run_pipeline, save_beams, run_all
from utils import deduplicate_beams


def _clean_reinf(reinf_list):
    clean = set()
    for r in reinf_list:
        if not r: continue
        r = re.sub(r"\(.*?\)|\[.*?\]", "", str(r).upper()).replace("+", " ")
        for m in re.findall(r"\d+\s*-\s*T\d+", r):
            clean.add(m.replace(" ", ""))
        for m in re.findall(r"\b(\d+)\s*T(\d+)\b", r):
            clean.add(f"{m[0]}-T{m[1]}")
    return sorted(clean)


def _clean_stirrups(stirrups):
    dia, spacing = set(), set()
    for d in stirrups.get("dia", []):
        if d: dia.add(d.strip().upper())
    for s in stirrups.get("spacing", []):
        if s:
            s = s.strip().upper()
            if "C/C" not in s:
                sc = s.replace(" ", "")
                if sc.isdigit(): s = f"{sc} C/C"
            spacing.add(s)
    return {"dia": sorted(dia), "spacing": sorted(spacing)}


def process_pdf(pdf_path):
    beams, folder, name = run_pipeline(pdf_path, "prompt_9.txt")
    beams = deduplicate_beams(beams)
    for beam in beams:
        beam["reinforcement"] = _clean_reinf(beam.get("reinforcement", []))
        beam["stirrups"]      = _clean_stirrups(beam.get("stirrups", {}))
    save_beams(beams, folder, name)


def main(): run_all(process_pdf)
if __name__ == "__main__": main()
