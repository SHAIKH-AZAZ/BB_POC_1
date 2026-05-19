from pipeline import run_pipeline, save_beams, run_all


def _clean_beam(beam):
    # Reinforcement
    reinf = []
    for r in beam.get("reinforcement", []):
        r = r.upper().replace("TH", "").replace("EX", "").strip()
        parts = r.split("-")
        temp = [parts[i] + "-" + parts[i+1]
                for i in range(0, len(parts)-1, 2)
                if "T" in parts[i] + "-" + parts[i+1]]
        reinf.extend(temp if temp else ([r] if "T" in r else []))
    beam["reinforcement"] = sorted(set(reinf))

    # Stirrups
    stirrups = beam.get("stirrups", {})
    dia, spacing = [], []
    for d in stirrups.get("dia", []):
        d = d.upper().strip()
        if "L-" in d or "T" in d:
            dia.append(d)
    for s in stirrups.get("spacing", []):
        s = s.strip()
        if s.isdigit():
            spacing.append(f"{s} C/C")
        elif "C/C" in s.upper():
            spacing.append(s.upper())
    beam["stirrups"] = {"dia": sorted(set(dia)), "spacing": sorted(set(spacing))}
    return beam


def process_pdf(pdf_path):
    beams, folder, name = run_pipeline(pdf_path, "prompt_8.txt")
    beams = [_clean_beam(b) for b in beams]
    save_beams(beams, folder, name)


def main(): run_all(process_pdf)
if __name__ == "__main__": main()
