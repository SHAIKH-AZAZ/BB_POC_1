from pipeline import run_pipeline, save_beams, run_all


def _normalize_reinf_7(reinf_list):
    cleaned = set()
    for item in reinf_list:
        if not item: continue
        item = item.strip().upper()
        if item == "-": continue
        for p in item.split("+"):
            p = p.strip()
            if p and p != "-":
                cleaned.add(p)
    return sorted(cleaned)


def process_pdf(pdf_path):
    beams, folder, name = run_pipeline(pdf_path, "prompt_7.txt")
    for beam in beams:
        beam["reinforcement"]      = _normalize_reinf_7(beam.get("reinforcement", []))
        beam["stirrups"]["dia"]     = sorted(set(beam["stirrups"].get("dia", [])))
        beam["stirrups"]["spacing"] = sorted(set(beam["stirrups"].get("spacing", [])))
    save_beams(beams, folder, name)


def main(): run_all(process_pdf)
if __name__ == "__main__": main()
