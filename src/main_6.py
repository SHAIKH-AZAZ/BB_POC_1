from pipeline import run_pipeline, save_beams, run_all


def _normalize_reinf_6(reinf_list):
    """Pattern 6/7: split on '+', strip '-' placeholders — no qty-Tdia formatting."""
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
    beams, folder, name = run_pipeline(pdf_path, "prompt_6.txt")
    # Pattern 6: clean each beam individually, no cross-merge
    for beam in beams:
        beam["reinforcement"]      = _normalize_reinf_6(beam.get("reinforcement", []))
        beam["stirrups"]["dia"]     = sorted(set(beam["stirrups"].get("dia", [])))
        beam["stirrups"]["spacing"] = sorted(set(beam["stirrups"].get("spacing", [])))
    save_beams(beams, folder, name)


def main(): run_all(process_pdf)
if __name__ == "__main__": main()
