from pipeline import run_pipeline, save_beams, run_all


def process_pdf(pdf_path):
    beams, folder, name = run_pipeline(pdf_path, "prompt_13.txt")
    cleaned = []
    for beam in beams:
        if beam["size"]["width"] is None and not beam["reinforcement"]:
            continue
        beam["reinforcement"]      = list(dict.fromkeys(beam.get("reinforcement", [])))
        beam["stirrups"]["dia"]     = list(dict.fromkeys(beam["stirrups"].get("dia", [])))
        beam["stirrups"]["spacing"] = list(dict.fromkeys(beam["stirrups"].get("spacing", [])))
        cleaned.append(beam)
    save_beams(cleaned, folder, name)


def main(): run_all(process_pdf)
if __name__ == "__main__": main()
