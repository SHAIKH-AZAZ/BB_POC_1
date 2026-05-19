from pipeline import run_pipeline, save_beams, run_all


def process_pdf(pdf_path):
    beams, folder, name = run_pipeline(pdf_path, "prompt_5.txt")
    # Drop empty beams (no size and no reinforcement)
    beams = [b for b in beams if b["size"]["width"] is not None or b["reinforcement"]]
    save_beams(beams, folder, name)


def main(): run_all(process_pdf)
if __name__ == "__main__": main()
