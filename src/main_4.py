from pipeline import run_pipeline, save_beams, run_all
from utils import normalize_reinforcement, deduplicate_beams


def process_pdf(pdf_path):
    beams, folder, name = run_pipeline(pdf_path, "prompt_4.txt")
    beams = deduplicate_beams(beams, normalize_fn=normalize_reinforcement)
    save_beams(beams, folder, name)


def main(): run_all(process_pdf)
if __name__ == "__main__": main()
