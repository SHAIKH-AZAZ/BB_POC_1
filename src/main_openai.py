"""
Production OpenAI workflow entry point.

This runner keeps the existing Beam project pipeline, but uses strict JSON
schema output from GPT-4.1 mini and deterministic Pydantic cleanup.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import INPUT_DIR, OPENAI_BATCH_WORKERS, OUTPUT_DIR
from pattern_detector import detect_pattern
from pipeline import run_structured_pipeline, save_beams


def _src_path(filename):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)


def _resolve_prompt(pdf_path, prompt_file=None, pattern=None):
    if prompt_file:
        return prompt_file

    if pattern is None:
        file_name = os.path.splitext(os.path.basename(pdf_path))[0]
        temp_folder = os.path.join(OUTPUT_DIR, "temp_detection", file_name)
        os.makedirs(temp_folder, exist_ok=True)
        print(f"Detecting pattern for {os.path.basename(pdf_path)}...")
        pattern = detect_pattern(pdf_path, temp_folder)

    prompt_file = f"prompt_{int(pattern)}.txt"
    if not os.path.exists(_src_path(prompt_file)):
        raise FileNotFoundError(f"Prompt file not found: {_src_path(prompt_file)}")
    return prompt_file


def _write_qa_report(qa_report, output_folder, file_name):
    qa_path = os.path.join(output_folder, f"{file_name}.qa.json")
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(qa_report, f, indent=2)
    print(f"QA report saved to {qa_path}")
    return qa_path


def process_pdf(pdf_path, prompt_file=None, pattern=None, workers=OPENAI_BATCH_WORKERS):
    prompt_file = _resolve_prompt(pdf_path, prompt_file=prompt_file, pattern=pattern)
    print(f"Using prompt: {prompt_file}")

    beams, folder, name, qa_report = run_structured_pipeline(
        pdf_path,
        prompt_file,
        max_workers=workers,
    )
    save_beams(beams, folder, name)
    _write_qa_report(qa_report, folder, name)
    return beams


def _pdf_paths(single_pdf=None):
    if single_pdf:
        return [os.path.abspath(single_pdf)]

    if not os.path.isdir(INPUT_DIR):
        return []

    return [
        os.path.join(INPUT_DIR, filename)
        for filename in os.listdir(INPUT_DIR)
        if filename.lower().endswith(".pdf")
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Extract RCC beam schedules using GPT-4.1 mini vision and strict JSON."
    )
    parser.add_argument("--pdf", help="Path to one PDF. Defaults to every PDF in input/.")
    parser.add_argument("--pattern", type=int, help="Known pattern number, e.g. 3.")
    parser.add_argument("--prompt-file", help="Prompt file in src/, e.g. prompt_3.txt.")
    parser.add_argument(
        "--workers",
        type=int,
        default=OPENAI_BATCH_WORKERS,
        help="Concurrent OpenAI requests per PDF page after slicing.",
    )
    parser.add_argument(
        "--pdf-workers",
        type=int,
        default=1,
        help="Concurrent PDFs to process from input/. Keep low to avoid rate limits.",
    )
    args = parser.parse_args()

    pdfs = _pdf_paths(args.pdf)
    if not pdfs:
        print("No PDF files found.")
        return

    workers = max(1, args.workers)
    pdf_workers = max(1, min(args.pdf_workers, len(pdfs)))

    if pdf_workers == 1:
        for pdf_path in pdfs:
            process_pdf(
                pdf_path,
                prompt_file=args.prompt_file,
                pattern=args.pattern,
                workers=workers,
            )
        return

    print(f"Processing {len(pdfs)} PDF(s) with {pdf_workers} PDF worker(s).")
    with ThreadPoolExecutor(max_workers=pdf_workers) as executor:
        futures = {
            executor.submit(
                process_pdf,
                pdf_path,
                args.prompt_file,
                args.pattern,
                workers,
            ): pdf_path
            for pdf_path in pdfs
        }

        for future in as_completed(futures):
            pdf_path = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"Failed: {os.path.basename(pdf_path)}: {exc}")


if __name__ == "__main__":
    main()
