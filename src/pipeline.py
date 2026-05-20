"""
pipeline.py
===========
Shared extraction pipeline used by all main_N.py pattern handlers.

Flow:
    PDF → images → smart_slice (model picks region + direction + count)
        → extract_with_tools per slice  (think + add_beam tool calls)
        → collect raw beams
        → caller does pattern-specific post-processing
        → save_beams()
"""

import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from config import INPUT_DIR, OPENAI_BATCH_WORKERS, OUTPUT_DIR
from pdf_to_images import convert_pdf_to_images
from beam_validator import build_qa_report, validate_beam_payload
from vision_extractor import (
    extract_from_image,
    extract_structured_from_image,
    extract_with_reflection,
    extract_with_tools,
)
from image_slicer import smart_slice, delete_temp_slices
from utils import safe_parse_json


# ─────────────────────────────────────────────────────────────────────────────
# FILE LOADER
# ─────────────────────────────────────────────────────────────────────────────

def _load_file(filename):
    """Load a text file from the same directory as this module (src/)."""
    src_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(src_dir, filename), "r", encoding="utf-8") as f:
        return f.read()


# ─────────────────────────────────────────────────────────────────────────────
# CORE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(pdf_path, prompt_file, output_dir=OUTPUT_DIR):
    """
    Run the standard extraction pipeline for one PDF.

    Each image slice is processed with extract_with_tools():
      1. Model calls think() to reason about table structure.
      2. Model calls add_beam() once per data row.
      3. Pipeline collects all add_beam results.

    Returns:
        all_beams          : list of raw beam dicts (not yet deduplicated)
        file_output_folder : path where images + JSON will be stored
        file_name          : base filename without extension
    """
    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    file_output_folder = os.path.join(output_dir, file_name)
    os.makedirs(file_output_folder, exist_ok=True)

    print(f"\n📄 Converting {file_name}.pdf to images...")
    image_paths = convert_pdf_to_images(pdf_path, file_output_folder)

    prompt    = _load_file(prompt_file)
    all_beams = []

    for img_path in tqdm(image_paths):
        # Model decides region, direction (horizontal/vertical), and slice count
        slice_paths = smart_slice(img_path, suggest_fn=extract_from_image)

        for slice_img in slice_paths:
            # Tool-based extraction: model uses think() then add_beam() per row.
            # Falls back to plain extraction if no tool calls are made.
            result = extract_with_tools(slice_img, prompt)

            parsed = safe_parse_json(result)
            if parsed and "beams" in parsed:
                all_beams.extend(parsed["beams"])
            elif result:
                print(f"  ⚠ Could not parse JSON from slice: {os.path.basename(slice_img)}")

        delete_temp_slices(slice_paths)

    return all_beams, file_output_folder, file_name


def _extract_structured_slice(slice_img, prompt, max_retries=2):
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            result = extract_structured_from_image(slice_img, prompt)
            parsed = safe_parse_json(result)
            if parsed and "beams" in parsed:
                return validate_beam_payload(parsed)["beams"]
            if result:
                print(f"  ⚠ Could not parse strict JSON from slice: {os.path.basename(slice_img)}")
            return []
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(2 ** attempt)

    raise last_error


def _extract_structured_slices(slice_paths, prompt, max_workers):
    if not slice_paths:
        return []

    worker_count = max(1, min(int(max_workers), len(slice_paths)))

    if worker_count == 1:
        all_beams = []
        for slice_img in slice_paths:
            all_beams.extend(_extract_structured_slice(slice_img, prompt))
        return all_beams

    print(f"  Processing {len(slice_paths)} slice(s) with {worker_count} worker(s)...")

    results_by_index = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_extract_structured_slice, slice_img, prompt): (index, slice_img)
            for index, slice_img in enumerate(slice_paths)
        }

        for future in as_completed(futures):
            index, slice_img = futures[future]
            try:
                results_by_index[index] = future.result()
            except Exception as exc:
                results_by_index[index] = []
                print(f"  ⚠ Extraction failed for {os.path.basename(slice_img)}: {exc}")

    all_beams = []
    for index in range(len(slice_paths)):
        all_beams.extend(results_by_index.get(index, []))
    return all_beams


def run_structured_pipeline(
    pdf_path,
    prompt_file,
    output_dir=OUTPUT_DIR,
    max_workers=OPENAI_BATCH_WORKERS,
):
    """
    Run the production OpenAI workflow:
      PDF -> page PNGs -> schedule crop/slices -> strict JSON schema extraction
      -> Pydantic validation/normalization -> QA report.

    Returns:
        beams              : validated beam dicts
        file_output_folder : path where images + JSON will be stored
        file_name          : base filename without extension
        qa_report          : deterministic sanity-check report
    """
    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    file_output_folder = os.path.join(output_dir, file_name)
    os.makedirs(file_output_folder, exist_ok=True)

    print(f"\n📄 Converting {file_name}.pdf to images...")
    image_paths = convert_pdf_to_images(pdf_path, file_output_folder)

    prompt = _load_file(prompt_file)
    all_beams = []

    for img_path in tqdm(image_paths):
        slice_paths = smart_slice(img_path, suggest_fn=extract_from_image)

        all_beams.extend(
            _extract_structured_slices(
                slice_paths,
                prompt,
                max_workers=max_workers,
            )
        )

        delete_temp_slices(slice_paths)

    validated_payload = validate_beam_payload({"beams": all_beams})
    beams = validated_payload["beams"]
    qa_report = build_qa_report(beams)

    return beams, file_output_folder, file_name, qa_report


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def save_beams(beams, file_output_folder, file_name):
    """Write the final beam list to JSON and print the output path."""
    output_file = os.path.join(file_output_folder, f"{file_name}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"beams": beams}, f, indent=2)
    print(f"✅ Output saved to {output_file}")
    return output_file


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY HELPER
# ─────────────────────────────────────────────────────────────────────────────

def run_all(process_pdf_fn, input_dir=INPUT_DIR, output_dir=OUTPUT_DIR):
    """
    Common main() implementation shared by all pattern handlers.
    Scans input_dir for PDFs and calls process_pdf_fn on each.
    """
    os.makedirs(output_dir, exist_ok=True)

    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("⚠ No PDF files found in input folder.")
        return

    for pdf in pdf_files:
        process_pdf_fn(os.path.join(input_dir, pdf))
