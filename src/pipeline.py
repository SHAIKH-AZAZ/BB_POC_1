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
from tqdm import tqdm

from config import INPUT_DIR, OUTPUT_DIR
from pdf_to_images import convert_pdf_to_images
from vision_extractor import extract_from_image, extract_with_tools, extract_with_reflection
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
