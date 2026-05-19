"""
pipeline.py
===========
Shared extraction pipeline used by all main_N.py pattern handlers.

Flow:
    PDF → images → smart_slice (model picks direction+count)
        → extract_with_reflection per slice
        → collect raw beams
        → caller does pattern-specific post-processing
        → save_beams()
"""

import os
import json
from tqdm import tqdm

from config import INPUT_DIR, OUTPUT_DIR
from pdf_to_images import convert_pdf_to_images
from vision_extractor import extract_from_image, extract_with_reflection
from image_slicer import smart_slice, delete_temp_slices
from utils import safe_parse_json


# ─────────────────────────────────────────────────────────────────────────────
# FILE LOADER
# ─────────────────────────────────────────────────────────────────────────────

def _load_file(filename):
    """Load a text file from the same directory as the caller's src/ folder."""
    src_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(src_dir, filename), "r", encoding="utf-8") as f:
        return f.read()


# ─────────────────────────────────────────────────────────────────────────────
# CORE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(pdf_path, prompt_file, max_rounds=1, output_dir=OUTPUT_DIR):
    """
    Run the standard extraction pipeline for one PDF.

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

    prompt        = _load_file(prompt_file)
    verify_prompt = _load_file("verify_prompt.txt")
    all_beams = []

    for img_path in tqdm(image_paths):
        # Model decides direction (horizontal / vertical) and slice count
        slice_paths = smart_slice(img_path, suggest_fn=extract_from_image)

        for slice_img in slice_paths:
            result = extract_with_reflection(
                slice_img,
                extract_prompt=prompt,
                verify_prompt_template=verify_prompt,
                max_rounds=max_rounds,
            )
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
