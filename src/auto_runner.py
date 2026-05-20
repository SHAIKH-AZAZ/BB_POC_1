"""
Step 0  →  ONE model call with full page image
           Model returns:
             region:    {"x1": 0.62, "y1": 0.02, "x2": 0.88, "y2": 0.52}
             direction: "horizontal"
             num_slices: 3

Step 1  →  Crop to region + 2% padding
           Result: ~1500×2000 px (was 5617×3974) — only the table, nothing else

Step 2  →  Slice the cropped region horizontally x3
           3 strips, each ~670 px tall, full table width

Step 3  →  extract_with_reflection on each of the 3 strips
           Every token the model sees is pure table content
"""


import os
import importlib

from config import INPUT_DIR, OUTPUT_DIR
from pattern_detector import detect_pattern


def run_pattern(pattern_number, pdf_path):

    module_name = f"main_{pattern_number}"
    module = importlib.import_module(module_name)

    print(f"Detected Pattern: {pattern_number}")
    print(f"Running {module_name}.py")

    module.process_pdf(pdf_path)


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdf_files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print("No PDF files found.")
        return

    for pdf in pdf_files:

        pdf_path = os.path.join(INPUT_DIR, pdf)
        temp_folder = os.path.join(OUTPUT_DIR, "temp_detection")

        os.makedirs(temp_folder, exist_ok=True)

        print(f"\nDetecting pattern for {pdf}...")

        pattern_number = detect_pattern(pdf_path, temp_folder)

        run_pattern(pattern_number, pdf_path)


if __name__ == "__main__":
    main()
