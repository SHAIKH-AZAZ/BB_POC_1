import os
import json
from tqdm import tqdm

from config import INPUT_DIR, OUTPUT_DIR
from pdf_to_images import convert_pdf_to_images
from vision_extractor import extract_from_image


# ==============================
# LOAD PROMPT
# ==============================

def load_prompt():
    with open(os.path.join(os.path.dirname(__file__), "prompt_5.txt"), "r", encoding="utf-8") as f:
        return f.read()


# ==============================
# SAFE JSON PARSER
# ==============================

def safe_parse_json(result):

    if not result:
        return None

    result = result.strip()

    if not result.startswith("{"):
        start = result.find("{")
        end = result.rfind("}")
        if start != -1 and end != -1:
            result = result[start:end+1]
        else:
            return None

    try:
        return json.loads(result)
    except:
        return None


# ==============================
# PROCESS PDF (NO SLICING)
# ==============================

def process_pdf(pdf_path):

    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    file_output_folder = os.path.join(OUTPUT_DIR, file_name)
    os.makedirs(file_output_folder, exist_ok=True)

    print(f"\n📄 Converting {file_name}.pdf to images...")
    image_paths = convert_pdf_to_images(pdf_path, file_output_folder)

    prompt = load_prompt()

    all_beams = []

    for img_path in tqdm(image_paths):

        result = extract_from_image(img_path, prompt)
        parsed = safe_parse_json(result)

        if parsed and "beams" in parsed:
            all_beams.extend(parsed["beams"])

    # Remove empty beams (like B6/B7 null rows)
    cleaned_beams = []
    for beam in all_beams:
        if beam["size"]["width"] is None and not beam["reinforcement"]:
            continue
        cleaned_beams.append(beam)

    final_output = {
        "beams": cleaned_beams
    }

    output_file = os.path.join(file_output_folder, f"{file_name}.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print(f"✅ Output saved to {output_file}")


# ==============================
# MAIN
# ==============================

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdf_files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print("⚠ No PDF files found.")
        return

    for pdf in pdf_files:
        process_pdf(os.path.join(INPUT_DIR, pdf))


if __name__ == "__main__":
    main()