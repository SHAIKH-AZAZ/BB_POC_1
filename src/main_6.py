import os
import json
from tqdm import tqdm

from config import INPUT_DIR, OUTPUT_DIR
from pdf_to_images import convert_pdf_to_images
from image_slicer import smart_slice, delete_temp_slices
from vision_extractor import extract_from_image, extract_with_reflection


# ==============================
# LOAD PROMPT
# ==============================

def load_prompt():
    with open(os.path.join(os.path.dirname(__file__), "prompt_6.txt"), "r") as f:
        return f.read()


def load_verify_prompt():
    with open(os.path.join(os.path.dirname(__file__), "verify_prompt.txt"), "r") as f:
        return f.read()



# ==============================
# NORMALIZE REINFORCEMENT
# ==============================

def normalize_reinforcement(reinf_list):
    cleaned = set()

    for item in reinf_list:
        if not item:
            continue

        item = item.strip().upper()

        if item == "-":
            continue

        parts = item.split("+")
        for p in parts:
            p = p.strip()
            if p and p != "-":
                cleaned.add(p)

    return sorted(list(cleaned))


# ==============================
# PROCESS PDF
# ==============================

def process_pdf(pdf_path):
    file_name = os.path.splitext(os.path.basename(pdf_path))[0]

    file_output_folder = os.path.join(OUTPUT_DIR, file_name)
    os.makedirs(file_output_folder, exist_ok=True)

    print(f"\n📄 Converting {file_name}.pdf to images...")
    image_paths = convert_pdf_to_images(pdf_path, file_output_folder)

    prompt = load_prompt()
    verify_prompt = load_verify_prompt()
    all_beams = []

    for img_path in tqdm(image_paths):
        slice_paths = smart_slice(img_path, suggest_fn=extract_from_image)
        for slice_img in slice_paths:

            result = extract_with_reflection(
                slice_img,
                extract_prompt=prompt,
                verify_prompt_template=verify_prompt,
                max_rounds=1,
            )
            result = result.strip()

            if result.startswith("{"):
                try:
                    parsed = json.loads(result)
                    if "beams" in parsed:
                        all_beams.extend(parsed["beams"])
                except:
                    print("⚠ Invalid JSON:", img_path)
            else:
                print("⚠ Non-JSON response skipped:", img_path)

        delete_temp_slices(slice_paths)

    # ==============================
    # CLEAN PER BEAM (NO CROSS MERGE)
    # ==============================

    for beam in all_beams:
        beam["reinforcement"] = normalize_reinforcement(beam["reinforcement"])
        beam["stirrups"]["dia"] = sorted(list(set(beam["stirrups"]["dia"])))
        beam["stirrups"]["spacing"] = sorted(list(set(beam["stirrups"]["spacing"])))

    final_output = {"beams": all_beams}

    output_file = os.path.join(file_output_folder, f"{file_name}.json")

    with open(output_file, "w") as f:
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