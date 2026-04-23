import os
import json
from tqdm import tqdm

from config import INPUT_DIR, OUTPUT_DIR
from pdf_to_images import convert_pdf_to_images
from vision_extractor import extract_from_image
from image_slicer import slice_image_horizontally, delete_temp_slices


# ==============================
# LOAD PROMPT
# ==============================

def load_prompt():
    with open(os.path.join(os.path.dirname(__file__), "prompt_4.txt"), "r", encoding="utf-8") as f:
        return f.read()


# ==============================
# NORMALIZE REINFORCEMENT
# ==============================

def normalize_reinforcement(reinf_list):
    """
    Cleans, normalizes and removes duplicates.
    Ensures strict format: quantity-Tdiameter
    """

    cleaned = set()

    for item in reinf_list:
        if not item:
            continue

        item = item.strip().upper()
        item = item.replace(" ", "")
        item = item.replace("TT", "T")

        # Convert 7T25 → 7-T25
        if "T" in item and "-" not in item:
            parts = item.split("T")
            if len(parts) == 2 and parts[0].isdigit():
                item = f"{parts[0]}-T{parts[1]}"

        cleaned.add(item)

    # Sort by diameter then quantity
    def sort_key(x):
        try:
            qty, dia = x.split("-T")
            return (int(dia), int(qty))
        except:
            return (999, 999)

    return sorted(list(cleaned), key=sort_key)


# ==============================
# SAFE JSON PARSER
# ==============================

def safe_parse_json(result, slice_img):
    """
    Attempts to safely parse LLM output into JSON.
    Skips non-JSON responses.
    """

    if not result:
        return None

    result = result.strip()

    # If model added extra text before JSON, try extracting JSON part
    if not result.startswith("{"):
        start_index = result.find("{")
        end_index = result.rfind("}")

        if start_index != -1 and end_index != -1:
            result = result[start_index:end_index+1]
        else:
            print("⚠ Non-JSON response skipped:", slice_img)
            return None

    try:
        return json.loads(result)
    except Exception:
        print("⚠ Invalid JSON format from slice:", slice_img)
        return None


# ==============================
# PROCESS SINGLE PDF
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

        # 🔥 Use 3 slices (more stable)
        slice_paths = slice_image_horizontally(img_path, num_slices=3)

        for slice_img in slice_paths:

            result = extract_from_image(slice_img, prompt)

            parsed = safe_parse_json(result, slice_img)

            if parsed and "beams" in parsed:
                all_beams.extend(parsed["beams"])

        delete_temp_slices(slice_paths)

    # ==============================
    # MERGE & DEDUPLICATE BEAMS
    # ==============================

    unique_beams = {}

    for beam in all_beams:

        beam_id = beam.get("beam_id")
        if not beam_id:
            continue

        if beam_id not in unique_beams:
            unique_beams[beam_id] = beam
        else:
            existing = unique_beams[beam_id]

            # Merge reinforcement
            merged_reinf = existing.get("reinforcement", []) + beam.get("reinforcement", [])
            existing["reinforcement"] = normalize_reinforcement(merged_reinf)

            # Merge stirrup dia
            existing["stirrups"]["dia"] = list(
                set(existing["stirrups"].get("dia", []) + beam["stirrups"].get("dia", []))
            )

            # Merge stirrup spacing
            existing["stirrups"]["spacing"] = list(
                set(existing["stirrups"].get("spacing", []) + beam["stirrups"].get("spacing", []))
            )

            unique_beams[beam_id] = existing

    # ==============================
    # FINAL CLEANUP PASS
    # ==============================

    for beam in unique_beams.values():
        beam["reinforcement"] = normalize_reinforcement(
            beam.get("reinforcement", [])
        )

        beam["stirrups"]["dia"] = sorted(
            list(set(beam["stirrups"].get("dia", [])))
        )

        beam["stirrups"]["spacing"] = sorted(
            list(set(beam["stirrups"].get("spacing", [])))
        )

    final_output = {
        "beams": list(unique_beams.values())
    }

    output_file = os.path.join(file_output_folder, f"{file_name}.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print(f"✅ Output saved to {output_file}")


# ==============================
# MAIN ENTRY
# ==============================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdf_files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print("⚠ No PDF files found in input folder.")
        return

    for pdf in pdf_files:
        process_pdf(os.path.join(INPUT_DIR, pdf))


if __name__ == "__main__":
    main()