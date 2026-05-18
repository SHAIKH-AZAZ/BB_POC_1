import os
import json
from tqdm import tqdm

from config import INPUT_DIR, OUTPUT_DIR
from pdf_to_images import convert_pdf_to_images
from vision_extractor import extract_from_image
from image_slicer import (
    slice_image_horizontally,
    delete_temp_slices,
    get_dynamic_slice_count,
)


# ==============================
# LOAD PROMPT
# ==============================

def load_prompt():
    with open(os.path.join(os.path.dirname(__file__), "prompt_1.txt"), "r") as f:
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

        # Fix duplicate T issue
        item = item.replace("TT", "T")

        # Ensure dash exists (convert 7T25 → 7-T25)
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
# PROCESS SINGLE PDF
# ==============================

def process_pdf(pdf_path):
    file_name = os.path.splitext(os.path.basename(pdf_path))[0]

    # Each file gets its own output folder
    file_output_folder = os.path.join(OUTPUT_DIR, file_name)
    os.makedirs(file_output_folder, exist_ok=True)

    print(f"\n📄 Converting {file_name}.pdf to images...")
    image_paths = convert_pdf_to_images(pdf_path, file_output_folder)

    prompt = load_prompt()
    all_beams = []

    for img_path in tqdm(image_paths):

        # Ask the model how many horizontal slices this page needs.
        num_slices = get_dynamic_slice_count(
            img_path,
            suggest_fn=extract_from_image,
            fallback_slices=6,
        )
        print(f"Using {num_slices} slice(s) for {os.path.basename(img_path)}")
        slice_paths = slice_image_horizontally(img_path, num_slices=num_slices)

        for slice_img in slice_paths:
            result = extract_from_image(slice_img, prompt)

            try:
                parsed = json.loads(result)

                if "beams" in parsed:
                    all_beams.extend(parsed["beams"])

            except Exception as e:
                print("⚠ JSON parse failed for slice:", slice_img)

        # 🧹 Delete temporary slices
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

            # 🔥 Merge reinforcement safely
            merged_reinf = existing["reinforcement"] + beam["reinforcement"]
            existing["reinforcement"] = normalize_reinforcement(merged_reinf)

            # 🔥 Merge stirrups dia
            existing["stirrups"]["dia"] = list(
                set(existing["stirrups"]["dia"] + beam["stirrups"]["dia"])
            )

            # 🔥 Merge stirrups spacing
            existing["stirrups"]["spacing"] = list(
                set(existing["stirrups"]["spacing"] + beam["stirrups"]["spacing"])
            )

            unique_beams[beam_id] = existing

    # ==============================
    # FINAL CLEANUP PASS
    # ==============================

    for beam in unique_beams.values():
        beam["reinforcement"] = normalize_reinforcement(beam["reinforcement"])
        beam["stirrups"]["dia"] = sorted(list(set(beam["stirrups"]["dia"])))
        beam["stirrups"]["spacing"] = sorted(list(set(beam["stirrups"]["spacing"])))

    final_output = {"beams": list(unique_beams.values())}

    # Save JSON inside same folder as images
    output_file = os.path.join(file_output_folder, f"{file_name}.json")

    with open(output_file, "w") as f:
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
