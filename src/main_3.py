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
    with open(os.path.join(os.path.dirname(__file__), "prompt_3.txt"), "r") as f:
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
        item = item.replace(" ", "")
        item = item.replace("TT", "T")

        # Convert 2T16 → 2-T16
        if "T" in item and "-" not in item:
            parts = item.split("T")
            if len(parts) == 2 and parts[0].isdigit():
                item = f"{parts[0]}-T{parts[1]}"

        cleaned.add(item)

    def sort_key(x):
        try:
            qty, dia = x.split("-T")
            return (int(dia), int(qty))
        except:
            return (999, 999)

    return sorted(list(cleaned), key=sort_key)


# ==============================
# STRICT REINFORCEMENT FILTER
# ==============================

def strict_filter_reinforcement(beam):

    """
    Removes clearly unrelated reinforcement entries
    by eliminating impossible quantities for Pattern-3.
    """

    valid = []

    for r in beam["reinforcement"]:

        try:
            qty, dia = r.split("-T")
            qty = int(qty)
            dia = int(dia)

            # Pattern-3 beams usually 2 or 3 bars only
            if qty <= 4 and dia <= 32:
                valid.append(r)

        except:
            continue

    return valid


# ==============================
# CLEAN STIRRUPS
# ==============================

def clean_stirrups(stirrups):

    dia_list = stirrups.get("dia", [])
    spacing_list = stirrups.get("spacing", [])

    cleaned_dia = set()
    cleaned_spacing = set()

    for d in dia_list:
        if not d:
            continue

        d = d.strip().upper().replace(" ", "")

        # Fix 8T → T8
        if d.endswith("T") and d[:-1].isdigit():
            d = f"T{d[:-1]}"

        cleaned_dia.add(d)

    for s in spacing_list:
        if not s:
            continue

        s = s.upper().replace(" ", "")
        s = s.replace("C/C", "")
        s = s.replace("C", "")

        if s.isdigit():
            cleaned_spacing.add(f"{s} C/C")

    return {
        "dia": sorted(list(cleaned_dia)),
        "spacing": sorted(list(cleaned_spacing))
    }


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
        result = extract_from_image(img_path, prompt)

        try:
            parsed = json.loads(result)
            if "beams" in parsed:
                all_beams.extend(parsed["beams"])
        except:
            print("⚠ JSON parse failed")

    # ==============================
    # DEDUPLICATE BY BEAM ID
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

            existing["reinforcement"] += beam.get("reinforcement", [])
            existing["stirrups"]["dia"] += beam.get("stirrups", {}).get("dia", [])
            existing["stirrups"]["spacing"] += beam.get("stirrups", {}).get("spacing", [])

    # ==============================
    # FINAL CLEANUP
    # ==============================

    final_beams = []

    for beam in unique_beams.values():

        # Normalize reinforcement
        beam["reinforcement"] = normalize_reinforcement(
            beam.get("reinforcement", [])
        )

        # Strict filter (prevents cross-row bleeding)
        beam["reinforcement"] = strict_filter_reinforcement(beam)

        # Clean stirrups
        beam["stirrups"] = clean_stirrups(
            beam.get("stirrups", {})
        )

        final_beams.append(beam)

    final_output = {"beams": final_beams}

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
        print("⚠ No PDF files found.")
        return

    for pdf in pdf_files:
        process_pdf(os.path.join(INPUT_DIR, pdf))


if __name__ == "__main__":
    main()