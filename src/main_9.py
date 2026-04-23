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
    with open(os.path.join(os.path.dirname(__file__), "prompt_9.txt"), "r") as f:
        return f.read()


# ==============================
# CLEAN REINFORCEMENT LIST
# ==============================

# ==============================
# CLEAN REINFORCEMENT LIST (FIXED)
# ==============================

def clean_reinforcement_list(reinf_list):

    import re

    clean_set = set()

    for r in reinf_list:
        if not r:
            continue

        r = r.strip().upper()

        # remove unwanted parts
        r = re.sub(r"\(.*?\)", "", r)   # remove (0.1L)
        r = re.sub(r"\[.*?\]", "", r)   # remove [LAP]
        r = r.replace("+", " ")

        # 🔥 FIX: normalize formats

        # case 1: already correct (2-T16)
        matches = re.findall(r"\d+\s*-\s*T\d+", r)

        for m in matches:
            clean_set.add(m.replace(" ", ""))

        # case 2: missing dash (2T16 → 2-T16)
        matches_no_dash = re.findall(r"\b(\d+)\s*T(\d+)\b", r)

        for m in matches_no_dash:
            clean_set.add(f"{m[0]}-T{m[1]}")

    return sorted(list(clean_set))


# ==============================
# CLEAN STIRRUPS
# ==============================

def clean_stirrups(stirrups):

    dia = set()
    spacing = set()

    for d in stirrups.get("dia", []):
        if d:
            dia.add(d.strip().upper())

    for s in stirrups.get("spacing", []):
        if s:
            s = s.strip().upper()

            if "C/C" not in s:
                s_clean = s.replace(" ", "")
                if s_clean.isdigit():
                    s = f"{s_clean} C/C"

            spacing.add(s)

    return {
        "dia": sorted(list(dia)),
        "spacing": sorted(list(spacing))
    }


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
    # DEDUPLICATION
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

            unique_beams[beam_id] = existing

    # ==============================
    # FINAL CLEAN
    # ==============================

    final_beams = []

    for beam in unique_beams.values():

        beam["reinforcement"] = clean_reinforcement_list(
            beam.get("reinforcement", [])
        )

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