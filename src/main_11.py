import os
import json
import re
from tqdm import tqdm

from config import INPUT_DIR, OUTPUT_DIR
from pdf_to_images import convert_pdf_to_images
from vision_extractor import extract_from_image


# ==============================
# LOAD PROMPT
# ==============================

def load_prompt():
    with open(os.path.join(os.path.dirname(__file__), "prompt_11.txt"), "r") as f:
        return f.read()


# ==============================
# SAFE JSON LOADER
# ==============================

def safe_json_load(text):

    try:
        return json.loads(text)
    except:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match:
        try:
            return json.loads(match.group())
        except:
            pass

    return None


# ==============================
# CLEAN REINFORCEMENT
# ==============================

def clean_reinforcement_list(reinf_list):

    clean_set = set()

    for r in reinf_list:
        if not r:
            continue

        # 🔥 FIX: handle int/string
        r = str(r).upper()

        r = re.sub(r"\(.*?\)", "", r)
        r = re.sub(r"\[.*?\]", "", r)
        r = r.replace("+", " ")

        # correct format
        matches = re.findall(r"\d+\s*-\s*T\d+", r)
        for m in matches:
            clean_set.add(m.replace(" ", ""))

        # fix missing dash
        matches2 = re.findall(r"\b(\d+)T(\d+)\b", r)
        for m in matches2:
            clean_set.add(f"{m[0]}-T{m[1]}")

    return sorted(list(clean_set))


# ==============================
# CLEAN STIRRUPS (FINAL FIX)
# ==============================

def clean_stirrups(stirrups):

    dia = set()
    spacing = set()

    combined = stirrups.get("dia", []) + stirrups.get("spacing", [])

    for item in combined:

        if item is None:
            continue

        # 🔥 FIX: handle int/string safely
        item = str(item).upper().strip()

        # ==============================
        # EMBEDDED (T10-100)
        # ==============================
        matches = re.findall(r"T(\d+)-(\d{2,3})(?!\d)", item)

        for d, s in matches:
            dia.add(f"T{d}")
            spacing.add(f"{s} C/C")

        # ==============================
        # STANDALONE DIA
        # ==============================
        if re.fullmatch(r"T\d+", item):
            dia.add(item)

        # ==============================
        # STANDALONE SPACING
        # ==============================
        if re.fullmatch(r"\d+", item):
            val = int(item)

            # realistic stirrup spacing
            if 75 <= val <= 300:
                spacing.add(f"{val} C/C")

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

        parsed = safe_json_load(result)

        # 🔥 retry if failed
        if not parsed:
            print("⚠ JSON parse failed → retrying...")
            result = extract_from_image(img_path, prompt)
            parsed = safe_json_load(result)

        if parsed and "beams" in parsed:
            all_beams.extend(parsed["beams"])
        else:
            print("❌ Still failed to extract valid JSON")

    # ==============================
    # DEDUPLICATION
    # ==============================

    unique = {}

    for beam in all_beams:
        beam_id = beam.get("beam_id")
        if beam_id:
            unique[beam_id] = beam

    final_beams = []

    for beam in unique.values():

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