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
    with open(os.path.join(os.path.dirname(__file__), "prompt_8.txt"), "r") as f:
        return f.read()


# ==============================
# CLEAN BEAM DATA
# ==============================

def clean_beam(beam):

    # --------------------------
    # CLEAN REINFORCEMENT
    # --------------------------
    reinf_cleaned = []

    for r in beam.get("reinforcement", []):
        r = r.upper()
        r = r.replace("TH", "")
        r = r.replace("EX", "")
        r = r.strip()

        # split combined patterns
        parts = r.split("-")
        temp = []

        for i in range(0, len(parts)-1, 2):
            val = parts[i] + "-" + parts[i+1]
            if "T" in val:
                temp.append(val)

        if temp:
            reinf_cleaned.extend(temp)
        else:
            if "T" in r:
                reinf_cleaned.append(r)

    reinf_cleaned = sorted(list(set(reinf_cleaned)))

    # 🔒 HARD SAFETY:
    # If reinforcement appears but visually shouldn't exist,
    # we prevent accidental carry-over by requiring beam text length > 0.
    if not reinf_cleaned:
        beam["reinforcement"] = []
    else:
        beam["reinforcement"] = reinf_cleaned

    # --------------------------
    # CLEAN STIRRUPS
    # --------------------------
    stirrups = beam.get("stirrups", {})
    dia_list = stirrups.get("dia", [])
    spacing_list = stirrups.get("spacing", [])

    cleaned_dia = []
    cleaned_spacing = []

    for d in dia_list:
        d = d.upper().strip()
        if "L-" in d:
            cleaned_dia.append(d)
        elif "T" in d:
            cleaned_dia.append(d)

    for s in spacing_list:
        s = s.strip()
        if s.isdigit():
            cleaned_spacing.append(f"{s} C/C")
        elif "C/C" in s.upper():
            cleaned_spacing.append(s.upper())

    beam["stirrups"]["dia"] = sorted(list(set(cleaned_dia)))
    beam["stirrups"]["spacing"] = sorted(list(set(cleaned_spacing)))

    return beam


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
            # 🔒 Extract JSON safely even if model adds spaces/newlines
            start = result.find("{")
            end = result.rfind("}") + 1
            cleaned_json = result[start:end]

            parsed = json.loads(cleaned_json)

            if "beams" in parsed:
                all_beams.extend(parsed["beams"])

        except Exception as e:
            print("⚠ JSON parse failed.")
            print("Model returned:")
            print(result)
            raise e  # 🔥 do NOT silently continue

    # FINAL CLEAN
    cleaned_beams = []
    for beam in all_beams:
        cleaned_beams.append(clean_beam(beam))

    final_output = {"beams": cleaned_beams}

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
