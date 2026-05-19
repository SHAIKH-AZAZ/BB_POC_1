import os
import json
from tqdm import tqdm

from config import INPUT_DIR, OUTPUT_DIR
from pdf_to_images import convert_pdf_to_images
from image_slicer import smart_slice, delete_temp_slices
from vision_extractor import extract_from_image, extract_with_reflection


def load_prompt():
    with open(os.path.join(os.path.dirname(__file__), "prompt_12.txt"), "r", encoding="utf-8") as f:
        return f.read()


def load_verify_prompt():
    with open(os.path.join(os.path.dirname(__file__), "verify_prompt.txt"), "r") as f:
        return f.read()



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
            parsed = safe_parse_json(result)

            if parsed and "beams" in parsed:
                all_beams.extend(parsed["beams"])

        delete_temp_slices(slice_paths)

    cleaned_beams = []
    for beam in all_beams:

        if beam["size"]["width"] is None and not beam["reinforcement"]:
            continue

        # 🔹 REMOVE DUPLICATE REINFORCEMENT
        if "reinforcement" in beam and beam["reinforcement"]:
            beam["reinforcement"] = list(dict.fromkeys(beam["reinforcement"]))

        # 🔹 REMOVE DUPLICATE STIRRUPS DIA + SPACING
        if "stirrups" in beam:
            if "dia" in beam["stirrups"]:
                beam["stirrups"]["dia"] = list(dict.fromkeys(beam["stirrups"]["dia"]))

            if "spacing" in beam["stirrups"]:
                beam["stirrups"]["spacing"] = list(dict.fromkeys(beam["stirrups"]["spacing"]))

        cleaned_beams.append(beam)

    final_output = {
        "beams": cleaned_beams
    }

    output_file = os.path.join(file_output_folder, f"{file_name}.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print(f"✅ Output saved to {output_file}")


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