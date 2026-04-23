import os
import json
from tqdm import tqdm

from config import INPUT_DIR, OUTPUT_DIR
from pdf_to_images import convert_pdf_to_images
from vision_extractor import extract_from_image


def load_prompt():
    with open(os.path.join(os.path.dirname(__file__), "prompt_10.txt"), "r") as f:
        return f.read()


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

            beams = parsed.get("beams", [])

            # 🔥 retry if empty
            if not beams:
                print("⚠ Empty output → retrying once...")

                result = extract_from_image(img_path, prompt)
                parsed = json.loads(result)
                beams = parsed.get("beams", [])

            all_beams.extend(beams)

        except Exception as e:
            print("⚠ JSON parse failed:", e)

    # ==============================
    # DEDUPLICATION
    # ==============================

    unique = {}
    for beam in all_beams:
        beam_id = beam.get("beam_id")
        if beam_id:
            unique[beam_id] = beam

    final_beams = list(unique.values())

    # 🔥 safety check
    if not final_beams:
        print("❌ ERROR: No beams extracted. Check prompt/image.")

    final_output = {"beams": final_beams}

    output_file = os.path.join(file_output_folder, f"{file_name}.json")

    with open(output_file, "w") as f:
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