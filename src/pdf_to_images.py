import fitz  # pymupdf
import os

def convert_pdf_to_images(pdf_path, output_folder):
    doc = fitz.open(pdf_path)
    image_paths = []

    for page_number, page in enumerate(doc):
        pix = page.get_pixmap(dpi=300)
        image_path = os.path.join(
            output_folder,
            f"page_{page_number + 1}.png"
        )
        pix.save(image_path)
        image_paths.append(image_path)

    return image_paths