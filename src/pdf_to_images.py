import math
import os

import fitz  # pymupdf


def _fit_dpi_to_pixel_limit(page, requested_dpi, max_pixels):
    width_pt = float(page.rect.width)
    height_pt = float(page.rect.height)
    scale = requested_dpi / 72.0
    estimated_pixels = width_pt * scale * height_pt * scale

    if estimated_pixels <= max_pixels:
        return int(requested_dpi)

    fitted_scale = math.sqrt(max_pixels / (width_pt * height_pt))
    fitted_dpi = int(fitted_scale * 72)
    return max(36, min(int(requested_dpi), fitted_dpi))


def convert_pdf_to_images(pdf_path, output_folder, dpi=600, max_pixels=120_000_000):
    doc = fitz.open(pdf_path)
    image_paths = []

    for page_number, page in enumerate(doc):
        render_dpi = _fit_dpi_to_pixel_limit(page, dpi, max_pixels)
        pix = page.get_pixmap(dpi=render_dpi)
        image_path = os.path.join(
            output_folder,
            f"page_{page_number + 1}.png",
        )
        pix.save(image_path)
        image_paths.append(image_path)

    return image_paths
