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


def convert_pdf_to_images(pdf_path, output_folder, dpi=300, max_pixels=50_000_000):
    """
    Convert every PDF page to a PNG.

    max_pixels cap (default 50 million):
      - Normal A4 at 300 DPI is ~8.7 MP -- well within limit, no change.
      - Wide A0 sheet (Pattern 10 etc.) is auto-capped to ~170 DPI,
        giving a ~7000x7000 px image (~15 MB) instead of ~360 MB at 600 DPI.
        This keeps the image small enough for PIL and the vision API.
    """
    doc = fitz.open(pdf_path)
    image_paths = []

    for page_number, page in enumerate(doc):
        render_dpi = _fit_dpi_to_pixel_limit(page, dpi, max_pixels)
        if render_dpi < dpi:
            print(f"  Page {page_number+1}: large page detected, "
                  f"rendering at {render_dpi} DPI (requested {dpi})")
        pix = page.get_pixmap(dpi=render_dpi)
        image_path = os.path.join(
            output_folder,
            f"page_{page_number + 1}.png",
        )
        pix.save(image_path)
        image_paths.append(image_path)
        print(f"  Page {page_number+1}: {pix.width}x{pix.height} px saved")

    return image_paths
