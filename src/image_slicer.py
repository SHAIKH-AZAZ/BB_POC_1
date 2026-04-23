from PIL import Image
import os
import uuid

def slice_image_horizontally(image_path, num_slices=8):
    """
    Splits image into horizontal strips.
    Returns list of temporary slice paths.
    """

    img = Image.open(image_path)
    width, height = img.size

    slice_height = height // num_slices
    slice_paths = []

    for i in range(num_slices):
        top = i * slice_height
        bottom = (i + 1) * slice_height if i < num_slices - 1 else height

        cropped = img.crop((0, top, width, bottom))

        temp_name = f"temp_slice_{uuid.uuid4().hex}.png"
        cropped.save(temp_name)

        slice_paths.append(temp_name)

    return slice_paths


def delete_temp_slices(slice_paths):
    for path in slice_paths:
        if os.path.exists(path):
            os.remove(path)