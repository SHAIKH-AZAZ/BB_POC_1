import io
import json
import base64
from PIL import Image
from openai import OpenAI
from beam_schema import BASE_BEAM_EXTRACTION_PROMPT, BEAM_EXTRACTION_SCHEMA
from config import OPENAI_API_KEY, OPENAI_IMAGE_DETAIL, OPENAI_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def encode_image(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode("utf-8")


def _image_content(base64_image, detail=OPENAI_IMAGE_DETAIL):
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{base64_image}",
            "detail": detail,
        },
    }


def _crop_image_b64(image_path, x1, y1, x2, y2):
    """
    Crop the image to normalized coordinates (0.0–1.0) and return as base64 PNG.

    The crop is upscaled to a minimum of 1200px on the longest side so that
    small, dense table text becomes readable when returned to the model.

    Args:
        image_path : source image
        x1, y1    : top-left corner (fractions of width/height)
        x2, y2    : bottom-right corner (fractions of width/height)

    Returns:
        base64-encoded PNG string
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # Clamp to valid range
    x1 = max(0.0, min(1.0, float(x1)))
    y1 = max(0.0, min(1.0, float(y1)))
    x2 = max(0.0, min(1.0, float(x2)))
    y2 = max(0.0, min(1.0, float(y2)))

    if x2 <= x1:
        x2 = min(1.0, x1 + 0.05)
    if y2 <= y1:
        y2 = min(1.0, y1 + 0.05)

    left   = int(x1 * w)
    top    = int(y1 * h)
    right  = int(x2 * w)
    bottom = int(y2 * h)

    # Enforce a minimum crop size (avoid 1-pixel crops)
    right  = max(right,  left   + 20)
    bottom = max(bottom, top    + 20)

    cropped = img.crop((left, top, right, bottom))

    # Upscale so smallest-readable text (≈8pt) fills enough pixels
    target_px = 1200
    longest   = max(cropped.size)
    if longest < target_px:
        scale    = target_px / longest
        new_size = (int(cropped.width * scale), int(cropped.height * scale))
        cropped  = cropped.resize(new_size, Image.LANCZOS)

    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# TOOL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

BEAM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "think",
            "description": (
                "Use this FIRST to reason step-by-step about the table before extracting. "
                "Identify: column headers, number of data rows, beam ID patterns "
                "(simple or grouped), reinforcement formats, stirrup notation, "
                "and any cells that look ambiguous or too small to read clearly. "
                "If anything is unclear, use zoom_region to get a closer look."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Your observations and reasoning about the table."
                    }
                },
                "required": ["reasoning"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "zoom_region",
            "description": (
                "Crop and zoom into a specific part of the current image to read it more clearly. "
                "Use this whenever: text looks too small to read, a cell value is ambiguous, "
                "you want to confirm a beam ID, reinforcement value, or stirrup notation. "
                "Coordinates are fractions of the full image size (0.0 = left/top, 1.0 = right/bottom). "
                "The zoomed image will be returned so you can read it and continue extraction."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "x1": {
                        "type": "number",
                        "description": "Left edge as fraction of image width (0.0–1.0)"
                    },
                    "y1": {
                        "type": "number",
                        "description": "Top edge as fraction of image height (0.0–1.0)"
                    },
                    "x2": {
                        "type": "number",
                        "description": "Right edge as fraction of image width (0.0–1.0)"
                    },
                    "y2": {
                        "type": "number",
                        "description": "Bottom edge as fraction of image height (0.0–1.0)"
                    },
                    "reason": {
                        "type": "string",
                        "description": "What you are trying to read or verify in this region."
                    }
                },
                "required": ["x1", "y1", "x2", "y2", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_beam",
            "description": (
                "Record one extracted beam row. "
                "Call this once per data row in the table. "
                "Do NOT call it for header rows, title rows, or empty rows. "
                "Use zoom_region first if any value is unclear."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "beam_id": {
                        "type": "string",
                        "description": (
                            "Exact beam label from the BEAM MARKED / BEAM NO column. "
                            "Copy character-for-character including all commas, "
                            "hyphens, underscores, and letter suffixes. "
                            "Grouped IDs like 'AB3,4,8,9,40,41,44,45' are ONE string."
                        )
                    },
                    "width": {
                        "type": ["number", "null"],
                        "description": "Beam width in mm (number only, no units). null if not given."
                    },
                    "depth": {
                        "type": ["number", "null"],
                        "description": "Beam depth in mm (number only, no units). null if not given."
                    },
                    "length": {
                        "type": ["number", "null"],
                        "description": "Beam length/span in mm. null if no span column."
                    },
                    "reinforcement": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "All rebar values from ALL reinforcement columns, "
                            "normalized to quantity-Tdiameter. "
                            "Examples: '3-T20', '2-T16', '5-T25'. "
                            "Skip '---' or blank cells. Remove duplicates."
                        )
                    },
                    "stirrups_dia": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Stirrup bar diameter(s). Format: 'T8', 'T10'. "
                            "If written as 'T8@150C/C', extract 'T8' here. "
                            "If legs given, use '2L-T8' format."
                        )
                    },
                    "stirrups_spacing": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Stirrup spacing(s). Format: '150 C/C', '200 C/C'. "
                            "Capture all unique spacings (UPTO L/4 and REST separately). "
                            "Skip '---' or blank."
                        )
                    }
                },
                "required": [
                    "beam_id",
                    "reinforcement",
                    "stirrups_dia",
                    "stirrups_spacing"
                ]
            }
        }
    }
]


# ─────────────────────────────────────────────────────────────────────────────
# TOOL-BASED EXTRACTION  (primary method)
# ─────────────────────────────────────────────────────────────────────────────

def extract_with_tools(image_path, prompt_text, max_iterations=80):
    """
    Tool-augmented extraction with chain-of-thought and image zoom.

    The model has three tools:
      think(reasoning)             — step-by-step reasoning scratchpad
      zoom_region(x1,y1,x2,y2)    — crop + upscale any part of the image
      add_beam(beam_id, ...)       — record one beam row

    Typical flow the model follows:
      1. think()          — survey the table: headers, row count, notation
      2. zoom_region()    — zoom into any cell that looks ambiguous or small
      3. add_beam()×N    — record each row after zooming as needed

    Falls back to extract_from_image() if no beams are collected.
    """
    base64_image = encode_image(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                _image_content(base64_image),
            ],
        }
    ]

    collected_beams = []
    zoom_count      = 0
    iteration       = 0

    while iteration < max_iterations:
        iteration += 1

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=BEAM_TOOLS,
            tool_choice="auto",
            temperature=0,
        )

        msg           = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        messages.append(msg)

        if not msg.tool_calls:
            break

        tool_results = []

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            # ── think ────────────────────────────────────────────────────────
            if fn_name == "think":
                snippet = args.get("reasoning", "")[:200]
                print(f"  💭 think: {snippet}...")
                result_content = "Reasoning noted. Use zoom_region if anything is unclear, then proceed with add_beam calls."

            # ── zoom_region ──────────────────────────────────────────────────
            elif fn_name == "zoom_region":
                x1     = args.get("x1", 0.0)
                y1     = args.get("y1", 0.0)
                x2     = args.get("x2", 1.0)
                y2     = args.get("y2", 1.0)
                reason = args.get("reason", "")
                zoom_count += 1

                print(f"  🔍 zoom_region [{zoom_count}] ({x1:.2f},{y1:.2f})→({x2:.2f},{y2:.2f}) — {reason[:80]}")

                cropped_b64 = _crop_image_b64(image_path, x1, y1, x2, y2)

                # Return cropped image directly in the tool response.
                # GPT-4.1 vision supports multimodal tool content.
                result_content = [
                    {
                        "type": "text",
                        "text": (
                            f"Zoomed region (x: {x1:.2f}–{x2:.2f}, y: {y1:.2f}–{y2:.2f}). "
                            f"Reason: {reason}. "
                            "Read the image below carefully, then continue."
                        )
                    },
                    _image_content(cropped_b64),
                ]

            # ── add_beam ─────────────────────────────────────────────────────
            elif fn_name == "add_beam":
                beam = {
                    "beam_id": args.get("beam_id", ""),
                    "size": {
                        "width":  args.get("width"),
                        "depth":  args.get("depth"),
                        "length": args.get("length"),
                    },
                    "reinforcement": args.get("reinforcement", []),
                    "stirrups": {
                        "dia":     args.get("stirrups_dia", []),
                        "spacing": args.get("stirrups_spacing", []),
                    },
                }
                collected_beams.append(beam)
                result_content = (
                    f"Beam '{beam['beam_id']}' recorded "
                    f"({len(collected_beams)} total so far). "
                    "Continue with the next row."
                )
                print(f"  ✔  add_beam: {beam['beam_id']}")

            else:
                result_content = f"Unknown tool '{fn_name}' — ignored."

            tool_results.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result_content,
            })

        messages.extend(tool_results)

        if finish_reason == "stop":
            break

    # ── assemble result ───────────────────────────────────────────────────────
    if collected_beams:
        print(f"  ✅ {len(collected_beams)} beam(s) collected  |  {zoom_count} zoom(s) used.")
        return json.dumps({"beams": collected_beams})

    print("  ⚠ Tool extraction returned 0 beams — falling back to direct extraction.")
    return extract_from_image(image_path, prompt_text)


# ─────────────────────────────────────────────────────────────────────────────
# PLAIN SINGLE-PASS EXTRACTION  (region detector + fallback)
# ─────────────────────────────────────────────────────────────────────────────

def extract_from_image(image_path, prompt_text):
    """Single-pass extraction: image + prompt → raw response string."""
    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    _image_content(base64_image),
                ],
            }
        ],
        temperature=0,
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────────────────────────────────────
# STRICT JSON-SCHEMA EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_structured_from_image(image_path, prompt_text=None, schema=None):
    """
    Single-pass extraction with OpenAI strict JSON schema output.

    This is the production path when callers want the model response itself to
    be schema-valid before deterministic post-processing runs.
    """
    base64_image = encode_image(image_path)
    prompt = prompt_text or BASE_BEAM_EXTRACTION_PROMPT
    json_schema = schema or BEAM_EXTRACTION_SCHEMA

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    _image_content(base64_image),
                ],
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "beam_schedule_extraction",
                "strict": True,
                "schema": json_schema,
            },
        },
        temperature=0,
    )

    return response.choices[0].message.content


# ─────────────────────────────────────────────────────────────────────────────
# REFLECTION LOOP  (optional, kept for backward compatibility)
# ─────────────────────────────────────────────────────────────────────────────

def extract_with_reflection(image_path, extract_prompt, verify_prompt_template, max_rounds=1):
    """
    Two-pass extraction with self-correction.

    Round 1 uses extract_with_tools (think + zoom + add_beam).
    Subsequent rounds send the previous JSON + verify prompt back with the
    original image for correction.
    """
    current_output = extract_with_tools(image_path, extract_prompt)

    if max_rounds < 1:
        return current_output

    base64_image = encode_image(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": extract_prompt},
                _image_content(base64_image),
            ],
        },
        {"role": "assistant", "content": current_output},
    ]

    for round_num in range(1, max_rounds + 1):
        try:
            json.loads(current_output)
        except json.JSONDecodeError:
            print(f"  ⚠ Reflection round {round_num}: skipped (output not valid JSON)")
            break

        verify_text = verify_prompt_template.replace("{extracted_json}", current_output)

        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": verify_text},
                _image_content(base64_image),
            ],
        })

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0,
        )
        corrected = response.choices[0].message.content
        messages.append({"role": "assistant", "content": corrected})
        current_output = corrected
        print(f"  🔄 Reflection round {round_num} complete.")

    return current_output
