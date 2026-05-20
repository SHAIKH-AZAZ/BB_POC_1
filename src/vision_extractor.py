import json
import base64
from openai import OpenAI
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def encode_image(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode("utf-8")


def _image_content(base64_image):
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
    }


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
                "and any cells that look ambiguous or blank."
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
            "name": "add_beam",
            "description": (
                "Record one extracted beam row. "
                "Call this once per data row in the table. "
                "Do NOT call it for header rows, title rows, or empty rows."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "beam_id": {
                        "type": "string",
                        "description": (
                            "Exact beam label from the BEAM MARKED / BEAM NO column. "
                            "Copy character-for-character including all commas, "
                            "hyphens, and letter suffixes. "
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
                        "description": "Beam length/span in mm. null if the table has no span column."
                    },
                    "reinforcement": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "All rebar values from ALL reinforcement columns, "
                            "normalized to quantity-Tdiameter format. "
                            "Examples: '3-T20', '2-T16', '5-T25'. "
                            "Skip cells that are '---' or blank. "
                            "Remove duplicates."
                        )
                    },
                    "stirrups_dia": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Stirrup bar diameter(s). "
                            "Format: 'T8', 'T10', 'T12'. "
                            "If stirrup is written as 'T8@150C/C', extract 'T8' here. "
                            "If legs are given, use '2L-T8' format."
                        )
                    },
                    "stirrups_spacing": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Stirrup spacing value(s). "
                            "Format: '150 C/C', '200 C/C'. "
                            "Extract all unique spacings (e.g. one for UPTO L/4, one for REST). "
                            "Skip '---' or blank cells."
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
    Tool-augmented extraction with chain-of-thought.

    Flow:
        1. Model receives image + extraction prompt + two tools.
        2. Model calls think(reasoning) to inspect the table structure first.
        3. Model calls add_beam(...) once per data row.
        4. Pipeline collects every add_beam call and assembles the JSON.

    The think tool acts as a scratchpad — the model writes out what it sees
    (column headers, row count, ambiguous cells) before committing to data.
    This prevents the model from hallucinating values it hasn't "looked at" yet.

    Falls back to extract_from_image() if no beams are collected via tools.

    Args:
        image_path     : path to image slice
        prompt_text    : extraction prompt (same as before)
        max_iterations : safety limit on the tool-call loop

    Returns:
        JSON string  {"beams": [...]}
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
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=BEAM_TOOLS,
            tool_choice="auto",
            temperature=0,
        )

        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # Append assistant turn to history (preserves tool_calls field)
        messages.append(msg)

        # No tool calls → model decided it's done
        if not msg.tool_calls:
            break

        # ── Process each tool call in this turn ──────────────────────────────
        tool_results = []
        for tc in msg.tool_calls:
            fn_name = tc.function.name

            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            if fn_name == "think":
                reasoning_snippet = args.get("reasoning", "")[:200]
                print(f"  💭 think: {reasoning_snippet}...")
                result_content = "Reasoning noted. Proceed with add_beam calls."

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
                result_content = f"Beam '{beam['beam_id']}' recorded ({len(collected_beams)} total)."
                print(f"  ✔ add_beam: {beam['beam_id']}")

            else:
                result_content = f"Unknown tool '{fn_name}' — ignored."

            tool_results.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_content,
            })

        messages.extend(tool_results)

        # OpenAI signals no more tool calls
        if finish_reason == "stop":
            break

    # ── Assemble result ───────────────────────────────────────────────────────
    if collected_beams:
        print(f"  ✅ Tool extraction: {len(collected_beams)} beam(s) collected.")
        return json.dumps({"beams": collected_beams})

    # Fallback — tool loop produced nothing (blank image, wrong region, etc.)
    print("  ⚠ Tool extraction returned 0 beams — falling back to direct extraction.")
    return extract_from_image(image_path, prompt_text)


# ─────────────────────────────────────────────────────────────────────────────
# PLAIN SINGLE-PASS EXTRACTION  (used by region detector + fallback)
# ─────────────────────────────────────────────────────────────────────────────

def extract_from_image(image_path, prompt_text):
    """
    Single-pass extraction: send image + prompt, return raw model response string.
    Used by the region/slice detector and as a fallback.
    """
    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
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
# REFLECTION LOOP  (kept for backward compatibility / optional use)
# ─────────────────────────────────────────────────────────────────────────────

def extract_with_reflection(image_path, extract_prompt, verify_prompt_template, max_rounds=1):
    """
    Two-pass extraction with self-correction.

    Round 1  →  extract_with_tools (chain-of-thought + structured tool calls)
    Round 2+ →  image + previous JSON + verify_prompt  →  corrected JSON

    The first pass now uses tools instead of a plain prompt, so the model
    already reasons carefully. The reflection pass catches any remaining errors.
    """
    # ── Round 1: tool-based extraction ───────────────────────────────────────
    current_output = extract_with_tools(image_path, extract_prompt)

    # ── Rounds 2+: reflection & correction ───────────────────────────────────
    if max_rounds < 1:
        return current_output

    base64_image = encode_image(image_path)

    # Seed the conversation for reflection with the tool-extraction result
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

        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": verify_text},
                    _image_content(base64_image),
                ],
            }
        )

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0,
        )
        corrected = response.choices[0].message.content
        messages.append({"role": "assistant", "content": corrected})
        current_output = corrected
        print(f"  🔄 Reflection round {round_num} complete.")

    return current_output
