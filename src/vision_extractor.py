import json
import base64
from openai import OpenAI
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)


def encode_image(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode("utf-8")


def extract_from_image(image_path, prompt_text):
    """
    Single-pass extraction: send image + prompt, return raw model response string.
    """
    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content


def extract_with_reflection(image_path, extract_prompt, verify_prompt_template, max_rounds=1):
    """
    Two-pass extraction with self-correction loop.

    Flow:
        Round 1 (Extract):
            Image + extract_prompt  →  model  →  raw JSON

        Round 2+ (Reflect & Correct):
            Image + previous JSON + verify_prompt  →  model  →  corrected JSON

    Args:
        image_path            : path to the image slice
        extract_prompt        : the standard extraction prompt (same as extract_from_image)
        verify_prompt_template: string containing {extracted_json} placeholder — loaded
                                from verify_prompt.txt at call time
        max_rounds            : how many correction passes to run (default 1 = one verify)

    Returns:
        Final raw JSON string (after all reflection rounds).
    """
    base64_image = encode_image(image_path)

    # ── Round 1: Initial extraction ───────────────────────────────────────────
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": extract_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
            ],
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0,
    )
    current_output = response.choices[0].message.content

    # Add model reply to conversation history so next round has full context
    messages.append({"role": "assistant", "content": current_output})

    # ── Rounds 2+: Reflection & Correction ───────────────────────────────────
    for round_num in range(1, max_rounds + 1):
        # Validate current output is parseable before sending for review
        try:
            json.loads(current_output)
        except json.JSONDecodeError:
            # Can't review unparseable output — return as-is
            print(f"  ⚠ Reflection round {round_num}: skipped (output not valid JSON yet)")
            break

        # Build the verify message: inject the current JSON + image again
        verify_text = verify_prompt_template.replace("{extracted_json}", current_output)

        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": verify_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
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
