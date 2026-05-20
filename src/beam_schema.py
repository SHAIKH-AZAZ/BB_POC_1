"""
Strict JSON schema and base prompt for GPT-4.1 mini beam extraction.
"""

BEAM_EXTRACTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "beams": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "beam_id": {"type": "string"},
                    "size": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "width": {"type": ["number", "null"]},
                            "depth": {"type": ["number", "null"]},
                            "length": {"type": ["number", "null"]},
                        },
                        "required": ["width", "depth", "length"],
                    },
                    "reinforcement": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "stirrups": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "dia": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "spacing": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["dia", "spacing"],
                    },
                },
                "required": ["beam_id", "size", "reinforcement", "stirrups"],
            },
        },
    },
    "required": ["beams"],
}


BASE_BEAM_EXTRACTION_PROMPT = """
You are an RCC beam reinforcement schedule extractor.

Extract only rows from the beam schedule table.
Ignore floor plan labels, grid labels, remarks, titles, and empty rows.

Rules:
- Keep grouped beam IDs exactly as written.
- Do not split grouped IDs.
- Size format: 200 x 600 means width 200, depth 600, length null.
- Reinforcement: collect bottom straight, bottom curtail, top straight,
  top extra left, and top extra right when those columns exist.
- Skip --- and blanks.
- Normalize reinforcement to quantity-Tdiameter, e.g. 2 T16 -> 2-T16.
- Stirrups like T8@150C/C become dia T8 and spacing 150 C/C.
- Return only valid JSON matching the supplied schema.
""".strip()
