import json
from pdf_to_images import convert_pdf_to_images
from vision_extractor import extract_from_image


def detect_pattern(pdf_path, temp_folder):

    image_paths = convert_pdf_to_images(pdf_path, temp_folder)

    if not image_paths:
        raise Exception("No image generated for detection.")

    first_image = image_paths[0]

    classification_prompt = """
You are an expert at identifying RCC beam schedule HEADER patterns.

IMPORTANT:
You must look ONLY at the HEADER structure.
Ignore all reinforcement values.
Ignore numbers.
Ignore row content.
Focus ONLY on column names.

There are EXACTLY 8 patterns.
Return ONLY one number from 1 to 8.
No explanation.
No extra text.
Only the number.

========================================================
PATTERN 1 (STRICT)
========================================================

Return 1 ONLY if:

- Simple table with:
  - BEAM NUMBERS
  - SIZE (WIDTH, DEPTH)
  - BOTTOM REINFORCEMENT (LEFT, MID, RIGHT)
  - TOP REINFORCEMENT (LEFT, MID, RIGHT)
  - SHEAR STIRRUPS (LEFT, MID, RIGHT)

AND MUST NOT contain:

- A, B, C, D1(mm), D2(mm)
- E, G
- Any labeled reinforcement like "TOP REINF A", "BOTTOM REINF B"

If any structured reinforcement labels (A/B/C/etc.) exist → DO NOT RETURN 1

========================================================
PATTERN 2
========================================================
Header contains:
- BEAM MARKED
- SIZE (B x D/d)   ← MUST contain "/d"
- Bottom (Straight)
- Bottom (Curtail)
- Top (Straight)
- Ex. Top (Straight)
- Top (Extra Over Support) Left
- Top (Extra Over Support) Right
- Stirrups (Upto L/4)
- Stirrups (Rest)

CRITICAL IDENTIFIER:
SIZE column contains "/d"

CRITICAL IDENTIFIER (STRICT):

Return 2 ONLY if:

- The SIZE column header explicitly contains "/d"
- Example: "B x D/d"

STRICT CONDITIONS:
- "/d" must be clearly visible inside the SIZE HEADER
- It must NOT be inferred
- It must NOT come from row values
- It must NOT come from OCR noise

If "/d" is not clearly visible in header → DO NOT RETURN 2

========================================================
PATTERN 3
========================================================
Header contains:
- Beam Marked
- Size (B x D)     ← MUST NOT contain "/d"
- Bottom (Straight)
- Bottom (Curtail)
- Top (Straight)
- Top (Extra Over Support - Left)
- Top (Extra Over Support - Right)
- Stirrups (Upto L/4)
- Stirrups (Rest)

CRITICAL IDENTIFIER:
SIZE column does NOT contain "/d"

If NO "/d" and above header exists → RETURN 3

CRITICAL IDENTIFIER:

Return 3 ONLY if:

- Size is "B x D"
- AND NO "/d" is present in header
- AND structure matches:
  Bottom (Straight)
  Bottom (Curtail)
  Top (Straight)
  Top (Extra Over Support - Left)
  Top (Extra Over Support - Right)

========================================================
PATTERN 4
========================================================
STRICT RULE:

Return 4 ONLY if ALL of the following exist:

- A
- B
- C
- D1(mm)
- G
- E
- D2(mm)

If ANY of these are missing → DO NOT RETURN 4

If only A and B exist → it is Pattern 5 (NOT 4)

========================================================
PATTERN 5
========================================================

Return 5 ONLY if:

- A exists
- B exists

AND NONE of the following exist:

- C
- D1(mm)
- D2(mm)
- G
- E

If any of these exist → it is NOT Pattern 5

========================================================
PATTERN 6
========================================================
Header contains:
- BEAM NO
BEAM SIZE - BREADTH (B)  -> BREADTH(B) AND WIDTH(W) ARE SAME.
BEAM SIZE - DEPTH (D)

BOTTOM REINFORCEMENT:
    - LEFT SUPPORT (Layer 1)
    - MID SPAN (Layer 1)
    - MID SPAN (Layer 2)
    - RIGHT SUPPORT (Layer 1)

TOP REINFORCEMENT:
    - LEFT SUPPORT (Layer 1)
    - LEFT SUPPORT (Layer 2)
    - MID SPAN (Layer 1)
    - RIGHT SUPPORT (Layer 1)
    - RIGHT SUPPORT (Layer 2)

STIRRUPS:
- NO OF LEGS
- DIA
- LEFT SUPPORT SPACING
- MID SPACING
- RIGHT SUPPORT SPACING

- SIDE FACE REINFORCEMENT ON EACH FACE

If GRID ID does NOT exist and above header exists → RETURN PATTERN 6

========================================================
PATTERN 7
========================================================
Header contains:
- BEAM NO
BEAM SIZE - BREADTH (B)  -> BREADTH(B) AND WIDTH(W) ARE SAME.
BEAM SIZE - DEPTH (D)
GRID ID

In this pattern GRID ID is a unique identifier. Rest everything is same as Pattern 6.

BOTTOM REINFORCEMENT:
    - LEFT SUPPORT (Layer 1)
    - MID SPAN (Layer 1)
    - MID SPAN (Layer 2)
    - RIGHT SUPPORT (Layer 1)

TOP REINFORCEMENT:
    - LEFT SUPPORT (Layer 1)
    - LEFT SUPPORT (Layer 2)
    - MID SPAN (Layer 1)
    - RIGHT SUPPORT (Layer 1)
    - RIGHT SUPPORT (Layer 2)

STIRRUPS:
- NO OF LEGS
- DIA
- LEFT SUPPORT SPACING
- MID SPACING
- RIGHT SUPPORT SPACING

- SIDE FACE REINFORCEMENT ON EACH FACE

========================================================
PATTERN 7 (STRICT)
========================================================

Return 7 ONLY if a column header EXACTLY contains:

"GRID ID"

STRICT CONDITIONS:
- Must be clearly visible as a COLUMN TITLE
- Must NOT be inferred
- Must NOT be guessed
- Must NOT be part of row text

If "GRID ID" is not clearly visible → DO NOT RETURN 7
→ RETURN 6 instead

========================================================
PATTERN 6 vs PATTERN 7 (CRITICAL DISTINCTION)
========================================================

These two patterns are almost identical.

The ONLY difference:

PATTERN 7 has a separate column named:
GRID ID

PATTERN 6 does NOT have GRID ID column.

STRICT RULE:

If the header contains a column explicitly labeled:
"GRID ID"
→ RETURN 7

If NO column named "GRID ID" exists
→ It CANNOT be Pattern 7
→ RETURN 6 (if other reinforcement structure matches)

Do NOT assume GRID ID.
It must be clearly visible in header.

========================================================
PATTERN 8
========================================================
NOT A TABLE.

Strip beam detail drawing.

BEAM NO. AND SIZE
LEGGED
NOS.
SPAC.c/c
STRP.DIA.

Contains:
- Beam segments like CB1a, CB1b, CB1c
- Size written in brackets like (300x600)
- Reinforcement written ABOVE each beam segment
- No tabular grid header

========================================================
PATTERN 9
========================================================
Header contains:


BEAM NOs.
BEAM TOP LEVEL
SIZE - B
SIZE - D
BOTTOM REINFORCEMENT
@LEFT SUPPORT
MID SPAN
@RIGHT SUPPORT
TOP REINFORCEMENT
@LEFT SUPPORT
MID SPAN
@RIGHT SUPPORT
SIDE FACE REINF. IN EACH FACE
STIRRUPS
LEFT SUP.
DIA
@C/C
DIST
CENTRE
DIA
@C/C
RIGHT SUP.
DIA
@C/C
DIST
BEAM TYPE
REMARKS

CRITICAL IDENTIFIER:
Presence of "BEAM TOP LEVEL" column is mandatory.
If this exists → ALWAYS RETURN 9

========================================================
PATTERN 10
========================================================
CRITICAL IDENTIFIER:
Very simple header with ONLY:
- BEAM NO.
- SIZE
- LEVEL

No reinforcement breakdown columns.

If reinforcement columns are missing → RETURN 10

========================================================
PATTERN 11
========================================================
Header contains:

BEAM NO.
FLOOR
GRADE
RCC BEAM SIZE
LVL.
B
D
STEEL MEMBER SIZE
b
d
tf
tw
EMBEDMENT LENGTH "Le"
LEFT SUPPORT 
RIGHT SUPPORT
BOTTOM REINF.
TOP REINF.
SFR (E.F.)
STIRRUPS
EXTRA BOUNDARY REINF. (E.F.)
REMARKS

========================================================
PATTERN 12
========================================================
Header contains:

TYPE
LINTEL LENGTH
SIZE (BXD)
TOP REIN.
BOTT. REIN.
RINGS

If LINTEL LENGTH is present → RETURN 12

========================================================
PATTERN 13
========================================================
Header contains:

TYPE
SIZE (BXD)
TOP REIN.
BOTT. REIN.
RINGS

If LINTEL LENGTH is NOT present → RETURN 13

========================================================
FINAL DECISION PRIORITY (VERY IMPORTANT)
========================================================

1) If header contains:
   "BEAM TOP LEVEL"
   → RETURN 9

2) If header contains:
   "LEVEL" AND "SIZE" AND only simple columns
   → RETURN 10

3) If header contains ONLY:
   - A
   - B
   AND does NOT contain:
   - C, D1, D2, E, G
   → RETURN 5

4) If you see "/d" inside size column
   → RETURN 2

5) Pattern 3 should be returned ONLY if:
   - Bottom (Straight)
   - Bottom (Curtail)
   - Top (Straight)
   - Top (Extra Over Support - Left)
   - Top (Extra Over Support - Right)
   AND
   NO columns like:
   - BEAM TOP LEVEL
   - SIDE FACE REINF
   - BEAM TYPE

6) Pattern 1 should be returned ONLY if NO labeled reinforcement
   (A/B/C/D1/D2/E/G) exists

7) If unsure → DO NOT default to 1 or 3

Match full header structure carefully.

Return ONLY the number.
"""

    result = extract_from_image(first_image, classification_prompt)
    result = result.strip()

    if not result.isdigit():
        raise Exception(f"Pattern detection failed. Model returned: {result}")

    return int(result)
