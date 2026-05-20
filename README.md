# Beam RCC Extraction

This project extracts RCC beam schedules from structural drawing PDFs with:

- PyMuPDF page rendering (`src/pdf_to_images.py`)
- GPT-assisted schedule region detection and cropping (`src/image_slicer.py`)
- GPT-4.1 mini vision extraction (`src/vision_extractor.py`)
- Strict JSON schema output (`src/beam_schema.py`)
- Pydantic validation, reinforcement normalization, stirrup cleanup, and QA checks (`src/beam_validator.py`)

## Tool mapping

The ChatGPT tools from the reference workflow map to repo code like this:

| Reference capability | Repo implementation |
| --- | --- |
| Uploaded-file parser / `file_search` | `src/table_extractor.py` for digital PDF text; PyMuPDF render fallback for visual extraction |
| Rendered PDF page image | `src/pdf_to_images.py` |
| Cropped table images | `src/image_slicer.py` model-selected region crop and slices |
| GPT-4.1 mini vision extraction | `src/vision_extractor.py` |
| Strict JSON schema | `src/beam_schema.py` plus `extract_structured_from_image()` |
| Manual rule validation | `src/beam_validator.py` and `src/utils.py` |
| QA pass | `src/beam_validator.py::build_qa_report()` and optional model reflection in `src/vision_extractor.py` |

## Environment

Create `.env` in the repo root:

```text
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4.1-mini
OPENAI_IMAGE_DETAIL=high
```

Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run strict OpenAI workflow

Put PDFs in `input/`, then run:

```powershell
.\.venv\Scripts\python.exe src\main_openai.py
```

Run one known pattern:

```powershell
.\.venv\Scripts\python.exe src\main_openai.py --pdf "input\20231215_RCC DETAILS OF ROOF & MIDLEVEL FLOOR (BLOCK-1A & 1B).pdf" --pattern 3
```

Run with a specific prompt:

```powershell
.\.venv\Scripts\python.exe src\main_openai.py --pdf "input\your-file.pdf" --prompt-file prompt_3.txt
```

Outputs are written to `output/<pdf-name>/<pdf-name>.json` with a deterministic QA report at `output/<pdf-name>/<pdf-name>.qa.json`.

## Existing runner

The original pattern runner is still available:

```powershell
.\.venv\Scripts\python.exe src\auto_runner.py
```

Use `main_openai.py` when you want strict schema output. Use `auto_runner.py` when you want the existing tool-call workflow that records beams through `add_beam`.
