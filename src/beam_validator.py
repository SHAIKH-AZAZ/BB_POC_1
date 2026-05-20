"""
Deterministic validation and normalization for model-extracted beam data.
"""

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from utils import deduplicate_beams, normalize_reinforcement


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return list(value.values())
    return [value]


def _blank_to_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value or value in {"-", "--", "---", "NULL", "NONE"}:
            return None
        value = re.sub(r"[^0-9.]", "", value)
        if not value:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_number(value: float | None) -> int | float | None:
    if value is None:
        return None
    if float(value).is_integer():
        return int(value)
    return value


def _dia_values(raw: Any) -> list[str]:
    values = []
    for item in _as_list(raw):
        if item is None:
            continue
        text = str(item).strip().upper().replace(" ", "")
        if not text or text in {"-", "--", "---"}:
            continue
        if "C/C" in text and "T" not in text and "L" not in text:
            continue
        if "@" in text:
            text = text.split("@", 1)[0]

        legs = re.search(r"(\d+)L[-_]*T?(\d+)", text)
        if legs:
            values.append(f"{legs.group(1)}L-T{legs.group(2)}")
            continue

        dia = re.search(r"T?(\d+)", text)
        if dia:
            values.append(f"T{dia.group(1)}")
    return values


def _spacing_values(raw: Any) -> list[str]:
    values = []
    for item in _as_list(raw):
        if item is None:
            continue
        text = str(item).strip().upper().replace(" ", "")
        if not text or text in {"-", "--", "---"}:
            continue

        source_had_at = "@" in text
        if source_had_at:
            text = text.split("@", 1)[1]

        if "T" in text and not source_had_at and "C/C" not in text:
            continue

        for number in re.findall(r"\d+(?:\.\d+)?", text):
            normalized = _normalize_number(float(number))
            values.append(f"{normalized} C/C")
    return values


def normalize_stirrups(stirrups: dict[str, Any] | None) -> dict[str, list[str]]:
    stirrups = stirrups or {}
    dia = set()
    spacing = set()

    for raw in _as_list(stirrups.get("dia")):
        dia.update(_dia_values(raw))
        spacing.update(_spacing_values(raw))

    for raw in _as_list(stirrups.get("spacing")):
        spacing.update(_spacing_values(raw))

    return {
        "dia": sorted(dia),
        "spacing": sorted(spacing),
    }


class BeamSize(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width: float | None = None
    depth: float | None = None
    length: float | None = None

    @field_validator("width", "depth", "length", mode="before")
    @classmethod
    def parse_dimension(cls, value: Any) -> float | None:
        return _blank_to_none(value)

    @field_validator("width", "depth", "length")
    @classmethod
    def simplify_dimension(cls, value: float | None) -> int | float | None:
        return _normalize_number(value)


class BeamStirrups(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dia: list[str] = Field(default_factory=list)
    spacing: list[str] = Field(default_factory=list)

    @field_validator("dia", "spacing", mode="before")
    @classmethod
    def force_list(cls, value: Any) -> list[Any]:
        return _as_list(value)


class Beam(BaseModel):
    model_config = ConfigDict(extra="forbid")

    beam_id: str
    size: BeamSize
    reinforcement: list[str] = Field(default_factory=list)
    stirrups: BeamStirrups = Field(default_factory=BeamStirrups)

    @field_validator("beam_id")
    @classmethod
    def clean_beam_id(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError("beam_id is required")
        return cleaned

    @field_validator("reinforcement", mode="before")
    @classmethod
    def force_reinforcement_list(cls, value: Any) -> list[Any]:
        return _as_list(value)

    @field_validator("reinforcement")
    @classmethod
    def clean_reinforcement(cls, value: list[str]) -> list[str]:
        return normalize_reinforcement(value)


class BeamExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    beams: list[Beam] = Field(default_factory=list)


def validate_beam_payload(payload: dict[str, Any] | str) -> dict[str, list[dict[str, Any]]]:
    if isinstance(payload, str):
        payload = json.loads(payload)

    parsed = BeamExtraction.model_validate(payload)
    beams = parsed.model_dump()["beams"]

    for beam in beams:
        beam["stirrups"] = normalize_stirrups(beam.get("stirrups"))

    beams = deduplicate_beams(beams, normalize_fn=normalize_reinforcement)
    for beam in beams:
        beam["stirrups"] = normalize_stirrups(beam.get("stirrups"))

    return {"beams": beams}


def build_qa_report(beams: list[dict[str, Any]]) -> dict[str, Any]:
    issues = []
    seen = set()

    for index, beam in enumerate(beams, start=1):
        beam_id = (beam.get("beam_id") or "").strip()
        if not beam_id:
            issues.append({"row": index, "issue": "missing beam_id"})
            continue
        if beam_id in seen:
            issues.append({"row": index, "beam_id": beam_id, "issue": "duplicate beam_id"})
        seen.add(beam_id)

        size = beam.get("size") or {}
        if size.get("width") is None and size.get("depth") is None:
            issues.append({"row": index, "beam_id": beam_id, "issue": "missing size"})

        if not beam.get("reinforcement") and not (beam.get("stirrups") or {}).get("dia"):
            issues.append({
                "row": index,
                "beam_id": beam_id,
                "issue": "no reinforcement or stirrup data",
            })

    return {
        "beam_count": len(beams),
        "issue_count": len(issues),
        "issues": issues,
    }
