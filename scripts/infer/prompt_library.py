from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_prompt_templates(path: str | Path) -> dict[str, Any]:
    import yaml

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Prompt template config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    categories = raw.get("categories", [])
    templates_by_id: dict[str, dict[str, Any]] = {}
    for cat in categories:
        cat_name = str(cat.get("name", ""))
        for item in cat.get("templates", []):
            template_id = str(item.get("id", "")).strip()
            if not template_id:
                continue
            templates_by_id[template_id] = {
                "id": template_id,
                "category": cat_name,
                "instruction_template": str(item.get("instruction_template", "")).strip(),
                "input_template": str(item.get("input_template", "")).strip(),
                "usage_notes": str(item.get("usage_notes", "")).strip(),
            }
    return {"categories": categories, "templates_by_id": templates_by_id}


def parse_slots_json(slots_json: str | None) -> dict[str, str]:
    if not slots_json:
        return {}
    data = json.loads(slots_json)
    if not isinstance(data, dict):
        raise ValueError("--template-slots-json must be a JSON object.")
    return {str(k): str(v) for k, v in data.items()}


def render_prompt(template: dict[str, str], slots: dict[str, str]) -> tuple[str, str]:
    try:
        instruction = template["instruction_template"].format(**slots)
    except KeyError as exc:
        missing = exc.args[0]
        raise KeyError(f"Missing slot for instruction_template: {missing}") from exc
    try:
        user_input = template["input_template"].format(**slots)
    except KeyError as exc:
        missing = exc.args[0]
        raise KeyError(f"Missing slot for input_template: {missing}") from exc
    return instruction, user_input

