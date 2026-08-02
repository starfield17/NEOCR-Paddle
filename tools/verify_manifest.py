#!/usr/bin/env python3
"""Dependency-free semantic checks for the fixed Milestone 4A model manifest."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

REQUIRED_ROLES = {
    "documentOrientation",
    "documentUnwarping",
    "textLineOrientation",
    "textDetection",
    "textRecognition",
}
SHA256 = re.compile(r"^[0-9a-f]{64}$")
INPUT_PATTERNS = {"random", "coordinateGradient"}


def validate(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if data.get("schemaVersion") != 1:
        errors.append("schemaVersion must be 1")
    if data.get("packageKind") != "model":
        errors.append("packageKind must be model")

    models = data.get("models")
    if not isinstance(models, list):
        errors.append("models must be an array")
        models = []
    roles = [model.get("role") for model in models if isinstance(model, dict)]
    if len(models) != 5 or set(roles) != REQUIRED_ROLES or len(roles) != len(set(roles)):
        errors.append("models must contain each required role exactly once")

    for index, model in enumerate(models):
        label = f"models[{index}]"
        if not isinstance(model, dict):
            errors.append(f"{label} must be an object")
            continue
        parsed = urlparse(str(model.get("sourceUrl", "")))
        if parsed.scheme != "https" or parsed.netloc != "paddle-model-ecology.bj.bcebos.com":
            errors.append(f"{label}.sourceUrl must use the pinned official HTTPS host")
        if not SHA256.fullmatch(str(model.get("sha256", ""))):
            errors.append(f"{label}.sha256 must be 64 lowercase hexadecimal characters")
        shape = model.get("inputShape")
        if not isinstance(shape, list) or len(shape) != 4 or any(
            not isinstance(value, int) or value <= 0 for value in shape
        ):
            errors.append(f"{label}.inputShape must contain four positive integers")
        source_directory = str(model.get("sourceDirectory", ""))
        if not source_directory or Path(source_directory).name != source_directory:
            errors.append(f"{label}.sourceDirectory must be one relative path segment")
        input_pattern = model.get("inputPattern", "random")
        if input_pattern not in INPUT_PATTERNS:
            errors.append(f"{label}.inputPattern must be random or coordinateGradient")
        if (
            input_pattern == "coordinateGradient"
            and isinstance(shape, list)
            and len(shape) == 4
            and shape[1] != 3
        ):
            errors.append(f"{label}.coordinateGradient requires three input channels")

    defaults = data.get("defaults", {})
    if any(defaults.get(role) is not False for role in REQUIRED_ROLES - {"textDetection", "textRecognition"}):
        errors.append("all three enhancement stages must default to false")
    document = data.get("presets", {}).get("document", {})
    if any(document.get(role) is not True for role in REQUIRED_ROLES - {"textDetection", "textRecognition"}):
        errors.append("the document preset must enable all three enhancement stages")

    parity = data.get("parity", {})
    if parity.get("onnxOpset") != 17:
        errors.append("Milestone 4A must use ONNX opset 17")
    for name in ("absoluteTolerance", "relativeTolerance"):
        value = parity.get(name)
        if not isinstance(value, (int, float)) or value <= 0:
            errors.append(f"parity.{name} must be positive")

    if errors:
        raise ValueError("\n".join(errors))
    return data


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: verify_manifest.py <manifest.json>", file=sys.stderr)
        return 2
    try:
        manifest = validate(Path(argv[1]))
    except (OSError, json.JSONDecodeError, ValueError) as exception:
        print(f"manifest invalid: {exception}", file=sys.stderr)
        return 1
    print(f"manifest valid: {manifest['id']}@{manifest['version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
