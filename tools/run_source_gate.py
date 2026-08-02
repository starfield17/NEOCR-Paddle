#!/usr/bin/env python3
"""Download, convert and capture Paddle reference tensors for Milestone 4A."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

from verify_manifest import validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paddlex-command", default="paddlex")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(model: dict, cache: Path) -> Path:
    archive = cache / f"{model['name']}.tar"
    if archive.exists() and sha256(archive) == model["sha256"]:
        return archive

    temporary = archive.with_suffix(".download")
    temporary.unlink(missing_ok=True)
    print(f"download {model['name']}", flush=True)
    with urllib.request.urlopen(model["sourceUrl"], timeout=120) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    actual = sha256(temporary)
    if actual != model["sha256"]:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"{model['name']}: SHA-256 {actual} did not match the manifest")
    os.replace(temporary, archive)
    return archive


def safe_extract(archive: Path, destination: Path) -> None:
    destination_root = destination.resolve()
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        for member in members:
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"{archive.name}: only regular files and directories are allowed: {member.name}")
            target = (destination / member.name).resolve()
            if target != destination_root and destination_root not in target.parents:
                raise ValueError(f"{archive.name}: unsafe archive path: {member.name}")
        tar.extractall(destination, members=members)


def source_directory(extraction_root: Path, expected_name: str) -> Path:
    matches = [path for path in extraction_root.rglob(expected_name) if path.is_dir()]
    if len(matches) != 1:
        raise ValueError(f"expected one extracted {expected_name} directory, found {len(matches)}")
    return matches[0]


def convert(source: Path, output: Path, opset: int, paddlex_command: str) -> Path:
    output.mkdir(parents=True)
    subprocess.run(
        [
            paddlex_command,
            "--paddle2onnx",
            "--paddle_model_dir",
            str(source),
            "--onnx_model_dir",
            str(output),
            "--opset_version",
            str(opset),
        ],
        check=True,
    )
    models = list(output.rglob("*.onnx"))
    if len(models) != 1:
        raise ValueError(f"conversion produced {len(models)} ONNX files under {output}")
    return models[0]


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def capture_reference(
    model: dict, source: Path, case_directory: Path, seed: int
) -> tuple[list[dict], list[dict]]:
    import numpy as np
    import paddle.inference as paddle_infer

    model_file = source / "inference.json"
    params_file = source / "inference.pdiparams"
    if not model_file.is_file() or not params_file.is_file():
        raise FileNotFoundError(f"{model['name']}: Paddle inference.json or parameters are missing")

    config = paddle_infer.Config(str(model_file), str(params_file))
    config.disable_gpu()
    config.disable_glog_info()
    config.set_cpu_math_library_num_threads(1)
    predictor = paddle_infer.create_predictor(config)
    input_names = predictor.get_input_names()
    if input_names != [model["inputName"]]:
        raise ValueError(
            f"{model['name']}: expected input {model['inputName']!r}, got {input_names!r}"
        )

    generator = np.random.default_rng(seed)
    values = generator.uniform(-1.0, 1.0, size=model["inputShape"]).astype("<f4")
    input_path = case_directory / "input.f32"
    values.tofile(input_path)
    input_handle = predictor.get_input_handle(input_names[0])
    input_handle.reshape(model["inputShape"])
    input_handle.copy_from_cpu(values)
    predictor.run()

    outputs: list[dict] = []
    for index, name in enumerate(predictor.get_output_names()):
        value = predictor.get_output_handle(name).copy_to_cpu()
        if not np.issubdtype(value.dtype, np.floating):
            raise TypeError(f"{model['name']}: output {name} has unsupported dtype {value.dtype}")
        value = value.astype("<f4", copy=False)
        if not np.isfinite(value).all():
            raise ValueError(f"{model['name']}: Paddle output {name} contains non-finite values")
        path = case_directory / f"output-{index}-{safe_name(name)}.f32"
        value.tofile(path)
        outputs.append({"name": name, "path": path.name, "shape": list(value.shape)})

    return [
        {
            "name": input_names[0],
            "path": input_path.name,
            "shape": model["inputShape"],
        }
    ], outputs


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def main() -> int:
    args = parse_args()
    manifest = validate(args.manifest)
    args.cache.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"output directory must be empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    index = {
        "schemaVersion": 1,
        "modelPackageId": manifest["id"],
        "modelPackageVersion": manifest["version"],
        "absoluteTolerance": manifest["parity"]["absoluteTolerance"],
        "relativeTolerance": manifest["parity"]["relativeTolerance"],
        "cases": [],
        "sourceEnvironment": {
            "paddlepaddle": package_version("paddlepaddle"),
            "paddleocr": package_version("paddleocr"),
            "paddlex": package_version("paddlex"),
            "paddle2onnx": package_version("paddle2onnx"),
        },
    }

    for offset, model in enumerate(manifest["models"]):
        archive = download(model, args.cache)
        extraction_root = args.cache / f"extract-{model['name']}"
        if extraction_root.exists():
            shutil.rmtree(extraction_root)
        extraction_root.mkdir()
        safe_extract(archive, extraction_root)
        source = source_directory(extraction_root, model["sourceDirectory"])
        case_directory = args.output / model["role"]
        case_directory.mkdir()
        conversion_root = args.cache / f"onnx-{model['name']}"
        if conversion_root.exists():
            shutil.rmtree(conversion_root)
        converted = convert(
            source,
            conversion_root,
            manifest["parity"]["onnxOpset"],
            args.paddlex_command,
        )
        model_path = case_directory / "model.onnx"
        shutil.copy2(converted, model_path)
        inputs, outputs = capture_reference(
            model,
            source,
            case_directory,
            manifest["parity"]["seed"] + offset,
        )
        case = {
            "name": model["name"],
            "modelPath": model_path.name,
            "inputs": inputs,
            "expectedOutputs": outputs,
        }
        case_path = case_directory / "case.json"
        case_path.write_text(json.dumps(case, indent=2) + "\n", encoding="utf-8")
        index["cases"].append(str(case_path.relative_to(args.output)))
        print(f"captured {model['name']}", flush=True)

    (args.output / "gate-index.json").write_text(
        json.dumps(index, indent=2) + "\n", encoding="utf-8"
    )
    print(f"gate bundle ready: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
