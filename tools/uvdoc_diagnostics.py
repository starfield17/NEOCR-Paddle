#!/usr/bin/env python3
"""Reproduce and localize UVDoc Paddle2ONNX parity differences.

This is a diagnostic-only Linux harness. It intentionally does not change or
call the five-model feasibility gate.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import itertools
import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from run_source_gate import create_input, download, safe_extract, sha256, source_directory
from verify_manifest import validate


REPORT_SCHEMA_VERSION = 1
OPTIMIZER_VARIANTS = ("None", "polygraphy", "onnxoptimizer")
SUPPORTED_PIR_OPS = frozenset(
    {
        "add",
        "batch_norm_",
        "bilinear_interp",
        "conv2d",
        "grid_sample",
        "pad3d",
        "prelu",
        "relu",
    }
)
CHECKPOINT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class CheckpointSelector:
    identifier: str
    op_type: str
    occurrence: int
    output_index: int = 0
    onnx_tensor: str | None = None


@dataclass(frozen=True)
class PirCheckpoint:
    selector: CheckpointSelector
    value_id: int
    shape: tuple[int, ...]
    tensor_type: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paddle2onnx-command", default="paddle2onnx")
    return parser.parse_args()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def load_checkpoint_selectors(path: Path) -> list[CheckpointSelector]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schemaVersion") != 1 or not isinstance(document.get("checkpoints"), list):
        raise ValueError("checkpoint file must use schemaVersion 1 and contain checkpoints")

    selectors: list[CheckpointSelector] = []
    identifiers: set[str] = set()
    for item in document["checkpoints"]:
        identifier = item.get("id")
        op_type = item.get("opType")
        occurrence = item.get("occurrence")
        output_index = item.get("outputIndex", 0)
        onnx_tensor = item.get("onnxTensor")
        if not isinstance(identifier, str) or not CHECKPOINT_ID.fullmatch(identifier):
            raise ValueError(f"invalid checkpoint id: {identifier!r}")
        if identifier in identifiers:
            raise ValueError(f"duplicate checkpoint id: {identifier}")
        if op_type not in SUPPORTED_PIR_OPS:
            raise ValueError(f"{identifier}: unsupported PIR op {op_type!r}")
        if not isinstance(occurrence, int) or occurrence < 0:
            raise ValueError(f"{identifier}: occurrence must be a non-negative integer")
        if not isinstance(output_index, int) or output_index < 0:
            raise ValueError(f"{identifier}: outputIndex must be a non-negative integer")
        if onnx_tensor is not None and (not isinstance(onnx_tensor, str) or not onnx_tensor):
            raise ValueError(f"{identifier}: onnxTensor must be a non-empty string")
        identifiers.add(identifier)
        selectors.append(
            CheckpointSelector(identifier, op_type, occurrence, output_index, onnx_tensor)
        )
    if not selectors:
        raise ValueError("checkpoint file must contain at least one checkpoint")
    return selectors


def pir_ops(program: dict) -> list[dict]:
    try:
        regions = program["program"]["regions"]
        blocks = regions[0]["blocks"]
        ops = blocks[0]["ops"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("unsupported Paddle PIR program structure") from error
    if len(regions) != 1 or len(blocks) != 1 or not isinstance(ops, list):
        raise ValueError("only a single-region, single-block Paddle PIR program is supported")
    return ops


def canonical_pir_op(op: dict) -> str:
    name = op.get("#")
    if not isinstance(name, str):
        raise ValueError("PIR op is missing its type")
    return name.split(".", 1)[1] if "." in name else name


def _tensor_descriptor(output: dict) -> tuple[int, tuple[int, ...], str]:
    try:
        value_id = output["%"]
        descriptor = output["TT"]["D"]
        tensor_type = descriptor[0]["#"]
        shape = descriptor[1]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("unsupported PIR tensor descriptor") from error
    if not isinstance(value_id, int) or not isinstance(shape, list) or not all(
        isinstance(dimension, int) for dimension in shape
    ):
        raise ValueError("invalid PIR tensor value or shape")
    if tensor_type != "0.t_f32":
        raise ValueError(f"only float32 checkpoints are supported, got {tensor_type!r}")
    return value_id, tuple(shape), tensor_type


def select_pir_checkpoint(program: dict, selector: CheckpointSelector) -> PirCheckpoint:
    if selector.op_type not in SUPPORTED_PIR_OPS:
        raise ValueError(f"{selector.identifier}: unsupported PIR op {selector.op_type!r}")
    matches = [op for op in pir_ops(program) if canonical_pir_op(op) == selector.op_type]
    if selector.occurrence >= len(matches):
        raise ValueError(
            f"{selector.identifier}: {selector.op_type} occurrence {selector.occurrence} "
            f"does not exist (found {len(matches)})"
        )
    outputs = matches[selector.occurrence].get("O")
    if not isinstance(outputs, list) or selector.output_index >= len(outputs):
        raise ValueError(
            f"{selector.identifier}: output {selector.output_index} does not exist on "
            f"{selector.op_type} occurrence {selector.occurrence}"
        )
    value_id, shape, tensor_type = _tensor_descriptor(outputs[selector.output_index])
    return PirCheckpoint(selector, value_id, shape, tensor_type)


def rewrite_pir_fetch(
    source_directory: Path, destination: Path, selector: CheckpointSelector
) -> PirCheckpoint:
    source_directory = source_directory.resolve()
    destination = destination.resolve()
    if (
        source_directory == destination
        or source_directory in destination.parents
        or destination in source_directory.parents
    ):
        raise ValueError("diagnostic source and destination directories must not overlap")
    if destination.exists():
        raise ValueError(f"diagnostic model destination already exists: {destination}")

    source_model = source_directory / "inference.json"
    source_digest = sha256(source_model)
    program = json.loads(source_model.read_text(encoding="utf-8"))
    checkpoint = select_pir_checkpoint(program, selector)
    fetches = [op for op in pir_ops(program) if canonical_pir_op(op) == "fetch"]
    if len(fetches) != 1:
        raise ValueError(f"expected exactly one PIR fetch op, found {len(fetches)}")
    fetch = fetches[0]
    if not isinstance(fetch.get("I"), list) or len(fetch["I"]) != 1:
        raise ValueError("unsupported PIR fetch input structure")
    if not isinstance(fetch.get("O"), list) or len(fetch["O"]) != 1:
        raise ValueError("unsupported PIR fetch output structure")

    shutil.copytree(source_directory, destination)
    fetch["I"] = [{"%": checkpoint.value_id}]
    fetch["O"][0]["TT"] = {
        "#": "0.t_dtensor",
        "D": [{"#": checkpoint.tensor_type}, list(checkpoint.shape), "NCHW", [], 0],
    }
    (destination / "inference.json").write_text(
        json.dumps(program, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    if sha256(source_model) != source_digest:
        raise RuntimeError("source Paddle model changed while creating a diagnostic copy")
    return checkpoint


def expected_onnx_tensor(selector: CheckpointSelector) -> str:
    return selector.onnx_tensor or (
        f"p2o.pd_op.{selector.op_type}.{selector.occurrence}.{selector.output_index}"
    )


def resolve_onnx_tensor(node_outputs: Iterable[str], selector: CheckpointSelector) -> str:
    expected = expected_onnx_tensor(selector)
    matches = [name for name in node_outputs if name == expected]
    if len(matches) != 1:
        raise ValueError(
            f"{selector.identifier}: expected exactly one ONNX tensor {expected!r}, "
            f"found {len(matches)}"
        )
    return matches[0]


def shapes_compatible(expected: Iterable[int], actual: Iterable[int | str | None]) -> bool:
    expected_tuple = tuple(expected)
    actual_tuple = tuple(actual)
    return len(expected_tuple) == len(actual_tuple) and all(
        source < 0 or target is None or isinstance(target, str) or source == target
        for source, target in zip(expected_tuple, actual_tuple, strict=True)
    )


def checkpoint_output_shape(
    expected: Iterable[int], inferred: Iterable[int | str | None]
) -> list[int | str | None]:
    expected_shape = tuple(expected)
    inferred_shape = tuple(inferred)
    if inferred_shape:
        if not shapes_compatible(expected_shape, inferred_shape):
            raise ValueError(
                f"Paddle shape {expected_shape} does not match ONNX shape {list(inferred_shape)}"
            )
        return list(inferred_shape)
    if not expected_shape:
        return []
    return [None if dimension < 0 else dimension for dimension in expected_shape]


def _shape_and_values(value) -> tuple[tuple[int, ...], list[float]]:
    if hasattr(value, "shape") and hasattr(value, "flat"):
        return tuple(int(dimension) for dimension in value.shape), [
            float(item) for item in value.flat
        ]

    def walk(item) -> tuple[tuple[int, ...], list[float]]:
        if not isinstance(item, (list, tuple)):
            return (), [float(item)]
        if not item:
            return (0,), []
        children = [walk(child) for child in item]
        child_shape = children[0][0]
        if any(shape != child_shape for shape, _ in children[1:]):
            raise ValueError("ragged arrays cannot be compared")
        return (len(item), *child_shape), [number for _, values in children for number in values]

    return walk(value)


def compare_arrays(expected, actual, absolute_tolerance: float, relative_tolerance: float) -> dict:
    expected_shape, expected_values = _shape_and_values(expected)
    actual_shape, actual_values = _shape_and_values(actual)
    if expected_shape != actual_shape:
        raise ValueError(f"shape mismatch: Paddle {expected_shape}, ONNX {actual_shape}")
    if not all(math.isfinite(value) for value in itertools.chain(expected_values, actual_values)):
        raise ValueError("cannot compare non-finite Paddle or ONNX values")
    differences = [
        abs(observed - reference)
        for reference, observed in zip(expected_values, actual_values)
    ]
    violations = sum(
        difference > absolute_tolerance + relative_tolerance * abs(reference)
        for reference, difference in zip(expected_values, differences)
    )
    count = len(expected_values)
    expected_mean = sum(expected_values) / count if count else 0.0
    actual_mean = sum(actual_values) / count if count else 0.0
    expected_variance = sum((value - expected_mean) ** 2 for value in expected_values)
    actual_variance = sum((value - actual_mean) ** 2 for value in actual_values)
    if count < 2 or expected_variance == 0 or actual_variance == 0:
        correlation = 1.0 if expected_values == actual_values else 0.0
    else:
        covariance = sum(
            (reference - expected_mean) * (observed - actual_mean)
            for reference, observed in zip(expected_values, actual_values)
        )
        correlation = covariance / math.sqrt(expected_variance * actual_variance)
    if not math.isfinite(correlation):
        raise ValueError("comparison produced a non-finite correlation")
    return {
        "passed": violations == 0,
        "shape": list(expected_shape),
        "elementCount": count,
        "violationCount": violations,
        "maxAbsoluteError": max(differences, default=0.0),
        "meanAbsoluteError": sum(differences) / count if count else 0.0,
        "rootMeanSquareError": math.sqrt(sum(value * value for value in differences) / count)
        if count
        else 0.0,
        "correlation": correlation,
    }


def diagnostic_passed(report: dict, expected_checkpoint_count: int) -> bool:
    variants = report.get("optimizerVariants", [])
    checkpoints = report.get("checkpoints", [])
    if report.get("errors") or len(variants) != len(OPTIMIZER_VARIANTS):
        return False
    if {item.get("optimizer") for item in variants} != set(OPTIMIZER_VARIANTS):
        return False
    if len(checkpoints) != expected_checkpoint_count:
        return False
    comparisons = [*variants, *checkpoints]
    return all(
        item.get("status") == "complete" and item["metrics"]["passed"]
        for item in comparisons
    )


def write_report(path: Path, report: dict, expected_checkpoint_count: int) -> None:
    report["passed"] = diagnostic_passed(report, expected_checkpoint_count)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def direct_convert(
    source: Path, output: Path, opset: int, optimizer: str, command: str
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            command,
            "--model_dir",
            str(source),
            "--model_filename",
            "inference.json",
            "--params_filename",
            "inference.pdiparams",
            "--save_file",
            str(output),
            "--opset_version",
            str(opset),
            "--enable_onnx_checker",
            "True",
            "--optimize_tool",
            optimizer,
        ],
        check=True,
    )
    if not output.is_file():
        raise ValueError(f"Paddle2ONNX did not create {output}")


def run_paddle(source: Path, input_name: str, input_shape: list[int], values):
    import numpy as np
    import paddle.inference as paddle_infer

    config = paddle_infer.Config(
        str(source / "inference.json"), str(source / "inference.pdiparams")
    )
    config.disable_gpu()
    config.disable_glog_info()
    config.switch_ir_optim(False)
    config.set_cpu_math_library_num_threads(1)
    predictor = paddle_infer.create_predictor(config)
    if predictor.get_input_names() != [input_name]:
        raise ValueError(f"unexpected Paddle inputs: {predictor.get_input_names()!r}")
    handle = predictor.get_input_handle(input_name)
    handle.reshape(input_shape)
    handle.copy_from_cpu(values)
    predictor.run()
    output_names = predictor.get_output_names()
    if len(output_names) != 1:
        raise ValueError(f"expected one Paddle output, found {output_names!r}")
    result = predictor.get_output_handle(output_names[0]).copy_to_cpu().astype("<f4", copy=False)
    if not np.isfinite(result).all():
        raise ValueError("Paddle output contains non-finite values")
    return result


def run_onnx(model_path: Path, input_name: str, values) -> dict[str, object]:
    import numpy as np
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    if [item.name for item in session.get_inputs()] != [input_name]:
        inputs = [item.name for item in session.get_inputs()]
        raise ValueError(f"unexpected ONNX inputs: {inputs!r}")
    names = [item.name for item in session.get_outputs()]
    outputs = session.run(names, {input_name: values})
    result = dict(zip(names, outputs, strict=True))
    if any(not np.isfinite(np.asarray(value)).all() for value in result.values()):
        raise ValueError("ONNX output contains non-finite values")
    return result


def expose_onnx_checkpoints(
    source: Path, output: Path, checkpoints: list[PirCheckpoint]
) -> dict[str, str]:
    import onnx
    from onnx import helper

    model = onnx.load(source)
    model = onnx.shape_inference.infer_shapes(model)
    node_outputs = [name for node in model.graph.node for name in node.output]
    value_info = {item.name: item for item in (*model.graph.value_info, *model.graph.output)}
    resolved: dict[str, str] = {}
    for checkpoint in checkpoints:
        name = resolve_onnx_tensor(node_outputs, checkpoint.selector)
        info = value_info.get(name)
        if info is None or not info.type.HasField("tensor_type"):
            raise ValueError(f"{checkpoint.selector.identifier}: ONNX tensor has no shape metadata")
        tensor_type = info.type.tensor_type
        if tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError(f"{checkpoint.selector.identifier}: ONNX tensor is not float32")
        shape = [
            dimension.dim_value
            if dimension.HasField("dim_value")
            else dimension.dim_param or None
            for dimension in tensor_type.shape.dim
        ]
        try:
            output_shape = checkpoint_output_shape(checkpoint.shape, shape)
        except ValueError as error:
            raise ValueError(f"{checkpoint.selector.identifier}: {error}") from error
        model.graph.output.append(
            helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, output_shape)
        )
        resolved[checkpoint.selector.identifier] = name
    onnx.checker.check_model(model)
    onnx.save(model, output)
    return resolved


def prepare_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def main() -> int:
    args = parse_args()
    manifest = validate(args.manifest)
    selectors = load_checkpoint_selectors(args.checkpoints)
    uvdoc_models = [model for model in manifest["models"] if model["role"] == "documentUnwarping"]
    if len(uvdoc_models) != 1 or uvdoc_models[0]["name"] != "UVDoc":
        raise ValueError("manifest must contain exactly the fixed UVDoc documentUnwarping model")
    model = uvdoc_models[0]
    if model.get("inputPattern") != "coordinateGradient":
        raise ValueError("UVDoc diagnostics require the coordinateGradient input")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"output directory must be empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)

    archive = download(model, args.cache)
    extraction = args.cache / "uvdoc-diagnostic-source"
    prepare_directory(extraction)
    safe_extract(archive, extraction)
    pristine_source = source_directory(extraction, model["sourceDirectory"])
    pristine_digest = sha256(pristine_source / "inference.json")

    import numpy as np

    values = create_input(model, manifest["parity"]["seed"] + 1, np)
    values.tofile(args.output / "input-coordinate-gradient.f32")
    reference = run_paddle(pristine_source, model["inputName"], model["inputShape"], values)
    reference.tofile(args.output / "paddle-final.f32")
    absolute_tolerance = manifest["parity"]["absoluteTolerance"]
    relative_tolerance = manifest["parity"]["relativeTolerance"]
    report = {
        "schemaVersion": REPORT_SCHEMA_VERSION,
        "model": {
            "name": model["name"],
            "sourceUrl": model["sourceUrl"],
            "sourceSha256": model["sha256"],
            "inputName": model["inputName"],
            "inputShape": model["inputShape"],
            "inputPattern": model["inputPattern"],
        },
        "parity": {
            "onnxOpset": manifest["parity"]["onnxOpset"],
            "absoluteTolerance": absolute_tolerance,
            "relativeTolerance": relative_tolerance,
            "paddleIrOptimization": False,
            "onnxRuntimeGraphOptimization": False,
        },
        "tools": {
            "paddlepaddle": package_version("paddlepaddle"),
            "paddle2onnx": package_version("paddle2onnx"),
            "onnx": package_version("onnx"),
            "onnxruntime": package_version("onnxruntime"),
            "polygraphy": package_version("polygraphy"),
            "onnxoptimizer": package_version("onnxoptimizer"),
        },
        "optimizerVariants": [],
        "checkpoints": [],
        "errors": [],
    }

    unoptimized_model: Path | None = None
    for optimizer in OPTIMIZER_VARIANTS:
        try:
            work_source = args.cache / f"uvdoc-source-{optimizer.lower()}"
            prepare_directory(work_source)
            shutil.copytree(pristine_source, work_source / model["sourceDirectory"])
            work_model = work_source / model["sourceDirectory"]
            onnx_path = args.output / f"uvdoc-{optimizer.lower()}.onnx"
            direct_convert(
                work_model,
                onnx_path,
                manifest["parity"]["onnxOpset"],
                optimizer,
                args.paddle2onnx_command,
            )
            outputs = run_onnx(onnx_path, model["inputName"], values)
            if len(outputs) != 1:
                raise ValueError(
                    f"{optimizer}: expected one final ONNX output, found {list(outputs)}"
                )
            actual = next(iter(outputs.values()))
            actual_path = args.output / f"onnx-{optimizer.lower()}-final.f32"
            np.asarray(actual, dtype="<f4").tofile(actual_path)
            metrics = compare_arrays(reference, actual, absolute_tolerance, relative_tolerance)
            report["optimizerVariants"].append(
                {
                    "optimizer": optimizer,
                    "status": "complete",
                    "modelPath": onnx_path.name,
                    "modelSha256": sha256(onnx_path),
                    "nodeCount": len(__import__("onnx").load(onnx_path).graph.node),
                    "metrics": metrics,
                }
            )
            if optimizer == "None":
                unoptimized_model = onnx_path
        except Exception as error:  # Preserve all successful variants in the diagnostic artifact.
            failure = {"optimizer": optimizer, "status": "error", "error": str(error)}
            report["optimizerVariants"].append(failure)
            report["errors"].append(
                {"phase": "optimizerVariant", "optimizer": optimizer, "message": str(error)}
            )

    report_path = args.output / "uvdoc-diagnostic-report.json"
    write_report(report_path, report, len(selectors))

    if unoptimized_model is None:
        if sha256(pristine_source / "inference.json") != pristine_digest:
            report["errors"].append(
                {"phase": "sourceIntegrity", "message": "pristine Paddle model was mutated"}
            )
            write_report(report_path, report, len(selectors))
        print(json.dumps(report, indent=2))
        return 1
    try:
        checkpoints: list[PirCheckpoint] = []
        paddle_values: dict[str, object] = {}
        checkpoint_root = args.cache / "uvdoc-checkpoints"
        prepare_directory(checkpoint_root)
        for selector in selectors:
            copied = checkpoint_root / selector.identifier
            checkpoint = rewrite_pir_fetch(pristine_source, copied, selector)
            checkpoints.append(checkpoint)
            paddle_value = run_paddle(copied, model["inputName"], model["inputShape"], values)
            paddle_values[selector.identifier] = paddle_value
            paddle_value.tofile(args.output / f"checkpoint-{selector.identifier}-paddle.f32")

        exposed_model = args.output / "uvdoc-none-checkpoints.onnx"
        resolved = expose_onnx_checkpoints(unoptimized_model, exposed_model, checkpoints)
        onnx_values = run_onnx(exposed_model, model["inputName"], values)
        for checkpoint in checkpoints:
            identifier = checkpoint.selector.identifier
            tensor_name = resolved[identifier]
            actual = onnx_values[tensor_name]
            np.asarray(actual, dtype="<f4").tofile(
                args.output / f"checkpoint-{identifier}-onnx.f32"
            )
            report["checkpoints"].append(
                {
                    "id": identifier,
                    "status": "complete",
                    "paddle": {
                        "opType": checkpoint.selector.op_type,
                        "occurrence": checkpoint.selector.occurrence,
                        "outputIndex": checkpoint.selector.output_index,
                        "valueId": checkpoint.value_id,
                        "shape": list(checkpoint.shape),
                    },
                    "onnxTensor": tensor_name,
                    "optimizer": "None",
                    "metrics": compare_arrays(
                        paddle_values[identifier], actual, absolute_tolerance, relative_tolerance
                    ),
                }
            )
    except Exception as error:
        report["errors"].append({"phase": "checkpointComparison", "message": str(error)})

    if sha256(pristine_source / "inference.json") != pristine_digest:
        report["errors"].append(
            {"phase": "sourceIntegrity", "message": "pristine Paddle model was mutated"}
        )
    write_report(report_path, report, len(selectors))
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
