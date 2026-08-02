import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from uvdoc_diagnostics import (  # noqa: E402
    CheckpointSelector,
    compare_arrays,
    diagnostic_passed,
    resolve_onnx_tensor,
    rewrite_pir_fetch,
    select_pir_checkpoint,
    shapes_compatible,
)


def tensor(value_id: int, shape: list[int]) -> dict:
    return {
        "%": value_id,
        "TT": {"#": "0.t_dtensor", "D": [{"#": "0.t_f32"}, shape, "NCHW", [], 0]},
    }


def program(*, include_conv: bool = True, fetch_count: int = 1) -> dict:
    ops = []
    if include_conv:
        ops.append({"#": "1.conv2d", "I": [{"%": 1}], "O": [tensor(2, [-1, 4, 8, 8])]})
    for index in range(fetch_count):
        ops.append(
            {
                "#": "1.fetch",
                "I": [{"%": 9}],
                "O": [tensor(10 + index, [-1, 3, 8, 8])],
            }
        )
    return {"base_code": {"magic": "pir"}, "program": {"regions": [{"blocks": [{"ops": ops}]}]}}


class PirRewriteTests(unittest.TestCase):
    def test_rewrite_uses_copy_and_preserves_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            original = json.dumps(program(), separators=(",", ":"))
            (source / "inference.json").write_text(original, encoding="utf-8")
            (source / "inference.pdiparams").write_bytes(b"parameters")
            destination = root / "rewritten"

            checkpoint = rewrite_pir_fetch(
                source, destination, CheckpointSelector("conv", "conv2d", 0)
            )

            self.assertEqual(original, (source / "inference.json").read_text(encoding="utf-8"))
            self.assertEqual(b"parameters", (destination / "inference.pdiparams").read_bytes())
            rewritten = json.loads((destination / "inference.json").read_text(encoding="utf-8"))
            fetch = rewritten["program"]["regions"][0]["blocks"][0]["ops"][-1]
            self.assertEqual([{"%": 2}], fetch["I"])
            self.assertEqual([-1, 4, 8, 8], fetch["O"][0]["TT"]["D"][1])
            self.assertEqual(2, checkpoint.value_id)

    def test_missing_target_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not exist"):
            select_pir_checkpoint(
                program(include_conv=False), CheckpointSelector("missing", "conv2d", 0)
            )

    def test_multiple_fetches_are_rejected_without_creating_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "inference.json").write_text(
                json.dumps(program(fetch_count=2)), encoding="utf-8"
            )
            (source / "inference.pdiparams").write_bytes(b"parameters")
            destination = root / "rewritten"

            with self.assertRaisesRegex(ValueError, "exactly one PIR fetch"):
                rewrite_pir_fetch(source, destination, CheckpointSelector("conv", "conv2d", 0))
            self.assertFalse(destination.exists())

    def test_overlapping_source_and_destination_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            with self.assertRaisesRegex(ValueError, "must not overlap"):
                rewrite_pir_fetch(source, source / "copy", CheckpointSelector("conv", "conv2d", 0))

    def test_unsupported_op_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported PIR op"):
            select_pir_checkpoint(program(), CheckpointSelector("bad", "fetch", 0))


class OnnxMappingTests(unittest.TestCase):
    def test_exact_unique_tensor_is_resolved(self) -> None:
        selector = CheckpointSelector("conv", "conv2d", 0)
        self.assertEqual(
            "p2o.pd_op.conv2d.0.0",
            resolve_onnx_tensor(["other", "p2o.pd_op.conv2d.0.0"], selector),
        )

    def test_missing_tensor_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "found 0"):
            resolve_onnx_tensor([], CheckpointSelector("conv", "conv2d", 0))

    def test_ambiguous_tensor_is_rejected(self) -> None:
        name = "p2o.pd_op.conv2d.0.0"
        with self.assertRaisesRegex(ValueError, "found 2"):
            resolve_onnx_tensor([name, name], CheckpointSelector("conv", "conv2d", 0))

    def test_dynamic_shapes_are_compatible_but_rank_and_static_dims_are_not(self) -> None:
        self.assertTrue(shapes_compatible([-1, 3, 8, 8], ["batch", 3, 8, 8]))
        self.assertFalse(shapes_compatible([-1, 3, 8, 8], [1, 4, 8, 8]))
        self.assertFalse(shapes_compatible([-1, 3, 8, 8], [1, 3, 8]))


class ReportMetricTests(unittest.TestCase):
    def test_tolerance_violation_cannot_be_reported_as_passed(self) -> None:
        metrics = compare_arrays(
            [0.0, 1.0],
            [0.0, 1.01],
            1e-4,
            1e-4,
        )
        self.assertFalse(metrics["passed"])
        self.assertEqual(1, metrics["violationCount"])
        self.assertTrue(math.isfinite(metrics["correlation"]))

    def test_shape_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            compare_arrays([0.0, 0.0], [[0.0, 0.0]], 1e-4, 1e-4)

    def test_non_finite_values_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-finite"):
            compare_arrays(
                [0.0],
                [float("nan")],
                1e-4,
                1e-4,
            )

    def test_incomplete_report_cannot_pass(self) -> None:
        report = {
            "optimizerVariants": [
                {"optimizer": "None", "status": "complete", "metrics": {"passed": True}}
            ],
            "checkpoints": [],
            "errors": [],
        }
        self.assertFalse(diagnostic_passed(report, 0))

    def test_report_with_error_cannot_pass(self) -> None:
        report = {
            "optimizerVariants": [
                {"optimizer": optimizer, "status": "complete", "metrics": {"passed": True}}
                for optimizer in ("None", "polygraphy", "onnxoptimizer")
            ],
            "checkpoints": [],
            "errors": [{"phase": "checkpointComparison", "message": "missing"}],
        }
        self.assertFalse(diagnostic_passed(report, 0))


if __name__ == "__main__":
    unittest.main()
