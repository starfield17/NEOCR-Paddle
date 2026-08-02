import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from run_source_gate import configure_paddle_reference, safe_extract  # noqa: E402


class RecordingConfig:
    def __init__(self) -> None:
        self.calls = []

    def __getattr__(self, name):
        return lambda *args: self.calls.append((name, args))


class SafeExtractionTests(unittest.TestCase):
    def test_parent_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "bad.tar"
            with tarfile.open(archive, "w") as output:
                info = tarfile.TarInfo("../../escaped")
                info.size = 1
                output.addfile(info, io.BytesIO(b"x"))

            with self.assertRaisesRegex(ValueError, "unsafe archive path"):
                safe_extract(archive, root / "extract")

    def test_symbolic_link_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "bad.tar"
            with tarfile.open(archive, "w") as output:
                info = tarfile.TarInfo("link")
                info.type = tarfile.SYMTYPE
                info.linkname = "target"
                output.addfile(info)

            with self.assertRaisesRegex(ValueError, "regular files"):
                safe_extract(archive, root / "extract")


class PaddleReferenceConfigTests(unittest.TestCase):
    def test_reference_disables_accelerated_and_graph_optimized_paths(self) -> None:
        config = RecordingConfig()

        configure_paddle_reference(config)

        self.assertEqual(
            [
                ("disable_gpu", ()),
                ("disable_mkldnn", ()),
                ("disable_glog_info", ()),
                ("switch_ir_optim", (False,)),
                ("set_cpu_math_library_num_threads", (1,)),
            ],
            config.calls,
        )


if __name__ == "__main__":
    unittest.main()
