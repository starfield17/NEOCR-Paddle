import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from run_source_gate import safe_extract  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
