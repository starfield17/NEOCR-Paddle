import copy
import json
import tempfile
import unittest
from pathlib import Path

from tools.verify_manifest import validate


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests/models/pp-ocrv5-mobile-full-v1.json"


class ManifestTests(unittest.TestCase):
    def test_checked_in_manifest_is_valid(self) -> None:
        data = validate(MANIFEST)
        self.assertEqual(5, len(data["models"]))

    def test_missing_pipeline_role_is_rejected(self) -> None:
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        invalid = copy.deepcopy(data)
        invalid["models"] = invalid["models"][:-1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "required role"):
                validate(path)

    def test_unpinned_or_nonofficial_source_is_rejected(self) -> None:
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        data["models"][0]["sourceUrl"] = "https://example.com/model.tar"
        data["models"][0]["sha256"] = "latest"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "official HTTPS host"):
                validate(path)

    def test_unknown_input_pattern_is_rejected(self) -> None:
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        data["models"][1]["inputPattern"] = "friendlyImage"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inputPattern"):
                validate(path)


if __name__ == "__main__":
    unittest.main()
