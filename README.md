# NEOCR-Paddle

Independent Paddle OCR distribution for [NEOCR](https://github.com/starfield17/NEOCR). This repository owns the production C# worker, runtime/model package contracts and conversion/parity harness. It does not place inference libraries inside the desktop host.

Milestone 4A is a stop/go gate. It pins five Paddle inference archives, converts each model to ONNX on Linux x64, captures Paddle raw outputs, and verifies the same ONNX outputs through a C# ONNX Runtime harness on Linux x64, macOS arm64 and Windows x64. No production worker is implemented until all five pass.

Current status: the four orientation/detection/recognition graphs pass, while UVDoc's converted grid-prediction graph fails the unchanged parity threshold. Milestone 4A is stopped; see [`docs/uvdoc-parity-blocker.md`](docs/uvdoc-parity-blocker.md) for reproducible evidence and the decisions required to continue.

The fixed first model set is:

- `PP-LCNet_x1_0_doc_ori`
- `UVDoc`
- `PP-LCNet_x1_0_textline_ori`
- `PP-OCRv5_mobile_det`
- `PP-OCRv5_mobile_rec`

The three enhancement stages are disabled by default in the eventual worker. The NEOCR “Document” preset enables all three together; detection and recognition are always required.

Local structural checks:

```sh
dotnet build NEOCR.Paddle.slnx
python3 -m unittest discover -s tests -v
python3 tools/verify_manifest.py manifests/models/pp-ocrv5-mobile-full-v1.json
```

Run the heavyweight `feasibility-gate` GitHub Actions workflow manually. It intentionally fails on the first conversion or parity failure; that failure is a design result, not a reason to downgrade the pipeline.
