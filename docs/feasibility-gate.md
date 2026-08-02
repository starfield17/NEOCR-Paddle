# Five-model ONNX feasibility gate

## Inputs

`manifests/models/pp-ocrv5-mobile-full-v1.json` is authoritative. Every source archive has an immutable URL, SHA-256, expected extracted directory and deterministic test tensor shape and pattern. The source documentation is PaddleOCR's version-3 OCR pipeline and module documentation.

The four classifier/detector/recognizer stages use seeded random tensors. UVDoc uses a deterministic three-channel coordinate gradient: its final operation is bilinear spatial resampling, so the first two output channels directly measure the predicted sampling coordinates. Random white noise is unsuitable for this stage because harmless subpixel coordinate drift produces unrelated neighboring pixel values and obscures the geometry being tested. The coordinate probe does not relax the common element-wise tolerance.

## Procedure

1. Download all five archives and verify SHA-256 before extraction.
2. Reject absolute paths, parent traversal and symbolic links in each tar archive.
3. Convert each static Paddle graph through `paddlex --paddle2onnx` at ONNX opset 17.
4. Run the Paddle generic CPU predictor once with both IR graph optimization and oneDNN disabled and the manifest-selected deterministic float32 tensor, then save every raw output. This compares the serialized Paddle operator semantics that Paddle2ONNX converts. Optimized Paddle execution is backend-specific and is covered later by end-to-end golden-image acceptance tests rather than raw graph parity.
5. Build a gate bundle containing ONNX graphs, input tensors, expected outputs, shapes, names and tool versions.
6. Run the C# CPU harness over the bundle on Linux x64, macOS arm64 and Windows x64.
7. Require matching output sets/shapes and `abs(actual - expected) <= atol + rtol * abs(expected)` for every element. Defaults are `atol=1e-4`, `rtol=1e-4`; a per-model relaxation requires an ADR with evidence.

## Stop conditions

Any download mismatch, unsafe archive entry, Paddle load failure, conversion failure, ONNX load failure, non-finite output, output-set/shape mismatch or tolerance failure stops Milestone 4A. Do not omit the failed stage, switch its implementation to Python, or publish a partial model pack.

## Platform scope

Source conversion/reference: Ubuntu x64, Python 3.11, Paddle generic CPU with oneDNN explicitly disabled. Shipping-harness verification: Ubuntu x64, macOS arm64 and Windows x64 using `Microsoft.ML.OnnxRuntime` CPU. Accelerated execution providers are outside this gate.

## Diagnostic separation

`tools/uvdoc_diagnostics.py` is a manual blocker-localization tool, not an alternate acceptance path. It may convert only UVDoc with several optimizer settings and expose intermediate graph values, but it cannot alter this gate's model set, input, opset, tolerances or pass result. A successful diagnostic identifies a converter fix to validate through the complete procedure above; it does not itself authorize worker development.
