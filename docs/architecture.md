# Architecture

## Distribution boundaries

NEOCR discovers a worker package. The worker is a separate process and receives explicit paths to one runtime package and one model package. The desktop application validates package metadata and launches the worker, but never downloads packages or loads inference libraries.

```text
NEOCR host -> worker package -> runtime package
                           \-> model package
```

The three package identities are independent:

- **Worker:** protocol implementation and OCR preprocessing/postprocessing.
- **Runtime:** ONNX Runtime native/managed binaries for one RID and execution-provider set.
- **Model:** ONNX graphs, dictionaries, transforms, licenses and provenance.

An installed worker/runtime/model combination is valid only when declared compatibility ranges and SHA-256 inventories match. A GitHub Release may contain several assets, but an asset cannot blur these identities.

## First pipeline

The required order is document orientation, UVDoc unwarping, text detection, per-line orientation, then text recognition. Orientation, unwarping and line orientation are individually represented but default off. A Document preset enables all three. Detection and recognition cannot be disabled.

## Milestone 4A gate

Linux x64 is the reproducible Paddle reference environment. It downloads source archives from pinned official URLs, verifies SHA-256, converts with the official PaddleX Paddle2ONNX path, and records deterministic float32 input/output tensors. C# ONNX Runtime then checks output names, shapes, finite values and element-wise tolerances on Linux x64, macOS arm64 and Windows x64.

The gate is all-or-nothing. Missing operators, conversion errors, output-shape drift or parity failures stop work before the worker is built. Generated artifacts are retained only as CI artifacts for diagnosis and are not release packages.

CPU is the first runtime. CUDA and a cross-vendor GPU option are separate later runtime packages; their availability must be stated per RID and never inferred from the model package.
