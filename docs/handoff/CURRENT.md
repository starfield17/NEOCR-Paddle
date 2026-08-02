# Current implementation state

Updated: 2026-08-02 (Asia/Shanghai)

Milestone 4A scaffolding is in progress on `main`.

## Working

- Separate public repository and Apache-2.0 code license.
- Fixed five-model manifest with official source URLs and verified SHA-256 values.
- Independent worker/runtime/model package architecture.
- Standard-library manifest verifier and tests.
- Linux conversion/Paddle-reference bundle generator.
- Cross-platform C# ONNX Runtime raw-output parity harness.
- CI for structural checks and a manually triggered heavyweight feasibility gate.

## Known gaps

- Heavyweight gate run `30736238229` stopped after successful document-orientation and UVDoc conversion because the manifest expected UVDoc input `img` while the static graph exposes `image`. This provenance error was corrected without changing the model, graph, shape or parity tolerance; a clean rerun is required.
- No production worker, preprocessing/postprocessing pipeline or NEOCR plugin package exists.
- No runtime or model GitHub Release asset exists.
- CUDA and cross-vendor GPU runtime packages are not designed yet.

## Next bounded task

Run the heavyweight gate. If any one of the five models fails conversion or parity, record the exact model/operator/output and stop for architecture reassessment. If all pass, freeze the gate bundle format and begin the CPU C# worker vertical slice.
