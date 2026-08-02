# Current implementation state

Updated: 2026-08-02 (Asia/Shanghai)

Milestone 4A is complete on `main`. Production-worker work is authorized; the gate bundle remains diagnostic evidence rather than a release package.

## Working

- Separate public repository and Apache-2.0 code license.
- Fixed five-model manifest with official source URLs and verified SHA-256 values.
- Independent worker/runtime/model package architecture.
- Standard-library manifest verifier and tests.
- Linux conversion/Paddle-reference bundle generator.
- Cross-platform C# ONNX Runtime raw-output parity harness.
- CI for structural checks and a manually triggered heavyweight feasibility gate.
- A separate manual UVDoc diagnostic harness that compares direct Paddle2ONNX optimizer variants and fail-closed matched PIR/ONNX checkpoints without changing the formal gate.
- Complete five-model parity at the unchanged common tolerance on Linux x64, macOS arm64 and Windows x64: GitHub Actions run `30747231341`, commit `b8a8fee`.

## Resolved investigation

- UVDoc's first bilinear resize was the first divergent checkpoint. Paddle2ONNX correctly emitted the serialized `align_corners=true` semantics; Paddle's oneDNN resize ignored that attribute. Explicitly disabling oneDNN reduced UVDoc final `maxAbs` from `0.022849` to `1.19e-6`.
- Paddle 3.0.0 and 3.2.1 reproduced the backend difference, so it was not a recent version regression. Polygraphy also changed 651 nodes to 259 without changing the final result, ruling out ONNX post-conversion optimization.
- The source gate and diagnostic harness now share one explicit Paddle reference configuration: CPU, oneDNN off, IR optimization off, one thread. `gate-index.json` records `paddleOneDnn=false` and `paddleIrOptimization=false`.
- No Paddle2ONNX fork or upstream PR was created because the mapper was not defective. Full evidence remains in `docs/uvdoc-parity-blocker.md`.

## Known gaps

- No production worker, preprocessing/postprocessing pipeline or NEOCR plugin package exists.
- No runtime or model GitHub Release asset exists.
- CUDA and cross-vendor GPU runtime packages are not designed yet.

## Next bounded task

Begin Milestone 4B with the smallest production vertical slice: define the independently versioned worker/runtime/model package manifests, then implement a CPU C# worker that loads only validated package paths and performs detection plus recognition through the existing NEOCR protocol. Add orientation, UVDoc and line-orientation stages after that baseline is end-to-end verified; do not combine CPU, CUDA and cross-vendor binaries into one runtime package.
