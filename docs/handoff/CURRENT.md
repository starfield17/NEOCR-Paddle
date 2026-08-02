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
- Rerun `30736340358` converted and captured all five models. Document orientation passed C# ORT parity (`maxAbs=1.7881393e-7`), then UVDoc failed at the first reported element (`abs=0.016703725` versus `1e-4` tolerance). Production-worker work is stopped while the complete bundle is diagnosed; thresholds and model selection remain unchanged.
- The downloaded full bundle was rerun on macOS arm64 with aggregate diagnostics. Four stages passed (`doc-orientation maxAbs=5.96e-8`, `textline-orientation=1.79e-7`, `detection=1.36e-7`, `recognition=2.01e-5`); only UVDoc failed (`maxAbs=1.827`, `MAE=0.440`, `98,285/98,304` violations). An independent bilinear sampler matched ONNX Runtime to `1.20e-7`, localizing the noisy-input discrepancy to the predicted sampling coordinates rather than the C# tensor reader or ORT `GridSample` kernel.
- UVDoc's gate input is now a deterministic coordinate gradient. This is a stricter geometry-specific probe at the unchanged `1e-4` element-wise tolerance; a clean heavyweight rerun is required before the gate can pass.
- Geometry-probe run `30736983229` still failed UVDoc (`maxAbs=0.02285`, `MAE=0.00777`, correlation `0.999986`) while the other four stages passed. Its coordinate channels prove that the converted graph predicts a sampling grid shifted by up to roughly `0.023` normalized units. The next diagnostic reference disables Paddle's default IR optimization so the raw, unfused source graph is compared with the graph Paddle2ONNX actually converts; model selection, opset and tolerance remain unchanged.
- No production worker, preprocessing/postprocessing pipeline or NEOCR plugin package exists.
- No runtime or model GitHub Release asset exists.
- CUDA and cross-vendor GPU runtime packages are not designed yet.

## Next bounded task

Run the heavyweight gate. If any one of the five models fails conversion or parity, record the exact model/operator/output and stop for architecture reassessment. If all pass, freeze the gate bundle format and begin the CPU C# worker vertical slice.
