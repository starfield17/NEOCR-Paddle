# Current implementation state

Updated: 2026-08-02 (Asia/Shanghai)

Milestone 4A is stopped at the UVDoc parity gate on `main`. No production-worker work is authorized under the current constraints.

## Working

- Separate public repository and Apache-2.0 code license.
- Fixed five-model manifest with official source URLs and verified SHA-256 values.
- Independent worker/runtime/model package architecture.
- Standard-library manifest verifier and tests.
- Linux conversion/Paddle-reference bundle generator.
- Cross-platform C# ONNX Runtime raw-output parity harness.
- CI for structural checks and a manually triggered heavyweight feasibility gate.
- A separate manual UVDoc diagnostic harness that compares direct Paddle2ONNX optimizer variants and fail-closed matched PIR/ONNX checkpoints without changing the formal gate.

## Known gaps

- Heavyweight gate run `30736238229` stopped after successful document-orientation and UVDoc conversion because the manifest expected UVDoc input `img` while the static graph exposes `image`. This provenance error was corrected without changing the model, graph, shape or parity tolerance; a clean rerun is required.
- Rerun `30736340358` converted and captured all five models. Document orientation passed C# ORT parity (`maxAbs=1.7881393e-7`), then UVDoc failed at the first reported element (`abs=0.016703725` versus `1e-4` tolerance). Production-worker work is stopped while the complete bundle is diagnosed; thresholds and model selection remain unchanged.
- The downloaded full bundle was rerun on macOS arm64 with aggregate diagnostics. Four stages passed (`doc-orientation maxAbs=5.96e-8`, `textline-orientation=1.79e-7`, `detection=1.36e-7`, `recognition=2.01e-5`); only UVDoc failed (`maxAbs=1.827`, `MAE=0.440`, `98,285/98,304` violations). An independent bilinear sampler matched ONNX Runtime to `1.20e-7`, localizing the noisy-input discrepancy to the predicted sampling coordinates rather than the C# tensor reader or ORT `GridSample` kernel.
- UVDoc's gate input is now a deterministic coordinate gradient. This is a stricter geometry-specific probe at the unchanged `1e-4` element-wise tolerance; a clean heavyweight rerun is required before the gate can pass.
- Geometry-probe run `30736983229` still failed UVDoc (`maxAbs=0.02285`, `MAE=0.00777`, correlation `0.999986`) while the other four stages passed. Its coordinate channels prove that the converted graph predicts a sampling grid shifted by up to roughly `0.023` normalized units. Run `30737222429` repeated the probe with Paddle's default IR optimization disabled; model selection, opset and tolerance remained unchanged.
- Unfused-reference run `30737222429` produced exactly the same UVDoc metrics as `30736983229`; Paddle IR optimization is not the cause. Choice 1 is selected: diagnose Paddle2ONNX before considering a model, runtime or acceptance-rule change. `docs/uvdoc-parity-blocker.md` is the authoritative evidence and reassessment record.
- The diagnostic harness has structural/unit coverage but has not yet produced a Linux heavyweight report. Its checked-in `p2o.pd_op.<type>.<occurrence>.<output>` mapping is intentionally strict; a missing tensor in the first run is evidence to correct an explicitly verified mapping, not permission to guess a nearby value.
- Initial diagnostic run `30746228042` proved the always-upload report path but could not evaluate the unoptimized graph because Python ONNX Runtime was only auto-installed later as a Polygraphy side effect. The diagnostic environment now pins ONNX Runtime 1.28.0 independently; no conclusion may be drawn from that incomplete run.
- Local verification passed on 2026-08-02: 20 Python tests, manifest validation, .NET build, .NET formatting verification, transitive NuGet vulnerability scan and `git diff --check`. All nine checked-in selectors also resolved uniquely against the pinned official UVDoc PIR source. Linux Paddle2ONNX/ORT execution remains intentionally pending in the manual workflow.
- No production worker, preprocessing/postprocessing pipeline or NEOCR plugin package exists.
- No runtime or model GitHub Release asset exists.
- CUDA and cross-vendor GPU runtime packages are not designed yet.

## Next bounded task

Run the manual `uvdoc-diagnostics` workflow and inspect `uvdoc-diagnostic-report.json`. First determine whether the divergence exists in Paddle2ONNX output before optimization or is introduced only by `polygraphy`/`onnxoptimizer`; then use the coarse checkpoints to identify the first divergent interval. If mapping fails, update only an explicitly verified tensor mapping and rerun. Do not modify the five-model gate or begin the worker.
