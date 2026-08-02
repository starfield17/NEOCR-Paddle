# UVDoc ONNX parity blocker

## Resolution

Resolved on 2026-08-02. Complete feasibility-gate run [`30747231341`](https://github.com/starfield17/NEOCR-Paddle/actions/runs/30747231341) passed all five models on Linux x64, macOS arm64 and Windows x64 at commit `b8a8fee`, with the unchanged common tolerance. Milestone 4A is complete and production-worker work may proceed.

The apparent UVDoc conversion failure came from a backend-contaminated Paddle reference: Paddle's oneDNN resize path ignored serialized interpolation attributes. Paddle2ONNX correctly preserved those attributes, so no converter fork or upstream patch was created. The formal reference now explicitly disables oneDNN and records that choice in the gate bundle. The investigation below is retained as historical evidence.

## Reproduction

Repository commit: `cd96bacce92042559f38d0a1f309fe50ebc87c86`

Run the manual GitHub Actions workflow `feasibility-gate`, or reproduce its two commands in the pinned Linux source environment:

```sh
python tools/run_source_gate.py \
  --manifest manifests/models/pp-ocrv5-mobile-full-v1.json \
  --cache .cache/gate \
  --output artifacts/gate-bundle

dotnet run --project src/NEOCR.Paddle.Gate -- \
  validate-bundle artifacts/gate-bundle
```

The source archive, SHA-256, input shape, deterministic input pattern, conversion opset and tolerances are all in the manifest. `gate-index.json` and `python-lock.txt` record the generated bundle's source environment.

## Evidence

| Run | Probe | Paddle IR optimization | Result |
| --- | --- | --- | --- |
| `30736340358` | Seeded random tensor | On (default) | UVDoc `maxAbs=1.827`, `MAE=0.440`, correlation `0.00665`; other four models passed when rerun with aggregate diagnostics |
| `30736983229` | Coordinate gradient | On (default) | UVDoc `maxAbs=0.022848845`, `MAE=0.007771069`, correlation `0.999986`; other four models passed |
| `30737222429` | Coordinate gradient | Off | Results were identical to run `30736983229` |

The coordinate-gradient probe is geometry-specific. Input channel 0 encodes horizontal coordinates, channel 1 vertical coordinates and channel 2 their mean. With bilinear sampling and `align_corners=true`, the first two output channels expose the predicted sampling grid directly. Channel consistency held within float32 noise.

Inspection of the converted ONNX graph found both spatial operations with the expected attributes:

- `Resize(mode=linear, coordinate_transformation_mode=align_corners)`
- `GridSample(mode=bilinear, padding_mode=zeros, align_corners=1)`

An independent NumPy bilinear sampler matched ONNX Runtime's UVDoc output to `1.20e-7`. ONNX Runtime's sampled coordinate-gradient output matched its exposed internal grid to `1.79e-7`. This rules out the C# tensor reader and the final ONNX Runtime `GridSample` implementation; the converted graph predicts a grid shifted from Paddle by as much as `0.02285` normalized units.

## Reassessment choices

Each choice changes an earlier product constraint and therefore requires an explicit decision before implementation:

1. Fix or fork Paddle2ONNX for UVDoc, retaining the all-ONNX C# worker design. This best preserves the current architecture but has unknown effort until the first divergent intermediate operator is localized.
2. Select a different document-unwarping model that passes the same cross-platform ONNX gate. This changes the fixed model set and needs quality benchmarks against UVDoc.
3. Package Paddle Inference as an alternative native runtime for the complete five-stage worker. This preserves UVDoc but abandons ONNX Runtime as the sole first CPU backend and materially increases native packaging work.
4. Redefine acceptance around golden document images and downstream OCR quality instead of raw `1e-4` UVDoc tensor parity. This may be product-valid but deliberately weakens the present gate and needs an ADR plus a representative corpus.

The recommended next investigation, if choice 1 is selected, is to expose matched Paddle and ONNX intermediate tensors from the UVDoc grid-prediction network and bisect from the first resize through the convolution blocks. Do not start this as incidental worker work.

## Root-cause diagnostic harness

Choice 1 is now the active investigation. The manual `uvdoc-diagnostics` workflow is intentionally separate from `feasibility-gate`; it does not change the five-model gate, model manifest or common tolerances.

The diagnostic performs two checks against one Paddle coordinate-gradient reference:

1. Directly converts the same pinned UVDoc source with Paddle2ONNX `--optimize_tool None`, `polygraphy` and `onnxoptimizer`, then compares each final output with ONNX Runtime graph optimization disabled.
2. Rewrites the sole PIR `fetch` in a fresh model copy for each checkpoint and exposes the corresponding tensor from the unoptimized ONNX graph. The original extracted model is hash-checked before and after the run and is never rewritten in place.

Checkpoint mapping is deliberately fail-closed. The checked-in selector identifies a supported Paddle op by type, zero-based occurrence and output index. Its ONNX counterpart defaults to `p2o.pd_op.<type>.<occurrence>.<output>`. Missing or duplicate tensors, unsupported ops, absent shape metadata, incompatible shapes and non-finite values make the schema-versioned report fail instead of selecting a nearby tensor.

Run it from the repository root in the pinned Linux source environment:

```sh
python tools/uvdoc_diagnostics.py \
  --manifest manifests/models/pp-ocrv5-mobile-full-v1.json \
  --checkpoints diagnostics/uvdoc-checkpoints-v1.json \
  --cache .cache/uvdoc-diagnostics \
  --output artifacts/uvdoc-diagnostics
```

The command is expected to return non-zero until every requested optimizer variant and checkpoint satisfies the unchanged parity rule. `uvdoc-diagnostic-report.json` records exact optimizer labels, package versions, tensor mappings and metrics. Generated ONNX graphs and float tensors remain ignored by Git and are retained only as the manual workflow artifact.

Diagnostic run `30746228042` is not parity evidence: the unoptimized variant ran before Polygraphy attempted its own ONNX Runtime auto-install and failed to import the package. The dedicated diagnostic requirements now pin Python ONNX Runtime 1.28.0 before any conversion so optimizer ordering cannot change dependency availability.

Corrected run `30746349174` produced identical final metrics for all three requested optimizer labels (`maxAbs=0.022849321`, `MAE=0.007771059`, 97,411 violations). The `None` and `onnxoptimizer` files were byte-identical; the latter CLI skipped its requested pass set after rejecting an unknown pass name, so this result must not be described as an applied ONNX Optimizer transformation. Polygraphy reduced the graph from 651 to 259 nodes but preserved the same output. Optimization is therefore not the source of the UVDoc difference.

That run stopped before checkpoint comparison because ONNX shape inference returned unknown rank for the correctly named `PRelu.0` output. The diagnostic now uses the Paddle PIR shape only when ONNX supplies no rank at all; known ONNX ranks and dimensions remain strictly checked, and the actual runtime tensor must still match Paddle before metrics can be emitted. Each optimizer variant also records its ONNX SHA-256 and node count so a requested optimizer label cannot be mistaken for an effective graph rewrite.

Run `30746603436` completed all nine checkpoints and localized the first divergence to UVDoc's first bilinear resize. The Paddle output matched an independent half-pixel implementation within `1.79e-7`; ONNX Runtime matched align-corners within `1.79e-7`. Paddle 3.0.0 comparison run `30746827134` was identical to Paddle 3.2.1, ruling out a recent runtime regression.

The serialized PIR explicitly contains `align_corners=true`, and Paddle2ONNX correctly emits ONNX `coordinate_transformation_mode=align_corners`. Paddle's generic CPU interpolation kernel consumes this attribute, while its oneDNN interpolation kernel marks both `align_corners` and `align_mode` unused. Diagnostic run `30747102151` explicitly disabled oneDNN: the first-resize difference fell to `2.98e-7` and final UVDoc difference to `1.19e-6`, passing the unchanged common tolerance for all three ONNX optimizer variants. The blocker was therefore a backend-contaminated Paddle reference, not a Paddle2ONNX mapper defect; no fork or upstream converter patch is justified.

The formal source gate now shares the same explicit reference configuration and records `paddleOneDnn=false` in `gate-index.json`. Complete run `30747231341` subsequently passed the source validation and all three cross-platform C# jobs, closing Milestone 4A.
