# Agent working agreement

Read these files before changing behavior:

1. `SPEC.md`
2. `docs/handoff/CURRENT.md`
3. `docs/architecture.md`
4. The manifest or gate document owned by the change

- Keep the worker, runtime package and model package independently versioned.
- Native inference stays out of the NEOCR host process.
- Do not replace a failed model with another model or a Python runtime. A gate failure stops the milestone and requires an explicit architecture decision.
- Generated models, downloaded archives, reference tensors and runtime binaries never enter Git.
- Pin every downloaded artifact by SHA-256 before it can enter a release package.
- Keep source conversion/reference inference in the Linux gate. Shipping inference code is C#.
- Update `docs/handoff/CURRENT.md` with verified commands, failures and the next bounded task before committing.

Required verification:

```sh
dotnet build NEOCR.Paddle.slnx
python3 -m unittest discover -s tests -v
python3 tools/verify_manifest.py manifests/models/pp-ocrv5-mobile-full-v1.json
dotnet format NEOCR.Paddle.slnx --no-restore --verify-no-changes
dotnet list NEOCR.Paddle.slnx package --vulnerable --include-transitive
```
