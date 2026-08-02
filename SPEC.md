# NEOCR Paddle

Intent: independently ship a C# Paddle OCR worker, ONNX Runtime packages and pinned model packs for the NEOCR host.

Done when:

1. The five-model document pipeline passes Paddle-to-ONNX conversion and raw-output parity gates.
2. A C# worker implements the NEOCR framed protocol using independently installed runtime and model packages.
3. CPU packages support macOS arm64 and Windows x64 before optional accelerated backends are added.

## Do not build

1. Do not load ONNX Runtime or model code into the NEOCR host process.
2. Do not ship Python as the production worker or silently fall back to Python.
3. Do not silently remove document orientation, UVDoc unwarping, text-line orientation, detection or recognition when a gate fails.
4. Do not auto-download runtime or model packages from the application.
5. Do not combine worker, runtime and model versions into one release identity.
6. Do not add CUDA, Vulkan or another accelerator before the CPU vertical slice is correct.
7. Do not commit model weights, generated ONNX files, golden tensors or native runtime binaries.
