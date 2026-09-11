---
status: superseded by ADR-0004
---

# Isolate and prefer CUDA for GPU mode

Keep the established CPU environment and single-session execution unchanged because controlled tests showed that thread tuning improved only 1.2%, while two concurrent sessions were 30% slower and nearly doubled model memory. Provide GPU acceleration only through a separate environment and explicit launch mode, prefer the local GPU's CUDA provider for every supported image format, and require at least a 15% measured speed improvement before GPU mode is delivered.

GPU mode initializes CUDA in the background before the first detection and may allow unsupported nodes to execute on CPU within the same ONNX Runtime session. Session creation failure visibly falls back to CPU; a runtime GPU failure instead marks the current image as unprocessed, never mixes GPU and CPU detections, rebuilds CUDA once before the next image, and stops the batch if rebuilding fails.
