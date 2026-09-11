---
status: accepted
---

# Require CUDA and retry unreliable static-image detection

Require the primary CUDA session to initialize before image processing and remove the standalone CPU mode and whole-image CPU fallback. Keep `CPUExecutionProvider` only as an internal fallback for CUDA-unsupported nodes within the same ONNX Runtime session. This supersedes ADR 0002 because silently changing the execution mode conflicts with predictable GPU-only behavior.

Initialize and retain the `isnet-anime` and `birefnet-portrait` sessions sequentially at startup. Static PNG, JPEG, and WebP images use `isnet-anime` first, then use `birefnet-portrait` once when the unpadded first suggestion is absent or covers at least 95% of the image. The two results are never combined; an absent or at-least-95% unpadded second suggestion falls back to the full image for manual adjustment. GIF files, including single-frame GIFs, never use the portrait session.

Failure of the optional portrait session does not prevent the primary workflow from starting. A portrait-session runtime failure falls back to manual adjustment for the current image and triggers one portrait-only CUDA rebuild before the next eligible static image; a failed rebuild disables secondary detection for the rest of the run. The larger portrait model increases startup time, download size, and GPU memory use in exchange for substantially better subject isolation on visually busy portrait screenshots.
