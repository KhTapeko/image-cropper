---
status: accepted
---

# Sample and union GIF person bounds

Animated GIFs use one shared crop box so their canvas does not move during playback. Detect the first frame, every fifth frame thereafter, and the final frame on fully composited images; union every valid unpadded person bound without cross-frame outlier rejection, then add padding once around the union. This deliberately favors never clipping a pose over producing the tightest crop, while bounding inference work for ordinary short GIFs without dropping any output frames.

Re-encode every composited frame inside that shared crop and preserve frame count, timing, looping, disposal, and transparency as decoded playback behavior rather than byte-for-byte palette identity. Write to a temporary GIF, fully decode and verify it, and only then replace the destination so a failed animation export cannot destroy an existing output.
