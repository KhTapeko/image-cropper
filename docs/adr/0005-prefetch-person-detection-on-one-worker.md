---
status: accepted
---

# Prefetch person detection on one persistent worker

Run image review and person detection as two main flows: the Tk UI owns display, crop, scale, and confirmation while one persistent background worker initializes the CUDA sessions and sequentially processes the entire detection queue. This hides later inference time behind the user's current-image adjustments without allowing concurrent model calls to compete for GPU memory.

Keep only the suggested crop, selected model, reliability, error, file size, and modification time as a prefetch result. Masks, decoded images, and GIF frames are not cached; the current GIF is decoded again on a short-lived preview-loading thread after its detection result is ready. Ordinary future-file failures are deferred until that file becomes current, while model rebuilds remain serialized with detection. GIF saving may continue to use its existing short-lived worker, so “two flows” does not impose a strict two-operating-system-thread limit.

Results are memory-only and are discarded after save or skip. A changed source file invalidates its result and is reprioritized after the currently running model call; files added after startup are not added to the queue.
