from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import PriorityQueue
from threading import Event, Lock, Thread
from typing import Callable

from PIL import Image

from .config import (
    CLOSEUP_MASK_MEAN_MIN,
    COMPONENT_JOIN_DISTANCE_RATIO,
    DARK_MASK_MEAN_MAX,
    DETECTION_MODEL,
    DETECTION_PADDING_RATIO,
    MASK_HIGH_THRESHOLD,
    MASK_LOW_THRESHOLD,
    MASK_THRESHOLD,
    MIN_COMPONENT_AREA_RATIO,
    MIN_COMPONENT_RELATIVE_AREA,
    MIN_FOREGROUND_AREA_RATIO,
    SECONDARY_DETECTION_MODEL,
    SMALL_BORDER_COMPONENT_MAX_AREA_RATIO,
    UNRELIABLE_CROP_AREA_RATIO,
)
from .core import (
    CropBox,
    add_crop_padding,
    composite_transparency_for_detection,
    gif_sample_indices,
    is_unreliable_suggested_crop,
    load_gif_animation,
    load_oriented_image,
    mask_to_suggested_crop,
    resolve_gif_crop,
)
from .detector import AnimePersonDetector


@dataclass(frozen=True)
class FileFingerprint:
    size: int
    modified_ns: int

    @classmethod
    def capture(cls, path: Path) -> FileFingerprint:
        stat = path.stat()
        return cls(size=stat.st_size, modified_ns=stat.st_mtime_ns)


@dataclass(frozen=True)
class DetectionResult:
    path: Path
    fingerprint: FileFingerprint | None
    kind: str
    image_size: tuple[int, int] | None
    crop_box: CropBox | None
    model_name: str | None
    reliable: bool
    outcome: str
    error: str | None = None
    successful_samples: int = 0
    total_samples: int = 0
    error_count: int = 0


@dataclass(frozen=True)
class DetectorInitializingPortrait:
    pass


@dataclass(frozen=True)
class DetectorInitialized:
    portrait_error: str | None
    error: str | None


@dataclass(frozen=True)
class DetectionCompleted:
    result: DetectionResult


@dataclass(frozen=True)
class GifDetectionProgress:
    path: Path
    completed: int
    total: int


@dataclass(frozen=True)
class PortraitAvailabilityChanged:
    error: str | None


@dataclass(frozen=True)
class PrimaryCudaUnavailable:
    error: str


class DetectionPrefetcher:
    """Initialize the models and sequentially detect every queued image."""

    def __init__(
        self,
        detector: AnimePersonDetector,
        paths: list[Path],
        event_sink: Callable[[object], None],
    ) -> None:
        self._detector = detector
        self._event_sink = event_sink
        self._jobs: PriorityQueue[tuple[int, int, int, Path | None]] = PriorityQueue()
        self._versions: dict[Path, int] = {}
        self._versions_lock = Lock()
        self._stop = Event()
        self._sequence = 0
        self._thread: Thread | None = None
        for path in paths:
            self._enqueue(path, priority=1, replace=False)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def reprioritize(self, path: Path) -> None:
        """Invalidate an old job/result and put this path at the queue front."""

        self._enqueue(path, priority=0, replace=True)

    def stop(self) -> None:
        self._stop.set()
        with self._versions_lock:
            self._sequence += 1
            sequence = self._sequence
        self._jobs.put((-1, sequence, -1, None))

    def _enqueue(self, path: Path, *, priority: int, replace: bool) -> None:
        with self._versions_lock:
            version = self._versions.get(path, -1)
            if replace or version < 0:
                version += 1
                self._versions[path] = version
            self._sequence += 1
            sequence = self._sequence
        self._jobs.put((priority, sequence, version, path))

    def _run(self) -> None:
        try:
            portrait_error = self._detector.initialize(
                lambda: self._event_sink(DetectorInitializingPortrait())
            )
        except Exception as exc:
            self._event_sink(DetectorInitialized(None, str(exc)))
            return

        self._event_sink(DetectorInitialized(portrait_error, None))
        while not self._stop.is_set():
            _priority, _sequence, version, path = self._jobs.get()
            if path is None or self._stop.is_set():
                return
            with self._versions_lock:
                if self._versions.get(path) != version:
                    continue

            result = self._detect_path(path)
            if self._stop.is_set():
                return
            self._event_sink(DetectionCompleted(result))

            if result.outcome == "primary_error":
                try:
                    self._detector.rebuild_cuda()
                except Exception as exc:
                    self._event_sink(PrimaryCudaUnavailable(str(exc)))
                    return
            elif result.outcome == "secondary_error":
                try:
                    self._detector.rebuild_portrait_cuda()
                    portrait_error = None
                except Exception as exc:
                    portrait_error = str(exc)
                self._event_sink(PortraitAvailabilityChanged(portrait_error))

    def _detect_path(self, path: Path) -> DetectionResult:
        try:
            fingerprint = FileFingerprint.capture(path)
        except Exception as exc:
            return self._read_error(path, None, exc)

        if path.suffix.casefold() == ".gif":
            return self._detect_gif(path, fingerprint)
        return self._detect_static(path, fingerprint)

    def _detect_static(
        self,
        path: Path,
        fingerprint: FileFingerprint,
    ) -> DetectionResult:
        try:
            image = load_oriented_image(path)
        except Exception as exc:
            return self._read_error(path, fingerprint, exc)

        try:
            mask = self._detector.create_mask(image)
            primary_crop = self._suggest_crop(mask, image.size)
        except Exception as exc:
            return DetectionResult(
                path=path,
                fingerprint=fingerprint,
                kind="static",
                image_size=image.size,
                crop_box=None,
                model_name=DETECTION_MODEL,
                reliable=False,
                outcome="primary_error",
                error=str(exc),
            )

        if not is_unreliable_suggested_crop(
            primary_crop,
            image.size,
            UNRELIABLE_CROP_AREA_RATIO,
        ):
            assert primary_crop is not None
            return DetectionResult(
                path=path,
                fingerprint=fingerprint,
                kind="static",
                image_size=image.size,
                crop_box=add_crop_padding(
                    primary_crop,
                    image.size,
                    DETECTION_PADDING_RATIO,
                ),
                model_name=DETECTION_MODEL,
                reliable=True,
                outcome="primary",
            )

        if not self._detector.portrait_available:
            return DetectionResult(
                path=path,
                fingerprint=fingerprint,
                kind="static",
                image_size=image.size,
                crop_box=None,
                model_name=DETECTION_MODEL,
                reliable=False,
                outcome="secondary_unavailable",
            )

        try:
            portrait_mask = self._detector.create_portrait_mask(image)
            portrait_crop = self._suggest_crop(portrait_mask, image.size)
        except Exception as exc:
            return DetectionResult(
                path=path,
                fingerprint=fingerprint,
                kind="static",
                image_size=image.size,
                crop_box=None,
                model_name=SECONDARY_DETECTION_MODEL,
                reliable=False,
                outcome="secondary_error",
                error=str(exc),
            )

        if is_unreliable_suggested_crop(
            portrait_crop,
            image.size,
            UNRELIABLE_CROP_AREA_RATIO,
        ):
            return DetectionResult(
                path=path,
                fingerprint=fingerprint,
                kind="static",
                image_size=image.size,
                crop_box=None,
                model_name=SECONDARY_DETECTION_MODEL,
                reliable=False,
                outcome="secondary_unreliable",
            )

        assert portrait_crop is not None
        return DetectionResult(
            path=path,
            fingerprint=fingerprint,
            kind="static",
            image_size=image.size,
            crop_box=add_crop_padding(
                portrait_crop,
                image.size,
                DETECTION_PADDING_RATIO,
            ),
            model_name=SECONDARY_DETECTION_MODEL,
            reliable=True,
            outcome="secondary",
        )

    def _detect_gif(
        self,
        path: Path,
        fingerprint: FileFingerprint,
    ) -> DetectionResult:
        try:
            animation = load_gif_animation(path)
        except Exception as exc:
            return self._read_error(path, fingerprint, exc, kind="gif")

        sample_indices = gif_sample_indices(animation.frame_count)
        suggestions: list[CropBox] = []
        error_count = 0
        for completed, frame_index in enumerate(sample_indices, start=1):
            if self._stop.is_set():
                return DetectionResult(
                    path,
                    fingerprint,
                    "gif",
                    animation.size,
                    None,
                    DETECTION_MODEL,
                    False,
                    "cancelled",
                )
            try:
                detection_image = composite_transparency_for_detection(
                    animation.frames[frame_index]
                )
                mask = self._detector.create_mask(detection_image)
                suggestion = self._suggest_crop(mask, animation.size)
            except Exception as exc:
                return DetectionResult(
                    path=path,
                    fingerprint=fingerprint,
                    kind="gif",
                    image_size=animation.size,
                    crop_box=None,
                    model_name=DETECTION_MODEL,
                    reliable=False,
                    outcome="primary_error",
                    error=str(exc),
                    successful_samples=completed - 1 - error_count,
                    total_samples=len(sample_indices),
                    error_count=error_count,
                )
            if suggestion is not None:
                suggestions.append(suggestion)
            self._event_sink(GifDetectionProgress(path, completed, len(sample_indices)))

        try:
            crop_box = resolve_gif_crop(
                suggestions,
                animation.size,
                error_count=error_count,
                padding_ratio=DETECTION_PADDING_RATIO,
            )
        except Exception as exc:
            return DetectionResult(
                path=path,
                fingerprint=fingerprint,
                kind="gif",
                image_size=animation.size,
                crop_box=None,
                model_name=DETECTION_MODEL,
                reliable=False,
                outcome="gif_error",
                error=str(exc),
                successful_samples=len(sample_indices) - error_count,
                total_samples=len(sample_indices),
                error_count=error_count,
            )

        return DetectionResult(
            path=path,
            fingerprint=fingerprint,
            kind="gif",
            image_size=animation.size,
            crop_box=crop_box,
            model_name=DETECTION_MODEL,
            reliable=bool(suggestions),
            outcome="gif",
            successful_samples=len(sample_indices) - error_count,
            total_samples=len(sample_indices),
            error_count=error_count,
        )

    @staticmethod
    def _read_error(
        path: Path,
        fingerprint: FileFingerprint | None,
        error: Exception,
        *,
        kind: str = "static",
    ) -> DetectionResult:
        return DetectionResult(
            path=path,
            fingerprint=fingerprint,
            kind=kind,
            image_size=None,
            crop_box=None,
            model_name=None,
            reliable=False,
            outcome="read_error",
            error=str(error),
        )

    @staticmethod
    def _suggest_crop(mask: Image.Image, image_size: tuple[int, int]) -> CropBox | None:
        return mask_to_suggested_crop(
            mask,
            image_size,
            threshold=MASK_THRESHOLD,
            low_threshold=MASK_LOW_THRESHOLD,
            high_threshold=MASK_HIGH_THRESHOLD,
            dark_mask_mean_max=DARK_MASK_MEAN_MAX,
            closeup_mask_mean_min=CLOSEUP_MASK_MEAN_MIN,
            min_foreground_area_ratio=MIN_FOREGROUND_AREA_RATIO,
            min_component_area_ratio=MIN_COMPONENT_AREA_RATIO,
            min_component_relative_area=MIN_COMPONENT_RELATIVE_AREA,
            component_join_distance_ratio=COMPONENT_JOIN_DISTANCE_RATIO,
            small_border_component_max_area_ratio=SMALL_BORDER_COMPONENT_MAX_AREA_RATIO,
            padding_ratio=0.0,
        )
