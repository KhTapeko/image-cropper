from __future__ import annotations

from pathlib import Path
from queue import Empty, Queue
import tempfile
import unittest
from unittest.mock import Mock

from PIL import Image

from image_cropper.core import CropBox
from image_cropper.prefetch import (
    DetectionCompleted,
    DetectionPrefetcher,
    DetectorInitialized,
)


class FakeDetector:
    def __init__(self, *, fail_first: bool = False) -> None:
        self.portrait_available = True
        self.fail_first = fail_first
        self.create_calls = 0
        self.rebuild_calls = 0

    def initialize(self, portrait_start_callback=None):
        if portrait_start_callback is not None:
            portrait_start_callback()
        return None

    def create_mask(self, image: Image.Image) -> Image.Image:
        self.create_calls += 1
        if self.fail_first and self.create_calls == 1:
            raise RuntimeError("CUDA failure")
        mask = Image.new("L", image.size, 0)
        mask.paste(255, (2, 1, image.width - 2, image.height - 1))
        return mask

    def create_portrait_mask(self, image: Image.Image) -> Image.Image:
        return self.create_mask(image)

    def rebuild_cuda(self) -> None:
        self.rebuild_calls += 1

    def rebuild_portrait_cuda(self) -> None:
        self.portrait_available = True


class DetectionPrefetcherTests(unittest.TestCase):
    def _collect_results(
        self,
        events: Queue[object],
        count: int,
    ) -> list[DetectionCompleted]:
        results: list[DetectionCompleted] = []
        while len(results) < count:
            try:
                event = events.get(timeout=3)
            except Empty as exc:
                self.fail(f"背景辨識沒有完成：{exc}")
            if isinstance(event, DetectionCompleted):
                results.append(event)
        return results

    def test_processes_all_paths_sequentially_and_caches_only_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / f"{index}.png" for index in range(3)]
            for path in paths:
                Image.new("RGB", (20, 10), "white").save(path)

            events: Queue[object] = Queue()
            detector = FakeDetector()
            prefetcher = DetectionPrefetcher(detector, paths, events.put)
            prefetcher.start()
            completed = self._collect_results(events, len(paths))
            prefetcher.stop()

            self.assertEqual([event.result.path for event in completed], paths)
            self.assertTrue(all(event.result.reliable for event in completed))
            self.assertTrue(
                all(event.result.crop_box == CropBox(1, 0, 19, 10) for event in completed)
            )
            self.assertTrue(
                all(not hasattr(event.result, "mask") for event in completed)
            )
            self.assertTrue(
                all(not hasattr(event.result, "image") for event in completed)
            )

    def test_primary_failure_rebuilds_once_and_continues_with_next_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "first.png", root / "second.png"]
            for path in paths:
                Image.new("RGB", (20, 10), "white").save(path)

            events: Queue[object] = Queue()
            detector = FakeDetector(fail_first=True)
            prefetcher = DetectionPrefetcher(detector, paths, events.put)
            prefetcher.start()
            completed = self._collect_results(events, len(paths))
            prefetcher.stop()

            self.assertEqual(completed[0].result.outcome, "primary_error")
            self.assertEqual(completed[1].result.outcome, "primary")
            self.assertEqual(detector.rebuild_calls, 1)

    def test_initialization_event_precedes_detection_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.png"
            Image.new("RGB", (20, 10), "white").save(path)
            events: Queue[object] = Queue()
            prefetcher = DetectionPrefetcher(FakeDetector(), [path], events.put)
            prefetcher.start()

            event_types: list[type[object]] = []
            while DetectionCompleted not in event_types:
                event_types.append(type(events.get(timeout=3)))
            prefetcher.stop()

            self.assertLess(
                event_types.index(DetectorInitialized),
                event_types.index(DetectionCompleted),
            )

    def test_gif_detection_never_calls_portrait_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.gif"
            frames = [
                Image.new("RGBA", (20, 10), "red"),
                Image.new("RGBA", (20, 10), "blue"),
            ]
            frames[0].save(
                path,
                format="GIF",
                save_all=True,
                append_images=frames[1:],
                duration=(30, 30),
                loop=0,
            )
            events: Queue[object] = Queue()
            detector = FakeDetector()
            detector.create_portrait_mask = Mock(
                side_effect=AssertionError("GIF 不可使用二次模型")
            )
            prefetcher = DetectionPrefetcher(detector, [path], events.put)
            prefetcher.start()
            completed = self._collect_results(events, 1)
            prefetcher.stop()

            self.assertEqual(completed[0].result.kind, "gif")
            self.assertEqual(completed[0].result.outcome, "gif")
            detector.create_portrait_mask.assert_not_called()


if __name__ == "__main__":
    unittest.main()
