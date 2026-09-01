from __future__ import annotations

from unittest.mock import patch
import unittest

from image_cropper.detector import AnimePersonDetector


class DetectorDeviceTests(unittest.TestCase):
    def test_cpu_mode_initializes_only_cpu(self) -> None:
        detector = AnimePersonDetector("cpu")
        cpu_session = object()
        with patch.object(detector, "_create_cpu_session", return_value=cpu_session) as create_cpu:
            fallback = detector.initialize()
        self.assertIsNone(fallback)
        self.assertTrue(detector.initialized)
        self.assertFalse(detector.uses_cuda)
        self.assertEqual(detector.device_label, "CPU")
        create_cpu.assert_called_once_with()

    def test_cuda_mode_uses_cuda_when_session_creation_succeeds(self) -> None:
        detector = AnimePersonDetector("cuda")
        cuda_session = object()
        with patch.object(detector, "_create_cuda_session", return_value=cuda_session):
            fallback = detector.initialize()
        self.assertIsNone(fallback)
        self.assertTrue(detector.uses_cuda)
        self.assertIn("CUDA", detector.device_label)

    def test_cuda_startup_failure_visibly_falls_back_to_cpu(self) -> None:
        detector = AnimePersonDetector("cuda")
        with (
            patch.object(detector, "_create_cuda_session", side_effect=RuntimeError("driver")),
            patch.object(detector, "_create_cpu_session", return_value=object()),
        ):
            fallback = detector.initialize()
        self.assertEqual(fallback, "driver")
        self.assertFalse(detector.uses_cuda)
        self.assertEqual(detector.device_label, "CPU")

    def test_runtime_rebuild_never_falls_back_to_cpu(self) -> None:
        detector = AnimePersonDetector("cuda")
        with patch.object(detector, "_create_cuda_session", return_value=object()):
            detector.initialize()
        with patch.object(detector, "_create_cuda_session", side_effect=RuntimeError("oom")):
            with self.assertRaisesRegex(RuntimeError, "oom"):
                detector.rebuild_cuda()
        self.assertFalse(detector.initialized)
        self.assertFalse(detector.uses_cuda)


if __name__ == "__main__":
    unittest.main()
