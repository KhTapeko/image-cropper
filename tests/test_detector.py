from __future__ import annotations

import sys
from unittest.mock import Mock, call, patch
import unittest

from image_cropper.config import DETECTION_MODEL, SECONDARY_DETECTION_MODEL
from image_cropper.detector import AnimePersonDetector


class DetectorDeviceTests(unittest.TestCase):
    def test_initializes_primary_then_portrait_cuda_sessions(self) -> None:
        detector = AnimePersonDetector()
        primary_session = object()
        portrait_session = object()
        with patch.object(
            detector,
            "_create_cuda_session",
            side_effect=(primary_session, portrait_session),
        ) as create:
            portrait_error = detector.initialize()

        self.assertIsNone(portrait_error)
        self.assertTrue(detector.initialized)
        self.assertTrue(detector.uses_cuda)
        self.assertTrue(detector.portrait_available)
        self.assertIn("CUDA", detector.device_label)
        self.assertEqual(
            create.call_args_list,
            [call(DETECTION_MODEL), call(SECONDARY_DETECTION_MODEL)],
        )

    def test_primary_cuda_failure_prevents_initialization(self) -> None:
        detector = AnimePersonDetector()
        with patch.object(
            detector,
            "_create_cuda_session",
            side_effect=RuntimeError("driver"),
        ):
            with self.assertRaisesRegex(RuntimeError, "driver"):
                detector.initialize()

        self.assertFalse(detector.initialized)
        self.assertFalse(detector.portrait_available)

    def test_portrait_failure_keeps_primary_cuda_available(self) -> None:
        detector = AnimePersonDetector()
        with patch.object(
            detector,
            "_create_cuda_session",
            side_effect=(object(), RuntimeError("portrait oom")),
        ):
            portrait_error = detector.initialize()

        self.assertEqual(portrait_error, "portrait oom")
        self.assertTrue(detector.initialized)
        self.assertFalse(detector.portrait_available)

    def test_primary_runtime_rebuild_never_falls_back_to_cpu(self) -> None:
        detector = AnimePersonDetector()
        with patch.object(detector, "_create_cuda_session", return_value=object()):
            detector.initialize()
        with patch.object(
            detector,
            "_create_cuda_session",
            side_effect=RuntimeError("oom"),
        ):
            with self.assertRaisesRegex(RuntimeError, "oom"):
                detector.rebuild_cuda()
        self.assertFalse(detector.initialized)
        self.assertTrue(detector.portrait_available)

    def test_portrait_runtime_rebuild_does_not_replace_primary(self) -> None:
        detector = AnimePersonDetector()
        with patch.object(detector, "_create_cuda_session", return_value=object()):
            detector.initialize()
        with patch.object(
            detector,
            "_create_cuda_session",
            side_effect=RuntimeError("portrait oom"),
        ):
            with self.assertRaisesRegex(RuntimeError, "portrait oom"):
                detector.rebuild_portrait_cuda()
        self.assertTrue(detector.initialized)
        self.assertFalse(detector.portrait_available)

    def test_only_portrait_session_limits_cudnn_workspace(self) -> None:
        detector = AnimePersonDetector()
        ort = Mock()
        ort.get_available_providers.return_value = ["CUDAExecutionProvider"]
        session = Mock()
        session.inner_session.get_providers.return_value = ["CUDAExecutionProvider"]
        rembg = Mock()
        rembg.new_session.return_value = session

        with patch.dict(sys.modules, {"onnxruntime": ort, "rembg": rembg}):
            detector._create_cuda_session(SECONDARY_DETECTION_MODEL)
            portrait_providers = rembg.new_session.call_args.kwargs["providers"]
            detector._create_cuda_session(DETECTION_MODEL)
            primary_providers = rembg.new_session.call_args.kwargs["providers"]

        self.assertEqual(
            portrait_providers,
            [
                (
                    "CUDAExecutionProvider",
                    {"cudnn_conv_use_max_workspace": "0"},
                ),
                "CPUExecutionProvider",
            ],
        )
        self.assertEqual(
            primary_providers,
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )


if __name__ == "__main__":
    unittest.main()
