from __future__ import annotations

import gc
from threading import Lock

from PIL import Image

from .config import DETECTION_MODEL


class AnimePersonDetector:
    """Creates and reuses one explicitly selected isnet-anime session."""

    def __init__(self, preferred_device: str = "cpu") -> None:
        if preferred_device not in {"cpu", "cuda"}:
            raise ValueError(f"不支援的辨識裝置：{preferred_device}")
        self.preferred_device = preferred_device
        self._session = None
        self._device = ""
        self._lock = Lock()

    @property
    def initialized(self) -> bool:
        with self._lock:
            return self._session is not None

    @property
    def uses_cuda(self) -> bool:
        with self._lock:
            return self._device == "cuda" and self._session is not None

    @property
    def device_label(self) -> str:
        with self._lock:
            if self._device == "cuda":
                return "本機 GPU（CUDA）"
            if self._device == "cpu":
                return "CPU"
            return "尚未準備"

    def initialize(self) -> str | None:
        """Initialize the preferred provider, visibly falling back only at startup."""

        if self.preferred_device == "cpu":
            with self._lock:
                self._session = self._create_cpu_session()
                self._device = "cpu"
            return None

        try:
            session = self._create_cuda_session()
        except Exception as exc:
            fallback_reason = str(exc)
            with self._lock:
                self._session = self._create_cpu_session()
                self._device = "cpu"
            return fallback_reason

        with self._lock:
            self._session = session
            self._device = "cuda"
        return None

    def rebuild_cuda(self) -> None:
        """Rebuild CUDA once after a runtime failure; never fall back here."""

        with self._lock:
            old_session = self._session
            self._session = None
            self._device = ""
        del old_session
        gc.collect()
        session = self._create_cuda_session()
        with self._lock:
            self._session = session
            self._device = "cuda"

    @staticmethod
    def _create_cpu_session():
        from rembg import new_session

        return new_session(DETECTION_MODEL, providers=["CPUExecutionProvider"])

    @staticmethod
    def _create_cuda_session():
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDAExecutionProvider 不可用")

        from rembg import new_session

        session = new_session(
            DETECTION_MODEL,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        inner_session = getattr(session, "inner_session", None)
        providers = inner_session.get_providers() if inner_session is not None else []
        if not providers or providers[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"CUDA session 未啟用：{providers}")
        return session

    def create_mask(self, image: Image.Image) -> Image.Image:
        from rembg import remove

        with self._lock:
            session = self._session
        if session is None:
            raise RuntimeError("人物辨識器尚未準備完成")

        result = remove(
            image.convert("RGB"),
            session=session,
            only_mask=True,
        )
        if not isinstance(result, Image.Image):
            raise TypeError("人物辨識器未回傳有效遮罩")
        return result.convert("L")
