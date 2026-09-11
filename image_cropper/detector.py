from __future__ import annotations

import gc
from threading import Lock
from typing import Callable

from PIL import Image

from .config import DETECTION_MODEL, SECONDARY_DETECTION_MODEL


class AnimePersonDetector:
    """Own the primary anime and optional portrait CUDA sessions."""

    def __init__(self) -> None:
        self._session = None
        self._portrait_session = None
        self._lock = Lock()

    @property
    def initialized(self) -> bool:
        with self._lock:
            return self._session is not None

    @property
    def portrait_available(self) -> bool:
        with self._lock:
            return self._portrait_session is not None

    @property
    def uses_cuda(self) -> bool:
        return self.initialized

    @property
    def device_label(self) -> str:
        return "本機 GPU（CUDA）" if self.initialized else "尚未準備"

    def initialize(
        self,
        portrait_start_callback: Callable[[], None] | None = None,
    ) -> str | None:
        """Initialize both models in order; only the portrait model is optional."""

        primary_session = self._create_cuda_session(DETECTION_MODEL)
        with self._lock:
            self._session = primary_session

        if portrait_start_callback is not None:
            portrait_start_callback()
        try:
            portrait_session = self._create_cuda_session(SECONDARY_DETECTION_MODEL)
        except Exception as exc:
            with self._lock:
                self._portrait_session = None
            return str(exc)

        with self._lock:
            self._portrait_session = portrait_session
        return None

    def rebuild_cuda(self) -> None:
        """Rebuild the primary CUDA session once; never switch to CPU mode."""

        with self._lock:
            old_session = self._session
            self._session = None
        del old_session
        gc.collect()
        session = self._create_cuda_session(DETECTION_MODEL)
        with self._lock:
            self._session = session

    def rebuild_portrait_cuda(self) -> None:
        """Rebuild only the optional portrait CUDA session."""

        with self._lock:
            old_session = self._portrait_session
            self._portrait_session = None
        del old_session
        gc.collect()
        session = self._create_cuda_session(SECONDARY_DETECTION_MODEL)
        with self._lock:
            self._portrait_session = session

    @staticmethod
    def _create_cuda_session(model_name: str):
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDAExecutionProvider 不可用")

        from rembg import new_session

        cuda_provider: str | tuple[str, dict[str, str]] = "CUDAExecutionProvider"
        if model_name == SECONDARY_DETECTION_MODEL:
            cuda_provider = (
                "CUDAExecutionProvider",
                {"cudnn_conv_use_max_workspace": "0"},
            )

        session = new_session(
            model_name,
            providers=[cuda_provider, "CPUExecutionProvider"],
        )
        inner_session = getattr(session, "inner_session", None)
        providers = inner_session.get_providers() if inner_session is not None else []
        if not providers or providers[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"CUDA session 未啟用：{providers}")
        return session

    def create_mask(self, image: Image.Image) -> Image.Image:
        with self._lock:
            session = self._session
        if session is None:
            raise RuntimeError("人物辨識器尚未準備完成")
        return self._create_mask_with_session(image, session)

    def create_portrait_mask(self, image: Image.Image) -> Image.Image:
        with self._lock:
            session = self._portrait_session
        if session is None:
            raise RuntimeError("二次人物辨識器無法使用")
        return self._create_mask_with_session(image, session)

    @staticmethod
    def _create_mask_with_session(image: Image.Image, session: object) -> Image.Image:
        from rembg import remove

        result = remove(
            image.convert("RGB"),
            session=session,
            only_mask=True,
        )
        if not isinstance(result, Image.Image):
            raise TypeError("人物辨識器未回傳有效遮罩")
        return result.convert("L")
