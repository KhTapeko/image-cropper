from __future__ import annotations

import threading
from queue import Empty, Queue
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageDraw, ImageTk

from .config import (
    CLOSEUP_MASK_MEAN_MIN,
    COMPONENT_JOIN_DISTANCE_RATIO,
    DARK_MASK_MEAN_MAX,
    DETECTION_PADDING_RATIO,
    MASK_HIGH_THRESHOLD,
    MASK_LOW_THRESHOLD,
    MASK_THRESHOLD,
    MIN_COMPONENT_AREA_RATIO,
    MIN_COMPONENT_RELATIVE_AREA,
    MIN_FOREGROUND_AREA_RATIO,
    OUTSIDE_SHADE_ALPHA,
    SMALL_BORDER_COMPONENT_MAX_AREA_RATIO,
)
from .core import (
    MAX_SCALE_PERCENT,
    MIN_SCALE_PERCENT,
    CropBox,
    GifAnimation,
    composite_transparency_for_detection,
    estimate_output_bytes,
    gif_sample_indices,
    load_gif_animation,
    load_oriented_image,
    mask_to_suggested_crop,
    output_size_for_crop,
    resolve_gif_crop,
    save_cropped_gif_atomic,
    save_cropped_image,
    scan_input_images,
)
from .detector import AnimePersonDetector


def shade_outside_crop(
    image: Image.Image,
    bounds: tuple[int, int, int, int],
    alpha: int,
) -> Image.Image:
    """Darken pixels outside bounds by alpha-compositing a black overlay."""

    display = image.convert("RGBA")
    overlay_alpha = min(255, max(0, int(alpha)))
    if overlay_alpha == 0:
        return display

    left, top, right, bottom = bounds
    overlay = Image.new("RGBA", display.size, (0, 0, 0, overlay_alpha))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle((left, top, right - 1, bottom - 1), fill=(0, 0, 0, 0))
    return Image.alpha_composite(display, overlay)


def format_file_size(byte_count: int) -> str:
    """Format a byte count as readable KB or MB text."""

    if byte_count < 1024 * 1024:
        kilobytes = 0 if byte_count <= 0 else max(1, (byte_count + 512) // 1024)
        return f"{kilobytes} KB"
    return f"{byte_count / (1024 * 1024):.2f} MB"


class ImageCropperApp:
    HANDLE_RADIUS = 7
    MIN_CROP_PIXELS = 10

    def __init__(
        self,
        root: tk.Tk,
        project_root: Path,
        *,
        preferred_device: str = "cpu",
        detector: AnimePersonDetector | None = None,
        auto_initialize_detector: bool = True,
    ) -> None:
        self.root = root
        self.project_root = project_root
        self.input_directory = project_root / "input"
        self.output_directory = project_root / "output"
        self.input_directory.mkdir(exist_ok=True)
        self.output_directory.mkdir(exist_ok=True)

        self.paths = scan_input_images(self.input_directory)
        self.index = 0
        self.success_count = 0
        self.failures: list[tuple[str, str]] = []
        self.completed = False

        self.detector = detector or AnimePersonDetector(preferred_device)
        self.preferred_device = preferred_device
        self._detector_ready = not auto_initialize_detector
        self._pending_detection: tuple[str, int, object] | None = None
        self._rebuilding_cuda = False
        self._worker_events: Queue[tuple[object, ...]] = Queue()
        self.current_path: Path | None = None
        self.current_image: Image.Image | None = None
        self.current_animation: GifAnimation | None = None
        self.current_frame_index = 0
        self.crop_box: CropBox | None = None
        self._generation = 0
        self._interaction_ready = False
        self._saving = False
        self._photo: ImageTk.PhotoImage | None = None
        self._display_scale = 1.0
        self._display_origin = (0.0, 0.0)
        self._drag_mode: str | None = None
        self._drag_start_image = (0, 0)
        self._drag_start_box: CropBox | None = None
        self._render_after_id: str | None = None
        self._animation_after_id: str | None = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(50, self._poll_worker_events)
        self.root.after(50, self._load_next)
        if auto_initialize_detector:
            self.device_label.configure(
                text=(
                    "辨識裝置：正在準備 NVIDIA GPU…"
                    if preferred_device == "cuda"
                    else "辨識裝置：正在準備 CPU…"
                )
            )
            threading.Thread(target=self._initialize_detector, daemon=True).start()
        else:
            self.device_label.configure(text="辨識裝置：測試模式")

    def _build_ui(self) -> None:
        self.root.title("人物圖片裁切器")
        self.root.geometry("1100x760")
        self.root.minsize(720, 520)

        header = ttk.Frame(self.root, padding=(12, 10, 12, 6))
        header.pack(fill="x")
        self.file_label = ttk.Label(header, text="準備中…", font=("Microsoft JhengHei UI", 12, "bold"))
        self.file_label.pack(side="left", fill="x", expand=True)
        self.progress_label = ttk.Label(header, text="", font=("Microsoft JhengHei UI", 10))
        self.progress_label.pack(side="right")
        self.device_label = ttk.Label(header, text="辨識裝置：準備中…", font=("Microsoft JhengHei UI", 9))
        self.device_label.pack(side="right", padx=(0, 16))

        self.canvas = tk.Canvas(self.root, background="#171717", highlightthickness=0, cursor="arrow")
        self.canvas.pack(fill="both", expand=True, padx=12, pady=6)
        self.canvas.bind("<Configure>", self._schedule_render)
        self.canvas.bind("<ButtonPress-1>", self._on_pointer_down)
        self.canvas.bind("<B1-Motion>", self._on_pointer_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_pointer_up)
        self.canvas.bind("<Motion>", self._on_hover)

        footer = ttk.Frame(self.root, padding=(12, 6, 12, 12))
        footer.pack(fill="x")
        self.status_label = ttk.Label(
            footer,
            text="",
            font=("Microsoft JhengHei UI", 10),
            anchor="w",
        )
        self.status_label.pack(fill="x")

        controls = ttk.Frame(footer)
        controls.pack(fill="x", pady=(6, 0))
        self.output_info_label = ttk.Label(
            controls,
            text="",
            font=("Microsoft JhengHei UI", 9),
            anchor="w",
        )
        self.output_info_label.pack(side="left", fill="x", expand=True)
        self.confirm_button = ttk.Button(
            controls,
            text="確認裁切與縮放",
            command=self._confirm_crop,
            state="disabled",
        )
        self.confirm_button.pack(side="right", ipadx=16, ipady=5)
        ttk.Label(controls, text="%").pack(side="right", padx=(3, 10))
        self.scale_percent_var = tk.StringVar(self.root, "100")
        self.scale_spinbox = ttk.Spinbox(
            controls,
            from_=MIN_SCALE_PERCENT,
            to=MAX_SCALE_PERCENT,
            increment=10,
            width=5,
            justify="right",
            textvariable=self.scale_percent_var,
            state="disabled",
        )
        self.scale_spinbox.pack(side="right")
        ttk.Label(controls, text="倍率").pack(side="right", padx=(10, 5))
        self.scale_percent_var.trace_add("write", self._on_scale_changed)

    def _parse_scale_percent(self) -> int | None:
        text = self.scale_percent_var.get().strip()
        if not text.isascii() or not text.isdecimal():
            return None
        value = int(text)
        if not MIN_SCALE_PERCENT <= value <= MAX_SCALE_PERCENT:
            return None
        return value

    def _on_scale_changed(self, *_args: object) -> None:
        self._update_output_info()

    def _set_scale_control_enabled(self, enabled: bool) -> None:
        self.scale_spinbox.configure(state="normal" if enabled else "disabled")
        self._update_output_info()

    def _update_output_info(self) -> None:
        if (
            self.current_path is None
            or self.current_image is None
            or self.crop_box is None
            or not self._interaction_ready
        ):
            self.output_info_label.configure(text="")
            return

        scale_percent = self._parse_scale_percent()
        if scale_percent is None:
            self.output_info_label.configure(
                text=f"倍率必須是 {MIN_SCALE_PERCENT}～{MAX_SCALE_PERCENT} 的整數"
            )
            self.confirm_button.configure(state="disabled")
            return

        box = self.crop_box.clamp(*self.current_image.size)
        output_size = output_size_for_crop(box, scale_percent)
        try:
            source_bytes = self.current_path.stat().st_size
        except OSError:
            source_bytes = 0
        estimated_bytes = estimate_output_bytes(
            source_bytes,
            self.current_image.size,
            output_size,
        )
        self.output_info_label.configure(
            text=(
                f"倍率 {scale_percent}%｜輸出 {output_size[0]} × {output_size[1]}｜"
                f"來源 {format_file_size(source_bytes)}｜"
                f"預估輸出約 {format_file_size(estimated_bytes)}"
            )
        )
        if not self._saving and not self.completed:
            self.confirm_button.configure(state="normal")

    def _initialize_detector(self) -> None:
        try:
            fallback_reason = self.detector.initialize()
            error = None
        except Exception as exc:
            fallback_reason = None
            error = str(exc)
        self._worker_events.put(
            ("detector_initialized", fallback_reason, error)
        )

    def _finish_detector_initialization(
        self,
        fallback_reason: str | None,
        error: str | None,
    ) -> None:
        if error is not None:
            self._stop_batch_for_device_error(f"辨識裝置初始化失敗：{error}")
            return

        self._detector_ready = True
        self.device_label.configure(text=f"辨識裝置：{self.detector.device_label}")
        if fallback_reason is not None:
            warning = f"GPU 無法使用，已改用 CPU：{fallback_reason}"
            self.status_label.configure(text=warning)
            messagebox.showwarning("GPU 無法使用", warning)
        self._start_pending_detection()

    def _start_pending_detection(self) -> None:
        pending = self._pending_detection
        if not self._detector_ready or pending is None:
            return
        self._pending_detection = None
        kind, generation, payload = pending
        if generation != self._generation:
            return
        if kind == "gif":
            assert isinstance(payload, Path)
            self._start_gif_detection(generation, payload)
        else:
            assert isinstance(payload, Image.Image)
            self._start_static_detection(generation, payload)

    def _waiting_for_detector_text(self) -> str:
        if self.preferred_device == "cuda":
            return "正在準備 NVIDIA GPU…"
        return "正在準備 CPU 人物辨識器…"

    def _load_next(self) -> None:
        self._cancel_animation_preview()
        if self.completed:
            return
        if self.index >= len(self.paths):
            self._show_completion()
            return

        self._generation += 1
        generation = self._generation
        self.current_path = self.paths[self.index]
        self.current_image = None
        self.current_animation = None
        self.current_frame_index = 0
        self.crop_box = None
        self._pending_detection = None
        self._interaction_ready = False
        self._saving = False
        self.scale_percent_var.set("100")
        self._set_scale_control_enabled(False)
        self.confirm_button.configure(state="disabled")
        self.file_label.configure(text=self.current_path.name)
        self.progress_label.configure(text=f"{self.index + 1} / {len(self.paths)}")
        self.canvas.delete("all")

        if self.current_path.suffix.casefold() == ".gif":
            if not self._detector_ready:
                self.status_label.configure(text=self._waiting_for_detector_text())
                self._pending_detection = ("gif", generation, self.current_path)
            else:
                self._start_gif_detection(generation, self.current_path)
            return

        self.status_label.configure(text="正在讀取圖片…")

        try:
            self.current_image = load_oriented_image(self.current_path)
        except Exception as exc:
            self.failures.append((self.current_path.name, f"讀取失敗：{exc}"))
            messagebox.showerror("圖片讀取失敗", f"無法讀取 {self.current_path.name}\n\n{exc}")
            self.index += 1
            self.root.after(1, self._load_next)
            return

        width, height = self.current_image.size
        self.file_label.configure(text=f"{self.current_path.name}　{width} × {height}")
        self.crop_box = CropBox(0, 0, width, height)
        self._render_current()

        image_for_detection = self.current_image.copy()
        if not self._detector_ready:
            self.status_label.configure(text=self._waiting_for_detector_text())
            self._pending_detection = ("static", generation, image_for_detection)
            return
        self._start_static_detection(generation, image_for_detection)

    def _start_static_detection(self, generation: int, image: Image.Image) -> None:
        self.status_label.configure(text="正在辨識人物…")
        worker = threading.Thread(
            target=self._detect_in_background,
            args=(generation, image),
            daemon=True,
        )
        worker.start()

    def _start_gif_detection(self, generation: int, path: Path) -> None:
        self.status_label.configure(text="正在讀取 GIF…")
        worker = threading.Thread(
            target=self._load_and_detect_gif,
            args=(generation, path),
            daemon=True,
        )
        worker.start()

    def _detect_in_background(self, generation: int, image: Image.Image) -> None:
        was_cuda = self.detector.uses_cuda
        try:
            mask = self.detector.create_mask(image)
            suggestion = self._suggest_crop(mask, image.size, DETECTION_PADDING_RATIO)
            error = None
            gpu_runtime_failure = False
        except Exception as exc:
            suggestion = None
            error = str(exc)
            gpu_runtime_failure = was_cuda
        self._worker_events.put(
            (
                "static_detection",
                generation,
                suggestion,
                error,
                gpu_runtime_failure,
            )
        )

    @staticmethod
    def _suggest_crop(
        mask: Image.Image,
        image_size: tuple[int, int],
        padding_ratio: float,
    ) -> CropBox | None:
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
            padding_ratio=padding_ratio,
        )

    def _load_and_detect_gif(self, generation: int, path: Path) -> None:
        try:
            animation = load_gif_animation(path)
            sample_indices = gif_sample_indices(animation.frame_count)
            suggestions: list[CropBox] = []
            error_count = 0
            for completed, frame_index in enumerate(sample_indices, start=1):
                was_cuda = self.detector.uses_cuda
                try:
                    detection_image = composite_transparency_for_detection(
                        animation.frames[frame_index]
                    )
                    mask = self.detector.create_mask(detection_image)
                    suggestion = self._suggest_crop(mask, animation.size, 0.0)
                except Exception as exc:
                    if was_cuda:
                        self._worker_events.put(
                            (
                                "gif_detection_complete",
                                generation,
                                None,
                                None,
                                0,
                                len(sample_indices),
                                0,
                                str(exc),
                                True,
                            )
                        )
                        return
                    suggestion = None
                    error_count += 1
                if suggestion is not None:
                    suggestions.append(suggestion)
                self._worker_events.put(
                    ("gif_detection_progress", generation, completed, len(sample_indices))
                )

            crop_box = resolve_gif_crop(
                suggestions,
                animation.size,
                error_count=error_count,
                padding_ratio=DETECTION_PADDING_RATIO,
            )

            self._worker_events.put(
                (
                    "gif_detection_complete",
                    generation,
                    animation,
                    crop_box,
                    len(sample_indices) - error_count,
                    len(sample_indices),
                    error_count,
                    None,
                    False,
                )
            )
        except Exception as exc:
            self._worker_events.put(
                (
                    "gif_detection_complete",
                    generation,
                    None,
                    None,
                    0,
                    0,
                    0,
                    str(exc),
                    False,
                )
            )

    def _poll_worker_events(self) -> None:
        try:
            while True:
                event = self._worker_events.get_nowait()
                event_type = event[0]
                if event_type == "detector_initialized":
                    _kind, fallback_reason, error = event
                    self._finish_detector_initialization(fallback_reason, error)
                elif event_type == "static_detection":
                    _kind, generation, suggestion, error, gpu_runtime_failure = event
                    self._finish_detection(
                        generation,
                        suggestion,
                        error,
                        gpu_runtime_failure,
                    )
                elif event_type == "gif_detection_progress":
                    _kind, generation, completed, total = event
                    self._show_gif_detection_progress(generation, completed, total)
                elif event_type == "gif_detection_complete":
                    self._finish_gif_detection(*event[1:])
                elif event_type == "gif_save_progress":
                    _kind, generation, completed, total = event
                    if generation == self._generation:
                        self.status_label.configure(
                            text=f"正在儲存 GIF：{completed} / {total} 個影格"
                        )
                elif event_type == "gif_save_validating":
                    _kind, generation = event
                    if generation == self._generation:
                        self.status_label.configure(text="正在驗證 GIF…")
                elif event_type == "gif_save_complete":
                    _kind, generation, error = event
                    self._finish_gif_save(generation, error)
                elif event_type == "cuda_rebuild_complete":
                    _kind, generation, error = event
                    self._finish_cuda_rebuild(generation, error)
        except Empty:
            pass
        if self.root.winfo_exists():
            self.root.after(50, self._poll_worker_events)

    def _finish_detection(
        self,
        generation: int,
        suggestion: CropBox | None,
        error: str | None,
        gpu_runtime_failure: bool = False,
    ) -> None:
        if generation != self._generation or self.current_image is None:
            return
        if gpu_runtime_failure:
            self._handle_gpu_runtime_failure(error or "未知 CUDA 錯誤")
            return
        if suggestion is None:
            width, height = self.current_image.size
            self.crop_box = CropBox(0, 0, width, height)
            if error:
                self.status_label.configure(text=f"人物辨識失敗，請手動調整裁切框：{error}")
            else:
                self.status_label.configure(text="未能可靠辨識人物，請手動調整裁切框")
        else:
            self.crop_box = suggestion
            self.status_label.configure(text="請調整裁切框與輸出倍率，確認後立即儲存")
        self._interaction_ready = True
        self._set_scale_control_enabled(True)
        self._render_current()

    def _show_gif_detection_progress(
        self,
        generation: int,
        completed: int,
        total: int,
    ) -> None:
        if generation != self._generation:
            return
        self.status_label.configure(
            text=f"正在辨識 GIF：{completed} / {total} 個取樣影格"
        )

    def _finish_gif_detection(
        self,
        generation: int,
        animation: GifAnimation | None,
        crop_box: CropBox | None,
        successful_samples: int,
        total_samples: int,
        error_count: int,
        error: str | None,
        gpu_runtime_failure: bool = False,
    ) -> None:
        if generation != self._generation:
            return
        if gpu_runtime_failure:
            self._handle_gpu_runtime_failure(error or "未知 CUDA 錯誤")
            return
        if error is not None or animation is None or crop_box is None:
            assert self.current_path is not None
            reason = error or "GIF 處理失敗"
            self.failures.append((self.current_path.name, reason))
            messagebox.showerror(
                "GIF 處理失敗",
                f"無法處理 {self.current_path.name}\n\n{reason}",
            )
            self.index += 1
            self.root.after(1, self._load_next)
            return

        self.current_animation = animation
        self.current_frame_index = 0
        self.current_image = animation.frames[0]
        self.crop_box = crop_box
        width, height = animation.size
        assert self.current_path is not None
        self.file_label.configure(text=f"{self.current_path.name}　{width} × {height}")
        if error_count:
            self.status_label.configure(
                text=(
                    f"已使用 {successful_samples} / {total_samples} 個取樣影格；"
                    f"{error_count} 格辨識失敗，請確認裁切範圍"
                )
            )
        else:
            self.status_label.configure(
                text="請調整共用裁切框與輸出倍率，確認後立即儲存"
            )
        self._interaction_ready = True
        self._set_scale_control_enabled(True)
        self._render_current()
        self._schedule_next_animation_frame()

    def _handle_gpu_runtime_failure(self, reason: str) -> None:
        if self._rebuilding_cuda or self.current_path is None:
            return
        self._cancel_animation_preview()
        self._interaction_ready = False
        self._set_scale_control_enabled(False)
        self.confirm_button.configure(state="disabled")
        self.failures.append((self.current_path.name, f"GPU 辨識失敗：{reason}"))
        messagebox.showerror(
            "GPU 辨識失敗",
            f"{self.current_path.name} 已列為未處理圖片。\n\n{reason}",
        )
        self._rebuilding_cuda = True
        self.status_label.configure(text="GPU 辨識失敗，正在重建 CUDA…")
        generation = self._generation

        def rebuild_worker() -> None:
            try:
                self.detector.rebuild_cuda()
                error = None
            except Exception as exc:
                error = str(exc)
            self._worker_events.put(("cuda_rebuild_complete", generation, error))

        threading.Thread(target=rebuild_worker, daemon=True).start()

    def _finish_cuda_rebuild(self, generation: int, error: str | None) -> None:
        if generation != self._generation:
            return
        self._rebuilding_cuda = False
        if error is not None:
            self.device_label.configure(text="辨識裝置：CUDA 無法重建")
            self._stop_batch_for_device_error(f"CUDA 重建失敗：{error}")
            return
        self.device_label.configure(text=f"辨識裝置：{self.detector.device_label}")
        self.index += 1
        self.root.after(1, self._load_next)

    def _stop_batch_for_device_error(self, reason: str) -> None:
        self._cancel_animation_preview()
        self.completed = True
        self._pending_detection = None
        self._interaction_ready = False
        self._set_scale_control_enabled(False)
        self.current_path = None
        self.current_image = None
        self.current_animation = None
        self.crop_box = None
        self.canvas.delete("all")
        self.file_label.configure(text="辨識裝置錯誤，批次已停止")
        self.progress_label.configure(text="")
        self.status_label.configure(text=reason)
        self.confirm_button.configure(text="關閉", state="normal", command=self.root.destroy)
        messagebox.showerror("批次已停止", reason)

    def _schedule_next_animation_frame(self) -> None:
        if self.current_animation is None or self.current_animation.frame_count <= 1:
            return
        duration = max(20, self.current_animation.durations[self.current_frame_index])
        self._animation_after_id = self.root.after(duration, self._advance_animation_frame)

    def _advance_animation_frame(self) -> None:
        self._animation_after_id = None
        if self.current_animation is None or self._saving:
            return
        self.current_frame_index = (
            self.current_frame_index + 1
        ) % self.current_animation.frame_count
        self.current_image = self.current_animation.frames[self.current_frame_index]
        self._render_current()
        self._schedule_next_animation_frame()

    def _cancel_animation_preview(self) -> None:
        if self._animation_after_id is not None:
            try:
                self.root.after_cancel(self._animation_after_id)
            except tk.TclError:
                pass
            self._animation_after_id = None

    def _schedule_render(self, _event=None) -> None:
        if self._render_after_id is not None:
            self.root.after_cancel(self._render_after_id)
        self._render_after_id = self.root.after(30, self._render_current)

    def _render_current(self) -> None:
        self._render_after_id = None
        if self.current_image is None or self.crop_box is None:
            return
        canvas_width = max(2, self.canvas.winfo_width())
        canvas_height = max(2, self.canvas.winfo_height())
        image_width, image_height = self.current_image.size
        scale = min(canvas_width / image_width, canvas_height / image_height)
        display_width = max(1, round(image_width * scale))
        display_height = max(1, round(image_height * scale))
        origin_x = (canvas_width - display_width) / 2
        origin_y = (canvas_height - display_height) / 2

        display = self.current_image.resize((display_width, display_height), Image.Resampling.LANCZOS).convert("RGBA")
        left = round(self.crop_box.left * scale)
        top = round(self.crop_box.top * scale)
        right = round(self.crop_box.right * scale)
        bottom = round(self.crop_box.bottom * scale)
        display = shade_outside_crop(
            display,
            (left, top, right, bottom),
            OUTSIDE_SHADE_ALPHA,
        )
        draw = ImageDraw.Draw(display, "RGBA")
        draw.rectangle((left, top, right - 1, bottom - 1), outline=(255, 211, 66, 255), width=3)

        radius = self.HANDLE_RADIUS
        handle_points = (
            (left, top),
            (right, top),
            (left, bottom),
            (right, bottom),
            ((left + right) // 2, top),
            (right, (top + bottom) // 2),
            ((left + right) // 2, bottom),
            (left, (top + bottom) // 2),
        )
        for x, y in handle_points:
            draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=(255, 211, 66, 255))

        self._photo = ImageTk.PhotoImage(display)
        self.canvas.delete("all")
        self.canvas.create_image(origin_x, origin_y, image=self._photo, anchor="nw")
        self._display_scale = scale
        self._display_origin = (origin_x, origin_y)

    def _canvas_to_image(self, canvas_x: float, canvas_y: float) -> tuple[int, int]:
        origin_x, origin_y = self._display_origin
        image_x = round((canvas_x - origin_x) / self._display_scale)
        image_y = round((canvas_y - origin_y) / self._display_scale)
        if self.current_image is None:
            return image_x, image_y
        width, height = self.current_image.size
        return min(max(0, image_x), width), min(max(0, image_y), height)

    def _hit_test(self, canvas_x: float, canvas_y: float) -> str | None:
        if self.crop_box is None:
            return None
        origin_x, origin_y = self._display_origin
        scale = self._display_scale
        points = {
            "nw": (origin_x + self.crop_box.left * scale, origin_y + self.crop_box.top * scale),
            "ne": (origin_x + self.crop_box.right * scale, origin_y + self.crop_box.top * scale),
            "sw": (origin_x + self.crop_box.left * scale, origin_y + self.crop_box.bottom * scale),
            "se": (origin_x + self.crop_box.right * scale, origin_y + self.crop_box.bottom * scale),
        }
        tolerance = self.HANDLE_RADIUS + 5
        for corner, (x, y) in points.items():
            if abs(canvas_x - x) <= tolerance and abs(canvas_y - y) <= tolerance:
                return corner

        left = origin_x + self.crop_box.left * scale
        top = origin_y + self.crop_box.top * scale
        right = origin_x + self.crop_box.right * scale
        bottom = origin_y + self.crop_box.bottom * scale
        if left <= canvas_x <= right and abs(canvas_y - top) <= tolerance:
            return "n"
        if left <= canvas_x <= right and abs(canvas_y - bottom) <= tolerance:
            return "s"
        if top <= canvas_y <= bottom and abs(canvas_x - left) <= tolerance:
            return "w"
        if top <= canvas_y <= bottom and abs(canvas_x - right) <= tolerance:
            return "e"

        image_x, image_y = self._canvas_to_image(canvas_x, canvas_y)
        if (
            self.crop_box.left <= image_x <= self.crop_box.right
            and self.crop_box.top <= image_y <= self.crop_box.bottom
        ):
            return "move"
        return None

    def _on_pointer_down(self, event: tk.Event) -> None:
        if not self._interaction_ready or self.crop_box is None:
            return
        mode = self._hit_test(event.x, event.y)
        if mode is None:
            return
        self._drag_mode = mode
        self._drag_start_image = self._canvas_to_image(event.x, event.y)
        self._drag_start_box = self.crop_box

    def _on_pointer_move(self, event: tk.Event) -> None:
        if (
            self._drag_mode is None
            or self._drag_start_box is None
            or self.current_image is None
        ):
            return
        image_x, image_y = self._canvas_to_image(event.x, event.y)
        width, height = self.current_image.size
        if self._drag_mode == "move":
            start_x, start_y = self._drag_start_image
            self.crop_box = self._drag_start_box.move(
                image_x - start_x,
                image_y - start_y,
                width,
                height,
            )
        else:
            self.crop_box = self._drag_start_box.resize_corner(
                self._drag_mode,
                image_x,
                image_y,
                width,
                height,
                self.MIN_CROP_PIXELS,
            )
        self._render_current()
        self._update_output_info()

    def _on_pointer_up(self, _event: tk.Event) -> None:
        self._drag_mode = None
        self._drag_start_box = None

    def _on_hover(self, event: tk.Event) -> None:
        if not self._interaction_ready:
            self.canvas.configure(cursor="arrow")
            return
        hit = self._hit_test(event.x, event.y)
        cursors = {
            "nw": "size_nw_se",
            "se": "size_nw_se",
            "ne": "size_ne_sw",
            "sw": "size_ne_sw",
            "n": "sb_v_double_arrow",
            "s": "sb_v_double_arrow",
            "e": "sb_h_double_arrow",
            "w": "sb_h_double_arrow",
            "move": "fleur",
        }
        self.canvas.configure(cursor=cursors.get(hit, "arrow"))

    def _confirm_crop(self) -> None:
        if self.current_path is None or self.crop_box is None:
            return
        scale_percent = self._parse_scale_percent()
        if scale_percent is None:
            self._update_output_info()
            return
        output_path = self.output_directory / self.current_path.name
        if output_path.exists() and not messagebox.askyesno(
            "確認覆寫",
            f"output 已有同名檔案：\n{self.current_path.name}\n\n是否覆寫？",
        ):
            self.status_label.configure(text="尚未儲存；可繼續調整後再次確認")
            return

        if self.current_animation is not None:
            self._start_gif_save(output_path, scale_percent)
            return

        self._saving = True
        self._interaction_ready = False
        self._set_scale_control_enabled(False)
        self.confirm_button.configure(state="disabled")
        try:
            save_cropped_image(
                self.current_path,
                output_path,
                self.crop_box,
                scale_percent=scale_percent,
            )
        except Exception as exc:
            messagebox.showerror("儲存失敗", f"無法儲存 {self.current_path.name}\n\n{exc}")
            self._saving = False
            self._interaction_ready = True
            self.status_label.configure(text="儲存失敗；請確認設定後再次嘗試")
            self._set_scale_control_enabled(True)
            return
        else:
            self.success_count += 1
        self._saving = False
        self.index += 1
        self.root.after(1, self._load_next)

    def _start_gif_save(self, output_path: Path, scale_percent: int) -> None:
        assert self.current_animation is not None
        assert self.crop_box is not None
        generation = self._generation
        animation = self.current_animation
        crop_box = self.crop_box
        self._cancel_animation_preview()
        self._saving = True
        self._interaction_ready = False
        self._set_scale_control_enabled(False)
        self.confirm_button.configure(state="disabled")
        self.status_label.configure(
            text=f"正在儲存 GIF：0 / {animation.frame_count} 個影格"
        )

        def save_worker() -> None:
            try:
                save_cropped_gif_atomic(
                    animation,
                    output_path,
                    crop_box,
                    scale_percent=scale_percent,
                    progress_callback=lambda completed, total: self._worker_events.put(
                        ("gif_save_progress", generation, completed, total)
                    ),
                    validating_callback=lambda: self._worker_events.put(
                        ("gif_save_validating", generation)
                    ),
                )
                error = None
            except Exception as exc:
                error = str(exc)
            self._worker_events.put(("gif_save_complete", generation, error))

        threading.Thread(target=save_worker, daemon=True).start()

    def _finish_gif_save(self, generation: int, error: str | None) -> None:
        if generation != self._generation:
            return
        self._saving = False
        assert self.current_path is not None
        if error is not None:
            messagebox.showerror(
                "儲存失敗",
                f"無法儲存 {self.current_path.name}\n\n{error}",
            )
            self._interaction_ready = True
            self.status_label.configure(text="GIF 儲存失敗；請確認設定後再次嘗試")
            self._set_scale_control_enabled(True)
            self._schedule_next_animation_frame()
            return
        self.success_count += 1
        self.index += 1
        self.root.after(1, self._load_next)

    def _show_completion(self) -> None:
        self._cancel_animation_preview()
        self.completed = True
        self.current_path = None
        self.current_image = None
        self.current_animation = None
        self.crop_box = None
        self._interaction_ready = False
        self._set_scale_control_enabled(False)
        self.canvas.delete("all")
        self.file_label.configure(text="全部處理完成")
        self.progress_label.configure(text="")
        summary = f"成功：{self.success_count}　失敗：{len(self.failures)}"
        if not self.paths:
            summary = "input 中沒有可處理的圖片"
        elif self.failures:
            failed_names = "、".join(name for name, _reason in self.failures)
            summary += f"\n未處理：{failed_names}"
        self.status_label.configure(text=summary)
        self.confirm_button.configure(text="關閉", state="normal", command=self.root.destroy)

    def _on_close(self) -> None:
        if self._saving:
            self.root.bell()
            return
        if self.completed or self.current_path is None:
            self._cancel_animation_preview()
            self.root.destroy()
            return
        if messagebox.askyesno("確認離開", "目前圖片尚未儲存，確定要離開嗎？"):
            self._cancel_animation_preview()
            self.root.destroy()


def run_app(project_root: Path, *, preferred_device: str = "cpu") -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    ImageCropperApp(root, project_root, preferred_device=preferred_device)
    root.mainloop()
