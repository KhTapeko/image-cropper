from __future__ import annotations

from pathlib import Path
import tempfile
import tkinter as tk
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from image_cropper.core import CropBox, GifAnimation
from image_cropper.gui import ImageCropperApp, format_file_size, shade_outside_crop
from image_cropper.prefetch import DetectionResult, FileFingerprint


class OutsideShadeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = Image.new("RGBA", (4, 4), (120, 100, 80, 255))
        self.bounds = (1, 1, 3, 3)

    def test_zero_alpha_does_not_change_image(self) -> None:
        shaded = shade_outside_crop(self.image, self.bounds, 0)
        self.assertEqual(shaded.getpixel((0, 0)), (120, 100, 80, 255))
        self.assertEqual(shaded.getpixel((1, 1)), (120, 100, 80, 255))

    def test_middle_alpha_partially_darkens_only_outside(self) -> None:
        shaded = shade_outside_crop(self.image, self.bounds, 128)
        self.assertEqual(shaded.getpixel((0, 0)), (60, 50, 40, 255))
        self.assertEqual(shaded.getpixel((1, 1)), (120, 100, 80, 255))

    def test_full_alpha_makes_outside_black(self) -> None:
        shaded = shade_outside_crop(self.image, self.bounds, 255)
        self.assertEqual(shaded.getpixel((0, 0)), (0, 0, 0, 255))
        self.assertEqual(shaded.getpixel((2, 2)), (120, 100, 80, 255))


class FileSizeFormattingTests(unittest.TestCase):
    def test_uses_kb_below_one_mb_and_mb_at_or_above_it(self) -> None:
        self.assertEqual(format_file_size(1), "1 KB")
        self.assertEqual(format_file_size(8192), "8 KB")
        self.assertEqual(format_file_size(3 * 1024 * 1024 + 200_000), "3.19 MB")


class GuiSmokeTests(unittest.TestCase):
    def test_cached_static_result_enables_interaction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sample.png"
            Image.new("RGB", (100, 60)).save(source)
            root = tk.Tk()
            root.withdraw()
            try:
                app = ImageCropperApp(root, Path(directory), auto_initialize_detector=False)
                app.current_path = source
                app.current_image = Image.new("RGB", (100, 60))
                result = DetectionResult(
                    path=source,
                    fingerprint=FileFingerprint.capture(source),
                    kind="static",
                    image_size=(100, 60),
                    crop_box=CropBox(20, 5, 80, 55),
                    model_name="isnet-anime",
                    reliable=True,
                    outcome="primary",
                )
                app._apply_static_prefetch_result(result)

                self.assertEqual(app.crop_box, CropBox(20, 5, 80, 55))
                self.assertTrue(app._interaction_ready)
                self.assertEqual(str(app.confirm_button.cget("state")), "normal")
            finally:
                root.destroy()

    def test_unreliable_cached_result_falls_back_to_manual_full_crop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = tk.Tk()
            root.withdraw()
            try:
                app = ImageCropperApp(root, Path(directory), auto_initialize_detector=False)
                app.current_path = Path(directory) / "sample.png"
                app.current_image = Image.new("RGB", (100, 60))
                result = DetectionResult(
                    path=app.current_path,
                    fingerprint=None,
                    kind="static",
                    image_size=(100, 60),
                    crop_box=None,
                    model_name="birefnet-portrait",
                    reliable=False,
                    outcome="secondary_unreliable",
                )
                app._apply_static_prefetch_result(result)

                self.assertEqual(app.crop_box, CropBox(0, 0, 100, 60))
                self.assertTrue(app._interaction_ready)
                self.assertIn("仍不可靠", app.status_label.cget("text"))
            finally:
                root.destroy()

    def test_changed_source_discards_result_and_prioritizes_redetection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sample.png"
            Image.new("RGB", (20, 10), "white").save(source)
            old_fingerprint = FileFingerprint.capture(source)
            source.write_bytes(source.read_bytes() + b"changed")
            root = tk.Tk()
            root.withdraw()
            try:
                app = ImageCropperApp(root, Path(directory), auto_initialize_detector=False)
                prefetcher = Mock()
                app._prefetcher = prefetcher
                app.current_path = source
                app.current_image = Image.new("RGB", (20, 10))
                app._prefetch_results[source] = DetectionResult(
                    path=source,
                    fingerprint=old_fingerprint,
                    kind="static",
                    image_size=(20, 10),
                    crop_box=CropBox(1, 1, 19, 9),
                    model_name="isnet-anime",
                    reliable=True,
                    outcome="primary",
                )

                app._consume_current_prefetch_result(app._generation)

                self.assertNotIn(source, app._prefetch_results)
                prefetcher.reprioritize.assert_called_once_with(source)
                self.assertFalse(app._interaction_ready)
                self.assertIn("重新辨識", app.status_label.cget("text"))
            finally:
                root.destroy()

    def test_loading_each_image_resets_scale_to_one_hundred_percent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sample.png"
            Image.new("RGB", (20, 10)).save(source)
            root = tk.Tk()
            root.withdraw()
            try:
                app = ImageCropperApp(root, Path(directory), auto_initialize_detector=False)
                app.paths = [source]
                app.scale_percent_var.set("180")
                app._load_next()
                self.assertEqual(app.scale_percent_var.get(), "100")
                self.assertEqual(str(app.scale_spinbox.cget("state")), "disabled")
            finally:
                root.destroy()

    def test_scale_control_updates_output_summary_and_rejects_invalid_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sample.png"
            source.write_bytes(b"x" * 8192)
            root = tk.Tk()
            root.withdraw()
            try:
                app = ImageCropperApp(root, Path(directory), auto_initialize_detector=False)
                app.current_path = source
                app.current_image = Image.new("RGBA", (100, 50))
                app.crop_box = CropBox(10, 10, 70, 40)
                app._interaction_ready = True
                app._set_scale_control_enabled(True)
                app.scale_percent_var.set("150")
                self.assertIn("倍率 150%", app.output_info_label.cget("text"))
                self.assertIn("輸出 90 × 45", app.output_info_label.cget("text"))
                self.assertIn("來源 8 KB", app.output_info_label.cget("text"))
                self.assertEqual(str(app.confirm_button.cget("state")), "normal")

                app.scale_percent_var.set("49")
                self.assertIn("50～200", app.output_info_label.cget("text"))
                self.assertEqual(str(app.confirm_button.cget("state")), "disabled")
            finally:
                root.destroy()

    def test_static_save_failure_keeps_current_image_and_scale(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sample.png"
            Image.new("RGB", (20, 10)).save(source)
            root = tk.Tk()
            root.withdraw()
            try:
                app = ImageCropperApp(root, Path(directory), auto_initialize_detector=False)
                app.current_path = source
                app.current_image = Image.new("RGB", (20, 10))
                app.crop_box = CropBox(0, 0, 20, 10)
                app._interaction_ready = True
                app.scale_percent_var.set("180")
                app._set_scale_control_enabled(True)
                with (
                    patch("image_cropper.gui.save_cropped_image", side_effect=OSError("disk full")),
                    patch("image_cropper.gui.messagebox.showerror"),
                ):
                    app._confirm_crop()
                self.assertEqual(app.index, 0)
                self.assertEqual(app.scale_percent_var.get(), "180")
                self.assertTrue(app._interaction_ready)
                self.assertEqual(str(app.scale_spinbox.cget("state")), "normal")
                self.assertEqual(str(app.confirm_button.cget("state")), "normal")
            finally:
                root.destroy()

    def test_static_save_passes_valid_scale_to_core(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sample.png"
            Image.new("RGB", (20, 10)).save(source)
            root = tk.Tk()
            root.withdraw()
            try:
                app = ImageCropperApp(root, Path(directory), auto_initialize_detector=False)
                app.current_path = source
                app.current_image = Image.new("RGB", (20, 10))
                app.crop_box = CropBox(1, 1, 19, 9)
                app._interaction_ready = True
                app.scale_percent_var.set("150")
                app._set_scale_control_enabled(True)
                with (
                    patch("image_cropper.gui.save_cropped_image") as save,
                    patch.object(root, "after") as after,
                ):
                    app._confirm_crop()
                save.assert_called_once_with(
                    source,
                    Path(directory) / "output" / "sample.png",
                    CropBox(1, 1, 19, 9),
                    scale_percent=150,
                )
                self.assertEqual(app.success_count, 1)
                self.assertEqual(app.index, 1)
                after.assert_called_once_with(1, app._load_next)
            finally:
                root.destroy()

    def test_gif_save_failure_keeps_scale_and_restarts_preview(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = tk.Tk()
            root.withdraw()
            try:
                app = ImageCropperApp(root, Path(directory), auto_initialize_detector=False)
                app.current_path = Path(directory) / "sample.gif"
                app.current_animation = GifAnimation(
                    frames=(Image.new("RGBA", (20, 10)), Image.new("RGBA", (20, 10))),
                    durations=(30, 40),
                    disposals=(1, 1),
                    loop=0,
                )
                app.current_image = app.current_animation.frames[0]
                app.crop_box = CropBox(0, 0, 20, 10)
                app.scale_percent_var.set("180")
                app._saving = True
                with (
                    patch("image_cropper.gui.messagebox.showerror"),
                    patch.object(app, "_schedule_next_animation_frame") as restart_preview,
                ):
                    app._finish_gif_save(app._generation, "disk full")
                self.assertEqual(app.index, 0)
                self.assertEqual(app.scale_percent_var.get(), "180")
                self.assertTrue(app._interaction_ready)
                self.assertEqual(str(app.scale_spinbox.cget("state")), "normal")
                restart_preview.assert_called_once_with()
            finally:
                root.destroy()

    def test_empty_input_shows_completion_screen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = tk.Tk()
            root.withdraw()
            try:
                app = ImageCropperApp(root, Path(directory), auto_initialize_detector=False)
                app._show_completion()
                root.update_idletasks()
                self.assertEqual(app.file_label.cget("text"), "全部處理完成")
                self.assertIn("沒有可處理的圖片", app.status_label.cget("text"))
                self.assertEqual(app.confirm_button.cget("text"), "關閉")
            finally:
                root.destroy()

    def test_all_four_edges_are_draggable_hit_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = tk.Tk()
            root.withdraw()
            try:
                app = ImageCropperApp(root, Path(directory), auto_initialize_detector=False)
                app.current_image = Image.new("RGB", (100, 100))
                app.crop_box = CropBox(20, 20, 80, 80)
                app._display_origin = (0.0, 0.0)
                app._display_scale = 1.0
                self.assertEqual(app._hit_test(50, 20), "n")
                self.assertEqual(app._hit_test(80, 50), "e")
                self.assertEqual(app._hit_test(50, 80), "s")
                self.assertEqual(app._hit_test(20, 50), "w")
                self.assertEqual(app._hit_test(50, 50), "move")
            finally:
                root.destroy()

    def test_finished_gif_detection_starts_looping_preview_with_warning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sample.gif"
            Image.new("RGBA", (20, 10), "red").save(source, format="GIF")
            root = tk.Tk()
            root.withdraw()
            try:
                app = ImageCropperApp(root, Path(directory), auto_initialize_detector=False)
                app.current_path = source
                frames = (
                    Image.new("RGBA", (20, 10), "red"),
                    Image.new("RGBA", (20, 10), "blue"),
                )
                animation = GifAnimation(
                    frames=frames,
                    durations=(0, 40),
                    disposals=(1, 1),
                    loop=0,
                )
                delays: list[int] = []
                original_after = root.after

                def capture_after(delay: int, callback=None, *args):
                    if callback == app._advance_animation_frame:
                        delays.append(delay)
                        return "preview-timer"
                    return original_after(delay, callback, *args)

                root.after = capture_after  # type: ignore[method-assign]
                result = DetectionResult(
                    path=source,
                    fingerprint=FileFingerprint.capture(source),
                    kind="gif",
                    image_size=(20, 10),
                    crop_box=CropBox(1, 1, 19, 9),
                    model_name="isnet-anime",
                    reliable=True,
                    outcome="gif",
                    successful_samples=2,
                    total_samples=3,
                    error_count=1,
                )
                app._finish_gif_preview_load(
                    app._generation,
                    source,
                    result,
                    animation,
                    FileFingerprint.capture(source),
                    None,
                )
                self.assertIs(app.current_animation, animation)
                self.assertTrue(app._interaction_ready)
                self.assertIn("已使用 2 / 3", app.status_label.cget("text"))
                self.assertIn("1 格辨識失敗", app.status_label.cget("text"))
                self.assertEqual(delays, [20])
                app._animation_after_id = None
            finally:
                root.destroy()

    def test_failed_cuda_rebuild_stops_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = tk.Tk()
            root.withdraw()
            try:
                app = ImageCropperApp(root, Path(directory), auto_initialize_detector=False)
                with patch("image_cropper.gui.messagebox.showerror"):
                    app._stop_batch_for_device_error(
                        "CUDA 重建失敗：driver reset failed"
                    )
                self.assertTrue(app.completed)
                self.assertIn("批次已停止", app.file_label.cget("text"))
                self.assertIn("CUDA 重建失敗", app.status_label.cget("text"))
            finally:
                root.destroy()


if __name__ == "__main__":
    unittest.main()
