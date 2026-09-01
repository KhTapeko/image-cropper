from __future__ import annotations

from pathlib import Path
import tempfile
import tkinter as tk
import unittest
from unittest.mock import patch

from PIL import Image

from image_cropper.core import CropBox, GifAnimation
from image_cropper.gui import ImageCropperApp, shade_outside_crop


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


class GuiSmokeTests(unittest.TestCase):
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
            root = tk.Tk()
            root.withdraw()
            try:
                app = ImageCropperApp(root, Path(directory), auto_initialize_detector=False)
                app.current_path = Path(directory) / "sample.gif"
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
                app._finish_gif_detection(
                    app._generation,
                    animation,
                    CropBox(1, 1, 19, 9),
                    2,
                    3,
                    1,
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

    def test_gpu_runtime_error_marks_current_image_and_rebuilds_without_cpu_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = tk.Tk()
            root.withdraw()
            try:
                app = ImageCropperApp(root, Path(directory), auto_initialize_detector=False)
                app.current_path = Path(directory) / "sample.png"
                app.current_image = Image.new("RGB", (20, 10))
                app.crop_box = CropBox(0, 0, 20, 10)
                with (
                    patch.object(app.detector, "rebuild_cuda", return_value=None) as rebuild,
                    patch("image_cropper.gui.messagebox.showerror"),
                ):
                    app._handle_gpu_runtime_failure("CUDA out of memory")
                    event = app._worker_events.get(timeout=2)
                self.assertEqual(len(app.failures), 1)
                self.assertIn("GPU 辨識失敗", app.failures[0][1])
                rebuild.assert_called_once_with()
                self.assertEqual(event[0], "cuda_rebuild_complete")
            finally:
                root.destroy()

    def test_failed_cuda_rebuild_stops_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = tk.Tk()
            root.withdraw()
            try:
                app = ImageCropperApp(root, Path(directory), auto_initialize_detector=False)
                with patch("image_cropper.gui.messagebox.showerror"):
                    app._finish_cuda_rebuild(app._generation, "driver reset failed")
                self.assertTrue(app.completed)
                self.assertIn("批次已停止", app.file_label.cget("text"))
                self.assertIn("CUDA 重建失敗", app.status_label.cget("text"))
            finally:
                root.destroy()


if __name__ == "__main__":
    unittest.main()
