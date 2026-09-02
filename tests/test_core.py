from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from PIL import Image, features

from image_cropper.core import (
    CropBox,
    GifAnimation,
    UnsupportedAnimatedImageError,
    composite_transparency_for_detection,
    estimate_output_bytes,
    gif_sample_indices,
    load_gif_animation,
    load_oriented_image,
    mask_to_suggested_crop,
    output_size_for_crop,
    resize_image_high_quality,
    resolve_gif_crop,
    save_cropped_gif_atomic,
    save_cropped_image,
    scan_input_images,
    sort_paths_by_creation_desc,
    union_crop_boxes,
)


class CropBoxTests(unittest.TestCase):
    def test_move_is_limited_to_image(self) -> None:
        box = CropBox(10, 20, 50, 80)
        self.assertEqual(box.move(-999, 999, 100, 100), CropBox(0, 40, 40, 100))

    def test_resize_corner_respects_minimum_and_bounds(self) -> None:
        box = CropBox(20, 20, 80, 80)
        resized = box.resize_corner("nw", 99, -20, 100, 100, min_size=10)
        self.assertEqual(resized, CropBox(70, 0, 80, 80))

    def test_resize_each_edge_changes_only_that_edge(self) -> None:
        box = CropBox(20, 20, 80, 80)
        self.assertEqual(box.resize_corner("n", 50, 10, 100, 100), CropBox(20, 10, 80, 80))
        self.assertEqual(box.resize_corner("s", 50, 90, 100, 100), CropBox(20, 20, 80, 90))
        self.assertEqual(box.resize_corner("w", 10, 50, 100, 100), CropBox(10, 20, 80, 80))
        self.assertEqual(box.resize_corner("e", 90, 50, 100, 100), CropBox(20, 20, 90, 80))


class OutputScalingTests(unittest.TestCase):
    def test_output_size_rounds_half_up_and_never_reaches_zero(self) -> None:
        self.assertEqual(output_size_for_crop(CropBox(0, 0, 333, 5), 50), (167, 3))
        self.assertEqual(output_size_for_crop(CropBox(0, 0, 1, 1), 50), (1, 1))
        self.assertEqual(output_size_for_crop(CropBox(0, 0, 7, 9), 200), (14, 18))

    def test_output_size_rejects_non_integer_or_out_of_range_percent(self) -> None:
        for invalid in (49, 201, 100.0, True):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    output_size_for_crop(CropBox(0, 0, 10, 10), invalid)  # type: ignore[arg-type]

    def test_output_size_estimate_uses_actual_output_pixel_ratio(self) -> None:
        self.assertEqual(estimate_output_bytes(4_000_000, (2000, 1000), (1500, 750)), 2_250_000)

    def test_transparent_resize_avoids_dark_color_fringe(self) -> None:
        image = Image.new("RGBA", (3, 1), (0, 0, 0, 0))
        image.putpixel((1, 0), (255, 255, 255, 255))
        resized = resize_image_high_quality(image, (9, 1))
        translucent_pixels = [
            pixel for pixel in resized.get_flattened_data() if 0 < pixel[3] < 255
        ]
        self.assertTrue(translucent_pixels)
        self.assertTrue(all(min(pixel[:3]) >= 250 for pixel in translucent_pixels))

class MaskSuggestionTests(unittest.TestCase):
    def test_adds_three_percent_padding_and_clamps(self) -> None:
        mask = Image.new("L", (100, 100), 0)
        mask.paste(255, (10, 20, 60, 80))
        suggestion = mask_to_suggested_crop(mask, mask.size)
        self.assertEqual(suggestion, CropBox(8, 18, 62, 82))

    def test_rejects_tiny_detection(self) -> None:
        mask = Image.new("L", (100, 100), 0)
        mask.paste(255, (0, 0, 5, 5))
        self.assertIsNone(mask_to_suggested_crop(mask, mask.size))

    def test_sparse_distant_noise_does_not_create_a_large_box(self) -> None:
        mask = Image.new("L", (100, 100), 0)
        mask.paste(255, (30, 25, 60, 75))
        mask.putpixel((0, 0), 255)
        mask.putpixel((99, 99), 255)
        suggestion = mask_to_suggested_crop(mask, mask.size, padding_ratio=0)
        self.assertEqual(suggestion, CropBox(30, 25, 60, 75))

    def test_rejects_sparse_foreground_even_when_its_bounds_are_large(self) -> None:
        mask = Image.new("L", (100, 100), 0)
        mask.paste(255, (10, 10, 15, 15))
        mask.paste(255, (85, 85, 90, 90))
        self.assertIsNone(mask_to_suggested_crop(mask, mask.size))

    def test_distant_regions_cannot_combine_to_pass_minimum_area(self) -> None:
        mask = Image.new("L", (100, 100), 0)
        mask.paste(255, (10, 10, 16, 20))
        mask.paste(255, (80, 80, 86, 90))
        self.assertIsNone(mask_to_suggested_crop(mask, mask.size))

    def test_nearby_fragments_form_one_reliable_subject_group(self) -> None:
        mask = Image.new("L", (100, 100), 0)
        mask.paste(255, (40, 30, 46, 40))
        mask.paste(255, (40, 42, 46, 52))
        suggestion = mask_to_suggested_crop(mask, mask.size, padding_ratio=0)
        self.assertEqual(suggestion, CropBox(40, 30, 46, 52))

    def test_tiny_nearby_noise_cannot_chain_into_subject_group(self) -> None:
        mask = Image.new("L", (100, 100), 0)
        mask.paste(255, (30, 25, 60, 75))
        mask.paste(255, (61, 25, 67, 35))
        suggestion = mask_to_suggested_crop(mask, mask.size, padding_ratio=0)
        self.assertEqual(suggestion, CropBox(30, 25, 60, 75))

    def test_low_mean_mask_uses_low_threshold_for_dark_subject(self) -> None:
        mask = Image.new("L", (100, 100), 0)
        mask.paste(16, (20, 20, 50, 70))
        suggestion = mask_to_suggested_crop(mask, mask.size, padding_ratio=0)
        self.assertEqual(suggestion, CropBox(20, 20, 50, 70))

    def test_high_mean_mask_uses_high_threshold_for_closeup_subject(self) -> None:
        mask = Image.new("L", (100, 100), 16)
        mask.paste(255, (20, 10, 80, 90))
        suggestion = mask_to_suggested_crop(mask, mask.size, padding_ratio=0)
        self.assertEqual(suggestion, CropBox(20, 10, 80, 90))

    def test_keeps_multiple_significant_subject_components(self) -> None:
        mask = Image.new("L", (100, 100), 0)
        mask.paste(255, (10, 20, 35, 70))
        mask.paste(255, (65, 30, 80, 65))
        suggestion = mask_to_suggested_crop(mask, mask.size, padding_ratio=0)
        self.assertEqual(suggestion, CropBox(10, 20, 80, 70))

    def test_drops_small_component_touching_image_border(self) -> None:
        mask = Image.new("L", (100, 100), 0)
        mask.paste(255, (30, 30, 65, 70))
        mask.paste(255, (0, 40, 5, 50))
        suggestion = mask_to_suggested_crop(mask, mask.size, padding_ratio=0)
        self.assertEqual(suggestion, CropBox(30, 30, 65, 70))

    def test_accepts_detection_that_fills_the_image(self) -> None:
        mask = Image.new("L", (100, 100), 255)
        self.assertEqual(mask_to_suggested_crop(mask, mask.size), CropBox(0, 0, 100, 100))


class GifCropSuggestionTests(unittest.TestCase):
    def test_sampling_includes_first_every_fifth_and_final_frame(self) -> None:
        self.assertEqual(gif_sample_indices(1), (0,))
        self.assertEqual(gif_sample_indices(6), (0, 5))
        self.assertEqual(gif_sample_indices(7), (0, 5, 6))
        self.assertEqual(gif_sample_indices(12), (0, 5, 10, 11))

    def test_union_adds_padding_once_after_combining_raw_bounds(self) -> None:
        result = union_crop_boxes(
            [CropBox(20, 30, 40, 70), CropBox(60, 20, 80, 60)],
            (100, 100),
            padding_ratio=0.1,
        )
        self.assertEqual(result, CropBox(14, 15, 86, 75))

    def test_detection_background_replaces_hidden_transparent_rgb(self) -> None:
        image = Image.new("RGBA", (2, 1), (255, 0, 0, 0))
        image.putpixel((1, 0), (10, 20, 30, 255))
        flattened = composite_transparency_for_detection(image)
        self.assertEqual(flattened.getpixel((0, 0)), (128, 128, 128))
        self.assertEqual(flattened.getpixel((1, 0)), (10, 20, 30))

    def test_valid_gif_crop_is_used_even_when_other_samples_error(self) -> None:
        result = resolve_gif_crop(
            [CropBox(20, 20, 80, 80)],
            (100, 100),
            error_count=2,
            padding_ratio=0,
        )
        self.assertEqual(result, CropBox(20, 20, 80, 80))

    def test_all_clean_misses_fall_back_to_full_gif_canvas(self) -> None:
        result = resolve_gif_crop([], (100, 80), error_count=0, padding_ratio=0.03)
        self.assertEqual(result, CropBox(0, 0, 100, 80))

    def test_any_model_error_without_a_valid_crop_fails_the_gif(self) -> None:
        with self.assertRaises(RuntimeError):
            resolve_gif_crop([], (100, 80), error_count=1, padding_ratio=0.03)


class InputScanningTests(unittest.TestCase):
    def test_creation_time_descending_with_filename_tiebreaker(self) -> None:
        paths = [Path("10.png"), Path("2.png"), Path("new.png")]
        times = {"10.png": 1, "2.png": 1, "new.png": 2}
        ordered = sort_paths_by_creation_desc(paths, lambda path: times[path.name])
        self.assertEqual([path.name for path in ordered], ["new.png", "10.png", "2.png"])

    def test_gifs_are_sorted_after_static_images_with_newest_gif_first(self) -> None:
        paths = [Path("new.gif"), Path("old.png"), Path("old.gif"), Path("new.jpg")]
        times = {"new.gif": 4, "old.png": 1, "old.gif": 2, "new.jpg": 3}
        ordered = sort_paths_by_creation_desc(paths, lambda path: times[path.name])
        self.assertEqual(
            [path.name for path in ordered],
            ["new.jpg", "old.png", "new.gif", "old.gif"],
        )

    def test_scanner_ignores_unsupported_files_and_subdirectories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("a.PNG", "b.jpg", "c.jpeg", "d.WebP", "e.GIF", "notes.txt"):
                (root / name).touch()
            (root / "nested").mkdir()
            (root / "nested" / "hidden.png").touch()
            found = scan_input_images(root)
            self.assertEqual(
                {path.name for path in found},
                {"a.PNG", "b.jpg", "c.jpeg", "d.WebP", "e.GIF"},
            )


class ImageSavingTests(unittest.TestCase):
    @staticmethod
    def _pattern(width: int = 20, height: int = 16) -> Image.Image:
        image = Image.new("RGBA", (width, height))
        for y in range(height):
            for x in range(width):
                image.putpixel((x, y), (x * 9 % 256, y * 13 % 256, (x + y) * 7 % 256, 50 + x * 5))
        return image

    def test_png_crop_is_exact_and_keeps_alpha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            output = root / "output.png"
            image = self._pattern()
            image.save(source)
            box = CropBox(3, 2, 14, 12)
            save_cropped_image(source, output, box)
            with Image.open(output) as result:
                self.assertEqual(result.mode, "RGBA")
                self.assertEqual(result.size, (11, 10))
                self.assertEqual(result.tobytes(), image.crop(box.as_tuple()).tobytes())

    def test_png_crop_can_be_resized_with_alpha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            output = root / "output.png"
            self._pattern().save(source)
            save_cropped_image(
                source,
                output,
                CropBox(3, 2, 14, 12),
                scale_percent=150,
            )
            with Image.open(output) as result:
                self.assertEqual(result.mode, "RGBA")
                self.assertEqual(result.size, (17, 15))

    def test_jpeg_preserves_quantization_tables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jpg"
            output = root / "output.jpg"
            self._pattern().convert("RGB").save(source, quality=87, subsampling=1)
            with Image.open(source) as before:
                expected_tables = before.quantization
            save_cropped_image(
                source,
                output,
                CropBox(2, 3, 18, 14),
                scale_percent=50,
            )
            with Image.open(output) as result:
                self.assertEqual(result.size, (8, 6))
                self.assertEqual(result.quantization, expected_tables)

    def test_webp_lossless_crop_has_identical_decoded_pixels(self) -> None:
        if not features.check("webp"):
            self.skipTest("Pillow 沒有 WebP 支援")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.webp"
            output = root / "output.webp"
            image = self._pattern()
            image.save(source, format="WEBP", lossless=True)
            box = CropBox(1, 1, 17, 15)
            save_cropped_image(source, output, box, scale_percent=200)
            with Image.open(source) as decoded_source, Image.open(output) as result:
                expected = resize_image_high_quality(
                    decoded_source.crop(box.as_tuple()).convert("RGBA"),
                    (32, 28),
                )
                self.assertEqual(result.size, expected.size)
                self.assertEqual(result.convert("RGBA").tobytes(), expected.tobytes())

    def test_png_resize_preserves_source_dpi(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            output = root / "output.png"
            self._pattern().save(source, dpi=(144, 144))
            save_cropped_image(
                source,
                output,
                CropBox(0, 0, 20, 16),
                scale_percent=50,
            )
            with Image.open(output) as result:
                self.assertEqual(result.size, (10, 8))
                self.assertAlmostEqual(result.info["dpi"][0], 144, delta=0.1)
                self.assertAlmostEqual(result.info["dpi"][1], 144, delta=0.1)


class GifAnimationTests(unittest.TestCase):
    @staticmethod
    def _animation_with_duplicate_frame() -> GifAnimation:
        first = Image.new("RGBA", (8, 6), (0, 0, 0, 0))
        first.paste((255, 0, 0, 255), (1, 1, 5, 5))
        second = first.copy()
        third = Image.new("RGBA", (8, 6), (0, 0, 0, 0))
        third.paste((0, 255, 0, 255), (3, 1, 7, 5))
        return GifAnimation(
            frames=(first, second, third),
            durations=(30, 40, 50),
            disposals=(1, 1, 1),
            loop=2,
            comment=b"crop-test",
        )

    def test_atomic_gif_save_keeps_duplicate_frames_and_playback_metadata(self) -> None:
        animation = self._animation_with_duplicate_frame()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.gif"
            output.write_bytes(b"old output")
            progress: list[tuple[int, int]] = []
            validating: list[bool] = []
            save_cropped_gif_atomic(
                animation,
                output,
                CropBox(1, 1, 7, 5),
                progress_callback=lambda current, total: progress.append((current, total)),
                validating_callback=lambda: validating.append(True),
            )

            self.assertEqual(progress, [(1, 3), (2, 3), (3, 3)])
            self.assertEqual(validating, [True])
            with Image.open(output) as result:
                self.assertEqual(result.size, (6, 4))
                self.assertEqual(result.n_frames, 3)
                self.assertEqual(result.info.get("loop"), 2)
                durations = []
                disposals = []
                decoded = []
                for index in range(result.n_frames):
                    result.seek(index)
                    durations.append(result.info.get("duration", 0))
                    disposals.append(result.disposal_method)
                    decoded.append(result.convert("RGBA").copy())
            self.assertEqual(durations, [30, 40, 50])
            self.assertEqual(disposals, [1, 1, 1])
            self.assertEqual(decoded[0].tobytes(), decoded[1].tobytes())
            self.assertEqual(decoded[0].getpixel((0, 0)), (255, 0, 0, 255))
            self.assertEqual(decoded[0].getpixel((5, 3))[3], 0)
            self.assertEqual(decoded[2].getpixel((5, 0)), (0, 255, 0, 255))

    def test_gif_resize_keeps_all_playback_metadata(self) -> None:
        animation = self._animation_with_duplicate_frame()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "scaled.gif"
            save_cropped_gif_atomic(
                animation,
                output,
                CropBox(1, 1, 7, 5),
                scale_percent=150,
            )
            loaded = load_gif_animation(output)
            self.assertEqual(loaded.size, (9, 6))
            self.assertEqual(loaded.frame_count, 3)
            self.assertEqual(loaded.durations, (30, 40, 50))
            self.assertEqual(loaded.disposals, (1, 1, 1))
            self.assertEqual(loaded.loop, 2)

    def test_opaque_gif_does_not_reduce_a_256_color_frame(self) -> None:
        frame = Image.new("RGBA", (16, 16))
        for value in range(256):
            frame.putpixel((value % 16, value // 16), (value, 255 - value, value // 2, 255))
        animation = GifAnimation(
            frames=(frame,),
            durations=(30,),
            disposals=(1,),
            loop=None,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "colors.gif"
            save_cropped_gif_atomic(animation, output, CropBox(0, 0, 16, 16))
            with Image.open(output) as result:
                decoded_colors = set(result.convert("RGB").get_flattened_data())
            self.assertEqual(len(decoded_colors), 256)

    def test_gif_loader_round_trips_complete_frames(self) -> None:
        animation = self._animation_with_duplicate_frame()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "source.gif"
            save_cropped_gif_atomic(animation, output, CropBox(0, 0, 8, 6))
            loaded = load_gif_animation(output)
            self.assertEqual(loaded.frame_count, 3)
            self.assertEqual(loaded.size, (8, 6))
            self.assertEqual(loaded.durations, (30, 40, 50))
            self.assertEqual(loaded.disposals, (1, 1, 1))
            self.assertEqual(loaded.loop, 2)
            self.assertEqual(loaded.frames[0].tobytes(), loaded.frames[1].tobytes())

    def test_gif_save_preserves_per_frame_disposal_values(self) -> None:
        frames = tuple(
            Image.new("RGBA", (5, 5), color)
            for color in ((255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255))
        )
        animation = GifAnimation(
            frames=frames,
            durations=(30, 40, 50),
            disposals=(1, 2, 3),
            loop=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "disposals.gif"
            save_cropped_gif_atomic(animation, output, CropBox(0, 0, 5, 5))
            loaded = load_gif_animation(output)
            self.assertEqual(loaded.disposals, (1, 2, 3))

    def test_gif_loader_expands_delta_frames_to_complete_canvases(self) -> None:
        frames = []
        for left in (0, 3, 6):
            frame = Image.new("RGB", (10, 6), "white")
            frame.paste("red", (left, 2, left + 2, 4))
            frames.append(frame)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "delta.gif"
            frames[0].save(
                source,
                format="GIF",
                save_all=True,
                append_images=frames[1:],
                duration=[30, 40, 50],
                disposal=[1, 1, 1],
                optimize=True,
            )
            loaded = load_gif_animation(source)
            self.assertEqual(loaded.frame_count, 3)
            for expected, actual in zip(frames, loaded.frames):
                self.assertEqual(actual.convert("RGB").tobytes(), expected.tobytes())

    def test_animated_webp_is_rejected_instead_of_flattened(self) -> None:
        if not features.check("webp"):
            self.skipTest("Pillow 沒有 WebP 支援")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "animated.webp"
            frames = [
                Image.new("RGB", (4, 4), "red"),
                Image.new("RGB", (4, 4), "blue"),
            ]
            frames[0].save(
                path,
                format="WEBP",
                save_all=True,
                append_images=frames[1:],
                duration=[40, 40],
                loop=0,
                lossless=True,
            )
            with self.assertRaises(UnsupportedAnimatedImageError):
                load_oriented_image(path)


if __name__ == "__main__":
    unittest.main()
