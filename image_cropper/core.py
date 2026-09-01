from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import tempfile
from typing import Callable, Iterable

import numpy as np
from PIL import GifImagePlugin, Image, ImageOps, JpegImagePlugin
from scipy import ndimage


SUPPORTED_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif"})


class UnsupportedAnimatedImageError(ValueError):
    """Raised when an animated format has no safe output implementation."""


@dataclass(frozen=True)
class CropBox:
    """A crop rectangle in oriented source-image pixels (right/bottom exclusive)."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom

    def clamp(self, image_width: int, image_height: int, min_size: int = 1) -> CropBox:
        image_width = max(1, image_width)
        image_height = max(1, image_height)
        min_width = min(max(1, min_size), image_width)
        min_height = min(max(1, min_size), image_height)

        left = min(max(0, self.left), image_width - min_width)
        top = min(max(0, self.top), image_height - min_height)
        right = min(max(left + min_width, self.right), image_width)
        bottom = min(max(top + min_height, self.bottom), image_height)
        return CropBox(left, top, right, bottom)

    def move(self, delta_x: int, delta_y: int, image_width: int, image_height: int) -> CropBox:
        width = min(self.width, image_width)
        height = min(self.height, image_height)
        left = min(max(0, self.left + delta_x), image_width - width)
        top = min(max(0, self.top + delta_y), image_height - height)
        return CropBox(left, top, left + width, top + height)

    def resize_corner(
        self,
        corner: str,
        image_x: int,
        image_y: int,
        image_width: int,
        image_height: int,
        min_size: int = 10,
    ) -> CropBox:
        left, top, right, bottom = self.as_tuple()
        if "w" in corner:
            left = min(max(0, image_x), right - min_size)
        if "e" in corner:
            right = max(min(image_width, image_x), left + min_size)
        if "n" in corner:
            top = min(max(0, image_y), bottom - min_size)
        if "s" in corner:
            bottom = max(min(image_height, image_y), top + min_size)
        return CropBox(left, top, right, bottom).clamp(image_width, image_height, min_size)


@dataclass(frozen=True)
class GifAnimation:
    """A fully decoded GIF plus the playback metadata needed for re-encoding."""

    frames: tuple[Image.Image, ...]
    durations: tuple[int, ...]
    disposals: tuple[int, ...]
    loop: int | None
    comment: bytes | str | None = None
    background_color: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        frame_count = len(self.frames)
        if frame_count == 0:
            raise ValueError("GIF 沒有可讀取的影格")
        if len(self.durations) != frame_count or len(self.disposals) != frame_count:
            raise ValueError("GIF 播放屬性與影格數量不一致")
        expected_size = self.frames[0].size
        if any(frame.size != expected_size for frame in self.frames):
            raise ValueError("GIF 影格畫布尺寸不一致")

    @property
    def size(self) -> tuple[int, int]:
        return self.frames[0].size

    @property
    def frame_count(self) -> int:
        return len(self.frames)


def mask_to_suggested_crop(
    mask: Image.Image,
    image_size: tuple[int, int],
    *,
    threshold: int = 32,
    low_threshold: int = 8,
    high_threshold: int = 64,
    dark_mask_mean_max: float = 4.0,
    closeup_mask_mean_min: float = 40.0,
    min_foreground_area_ratio: float = 0.01,
    min_component_area_ratio: float = 0.0002,
    min_component_relative_area: float = 0.05,
    component_join_distance_ratio: float = 0.03,
    small_border_component_max_area_ratio: float = 0.01,
    padding_ratio: float = 0.03,
) -> CropBox | None:
    """Turn a cleaned, adaptively thresholded person mask into a crop."""

    image_width, image_height = image_size
    if mask.size != image_size:
        mask = mask.resize(image_size, Image.Resampling.BILINEAR)
    grayscale = np.asarray(mask.convert("L"), dtype=np.uint8)
    mask_mean = float(grayscale.mean())
    if mask_mean < dark_mask_mean_max:
        selected_threshold = low_threshold
    elif mask_mean >= closeup_mask_mean_min:
        selected_threshold = high_threshold
    else:
        selected_threshold = threshold

    binary = grayscale >= selected_threshold
    labels, component_count = ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
    if component_count == 0:
        return None

    image_area = max(1, image_width * image_height)
    component_areas = np.bincount(labels.ravel())[1:]
    component_slices = ndimage.find_objects(labels)
    minimum_component_area = max(1, math.ceil(image_area * min_component_area_ratio))

    candidates: list[tuple[int, int, int, int, int, int]] = []
    for component_id, (area, component_slice) in enumerate(
        zip(component_areas, component_slices),
        start=1,
    ):
        if component_slice is None or area < minimum_component_area:
            continue
        row_slice, column_slice = component_slice
        left = column_slice.start
        top = row_slice.start
        right = column_slice.stop
        bottom = row_slice.stop
        touches_border = left == 0 or top == 0 or right == image_width or bottom == image_height
        if touches_border and area / image_area < small_border_component_max_area_ratio:
            continue
        candidates.append((component_id, int(area), left, top, right, bottom))

    if not candidates:
        return None

    largest_component_area = max(area for _component_id, area, *_bounds in candidates)
    component_relative_minimum = largest_component_area * min_component_relative_area
    candidates = [
        component for component in candidates if component[1] >= component_relative_minimum
    ]

    join_distance = max(image_width, image_height) * component_join_distance_ratio
    parent = list(range(len(candidates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    def bounds_distance(
        first: tuple[int, int, int, int, int, int],
        second: tuple[int, int, int, int, int, int],
    ) -> float:
        horizontal_gap = max(0, first[2] - second[4], second[2] - first[4])
        vertical_gap = max(0, first[3] - second[5], second[3] - first[5])
        return math.hypot(horizontal_gap, vertical_gap)

    for first_index, first in enumerate(candidates):
        for second_index in range(first_index + 1, len(candidates)):
            if bounds_distance(first, candidates[second_index]) <= join_distance:
                union(first_index, second_index)

    groups: dict[int, list[tuple[int, int, int, int, int, int]]] = {}
    for index, component in enumerate(candidates):
        groups.setdefault(find(index), []).append(component)

    grouped = []
    for components in groups.values():
        grouped.append(
            (
                sum(component[1] for component in components),
                min(component[2] for component in components),
                min(component[3] for component in components),
                max(component[4] for component in components),
                max(component[5] for component in components),
            )
        )

    largest_group_area = max(group[0] for group in grouped)
    if largest_group_area / image_area < min_foreground_area_ratio:
        return None

    relative_minimum = largest_group_area * min_component_relative_area
    kept_groups = [group for group in grouped if group[0] >= relative_minimum]
    left = min(group[1] for group in kept_groups)
    top = min(group[2] for group in kept_groups)
    right = max(group[3] for group in kept_groups)
    bottom = max(group[4] for group in kept_groups)
    pad_x = math.ceil((right - left) * padding_ratio)
    pad_y = math.ceil((bottom - top) * padding_ratio)
    return CropBox(
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(image_width, right + pad_x),
        min(image_height, bottom + pad_y),
    )


def union_crop_boxes(
    boxes: Iterable[CropBox],
    image_size: tuple[int, int],
    *,
    padding_ratio: float = 0.0,
) -> CropBox | None:
    """Union crop boxes, then add padding once around the combined bounds."""

    collected = list(boxes)
    if not collected:
        return None
    image_width, image_height = image_size
    left = min(box.left for box in collected)
    top = min(box.top for box in collected)
    right = max(box.right for box in collected)
    bottom = max(box.bottom for box in collected)
    pad_x = math.ceil((right - left) * max(0.0, padding_ratio))
    pad_y = math.ceil((bottom - top) * max(0.0, padding_ratio))
    return CropBox(
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(image_width, right + pad_x),
        min(image_height, bottom + pad_y),
    ).clamp(image_width, image_height)


def resolve_gif_crop(
    suggestions: Iterable[CropBox],
    image_size: tuple[int, int],
    *,
    error_count: int,
    padding_ratio: float,
) -> CropBox:
    """Resolve valid GIF detections according to the agreed failure fallback."""

    combined = union_crop_boxes(
        suggestions,
        image_size,
        padding_ratio=padding_ratio,
    )
    if combined is not None:
        return combined
    if error_count:
        raise RuntimeError(
            f"沒有有效人物範圍，且 {error_count} 個取樣影格辨識失敗"
        )
    return CropBox(0, 0, *image_size)


def gif_sample_indices(frame_count: int, interval: int = 5) -> tuple[int, ...]:
    """Return zero-based GIF sample indices, always including first and last."""

    if frame_count < 1:
        return ()
    if interval < 1:
        raise ValueError("取樣間隔必須大於 0")
    indices = list(range(0, frame_count, interval))
    final_index = frame_count - 1
    if indices[-1] != final_index:
        indices.append(final_index)
    return tuple(indices)


def composite_transparency_for_detection(
    image: Image.Image,
    background: tuple[int, int, int] = (128, 128, 128),
) -> Image.Image:
    """Flatten transparency onto a neutral background for person detection."""

    foreground = image.convert("RGBA")
    backdrop = Image.new("RGBA", foreground.size, (*background, 255))
    return Image.alpha_composite(backdrop, foreground).convert("RGB")


def sort_paths_by_creation_desc(
    paths: Iterable[Path],
    creation_time_getter: Callable[[Path], int] | None = None,
) -> list[Path]:
    getter = creation_time_getter or (lambda path: path.stat().st_ctime_ns)
    return sorted(
        paths,
        key=lambda path: (
            path.suffix.casefold() == ".gif",
            -getter(path),
            path.name.casefold(),
        ),
    )


def scan_input_images(input_directory: Path) -> list[Path]:
    if not input_directory.exists():
        return []
    candidates = (
        path
        for path in input_directory.iterdir()
        if path.is_file() and path.suffix.casefold() in SUPPORTED_EXTENSIONS
    )
    return sort_paths_by_creation_desc(candidates)


def load_oriented_image(path: Path) -> Image.Image:
    with Image.open(path) as source:
        if (source.format or "").upper() == "WEBP" and getattr(source, "is_animated", False):
            raise UnsupportedAnimatedImageError("動畫 WebP 尚未支援")
        source.load()
        return ImageOps.exif_transpose(source).copy()


def load_gif_animation(path: Path) -> GifAnimation:
    """Decode every GIF frame as the complete canvas seen during playback."""

    frames: list[Image.Image] = []
    durations: list[int] = []
    disposals: list[int] = []
    with Image.open(path) as source:
        if (source.format or "").upper() != "GIF":
            raise ValueError("檔案不是 GIF")
        global_info = dict(source.info)
        background_color = None
        background_index = global_info.get("background")
        source_palette = source.getpalette()
        if isinstance(background_index, int) and source_palette is not None:
            offset = background_index * 3
            if offset + 2 < len(source_palette):
                alpha = 0 if global_info.get("transparency") == background_index else 255
                background_color = (
                    source_palette[offset],
                    source_palette[offset + 1],
                    source_palette[offset + 2],
                    alpha,
                )
        frame_count = int(getattr(source, "n_frames", 1))
        for frame_index in range(frame_count):
            source.seek(frame_index)
            frames.append(source.convert("RGBA").copy())
            durations.append(int(source.info.get("duration", 0)))
            disposals.append(int(getattr(source, "disposal_method", 0)))

    return GifAnimation(
        frames=tuple(frames),
        durations=tuple(durations),
        disposals=tuple(disposals),
        loop=int(global_info["loop"]) if "loop" in global_info else None,
        comment=global_info.get("comment"),
        background_color=background_color,
    )


def save_cropped_image(source_path: Path, output_path: Path, crop_box: CropBox) -> None:
    """Crop without resizing and preserve useful source metadata when supported."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as source:
        source.load()
        source_format = (source.format or source_path.suffix.lstrip(".")).upper()
        if source_format == "JPG":
            source_format = "JPEG"

        oriented = ImageOps.exif_transpose(source)
        box = crop_box.clamp(*oriented.size)
        cropped = oriented.crop(box.as_tuple())

        save_options: dict[str, object] = {}
        if source.info.get("icc_profile"):
            save_options["icc_profile"] = source.info["icc_profile"]
        if source.info.get("dpi"):
            save_options["dpi"] = source.info["dpi"]

        exif = oriented.getexif()
        if exif:
            exif[274] = 1
            save_options["exif"] = exif.tobytes()

        if source_format == "PNG":
            cropped.save(output_path, format="PNG", **save_options)
            return

        if source_format == "JPEG":
            if cropped.mode not in ("RGB", "L", "CMYK"):
                cropped = cropped.convert("RGB")
            cropped.format = "JPEG"
            if hasattr(source, "quantization"):
                cropped.quantization = source.quantization
            save_options.update(
                quality="keep",
                subsampling=JpegImagePlugin.get_sampling(source),
                qtables=getattr(source, "quantization", None),
            )
            cropped.save(output_path, format="JPEG", **save_options)
            return

        if source_format == "WEBP":
            save_options.update(lossless=True, quality=100, method=6, exact=True)
            cropped.save(output_path, format="WEBP", **save_options)
            return

        raise ValueError(f"不支援的圖片格式：{source_format}")


def _palette_gif_frame(frame: Image.Image) -> Image.Image:
    """Quantize one RGBA frame while retaining a GIF transparency index."""

    rgba = frame.convert("RGBA")
    alpha = rgba.getchannel("A")
    has_transparency = alpha.getextrema()[0] < 255
    paletted = rgba.convert("RGB").quantize(
        colors=255 if has_transparency else 256,
        method=Image.Quantize.MEDIANCUT,
    )
    if has_transparency:
        transparency_index = 255
        transparent_pixels = alpha.point(lambda value: 255 if value < 128 else 0)
        paletted.paste(transparency_index, mask=transparent_pixels)
        palette_data = list(paletted.getpalette() or [])
        palette_data.extend([0] * (768 - len(palette_data)))
        palette_data[transparency_index * 3 : transparency_index * 3 + 3] = [0, 0, 0]
        paletted.putpalette(palette_data)
        paletted.info["transparency"] = transparency_index
    return paletted


def _write_gif_without_frame_coalescing(
    path: Path,
    frames: list[Image.Image],
    animation: GifAnimation,
    progress_callback: Callable[[int, int], None] | None = None,
) -> None:
    """Write every GIF frame explicitly so Pillow cannot merge duplicates."""

    first_frame = _palette_gif_frame(frames[0])
    header_info: dict[str, object] = {"optimize": False}
    if animation.loop is not None:
        header_info["loop"] = animation.loop
    if animation.comment is not None:
        header_info["comment"] = animation.comment
    first_transparency = first_frame.info.get("transparency")
    if first_transparency is not None:
        header_info["transparency"] = first_transparency
    if animation.background_color is not None and first_frame.palette is not None:
        background_index = first_frame.palette.colors.get(animation.background_color)
        if background_index is None and animation.background_color[3] == 255:
            background_index = first_frame.palette.colors.get(animation.background_color[:3])
        if background_index is None and animation.background_color[3] == 0:
            background_index = first_transparency
        if background_index is not None:
            header_info["background"] = background_index

    with path.open("wb") as output:
        header, _used_palette = GifImagePlugin.getheader(
            first_frame,
            info=header_info,
        )
        for block in header:
            output.write(block)

        for frame_index, source_frame in enumerate(frames):
            frame = first_frame if frame_index == 0 else _palette_gif_frame(source_frame)
            frame_info: dict[str, object] = {
                "duration": animation.durations[frame_index],
                "disposal": animation.disposals[frame_index],
            }
            transparency = frame.info.get("transparency")
            if transparency is not None:
                frame_info["transparency"] = transparency
            if frame_index > 0:
                frame_info["include_color_table"] = True
            for block in GifImagePlugin.getdata(frame, **frame_info):
                output.write(block)
            if progress_callback is not None:
                progress_callback(frame_index + 1, len(frames))
        output.write(b";")


def validate_cropped_gif(
    path: Path,
    animation: GifAnimation,
    expected_size: tuple[int, int],
) -> None:
    """Fully decode a GIF and verify its required playback contract."""

    with Image.open(path) as result:
        if (result.format or "").upper() != "GIF":
            raise ValueError("暫存輸出不是 GIF")
        if result.size != expected_size:
            raise ValueError(f"GIF 畫布尺寸不符：{result.size}，預期 {expected_size}")
        if int(getattr(result, "n_frames", 1)) != animation.frame_count:
            raise ValueError("GIF 影格數量不符")
        actual_loop = int(result.info["loop"]) if "loop" in result.info else None
        if actual_loop != animation.loop:
            raise ValueError("GIF 循環設定不符")

        actual_durations: list[int] = []
        actual_disposals: list[int] = []
        for frame_index in range(animation.frame_count):
            result.seek(frame_index)
            result.load()
            if result.size != expected_size:
                raise ValueError(f"GIF 第 {frame_index + 1} 格畫布尺寸不符")
            actual_durations.append(int(result.info.get("duration", 0)))
            actual_disposals.append(int(getattr(result, "disposal_method", 0)))

    if tuple(actual_durations) != animation.durations:
        raise ValueError("GIF 影格播放時間不符")
    if tuple(actual_disposals) != animation.disposals:
        raise ValueError("GIF disposal 設定不符")


def save_cropped_gif_atomic(
    animation: GifAnimation,
    output_path: Path,
    crop_box: CropBox,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
    validating_callback: Callable[[], None] | None = None,
) -> None:
    """Crop, encode, validate, and atomically replace one animated GIF."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    box = crop_box.clamp(*animation.size)
    cropped_frames: list[Image.Image] = []
    for frame in animation.frames:
        cropped_frames.append(frame.crop(box.as_tuple()))

    temporary_file = tempfile.NamedTemporaryFile(
        prefix=f".{output_path.stem}-",
        suffix=".gif",
        dir=output_path.parent,
        delete=False,
    )
    temporary_path = Path(temporary_file.name)
    temporary_file.close()
    try:
        _write_gif_without_frame_coalescing(
            temporary_path,
            cropped_frames,
            animation,
            progress_callback,
        )
        if validating_callback is not None:
            validating_callback()
        validate_cropped_gif(temporary_path, animation, (box.width, box.height))
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
