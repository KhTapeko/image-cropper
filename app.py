import argparse
from pathlib import Path

from image_cropper.gui import run_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="人物圖片裁切器")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="人物辨識裝置；GPU 模式請使用 cuda。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_app(Path(__file__).resolve().parent, preferred_device=arguments.device)
