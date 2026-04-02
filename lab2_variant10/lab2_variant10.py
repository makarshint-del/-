

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


WINDOW_SIZE = 3
K_WOLF = 0.5


def rgb_to_gray_manual(image: Image.Image) -> np.ndarray:
    """Ручной перевод RGBA/RGB -> grayscale."""
    array = np.asarray(image, dtype=np.uint8)
    rgb = array[..., :3].astype(np.float32)

    gray = np.rint(
        0.299 * rgb[..., 0] +
        0.587 * rgb[..., 1] +
        0.114 * rgb[..., 2]
    )
    return np.clip(gray, 0, 255).astype(np.uint8)


def local_mean_std(gray: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:

    if window_size % 2 == 0:
        raise ValueError("Размер окна должен быть нечётным.")

    radius = window_size // 2
    padded = np.pad(gray.astype(np.float64), ((radius, radius), (radius, radius)), mode="edge")
    padded_sq = padded * padded

    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    integral_sq = np.pad(padded_sq, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)

    h, w = gray.shape
    y0 = np.arange(0, h)
    x0 = np.arange(0, w)
    y1 = y0 + window_size
    x1 = x0 + window_size

    area_sum = (
        integral[y1[:, None], x1[None, :]]
        - integral[y0[:, None], x1[None, :]]
        - integral[y1[:, None], x0[None, :]]
        + integral[y0[:, None], x0[None, :]]
    )

    area_sum_sq = (
        integral_sq[y1[:, None], x1[None, :]]
        - integral_sq[y0[:, None], x1[None, :]]
        - integral_sq[y1[:, None], x0[None, :]]
        + integral_sq[y0[:, None], x0[None, :]]
    )

    area = float(window_size * window_size)
    mean = area_sum / area
    variance = np.maximum(area_sum_sq / area - mean * mean, 0.0)
    std = np.sqrt(variance)

    return mean, std


def wolf_binarization(gray: np.ndarray, window_size: int = WINDOW_SIZE, k: float = K_WOLF) -> np.ndarray:

    mean, std = local_mean_std(gray, window_size)

    max_std = float(std.max()) if std.max() > 0 else 1.0
    min_gray = float(gray.min())

    threshold = mean + k * ((std / max_std) - 1.0) * (mean - min_gray)

    # Тёмные пиксели считаются объектом, светлые - фоном.
    binary = np.where(gray < threshold, 0, 255).astype(np.uint8)
    return binary


def save_comparison(original_path: Path, gray_path: Path, binary_path: Path, output_path: Path) -> None:

    original = Image.open(original_path).convert("RGB")
    gray = Image.open(gray_path).convert("RGB")
    binary = Image.open(binary_path).convert("RGB")

    max_width = 500
    labels = ["Оригинал", "Полутон", "Бинаризация Вульфа 3x3"]
    images = []

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
    except OSError:
        font = ImageFont.load_default()

    for index, image in enumerate([original, gray, binary]):
        width, height = image.size
        scale = min(max_width / width, 1.0)
        resized = image.resize(
            (int(width * scale), int(height * scale)),
            Image.Resampling.NEAREST if index == 2 else Image.Resampling.LANCZOS,
        )
        images.append(resized)

    gap = 20
    caption_height = 48
    canvas_width = sum(img.width for img in images) + gap * 4
    canvas_height = max(img.height for img in images) + caption_height + gap * 2

    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)

    x = gap
    for image, label in zip(images, labels):
        draw.text((x, 10), label, fill="black", font=font)
        canvas.paste(image, (x, caption_height))
        x += image.width + gap

    canvas.save(output_path)


def process_image(input_path: Path, output_root: Path) -> None:
    image = Image.open(input_path)

    gray = rgb_to_gray_manual(image)
    gray_image = Image.fromarray(gray, mode="L")

    binary = wolf_binarization(gray, window_size=WINDOW_SIZE, k=K_WOLF)
    binary_image = Image.fromarray(binary, mode="L")

    gray_dir = output_root / "grayscale"
    binary_dir = output_root / "binary_wolf_3x3"
    compare_dir = output_root / "comparisons"

    gray_dir.mkdir(parents=True, exist_ok=True)
    binary_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem
    gray_path = gray_dir / f"{stem}_gray.bmp"
    binary_path = binary_dir / f"{stem}_wolf3x3.bmp"
    comparison_path = compare_dir / f"{stem}_comparison.png"

    gray_image.save(gray_path, format="BMP")
    binary_image.save(binary_path, format="BMP")
    save_comparison(input_path, gray_path, binary_path, comparison_path)


def main() -> None:
    project_root = Path(__file__).resolve().parent
    input_dir = project_root / "input"
    output_dir = project_root / "results"

    images = sorted(input_dir.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"В папке {input_dir} нет PNG-изображений.")

    for image_path in images:
        print(f"Обработка: {image_path.name}")
        process_image(image_path, output_dir)

    print("Готово. Результаты сохранены в папке results.")


if __name__ == "__main__":
    main()
