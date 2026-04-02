# -*- coding: utf-8 -*-
"""
Лабораторная работа №1
Тема: цветовые модели и передискретизация изображений

Требования выполнены:
- исходное изображение: полноцветное трёхканальное PNG;
- библиотечные функции передискретизации НЕ используются;
- каждая операция сохраняется в отдельный файл.
"""

from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import os

OUT_DIR = "results"
M = 3
N = 2
K = M / N


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_rgb(arr: np.ndarray, path: str) -> None:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def save_gray(arr: np.ndarray, path: str) -> None:
    arr = np.clip(arr, 0, 1)
    gray = (arr * 255).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)
    Image.fromarray(rgb, mode="RGB").save(path)


def create_demo_image(width: int = 480, height: int = 320) -> Image.Image:
    """Создаёт собственное тестовое изображение в формате RGB."""
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Небо / фон
    for y in range(height):
        if y < int(height * 0.55):
            t = y / (height * 0.55)
            color = (
                int(20 + 80 * t),
                int(100 + 80 * t),
                int(220 + 20 * t),
            )
        else:
            t = (y - height * 0.55) / (height * 0.45)
            color = (
                int(20 + 40 * t),
                int(120 + 60 * t),
                int(80 + 40 * t),
            )
        draw.line([(0, y), (width, y)], fill=color)

    # Солнце
    draw.ellipse((35, 30, 115, 110), fill=(255, 220, 60))

    # Горы
    draw.polygon([(0, 180), (80, 80), (180, 190)], fill=(90, 80, 130))
    draw.polygon([(120, 190), (240, 70), (360, 200)], fill=(70, 100, 140))
    draw.polygon([(260, 200), (390, 90), (480, 190)], fill=(100, 90, 150))

    # Заснеженные вершины
    draw.polygon([(55, 110), (80, 80), (102, 115)], fill=(245, 245, 250))
    draw.polygon([(212, 110), (240, 70), (268, 112)], fill=(245, 245, 250))
    draw.polygon([(360, 120), (390, 90), (415, 123)], fill=(245, 245, 250))

    # Озеро
    draw.rectangle((0, 200, width, 270), fill=(40, 130, 190))
    for y in range(200, 270, 4):
        draw.line([(0, y), (width, y)], fill=(60, 150, 210), width=1)

    # Трава
    draw.rectangle((0, 270, width, height), fill=(60, 160, 70))

    # Дерево
    draw.rectangle((365, 180, 385, 285), fill=(110, 70, 30))
    for box in [(330, 130, 420, 220), (315, 155, 435, 245), (340, 105, 410, 175)]:
        draw.ellipse(box, fill=(30, 120, 50))

    # Тропинка
    draw.polygon([(180, height), (260, height), (240, 270), (200, 270)], fill=(180, 140, 90))

    # Цветы
    rng = np.random.default_rng(4)
    palette = np.array([
        [240, 50, 70],
        [255, 255, 255],
        [250, 220, 40],
        [180, 80, 220],
    ], dtype=np.uint8)
    for _ in range(120):
        x = int(rng.integers(0, width))
        y = int(rng.integers(275, height))
        color = tuple(int(v) for v in palette[int(rng.integers(0, len(palette)))])
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)

    return img.filter(ImageFilter.GaussianBlur(radius=0.3))


def rgb_to_hsi(img_rgb: np.ndarray):
    """Преобразование RGB -> HSI, компоненты в диапазоне [0, 1]."""
    eps = 1e-8
    r = img_rgb[..., 0]
    g = img_rgb[..., 1]
    b = img_rgb[..., 2]

    intensity = (r + g + b) / 3.0
    min_rgb = np.minimum(np.minimum(r, g), b)
    saturation = 1.0 - np.where(intensity > eps, min_rgb / (intensity + eps), 0.0)

    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b)) + eps
    theta = np.arccos(np.clip(num / den, -1.0, 1.0))
    hue = np.where(b <= g, theta, 2.0 * np.pi - theta)

    return hue, saturation, intensity


def hsi_to_rgb(h: np.ndarray, s: np.ndarray, i: np.ndarray) -> np.ndarray:
    """Преобразование HSI -> RGB, результат в диапазоне [0, 1]."""
    eps = 1e-8
    h = h % (2.0 * np.pi)

    r = np.zeros_like(i)
    g = np.zeros_like(i)
    b = np.zeros_like(i)

    # Сектор RG
    mask1 = (h >= 0) & (h < 2.0 * np.pi / 3.0)
    h1 = h[mask1]
    b[mask1] = i[mask1] * (1.0 - s[mask1])
    r[mask1] = i[mask1] * (1.0 + (s[mask1] * np.cos(h1)) / (np.cos(np.pi / 3.0 - h1) + eps))
    g[mask1] = 3.0 * i[mask1] - (r[mask1] + b[mask1])

    # Сектор GB
    mask2 = (h >= 2.0 * np.pi / 3.0) & (h < 4.0 * np.pi / 3.0)
    h2 = h[mask2] - 2.0 * np.pi / 3.0
    r[mask2] = i[mask2] * (1.0 - s[mask2])
    g[mask2] = i[mask2] * (1.0 + (s[mask2] * np.cos(h2)) / (np.cos(np.pi / 3.0 - h2) + eps))
    b[mask2] = 3.0 * i[mask2] - (r[mask2] + g[mask2])

    # Сектор BR
    mask3 = ~(mask1 | mask2)
    h3 = h[mask3] - 4.0 * np.pi / 3.0
    g[mask3] = i[mask3] * (1.0 - s[mask3])
    b[mask3] = i[mask3] * (1.0 + (s[mask3] * np.cos(h3)) / (np.cos(np.pi / 3.0 - h3) + eps))
    r[mask3] = 3.0 * i[mask3] - (g[mask3] + b[mask3])

    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0)


def bilinear_resize_manual(src: np.ndarray, scale: float) -> np.ndarray:
    """
    Ручная билинейная интерполяция.
    Используются собственные формулы без библиотечных функций resize.
    """
    in_h, in_w, channels = src.shape
    out_h = max(1, int(round(in_h * scale)))
    out_w = max(1, int(round(in_w * scale)))
    dst = np.zeros((out_h, out_w, channels), dtype=np.float32)

    for y in range(out_h):
        src_y = y / scale
        y0 = int(np.floor(src_y))
        y1 = min(y0 + 1, in_h - 1)
        dy = src_y - y0
        if y0 >= in_h:
            y0 = in_h - 1
            y1 = in_h - 1
            dy = 0.0

        for x in range(out_w):
            src_x = x / scale
            x0 = int(np.floor(src_x))
            x1 = min(x0 + 1, in_w - 1)
            dx = src_x - x0
            if x0 >= in_w:
                x0 = in_w - 1
                x1 = in_w - 1
                dx = 0.0

            top = src[y0, x0] * (1.0 - dx) + src[y0, x1] * dx
            bottom = src[y1, x0] * (1.0 - dx) + src[y1, x1] * dx
            dst[y, x] = top * (1.0 - dy) + bottom * dy

    return np.clip(dst, 0, 255).astype(np.uint8)


def decimate_manual(src: np.ndarray, factor: int) -> np.ndarray:
    """Ручная децимация: оставляем каждый factor-й пиксель."""
    return src[::factor, ::factor].astype(np.uint8)


def main() -> None:
    ensure_dir(OUT_DIR)

    # 1. Исходное изображение
    img = create_demo_image()
    img.save(os.path.join(OUT_DIR, "source_image.png"))
    rgb = np.asarray(img, dtype=np.float32) / 255.0

    # 2. Компоненты RGB
    save_gray(rgb[..., 0], os.path.join(OUT_DIR, "R_component.png"))
    save_gray(rgb[..., 1], os.path.join(OUT_DIR, "G_component.png"))
    save_gray(rgb[..., 2], os.path.join(OUT_DIR, "B_component.png"))

    # 3. Модель HSI
    h, s, i = rgb_to_hsi(rgb)
    save_gray(i, os.path.join(OUT_DIR, "HSI_intensity.png"))

    # 4. Инверсия яркости в исходном изображении
    i_inv = 1.0 - i
    inv_rgb = hsi_to_rgb(h, s, i_inv)
    save_rgb(inv_rgb * 255.0, os.path.join(OUT_DIR, "inverted_intensity_result.png"))

    # 5. Передискретизация
    src = rgb * 255.0
    stretched = bilinear_resize_manual(src, M)
    compressed = decimate_manual(src, N)
    two_pass = decimate_manual(stretched.astype(np.float32), N)
    one_pass = bilinear_resize_manual(src, K)

    save_rgb(stretched, os.path.join(OUT_DIR, f"stretch_M{M}.png"))
    save_rgb(compressed, os.path.join(OUT_DIR, f"compress_N{N}.png"))
    save_rgb(two_pass, os.path.join(OUT_DIR, f"resample_two_pass_K{K:.2f}.png"))
    save_rgb(one_pass, os.path.join(OUT_DIR, f"resample_one_pass_K{K:.2f}.png"))

    print("Готово. Результаты сохранены в папку:", OUT_DIR)


if __name__ == "__main__":
    main()
