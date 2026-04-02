from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='ЛР4, вариант 10: выделение контуров оператором Прюитта 5x5.'
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('input/05.png'),
        help='Путь к исходному цветному изображению.'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('output'),
        help='Папка для сохранения результатов.'
    )
    parser.add_argument(
        '--threshold',
        type=int,
        default=110,
        help='Порог бинаризации для нормализованной матрицы G.'
    )
    return parser.parse_args()


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    min_val = float(arr.min())
    max_val = float(arr.max())
    if np.isclose(max_val, min_val):
        return np.zeros(arr.shape, dtype=np.uint8)
    normalized = (arr - min_val) / (max_val - min_val)
    normalized = np.clip(normalized * 255.0, 0, 255)
    return normalized.astype(np.uint8)


def save_image(path: Path, image: np.ndarray, is_rgb: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_rgb:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr)
    else:
        cv2.imwrite(str(path), image)


def create_summary_figure(
    original_rgb: np.ndarray,
    gray: np.ndarray,
    gx_norm: np.ndarray,
    gy_norm: np.ndarray,
    g_norm: np.ndarray,
    binary: np.ndarray,
    threshold: int,
    save_path: Path,
) -> None:
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, axes = plt.subplots(2, 3, figsize=(12, 15))

    items = [
        (original_rgb, 'Исходное цветное изображение', None),
        (gray, 'Полутоновое изображение', 'gray'),
        (gx_norm, 'Нормализованная матрица Gx', 'gray'),
        (gy_norm, 'Нормализованная матрица Gy', 'gray'),
        (g_norm, 'Нормализованная матрица G', 'gray'),
        (binary, f'Бинаризованная матрица G (T = {threshold})', 'gray'),
    ]

    for ax, (image, title, cmap) in zip(axes.ravel(), items):
        if cmap is None:
            ax.imshow(image)
        else:
            ax.imshow(image, cmap=cmap)
        ax.set_title(title, fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.imread(str(args.input))
    if image_bgr is None:
        raise FileNotFoundError(f'Не удалось открыть изображение: {args.input}')

    original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    kernel_gx = np.array([
        [-1, -1, -1, -1, -1],
        [ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0],
        [ 1,  1,  1,  1,  1],
    ], dtype=np.float32)

    kernel_gy = np.array([
        [-1,  0,  0,  0,  1],
        [-1,  0,  0,  0,  1],
        [-1,  0,  0,  0,  1],
        [-1,  0,  0,  0,  1],
        [-1,  0,  0,  0,  1],
    ], dtype=np.float32)

    gray_f32 = gray.astype(np.float32)
    gx = cv2.filter2D(gray_f32, ddepth=cv2.CV_32F, kernel=kernel_gx, borderType=cv2.BORDER_REPLICATE)
    gy = cv2.filter2D(gray_f32, ddepth=cv2.CV_32F, kernel=kernel_gy, borderType=cv2.BORDER_REPLICATE)

    # Вариант 10: G = |Gx| + |Gy|
    g = np.abs(gx) + np.abs(gy)

    gx_norm = normalize_to_uint8(gx)
    gy_norm = normalize_to_uint8(gy)
    g_norm = normalize_to_uint8(g)

    _, binary = cv2.threshold(g_norm, args.threshold, 255, cv2.THRESH_BINARY)

    save_image(output_dir / '01_original.png', original_rgb, is_rgb=True)
    save_image(output_dir / '02_gray.png', gray)
    save_image(output_dir / '03_gx.png', gx_norm)
    save_image(output_dir / '04_gy.png', gy_norm)
    save_image(output_dir / '05_g.png', g_norm)
    save_image(output_dir / '06_binary.png', binary)

    create_summary_figure(
        original_rgb=original_rgb,
        gray=gray,
        gx_norm=gx_norm,
        gy_norm=gy_norm,
        g_norm=g_norm,
        binary=binary,
        threshold=args.threshold,
        save_path=output_dir / '07_summary.png',
    )

    with open(output_dir / 'info.txt', 'w', encoding='utf-8') as f:
        f.write('Лабораторная работа №4. Выделение контуров на изображении\n')
        f.write('Вариант: 10\n')
        f.write('Оператор: Прюитта 5x5\n')
        f.write('Формула градиента: G = |Gx| + |Gy|\n')
        f.write(f'Исходное изображение: {args.input.name}\n')
        f.write(f'Размер изображения: {original_rgb.shape[1]}x{original_rgb.shape[0]}\n')
        f.write(f'Порог бинаризации: {args.threshold}\n')
        f.write('Граница при свёртке: BORDER_REPLICATE\n')


if __name__ == '__main__':
    main()
