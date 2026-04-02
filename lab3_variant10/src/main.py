from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

MASK_OFFSETS = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]  # diagonal cross
RANK_INDEX = 3  # variant 10: rank 4/5, 0-based index = 3


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def rgba_to_rgb_white(path: Path) -> Image.Image:
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, "white")
    return Image.alpha_composite(bg, im).convert("RGB")


def rgb_to_gray_array(rgb: Image.Image) -> np.ndarray:
    return np.asarray(rgb.convert("L"), dtype=np.uint8)


def otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    sigma_b2 = np.zeros(256, dtype=np.float64)
    denom = omega * (1.0 - omega)
    valid = denom > 0
    sigma_b2[valid] = (mu_t * omega[valid] - mu[valid]) ** 2 / denom[valid]
    return int(np.argmax(sigma_b2))


def to_binary(gray: np.ndarray, threshold: int) -> np.ndarray:
    # Text is darker than background; foreground becomes black (0), background white (255)
    return np.where(gray <= threshold, 0, 255).astype(np.uint8)


def rank_filter_diag_cross(arr: np.ndarray) -> np.ndarray:
    padded = np.pad(arr, ((1, 1), (1, 1)), mode="edge")
    h, w = arr.shape
    samples = []
    for dy, dx in MASK_OFFSETS:
        y0 = 1 + dy
        x0 = 1 + dx
        samples.append(padded[y0:y0 + h, x0:x0 + w])
    stacked = np.stack(samples, axis=0)
    filtered = np.sort(stacked, axis=0)[RANK_INDEX]
    return filtered.astype(np.uint8)


def gray_difference(src: np.ndarray, filtered: np.ndarray) -> np.ndarray:
    return np.abs(src.astype(np.int16) - filtered.astype(np.int16)).astype(np.uint8)


def xor_difference(src_bin: np.ndarray, filtered_bin: np.ndarray) -> np.ndarray:
    return np.bitwise_xor(src_bin, filtered_bin).astype(np.uint8)


def enhance_for_display(diff: np.ndarray) -> np.ndarray:
    mx = int(diff.max())
    if mx == 0:
        return diff.copy()
    if mx < 32:
        enhanced = np.clip(diff.astype(np.int16) * 8, 0, 255).astype(np.uint8)
        return enhanced
    return diff


def array_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr, mode="L")


def fit_to_box(im: Image.Image, box_w: int, box_h: int, bg: str = "white") -> Image.Image:
    canvas = Image.new("RGB", (box_w, box_h), bg)
    w, h = im.size
    scale = min(box_w / w, box_h / h)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    resized = im.resize(new_size, Image.Resampling.LANCZOS).convert("RGB")
    x = (box_w - new_size[0]) // 2
    y = (box_h - new_size[1]) // 2
    canvas.paste(resized, (x, y))
    return canvas


def get_font(size: int, bold: bool = False):
    candidates = []
    if bold:
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ]
    else:
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ]
    for cand in candidates:
        p = Path(cand)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def draw_text_center(draw: ImageDraw.ImageDraw, x_center: int, y: int, text: str, font, fill="black"):
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    draw.text((x_center - w / 2, y), text, font=font, fill=fill)


def make_contact_sheet(out_path: Path, title: str, panels: list[tuple[str, Image.Image]]) -> None:
    cols = 3
    rows = 2
    panel_w = 520
    panel_h = 640
    pad = 24
    title_h = 66
    label_h = 42
    sheet_w = pad + cols * panel_w + (cols - 1) * pad + pad
    sheet_h = title_h + pad + rows * (label_h + panel_h) + (rows - 1) * pad + pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    title_font = get_font(28, bold=True)
    label_font = get_font(22, bold=True)

    draw_text_center(draw, sheet_w // 2, 16, title, title_font)

    for idx, (label, im) in enumerate(panels):
        r = idx // cols
        c = idx % cols
        x = pad + c * (panel_w + pad)
        y = title_h + pad + r * (label_h + panel_h + pad)
        draw_text_center(draw, x + panel_w // 2, y, label, label_font)
        boxed = fit_to_box(im, panel_w, panel_h)
        sheet.paste(boxed, (x, y + label_h))
        draw.rectangle([x, y + label_h, x + panel_w, y + label_h + panel_h], outline="#cfcfcf", width=2)

    sheet.save(out_path)


def process_image(path: Path, out_root: Path) -> dict:
    name = path.stem
    rgb = rgba_to_rgb_white(path)
    gray = rgb_to_gray_array(rgb)
    threshold = otsu_threshold(gray)
    binary = to_binary(gray, threshold)

    gray_filtered = rank_filter_diag_cross(gray)
    gray_diff_raw = gray_difference(gray, gray_filtered)
    gray_diff = enhance_for_display(gray_diff_raw)

    binary_filtered = rank_filter_diag_cross(binary)
    binary_xor = xor_difference(binary, binary_filtered)

    gray_dir = out_root / "grayscale"
    bin_dir = out_root / "binary"
    sheets_dir = out_root / "contact_sheets"
    ensure_dir(gray_dir)
    ensure_dir(bin_dir)
    ensure_dir(sheets_dir)

    array_to_pil(gray).save(gray_dir / f"{name}_input_gray.png")
    array_to_pil(gray_filtered).save(gray_dir / f"{name}_filtered_gray.png")
    array_to_pil(gray_diff_raw).save(gray_dir / f"{name}_difference_gray_raw.png")
    array_to_pil(gray_diff).save(gray_dir / f"{name}_difference_gray.png")

    array_to_pil(binary).save(bin_dir / f"{name}_input_binary.png")
    array_to_pil(binary_filtered).save(bin_dir / f"{name}_filtered_binary.png")
    array_to_pil(binary_xor).save(bin_dir / f"{name}_xor_binary.png")

    panels = [
        ("Полутон: исходное", array_to_pil(gray)),
        ("Полутон: фильтр", array_to_pil(gray_filtered)),
        ("Полутон: |Δ|", array_to_pil(gray_diff)),
        ("Монохром: исходное", array_to_pil(binary)),
        ("Монохром: фильтр", array_to_pil(binary_filtered)),
        ("Монохром: XOR", array_to_pil(binary_xor)),
    ]
    make_contact_sheet(sheets_dir / f"{name}_summary.png", f"Вариант 10 - {path.name}", panels)

    gray_changed_pct = float((gray_diff_raw > 0).mean() * 100.0)
    bin_changed_pct = float((binary_xor > 0).mean() * 100.0)
    metrics = {
        "file": path.name,
        "width": gray.shape[1],
        "height": gray.shape[0],
        "otsu_threshold": threshold,
        "gray_mean_abs_diff": round(float(gray_diff_raw.mean()), 4),
        "gray_max_abs_diff": int(gray_diff_raw.max()),
        "gray_changed_pct": round(gray_changed_pct, 4),
        "binary_changed_pct": round(bin_changed_pct, 4),
    }
    return metrics


def write_metrics_csv(metrics: list[dict], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)


def copy_inputs(input_files: Iterable[Path], dst_dir: Path) -> None:
    ensure_dir(dst_dir)
    for path in input_files:
        rgba_to_rgb_white(path).save(dst_dir / path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab 3, variant 10: rank filter with diagonal-cross mask, rank 4/5.")
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    input_files = sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
    if not input_files:
        raise SystemExit("No input images found.")

    ensure_dir(args.output_dir)
    copy_inputs(input_files, args.output_dir / "input")

    metrics = []
    for path in input_files:
        metrics.append(process_image(path, args.output_dir / "output"))

    write_metrics_csv(metrics, args.output_dir / "output" / "summary_metrics.csv")


if __name__ == "__main__":
    main()
