import argparse
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def enhance_angiogram(image: np.ndarray, scale: float = 1.0) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    if scale != 1.0:
        height, width = gray.shape[:2]
        new_size = (round(width * scale), round(height * scale))
        gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)

    denoised = cv2.fastNlMeansDenoising(
        contrast,
        None,
        h=7,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    blurred = cv2.GaussianBlur(denoised, (0, 0), sigmaX=1.2)
    sharpened = cv2.addWeighted(denoised, 1.6, blurred, -0.6, 0)

    return cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)


def preprocess_folder(input_dir: Path, output_dir: Path, scale: float) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    count = 0
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        enhanced = enhance_angiogram(image, scale=scale)
        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), enhanced)
        count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enhance raw angiogram images for easier manual annotation."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("new_raw_images"),
        help="Folder containing raw angiogram images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("new_preprocessed_images"),
        help="Folder where enhanced images will be saved.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Upscale factor before enhancement. Use 2 for 1024x1024 from 512x512.",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {args.input_dir}")
    if args.scale <= 0:
        raise ValueError("--scale must be greater than 0")

    count = preprocess_folder(args.input_dir, args.output_dir, scale=args.scale)
    print(f"Saved {count} enhanced images to: {args.output_dir}")


if __name__ == "__main__":
    main()
