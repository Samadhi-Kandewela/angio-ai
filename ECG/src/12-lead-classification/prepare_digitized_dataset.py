"""
prepare_digitized_dataset.py
----------------------------
Converts clean PTB-XL .npy signals into digitization-artifact-corrupted
versions that match what the pipeline produces from scanned ECG images.

Run after prepare_ptbxl_dataset.py has already downloaded the data.

Usage:
    python src/prepare_digitized_dataset.py
"""

import numpy as np
from pathlib import Path
from scipy.ndimage import median_filter

ROOT = Path(__file__).parent.parent
DATASET = ROOT / "dataset"


def simulate_digitization(signal: np.ndarray, dpi: int = 200,
                          noise_sigma: float = 0.5,
                          baseline_drift_mv: float = 0.05) -> np.ndarray:
    """
    Simulate the argmin-based pipeline artifacts on a clean 1-D ECG signal.

    1. Quantise to discrete pixel levels (finite DPI resolution).
    2. Add Gaussian noise (printing + scanning).
    3. Add slow baseline drift (imperfect baseline detection).
    4. Apply a 7-sample median filter (as in stage3_digitize).
    """
    px_per_mm = dpi / 25.4          # pixels per mm
    mv_per_px = 1.0 / (px_per_mm * 10.0)   # 10 mm/mV standard ECG

    # 1. Quantise
    px = np.round(signal / mv_per_px)

    # 2. Pixel-level Gaussian noise
    px += np.random.normal(0, noise_sigma, len(px))

    # 3. Slow baseline drift (sinusoidal, random phase)
    t = np.linspace(0, 2 * np.pi, len(px))
    drift_px = (baseline_drift_mv / mv_per_px) * np.sin(
        t * np.random.uniform(0.5, 2.0) + np.random.uniform(0, 2 * np.pi)
    )
    px += drift_px

    # 4. Median filter (matches stage3 pipeline)
    px = median_filter(px, size=7)

    return px * mv_per_px


def process_split(signals_path: Path, labels_path: Path,
                  out_signals: Path, out_labels: Path,
                  n_augments: int = 2):
    """
    For every signal, produce n_augments randomly-corrupted copies.
    Labels are replicated to match.
    """
    X = np.load(signals_path)   # (N, 1000)
    y = np.load(labels_path)    # (N,)
    print(f"  Loaded {len(X)} samples from {signals_path.name}")

    aug_X, aug_y = [], []
    for i, (sig, label) in enumerate(zip(X, y)):
        for _ in range(n_augments):
            aug_X.append(simulate_digitization(sig))
            aug_y.append(label)
        if i % 2000 == 0:
            print(f"    {i}/{len(X)} processed...")

    aug_X = np.array(aug_X, dtype=np.float32)
    aug_y = np.array(aug_y, dtype=np.int64)
    np.save(out_signals, aug_X)
    np.save(out_labels,  aug_y)
    print(f"  Saved {len(aug_X)} augmented samples -> {out_signals.name}")


if __name__ == "__main__":
    print("Preparing digitized training dataset...")

    process_split(
        DATASET / "ptbxl_train_signals.npy",
        DATASET / "ptbxl_train_labels.npy",
        DATASET / "ptbxl_train_digitized_signals.npy",
        DATASET / "ptbxl_train_digitized_labels.npy",
        n_augments=2,
    )
    process_split(
        DATASET / "ptbxl_val_signals.npy",
        DATASET / "ptbxl_val_labels.npy",
        DATASET / "ptbxl_val_digitized_signals.npy",
        DATASET / "ptbxl_val_digitized_labels.npy",
        n_augments=1,
    )
    print("Done.")
