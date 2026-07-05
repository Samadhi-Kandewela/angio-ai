"""
pipeline.py
-----------
End-to-end single-lead ECG analysis pipeline.

Chains Stage I -> Stage III -> Stage IV and produces:
  - Annotated diagnostic visualization PNG
  - JSON report with digitized signal stats and classification result
  - CSV of the extracted time-series signal

Usage:
    python pipeline.py --image ../sample.png
    python pipeline.py --image ../sample.png --model ../checkpoints/ecg_resnet1d_best.pth
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stage1_localize import localize
from stage3_digitize import digitize
from stage4_classify import ECGResNet1D, load_model, classify, CLASS_NAMES


def run_pipeline(image_path: str, model_path: str = None, dpi: int = 200) -> dict:
    """
    Run the complete single-lead ECG pipeline.

    Args:
        image_path : path to scanned ECG strip image
        model_path : path to trained Stage IV model weights (.pth)
                     If None, classification is skipped.
        dpi        : scanning DPI of the input image

    Returns a dict containing all results.
    """
    image_path = str(image_path)
    stem = Path(image_path).stem
    out_dir = Path(image_path).parent / f"{stem}_results"
    out_dir.mkdir(exist_ok=True)
    prefix = str(out_dir / stem)

    print("=" * 60)
    print(f"  ECG Analysis Pipeline")
    print(f"  Input : {image_path}")
    print(f"  Output: {out_dir}")
    print("=" * 60)

    # ── Stage I: Localization ───────────────────────────────────────────────
    print("\n[ Stage I ] Layout Localization")
    s1 = localize(image_path, visualize=True)

    baseline_local = s1["baseline_y"] - s1["crop_top"]

    # ── Stage III: Digitization ─────────────────────────────────────────────
    print("\n[ Stage III ] Waveform Digitization")
    s3 = digitize(
        cropped_binary=s1["cropped_binary"],
        baseline_y_local=baseline_local,
        dpi=dpi,
        visualize=True,
        output_prefix=prefix,
        cropped_original_bgr=s1.get("cropped_original"),
    )

    # ── Stage IV: Classification (optional) ────────────────────────────────
    classification = None
    if model_path and os.path.exists(model_path):
        print("\n[ Stage IV ] Diagnostic Classification")
        model = load_model(model_path)
        classification = classify(s3["voltage_mv"], model)
        print(f"  -> Prediction : {classification['label']}")
        print(f"  -> Confidence : {classification['confidence']:.1%}")
    elif model_path:
        print(f"\n[ Stage IV ] Model not found at {model_path} — skipping classification.")
    else:
        print("\n[ Stage IV ] No model provided — skipping classification.")

    # ── Save CSV ────────────────────────────────────────────────────────────
    csv_path = f"{prefix}_signal.csv"
    data = np.column_stack([s3["time_ms"], s3["voltage_mv"]])
    np.savetxt(csv_path, data, delimiter=",",
               header="time_ms,voltage_mv", comments="")
    print(f"\n[Pipeline] Signal CSV saved -> {csv_path}")

    # ── Build Report ────────────────────────────────────────────────────────
    report = {
        "timestamp": datetime.now().isoformat(),
        "input_image": image_path,
        "stage1": {
            "baseline_y_px": int(s1["baseline_y"]),
            "crop_top_px": int(s1["crop_top"]),
            "crop_bottom_px": int(s1["crop_bottom"]),
        },
        "stage3": {
            "signal_length_samples": len(s3["time_ms"]),
            "duration_ms": float(s3["time_ms"][-1]),
            "voltage_min_mv": float(s3["voltage_mv"].min()),
            "voltage_max_mv": float(s3["voltage_mv"].max()),
            "voltage_peak_to_peak_mv": float(
                s3["voltage_mv"].max() - s3["voltage_mv"].min()
            ),
            "csv_path": csv_path,
        },
        "stage4": classification,
    }

    report_path = f"{prefix}_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[Pipeline] Report saved -> {report_path}")

    # ── Final Summary Plot ──────────────────────────────────────────────────
    _save_summary_plot(s1, s3, classification, prefix)

    print("\n" + "=" * 60)
    print("  Pipeline Complete!")
    print("=" * 60)
    return report


def _save_summary_plot(s1: dict, s3: dict, classification: dict, prefix: str):
    """Save a clean final summary figure combining signal and classification."""
    fig = plt.figure(figsize=(14, 7), facecolor="#0d1b2a")

    # ── Signal panel ───────────────────────────────────────────────────────
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.set_facecolor("#111827")
    ax1.imshow(s1["original_bgr"][:, :, ::-1], aspect="auto")
    ax1.set_title("Original ECG Strip", color="white", fontsize=12, pad=6)
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.set_facecolor("#111827")
    ax2.plot(s3["time_ms"], s3["voltage_mv"], color="#00e676", linewidth=1.2)
    ax2.axhline(0, color="#ff5252", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("Time (ms)", color="#aaa", fontsize=10)
    ax2.set_ylabel("Voltage (mV)", color="#aaa", fontsize=10)
    ax2.tick_params(colors="#aaa")
    ax2.spines[:].set_color("#333")

    title = "Digitized ECG Signal"
    if classification:
        lbl = classification["label"]
        conf = classification["confidence"]
        color_map = {
            "NORM": "#00e676", "MI": "#ff5252", "STTC": "#ff9800",
            "CD": "#7c4dff", "HYP": "#40c4ff"
        }
        c = color_map.get(lbl, "white")
        title += f"  |  Diagnosis: [{lbl}] ({conf:.1%})"
        ax2.set_title(title, color=c, fontsize=12, pad=6, fontweight="bold")
    else:
        ax2.set_title(title, color="white", fontsize=12, pad=6)

    plt.tight_layout(rect=[0, 0, 1, 1])
    out = f"{prefix}_summary.png"
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Pipeline] Summary plot saved -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-Lead ECG Analysis Pipeline")
    parser.add_argument("--image", required=True, help="Path to ECG strip image")
    parser.add_argument("--model", default=None,
                        help="Path to trained Stage IV model (.pth)")
    parser.add_argument("--dpi", type=int, default=200,
                        help="Scanning DPI of the image (default: 200)")
    args = parser.parse_args()

    report = run_pipeline(args.image, args.model, args.dpi)
    print("\nFinal Report:")
    print(json.dumps(report, indent=2))
