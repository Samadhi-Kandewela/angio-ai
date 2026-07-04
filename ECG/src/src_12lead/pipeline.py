"""
pipeline.py (12-Lead Version)
-----------------------------
End-to-end 12-lead ECG analysis pipeline.

Run with no arguments to open a file-browser dialog for image selection.
The model is auto-discovered from the sibling ecg-12-lead-model/ folder.
Outputs are written to ecg-12-lead-model/pipeline_outputs/<stem>_<timestamp>/.

Alternatively supply CLI args:
    python pipeline.py --img_dir <image_or_dir> [--model <pth>] [--output_dir <dir>] [--dpi 200]
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

from stage1_localize import localize
from stage3_digitize import digitize
from stage4_classify import load_model, predict, LEAD_NAMES, SIGNAL_LENGTH

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE          = Path(__file__).resolve().parent          # .../ECG/src/src_12lead
_ECG_ROOT      = _HERE.parent.parent                      # .../ECG
_MODEL_DIR     = _ECG_ROOT / "ecg-12-lead-model"
_OUTPUT_ROOT   = _MODEL_DIR / "pipeline_outputs"
_DEFAULT_MODEL = _MODEL_DIR / "ecg_resnet1d_best.pth"


# ── File browser ──────────────────────────────────────────────────────────────
def _browse_for_image() -> str:
    """Open a tkinter file dialog and return the chosen image path."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        sys.exit("tkinter is not available in this Python installation. "
                 "Pass --img_dir on the command line instead.")

    root = tk.Tk()
    root.withdraw()          # hide the blank root window
    root.lift()
    root.attributes("-topmost", True)

    path = filedialog.askopenfilename(
        title="Select ECG image",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
            ("All files",   "*.*"),
        ],
        initialdir=str(_ECG_ROOT),
    )
    root.destroy()

    if not path:
        sys.exit("No image selected. Exiting.")
    return path


# ── Core pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(input_path: str,
                 model_path: str  = None,
                 dpi:        int  = 200,
                 output_dir: str  = None) -> dict:

    input_path = str(Path(input_path).resolve())
    is_file    = os.path.isfile(input_path)

    # Resolve model
    if model_path is None:
        model_path = str(_DEFAULT_MODEL)
    if not os.path.exists(model_path):
        sys.exit(f"Model checkpoint not found: {model_path}\n"
                 f"Place ecg_resnet1d_best.pth in {_MODEL_DIR} or pass --model.")

    # Resolve output directory
    if output_dir is None:
        stem = Path(input_path).stem if is_file else Path(input_path).name
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = str(_OUTPUT_ROOT / f"{stem}_{ts}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  12-Lead ECG Analysis Pipeline")
    print(f"  Input : {input_path}")
    print(f"  Model : {model_path}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    # ── 1. Digitization ───────────────────────────────────────────────────────
    signal_12lead_mv   = np.zeros((12, SIGNAL_LENGTH))
    digitization_reports = {}

    single_file_lead = None
    if is_file:
        basename = Path(input_path).stem
        if basename in LEAD_NAMES:
            single_file_lead = basename
        else:
            print(f"  Filename '{basename}' is not a standard lead name — treating as Lead I.")
            single_file_lead = "I"

    for i, lead_name in enumerate(LEAD_NAMES):
        img_path = None

        if is_file:
            if lead_name == single_file_lead:
                img_path = input_path
        else:
            for ext in (".png", ".jpg", ".jpeg"):
                candidate = os.path.join(input_path, f"{lead_name}{ext}")
                if os.path.exists(candidate):
                    img_path = candidate
                    break

        if img_path is None:
            print(f"  [Lead {lead_name:>4}]  missing — zero-padded")
            continue

        print(f"\n--- Lead {lead_name} ---")
        try:
            stage1  = localize(img_path, visualize=True)
            baseline_local = stage1["baseline_y"] - stage1["crop_top"]

            stage3  = digitize(
                cropped_binary       = stage1["cropped_binary"],
                baseline_y_local     = baseline_local,
                dpi                  = dpi,
                visualize            = True,
                output_prefix        = os.path.join(output_dir, f"lead_{lead_name}"),
                cropped_original_bgr = stage1["cropped_original"],
            )

            sig = stage3["voltage_mv"]
            slen = len(sig)
            if slen >= SIGNAL_LENGTH:
                signal_12lead_mv[i] = sig[:SIGNAL_LENGTH]
            else:
                signal_12lead_mv[i, :slen] = sig

            digitization_reports[lead_name] = {
                "signal_length_extracted":  slen,
                "voltage_min_mv":           float(sig.min()),
                "voltage_max_mv":           float(sig.max()),
                "voltage_peak_to_peak_mv":  float(sig.max() - sig.min()),
            }
        except Exception as exc:
            print(f"  [Lead {lead_name}] Error: {exc}  — zero-padded")

    # ── 2. Visualise all 12 signals ───────────────────────────────────────────
    _save_signal_plot(signal_12lead_mv, output_dir)

    # ── 3. Classification ─────────────────────────────────────────────────────
    print(f"\n[ Stage IV ] Diagnostic Classification")
    try:
        model  = load_model(model_path)
        result = predict(model, signal_12lead_mv)
        print(f"  Prediction : {result['label']}")
        print(f"  Confidence : {result['confidence']*100:.1f}%")
        print(f"  Probabilities:")
        for cls, p in result.get("probabilities", {}).items():
            bar = "#" * int(p * 30)
            print(f"    {cls:<6} {p*100:5.1f}%  {bar}")
    except Exception as exc:
        print(f"  [Stage IV] Classification failed: {exc}")
        result = {"error": str(exc)}

    # ── 4. Save outputs ───────────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, "signal_12lead.csv")
    np.savetxt(csv_path, signal_12lead_mv.T, delimiter=",",
               header=",".join(LEAD_NAMES), comments="")

    report = {
        "timestamp":      datetime.now().isoformat(),
        "input_path":     input_path,
        "model_path":     model_path,
        "output_dir":     output_dir,
        "digitization":   digitization_reports,
        "classification": result,
    }
    report_path = os.path.join(output_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Signal CSV  -> {csv_path}")
    print(f"  Report JSON -> {report_path}")
    print(f"\n{'='*60}")
    print(f"  Pipeline complete!")
    print(f"  All outputs -> {output_dir}")
    print(f"{'='*60}\n")
    return report


# ── Signal overview plot ──────────────────────────────────────────────────────
def _save_signal_plot(signal_12lead_mv: np.ndarray, output_dir: str):
    fig, axes = plt.subplots(12, 1, figsize=(14, 18), facecolor="#0d1b2a")
    fig.suptitle("12-Lead ECG — Digitised Signals", color="white", fontsize=13, y=0.995)
    t = np.linspace(0, 10, SIGNAL_LENGTH)
    for i, (ax, name) in enumerate(zip(axes, LEAD_NAMES)):
        ax.set_facecolor("#111827")
        ax.plot(t, signal_12lead_mv[i], color="#00e5ff", linewidth=0.9)
        ax.set_ylabel(name, color="#aaa", fontsize=8, rotation=0, labelpad=28)
        ax.tick_params(colors="#555", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#333")
        if i < 11:
            ax.set_xticks([])
    axes[-1].set_xlabel("Time (s)", color="#aaa", fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.995])
    out = os.path.join(output_dir, "signals_12lead.png")
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Signal plot -> {out}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="12-Lead ECG Analysis Pipeline")
    parser.add_argument("--img_dir",    default=None,
                        help="ECG image path or directory of per-lead images. "
                             "Omit to open a file-browser dialog.")
    parser.add_argument("--model",      default=None,
                        help=f"Path to model checkpoint "
                             f"(default: {_DEFAULT_MODEL})")
    parser.add_argument("--dpi",        type=int, default=200,
                        help="Scanning DPI of the image (default: 200)")
    parser.add_argument("--output_dir", default=None,
                        help=f"Output directory "
                             f"(default: {_OUTPUT_ROOT}/<name>_<timestamp>/)")
    args = parser.parse_args()

    # If no image path given, open a file-browser dialog
    img_path = args.img_dir if args.img_dir else _browse_for_image()

    run_pipeline(
        input_path = img_path,
        model_path = args.model,
        dpi        = args.dpi,
        output_dir = args.output_dir,
    )
