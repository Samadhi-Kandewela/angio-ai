"""
train_classifier.py
-------------------
Trains the Stage IV 1D-ResNet ECG classifier using the .npy files
produced by prepare_ptbxl_dataset.py.

Each run is saved to a timestamped folder:
  runs/run_YYYYMMDD_HHMMSS/
    ecg_resnet1d_best.pth    <- best model weights
    history.json             <- epoch-by-epoch loss and accuracy
    training_curves.png      <- loss + accuracy plot
    confusion_matrix.png     <- per-class confusion matrix
    classification_report.txt<- precision / recall / F1 per class
    run_config.json          <- hyperparameters used for this run

A symlink 'checkpoints/ecg_resnet1d_best.pth' always points to the
latest best model so pipeline.py can find it without knowing the run name.

Usage:
    python train_classifier.py
    python train_classifier.py --epochs 50 --batch-size 32 --lr 0.001
"""

import os
import sys
import json
import shutil
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage4_classify import train_model, ECGResNet1D, CLASS_NAMES, SIGNAL_LENGTH

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from stage4_classify import ECGDataset

SCRIPT_DIR  = Path(__file__).parent
ROOT_DIR    = SCRIPT_DIR.parent
DATASET_DIR = ROOT_DIR / "dataset"
RUNS_DIR    = ROOT_DIR / "runs"
CKPT_DIR    = ROOT_DIR / "checkpoints"   # always holds the latest best model


# ─── Plotting helpers ─────────────────────────────────────────────────────────

def _dark_ax(ax):
    ax.set_facecolor("#111827")
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_color("#333")


def plot_training_curves(history: dict, run_dir: Path):
    """Loss + accuracy curves on a dark background."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    fig.patch.set_facecolor("#0d1b2a")
    _dark_ax(ax1); _dark_ax(ax2)

    epochs = range(1, len(history["train_loss"]) + 1)
    best_acc = max(history["val_acc"])

    ax1.plot(epochs, history["train_loss"], color="#ff6b6b", linewidth=1.8, label="Train loss")
    ax1.set_title("Training Loss", color="white", fontsize=12)
    ax1.set_xlabel("Epoch", color="#aaa")
    ax1.set_ylabel("Weighted Cross-Entropy", color="#aaa")
    ax1.legend(facecolor="#111827", labelcolor="white")

    ax2.plot(epochs, history["val_acc"], color="#00e676", linewidth=1.8, label="Val accuracy")
    ax2.axhline(best_acc, color="#ffd54f", linewidth=0.9, linestyle="--",
                label=f"Best: {best_acc:.3f} ({best_acc*100:.1f}%)")
    ax2.set_title("Validation Accuracy", color="white", fontsize=12)
    ax2.set_xlabel("Epoch", color="#aaa")
    ax2.set_ylabel("Accuracy", color="#aaa")
    ax2.set_ylim(0, 1)
    ax2.legend(facecolor="#111827", labelcolor="white")

    plt.tight_layout()
    out = run_dir / "training_curves.png"
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: training_curves.png")
    return out


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, run_dir: Path):
    """Normalised confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0d1b2a")
    ax.set_facecolor("#0d1b2a")

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, color="white", fontsize=11)
    ax.set_yticklabels(CLASS_NAMES, color="white", fontsize=11)
    ax.set_xlabel("Predicted", color="#aaa", fontsize=11)
    ax.set_ylabel("True", color="#aaa", fontsize=11)
    ax.set_title("Confusion Matrix (normalised)", color="white", fontsize=12, pad=10)

    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            val = cm_norm[i, j]
            count = cm[i, j]
            color = "white" if val < 0.5 else "#111"
            ax.text(j, i, f"{val:.2f}\n({count})",
                    ha="center", va="center", color=color, fontsize=9)

    plt.tight_layout()
    out = run_dir / "confusion_matrix.png"
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: confusion_matrix.png")
    return out


def plot_class_metrics(report_dict: dict, run_dir: Path):
    """Horizontal bar chart of per-class F1, precision, recall."""
    classes = CLASS_NAMES
    metrics = ["precision", "recall", "f1-score"]
    colors  = ["#42a5f5", "#66bb6a", "#ffa726"]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0d1b2a")
    _dark_ax(ax)

    x = np.arange(len(classes))
    width = 0.25
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [report_dict.get(cls, {}).get(metric, 0) for cls in classes]
        bars = ax.bar(x + i * width, vals, width, label=metric.capitalize(), color=color, alpha=0.9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8, color="white")

    ax.set_xticks(x + width)
    ax.set_xticklabels(classes, color="white", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", color="#aaa")
    ax.set_title("Per-Class Precision / Recall / F1", color="white", fontsize=12)
    ax.legend(facecolor="#111827", labelcolor="white")

    plt.tight_layout()
    out = run_dir / "class_metrics.png"
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: class_metrics.png")
    return out


def evaluate_model(model_path: Path, val_signals: Path, val_labels: Path, run_dir: Path):
    """Run validation pass and save confusion matrix + classification report."""
    from sklearn.metrics import classification_report, confusion_matrix

    val_ds = ECGDataset(str(val_signals), str(val_labels), augment=False)
    loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    model = ECGResNet1D()
    model.load_state_dict(torch.load(str(model_path), map_location="cpu"))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for sigs, labs in loader:
            preds = model(sigs).argmax(dim=1)
            all_preds.extend(preds.numpy().tolist())
            all_labels.extend(labs.numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # Text report
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                   zero_division=0)
    report_path = run_dir / "classification_report.txt"
    report_path.write_text(report)
    print(f"  Saved: classification_report.txt")
    print("\n" + report)

    # Parse report dict for bar chart
    report_dict = classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                        output_dict=True, zero_division=0)

    # Plots
    plot_confusion_matrix(y_true, y_pred, run_dir)
    plot_class_metrics(report_dict, run_dir)

    return report_dict


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stage IV ECG Classifier")
    parser.add_argument("--epochs",     type=int,   default=50,   help="Max epochs (default: 50)")
    parser.add_argument("--batch-size", type=int,   default=32,   help="Batch size (default: 32)")
    parser.add_argument("--lr",         type=float, default=1e-3, help="Learning rate (default: 0.001)")
    parser.add_argument("--patience",   type=int,   default=20,   help="Early stop patience (default: 20)")
    args = parser.parse_args()

    # ── Verify dataset ────────────────────────────────────────────────────────
    required = [
        DATASET_DIR / "ptbxl_train_signals.npy",
        DATASET_DIR / "ptbxl_train_labels.npy",
        DATASET_DIR / "ptbxl_val_signals.npy",
        DATASET_DIR / "ptbxl_val_labels.npy",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("ERROR: Dataset files not found. Run prepare_ptbxl_dataset.py first.")
        for m in missing:
            print(f"  Missing: {m}")
        sys.exit(1)

    # ── Create timestamped run directory ──────────────────────────────────────
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir  = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nRun directory: {run_dir}\n")

    # ── Print summary ─────────────────────────────────────────────────────────
    y_train = np.load(required[1])
    y_val   = np.load(required[3])
    print("=" * 60)
    print("  Stage IV: 1D-ResNet ECG Classifier Training")
    print("=" * 60)
    print(f"  Train samples : {len(y_train)}")
    print(f"  Val samples   : {len(y_val)}")
    print(f"  Classes       : {CLASS_NAMES}")
    print(f"  Epochs (max)  : {args.epochs}")
    print(f"  Batch size    : {args.batch_size}")
    print(f"  Learning rate : {args.lr}")
    print(f"  Early stopping: patience={args.patience}")
    print("=" * 60)

    print("\nClass distribution (train):")
    for cls_idx, count in sorted(Counter(y_train.tolist()).items()):
        pct = count / len(y_train) * 100
        print(f"  {CLASS_NAMES[cls_idx]:6s}: {count:5d} ({pct:.1f}%)")

    # Save run config
    config = {
        "run_name":    run_name,
        "timestamp":   datetime.now().isoformat(),
        "epochs_max":  args.epochs,
        "batch_size":  args.batch_size,
        "lr":          args.lr,
        "patience":    args.patience,
        "train_samples": int(len(y_train)),
        "val_samples":   int(len(y_val)),
        "classes":     CLASS_NAMES,
        "class_distribution": {
            CLASS_NAMES[k]: int(v)
            for k, v in sorted(Counter(y_train.tolist()).items())
        },
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2))

    # ── Train ─────────────────────────────────────────────────────────────────
    train_model(
        signals_train=str(required[0]),
        labels_train=str(required[1]),
        signals_val=str(required[2]),
        labels_val=str(required[3]),
        checkpoint_dir=str(run_dir),   # save best.pth inside run dir
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )

    # ── Copy best model to stable checkpoints/ location ───────────────────────
    best_src = run_dir / "ecg_resnet1d_best.pth"
    best_dst = CKPT_DIR / "ecg_resnet1d_best.pth"
    if best_src.exists():
        shutil.copy2(best_src, best_dst)
        print(f"\nBest model copied -> {best_dst}")

    # ── Load history and plot curves ──────────────────────────────────────────
    history_path = run_dir / "history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        # Also copy history.json to run dir (it's already there)
        print("\nSaving training artifacts...")
        plot_training_curves(history, run_dir)

    # ── Evaluate and save confusion matrix + per-class metrics ────────────────
    if best_src.exists():
        evaluate_model(best_src, required[2], required[3], run_dir)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Training complete!")
    print(f"  Run artifacts saved -> {run_dir}")
    print()
    print(f"  Files saved:")
    for f in sorted(run_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:<35s} {size_kb:6.1f} KB")
    print()
    print(f"  Best model (stable path):")
    print(f"    {best_dst}")
    print()
    print(f"  Run the full pipeline:")
    print(f"    python src/pipeline.py --image sample.png --model checkpoints/ecg_resnet1d_best.pth")
    print("=" * 60)
