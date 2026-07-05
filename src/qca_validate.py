"""
qca_validate.py  —  QCA validation: measurement accuracy + stenosis detection

Run:
    python qca_validate.py
"""

import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from qca import QCAConfig, qca_from_mask


# ──────────────────────────────────────────────────────────
# Phantom builders
# ──────────────────────────────────────────────────────────

def make_vessel(length=200, prox_r=10.0, stenosis_pct=None,
                stenosis_start=0.38, stenosis_end=0.62, h=60):
    """
    Straight horizontal vessel with optional focal stenosis.
    Returns (mask uint8, true_ds_pct).  stenosis_pct=None = healthy vessel.
    """
    if stenosis_pct is None:
        stenosis_r = prox_r          # no narrowing
        true_ds    = 0.0
    else:
        stenosis_r = prox_r * (1.0 - stenosis_pct / 100.0)
        true_ds    = stenosis_pct

    w    = length + 40
    cx   = h // 2
    mask = np.zeros((h, w), dtype=np.uint8)
    for x in range(w):
        frac = max(0.0, min(1.0, (x - 20) / length))
        if stenosis_pct and stenosis_start <= frac <= stenosis_end:
            t = (frac - stenosis_start) / (stenosis_end - stenosis_start)
            r = prox_r - (prox_r - stenosis_r) * (0.5 - 0.5 * math.cos(math.pi * t))
        else:
            r = prox_r
        r = max(1.0, r)
        mask[int(cx - r): int(cx + r) + 1, x] = 255
    return mask, true_ds


# ──────────────────────────────────────────────────────────
# Part A — Measurement accuracy
#   Phantoms with a known stenosis; compare true DS% vs detected DS%
# ──────────────────────────────────────────────────────────

def run_measurement_accuracy(cfg, tolerance=10.0):
    """
    For each phantom, check whether the algorithm measures the DS%
    within 'tolerance' percentage points of the true value.
    """
    cases = [
        # (label,         true_DS%)
        ("Mild  30%",     30),
        ("Mild  40%",     40),
        ("Moderate  50%", 50),
        ("Moderate  60%", 60),
        ("Severe  70%",   70),
        ("Severe  80%",   80),
        ("Severe  90%",   90),
    ]

    true_vals, det_vals, errors = [], [], []
    rows = []

    for label, true_ds in cases:
        mask, _ = make_vessel(stenosis_pct=true_ds)
        _, lesions, _ = qca_from_mask(mask, cfg)

        if not lesions:
            rows.append((label, true_ds, None, None, "MISS"))
            continue

        # Pick the lesion closest to the true DS
        best    = min(lesions, key=lambda l: abs(l["DS_percent"] - true_ds))
        det_ds  = best["DS_percent"]
        err     = det_ds - true_ds
        status  = "PASS" if abs(err) <= tolerance else "FAIL"

        true_vals.append(true_ds)
        det_vals.append(det_ds)
        errors.append(err)
        rows.append((label, true_ds, det_ds, err, status))

    # Summary stats
    mae  = float(np.mean(np.abs(errors))) if errors else 0.0
    bias = float(np.mean(errors))          if errors else 0.0
    passed = sum(1 for r in rows if r[4] == "PASS")

    return rows, mae, bias, passed, len(cases), true_vals, det_vals


# ──────────────────────────────────────────────────────────
# Part B — Stenosis identification (detection)
#   Sensitivity : did the algorithm find stenoses that exist?
#   Specificity : did it avoid flagging healthy vessels?
# ──────────────────────────────────────────────────────────

def run_detection_accuracy(cfg):
    """
    Build phantoms WITH and WITHOUT stenosis.
    Count True Positives, False Negatives, True Negatives, False Positives.
    Report Sensitivity and Specificity.
    """

    # Phantoms that HAVE a stenosis (expect at least one lesion detected)
    positive_cases = [
        ("Focal severe  70%",    make_vessel(stenosis_pct=70)[0]),
        ("Focal severe  80%",    make_vessel(stenosis_pct=80)[0]),
        ("Focal moderate  55%",  make_vessel(stenosis_pct=55)[0]),
        ("Focal moderate  65%",  make_vessel(stenosis_pct=65)[0]),
        ("Focal mild  35%",      make_vessel(stenosis_pct=35)[0]),
    ]

    # Phantoms that have NO stenosis (expect zero lesions)
    negative_cases = [
        ("Healthy uniform",
            make_vessel(stenosis_pct=None, prox_r=10)[0]),
        ("Healthy wide",
            make_vessel(stenosis_pct=None, prox_r=14)[0]),
        ("Healthy narrow",
            make_vessel(stenosis_pct=None, prox_r=6)[0]),
        ("Healthy tapered",       _tapered_no_stenosis()),
        ("Very short vessel",     _short_vessel()),
    ]

    tp = fn = tn = fp = 0
    rows = []

    for label, mask in positive_cases:
        _, lesions, _ = qca_from_mask(mask, cfg)
        detected = len(lesions) > 0
        if detected:
            tp += 1
            rows.append(("STENOTIC",  label, "Detected",     "TP"))
        else:
            fn += 1
            rows.append(("STENOTIC",  label, "Missed",       "FN"))

    for label, mask in negative_cases:
        _, lesions, _ = qca_from_mask(mask, cfg)
        # Only count lesions that would be displayed (>=50% DS = MODERATE+)
        flagged = any(l["DS_percent"] >= cfg.moderate_threshold for l in lesions)
        if not flagged:
            tn += 1
            rows.append(("HEALTHY", label, "Correctly clear", "TN"))
        else:
            fp += 1
            ds = max(l["DS_percent"] for l in lesions)
            rows.append(("HEALTHY", label, f"False alarm {ds:.1f}%", "FP"))

    sensitivity = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0.0
    return rows, sensitivity, specificity, tp, fn, tn, fp


def _tapered_no_stenosis(length=180, h=60):
    """Healthy vessel that linearly tapers from r=12 to r=7 — no stenosis."""
    mask = np.zeros((h, length + 40), dtype=np.uint8)
    cx = h // 2
    for x in range(length + 40):
        frac = max(0.0, min(1.0, (x - 20) / length))
        r = max(1.0, 12.0 - 5.0 * frac)
        mask[int(cx - r): int(cx + r) + 1, x] = 255
    return mask


def _short_vessel(h=40):
    """Very short vessel — should produce no lesions."""
    mask = np.zeros((h, 60), dtype=np.uint8)
    mask[15:25, 5:55] = 255
    return mask


# ──────────────────────────────────────────────────────────
# Print helpers
# ──────────────────────────────────────────────────────────

LINE  = "-" * 60
DLINE = "=" * 60


def print_measurement_results(rows, mae, bias, passed, total, tol):
    print()
    print(DLINE)
    print("PART A  -  MEASUREMENT ACCURACY")
    print("  How close is the detected DS% to the true DS%?")
    print(DLINE)
    print(f"  {'Phantom':<22}  {'True':>6}  {'Detected':>9}  {'Error':>7}  {'Result':>6}")
    print(LINE)
    for label, true_ds, det_ds, err, status in rows:
        if det_ds is None:
            print(f"  {label:<22}  {true_ds:>5.0f}%  {'not found':>9}  {'':>7}  {'MISS':>6}")
        else:
            sign = "+" if err >= 0 else ""
            print(f"  {label:<22}  {true_ds:>5.0f}%  {det_ds:>8.1f}%  "
                  f"{sign}{err:>5.1f}%  {status:>6}")
    print(LINE)
    print(f"  Passed : {passed}/{total}  (within +/-{tol:.0f}% tolerance)")
    print(f"  MAE    : {mae:.1f}%  (mean absolute error)")
    print(f"  Bias   : {bias:+.1f}%  (negative = algorithm underestimates DS%)")
    print()


def print_detection_results(rows, sensitivity, specificity, tp, fn, tn, fp):
    print()
    print(DLINE)
    print("PART B  -  STENOSIS IDENTIFICATION")
    print("  Does the algorithm detect real stenoses and ignore healthy vessels?")
    print(DLINE)
    print(f"  {'Vessel type':<12}  {'Phantom':<28}  {'Outcome':<22}  {'Code':>4}")
    print(LINE)
    for vessel_type, label, outcome, code in rows:
        print(f"  {vessel_type:<12}  {label:<28}  {outcome:<22}  {code:>4}")
    print(LINE)
    print()
    print(f"  True Positives  (real stenosis detected)   : {tp}")
    print(f"  False Negatives (real stenosis missed)     : {fn}")
    print(f"  True Negatives  (healthy vessel clear)     : {tn}")
    print(f"  False Positives (healthy vessel flagged)   : {fp}")
    print()
    print(f"  Sensitivity : {sensitivity:.0f}%  "
          f"-- of {tp+fn} stenotic vessels, {tp} were detected")
    print(f"  Specificity : {specificity:.0f}%  "
          f"-- of {tn+fp} healthy vessels, {tn} were correctly left clear")
    print()


# -------------------------------------------------------
# Plot
# -------------------------------------------------------

def save_plots(rows_A, true_vals, det_vals, rows_B, out_dir, tol):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Plot A: true vs detected DS% ──────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    if true_vals:
        t = np.array(true_vals)
        d = np.array(det_vals)
        colors = ["green" if abs(d[i]-t[i]) <= tol else "red"
                  for i in range(len(t))]
        ax.scatter(t, d, c=colors, s=80, zorder=3)
        lim = [0, 105]
        ax.plot(lim, lim, "k--", lw=1, label="Perfect agreement")
        ax.fill_between(lim,
                        [lim[0]-tol, lim[1]-tol],
                        [lim[0]+tol, lim[1]+tol],
                        alpha=0.10, color="blue", label=f"±{tol:.0f}% band")
        ax.axvline(70, color="red",    ls=":", lw=1)
        ax.axvline(50, color="orange", ls=":", lw=1)
        ax.set_xlabel("True DS%")
        ax.set_ylabel("Detected DS%")
        ax.set_title("Measurement Accuracy\n(green = within tolerance)")
        ax.set_xlim(0, 105); ax.set_ylim(0, 105)
        ax.legend(fontsize=9)

    # ── Plot B: detection confusion ───────────────────────
    ax2 = axes[1]
    counts = {"TP": 0, "FN": 0, "TN": 0, "FP": 0}
    for *_, code in rows_B:
        counts[code] += 1

    bar_labels = ["TP\n(found)", "FN\n(missed)", "TN\n(clear)", "FP\n(false alarm)"]
    bar_vals   = [counts["TP"], counts["FN"], counts["TN"], counts["FP"]]
    bar_colors = ["#2ecc71", "#e74c3c", "#3498db", "#e67e22"]

    bars = ax2.bar(bar_labels, bar_vals, color=bar_colors, width=0.5)
    for bar, val in zip(bars, bar_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 str(val), ha="center", fontsize=13, fontweight="bold")

    total_pos = counts["TP"] + counts["FN"]
    total_neg = counts["TN"] + counts["FP"]
    sens = counts["TP"] / total_pos * 100 if total_pos else 0
    spec = counts["TN"] / total_neg * 100 if total_neg else 0
    ax2.set_title(f"Stenosis Detection\nSensitivity {sens:.0f}%  |  Specificity {spec:.0f}%")
    ax2.set_ylim(0, max(bar_vals) + 1.5 if bar_vals else 5)
    ax2.set_ylabel("Count")

    plt.suptitle("QCA Validation Summary", fontweight="bold", fontsize=13)
    plt.tight_layout()
    path = out_dir / "qca_validation_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved -> {path}")


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = QCAConfig()
    TOL = 10.0                      # acceptable DS% error in percentage points
    OUT = "qca_validation/"

    # Part A
    rows_A, mae, bias, passed, total, true_vals, det_vals = \
        run_measurement_accuracy(cfg, tolerance=TOL)
    print_measurement_results(rows_A, mae, bias, passed, total, TOL)

    # Part B
    rows_B, sens, spec, tp, fn, tn, fp = run_detection_accuracy(cfg)
    print_detection_results(rows_B, sens, spec, tp, fn, tn, fp)

    # Summary line
    print(DLINE)
    print("  SUMMARY")
    print(LINE)
    print(f"  Measurement accuracy : {passed}/{total} phantoms within +/-{TOL:.0f}%  |  MAE {mae:.1f}%")
    print(f"  Detection accuracy   : Sensitivity {sens:.0f}%  |  Specificity {spec:.0f}%")
    print(DLINE)
    print()

    save_plots(rows_A, true_vals, det_vals, rows_B, OUT, TOL)
