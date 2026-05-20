import matplotlib.pyplot as plt
import numpy as np
import os
from adjustText import adjust_text

# -----------------------------
# Data
# -----------------------------
models = {
    "Standard U-Net": {"latency": 2157, "dice": 0.789, "type": "baseline"},
    "DSC-UNet": {"latency": 377, "dice": 0.776, "type": "from_scratch"},
    "MobileUNet-v1": {"latency": 530, "dice": 0.812, "type": "lightweight"},
    "MobileUNet-v2": {"latency": 562, "dice": 0.814, "type": "lightweight"},
    "MobileUNet-v3 (Ours)": {"latency": 548, "dice": 0.824, "type": "proposed_base"},
    "MobileUNet-v3 + Aug": {"latency": 548, "dice": 0.902, "type": "proposed_aug"},
    "DeepLabV3+": {"latency": 3000, "dice": 0.834, "type": "heavy"},
    "ResU-Net": {"latency": 1400, "dice": 0.702, "type": "heavy"},
    "SegFormer-B4": {"latency": 2500, "dice": 0.8899, "type": "heavy"},
}

styles = {
    "baseline": {"color": "#6c757d", "marker": "s", "size": 110, "label": "Baseline"},
    "from_scratch": {"color": "#f4a261", "marker": "v", "size": 140, "label": "Efficiency (Scratch)"},
    "lightweight": {"color": "#2a9d8f", "marker": "o", "size": 120, "label": "Lightweight"},
    "proposed_base": {"color": "#2b9348", "marker": "*", "size": 260, "label": "Proposed (Base)"},
    "proposed_aug": {"color": "#d00000", "marker": "*", "size": 320, "label": "Proposed (Augmented)"},
    "heavy": {"color": "#6a4c93", "marker": "D", "size": 110, "label": "High-Capacity (SOTA)"},
}

# -----------------------------
# Professional Styling
# -----------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11
})

fig, ax = plt.subplots(figsize=(11, 7))

plotted_categories = set()
texts = []

# -----------------------------
# Plot Points
# -----------------------------
for name, data in models.items():
    cat = data["type"]
    style = styles[cat]

    label = style["label"] if cat not in plotted_categories else None

    ax.scatter(
        data["latency"],
        data["dice"],
        color=style["color"],
        marker=style["marker"],
        s=style["size"],
        label=label,
        edgecolors="black",
        linewidth=0.6,
        zorder=3
    )

    plotted_categories.add(cat)

    texts.append(
        ax.text(
            data["latency"],
            data["dice"],
            name,
            fontsize=9.5,
            weight="bold" if "MobileUNet-v3" in name else "normal",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.2)
        )
    )

# -----------------------------
# Quadrant Lines
# -----------------------------
ax.axhline(0.85, color="gray", linestyle="--", linewidth=1)
ax.axvline(1000, color="gray", linestyle="--", linewidth=1)

ax.fill_betweenx(
    y=[0.85, 0.92],
    x1=0,
    x2=1000,
    color="#90be6d",
    alpha=0.12
)

# -----------------------------
# Optimal Region Label (Axes Coordinates — No Collision)
# -----------------------------
ax.text(
    500, 0.885,
    "Optimal Real-Time Region\n(Dice > 0.85, Latency < 1000 ms)",
    ha="center",
    va="center",
    fontsize=11,
    color="#1b7f3b",
    bbox=dict(
        facecolor="white",
        edgecolor="#1b7f3b",
        boxstyle="round,pad=0.35",
        alpha=0.95
    ),
    zorder=4
)

# -----------------------------
# Labels & Title
# -----------------------------
ax.set_title(
    "Accuracy–Latency Trade-off in Coronary Segmentation",
    fontsize=15,
    weight="bold",
    pad=15
)

ax.set_xlabel("Inference Latency (ms/frame) – CPU", fontsize=12)
ax.set_ylabel("Segmentation Accuracy (Dice Score)", fontsize=12)

ax.set_xlim(0, 3200)
ax.set_ylim(0.68, 0.92)

ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
ax.legend(title="Model Categories", loc="lower right", frameon=True)

# -----------------------------
# FINAL: Adjust Labels (Run LAST)
# -----------------------------
adjust_text(
    texts,
    ax=ax,
    arrowprops=dict(arrowstyle="-", color='gray', lw=0.6),
    expand_points=(1.6, 1.8),
    expand_text=(1.4, 1.6),
    force_points=1.0,
    force_text=1.2
)

plt.tight_layout()

# -----------------------------
# Save High-Resolution Figure
# -----------------------------
OUTPUT_PATH = "E:/Research/figures/Figure_4_Accuracy_Latency_Tradeoff_Professional.png"
os.makedirs("E:/Research/figures", exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=600, bbox_inches="tight")
plt.show()

print(f"Professional figure saved to: {OUTPUT_PATH}")