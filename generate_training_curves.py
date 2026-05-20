import matplotlib.pyplot as plt
import os

# -----------------------------
# Data
# -----------------------------
epochs_orig = list(range(1, 51))
dice_orig = [
    0.5518, 0.6392, 0.7568, 0.7850, 0.7912, 0.7992, 0.8019, 0.8080, 0.8007, 0.8112,
    0.7953, 0.8026, 0.8079, 0.8242, 0.8239, 0.8222, 0.8217, 0.8226, 0.8234, 0.8242,
    0.8241, 0.8242, 0.8239, 0.8240, 0.8242, 0.8239, 0.8239, 0.8239, 0.8238, 0.8237,
    0.8237, 0.8240, 0.8241, 0.8238, 0.8242, 0.8240, 0.8240, 0.8239, 0.8239, 0.8240,
    0.8239, 0.8239, 0.8239, 0.8239, 0.8239, 0.8241, 0.8240, 0.8240, 0.8240, 0.8239
]

epochs_aug = list(range(1, 51))
dice_aug = [
    0.7637, 0.7882, 0.7930, 0.8014, 0.8086, 0.8168, 0.8231, 0.8244, 0.8231, 0.8355,
    0.8069, 0.8288, 0.8374, 0.8366, 0.8447, 0.8442, 0.8487, 0.8470, 0.8578, 0.8549,
    0.8564, 0.8559, 0.8799, 0.8805, 0.8836, 0.8852, 0.8865, 0.8872, 0.8900, 0.8898,
    0.8910, 0.8924, 0.8922, 0.8933, 0.8940, 0.8940, 0.8961, 0.8964, 0.8982, 0.8963,
    0.8986, 0.8975, 0.8991, 0.9002, 0.9014, 0.9001, 0.9008, 0.9025, 0.9018, 0.9027
]

# -----------------------------
# Professional Styling (MATCH PREVIOUS FIGURE)
# -----------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11
})

fig, ax = plt.subplots(figsize=(10, 6))

# -----------------------------
# Plot Curves (Professional Colors)
# -----------------------------
ax.plot(
    epochs_orig,
    dice_orig,
    label="MobileUNet-v3 (Original Data, n=1,000)",
    color="#2b9348",
    linewidth=2.5
)

ax.plot(
    epochs_aug,
    dice_aug,
    label="MobileUNet-v3 (Augmented Data, n=6,000)",
    color="#d00000",
    linewidth=2.5
)

# -----------------------------
# Highlight Peak Performance
# -----------------------------
peak_orig = max(dice_orig)
epoch_peak_orig = dice_orig.index(peak_orig) + 1

ax.scatter(epoch_peak_orig, peak_orig, color="#2b9348", s=70, zorder=4)

ax.annotate(
    f"Peak: {peak_orig:.3f}",
    xy=(epoch_peak_orig, peak_orig),
    xytext=(epoch_peak_orig - 12, peak_orig - 0.02),
    arrowprops=dict(arrowstyle="-", color="#2b9348", lw=1),
    fontsize=10,
    weight="bold",
    color="#1b5e20"
)

peak_aug = max(dice_aug)
epoch_peak_aug = dice_aug.index(peak_aug) + 1

ax.scatter(epoch_peak_aug, peak_aug, color="#d00000", s=70, zorder=4)

ax.annotate(
    f"Peak: {peak_aug:.3f}",
    xy=(epoch_peak_aug, peak_aug),
    xytext=(epoch_peak_aug - 15, peak_aug - 0.035),
    arrowprops=dict(arrowstyle="-", color="#d00000", lw=1),
    fontsize=10,
    weight="bold",
    color="#7f0000"
)

# -----------------------------
# Formatting
# -----------------------------
ax.set_title(
    "Impact of Data Augmentation on Validation Dice (MobileUNet-v3)",
    fontsize=14,
    weight="bold",
    pad=15
)

ax.set_xlabel("Training Epochs", fontsize=12)
ax.set_ylabel("Validation Dice Score", fontsize=12)

ax.set_xlim(1, 50)
ax.set_ylim(0.5, 0.93)

ax.set_xticks(range(0, 51, 5))
ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)

# Legend
ax.legend(
    loc="lower right",
    frameon=True,
    edgecolor="gray",
    fontsize=11
)

plt.tight_layout()

# -----------------------------
# Save High Resolution
# -----------------------------
OUTPUT_PATH = "E:/Research/figures/Figure_5_Training_Curves_Professional.png"
os.makedirs("E:/Research/figures", exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=600, bbox_inches="tight")
plt.show()

print(f"Professional figure saved to: {OUTPUT_PATH}")