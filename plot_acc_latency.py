import matplotlib.pyplot as plt
import os

def generate_tradeoff_plot():
    # Model name, FPS (Latency scale), Validation Dice
    data = [
        ("Standard U-Net", 0.5, 0.789),
        ("DSC-UNet", 2.6, 0.776),
        ("MobileNetV2", 1.9, 0.812),
        ("MNv2 + CBAM", 1.77, 0.814),
        ("DeepLabV3+", 0.3, 0.834),
        ("SegFormer-B4", 0.4, 0.889),
        ("ResU-Net", 0.7, 0.702),
        ("OUR MODEL", 1.82, 0.902)
    ]

    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot background grid
    ax.grid(True, linestyle='--', alpha=0.6)

    # Plot points
    for name, fps, dice in data:
        if name == "OUR MODEL":
            # Highlight OUR MODEL
            ax.scatter(fps, dice, color='#EA4335', s=250, marker='*', zorder=5, edgecolors='black')
            ax.annotate(name, (fps, dice), xytext=(5, 10), textcoords='offset points',
                        fontsize=12, fontweight='bold', color='#EA4335')
        else:
            ax.scatter(fps, dice, color='#4285F4', s=100, zorder=4, edgecolors='black', alpha=0.8)
            
            # Manual fine-tuning for text labels so they don't overlap too much
            offset_y = 5
            offset_x = 5
            if name == "MobileNetV2":
                offset_y = -15
            elif name == "DeepLabV3+":
                offset_y = 8
            elif name == "ResU-Net":
                offset_y = -15
            elif name == "MNv2 + CBAM":
                offset_x = -75
                offset_y = -5
            elif name == "DSC-UNet":
                offset_y = 5
                offset_x = -35
            
            ax.annotate(name, (fps, dice), xytext=(offset_x, offset_y), textcoords='offset points',
                        fontsize=10, color='#333333', fontweight='medium')

    ax.set_title("Model Comparison: Accuracy vs Inference Speed Trade-off", fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Inference Speed (FPS) $\\rightarrow$ Faster", fontsize=12, fontweight='bold')
    ax.set_ylabel("Validation Accuracy (Dice Score) $\\rightarrow$ Better", fontsize=12, fontweight='bold')
    
    # Highlight the "Ideal" quadrant direction
    ax.annotate("Ideal Region\n(High Accuracy, High Speed)", xy=(2.4, 0.88), xytext=(2.2, 0.86),
                arrowprops=dict(facecolor='green', shrink=0.05, alpha=0.5, edgecolor='none'),
                fontsize=11, color="green", fontweight='bold', alpha=0.7, ha='center')

    # Spines style
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    
    os.makedirs("figures", exist_ok=True)
    out_path = os.path.join("figures", "Accuracy_vs_Latency_Tradeoff_Graph.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved tradeoff plot to: {out_path}")

if __name__ == "__main__":
    generate_tradeoff_plot()
