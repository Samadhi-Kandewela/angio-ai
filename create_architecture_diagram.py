import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def draw_rounded_rect(ax, x, y, w, h, text, facecolor, edgecolor="black"):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1", 
        linewidth=2,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=2
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, 
            ha='center', va='center', 
            fontsize=12, fontweight='bold', color='white',
            family='sans-serif', zorder=3)

def draw_arrow(ax, start_x, start_y, end_x, end_y, text="", curved=False):
    if curved:
        ax.annotate(text, xy=(end_x, end_y), xytext=(start_x, start_y),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.3",
                                    color="gray", lw=2),
                    fontsize=10, fontweight='bold', color="gray", ha="center", va="bottom", zorder=1)
    else:
        ax.annotate(text, xy=(end_x, end_y), xytext=(start_x, start_y),
                    arrowprops=dict(arrowstyle="->", color="black", lw=2),
                    fontsize=10, fontweight='bold', color="black", ha="center", va="bottom", zorder=1)

def generate_diagram():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Title
    ax.text(50, 90, "MobileUNetv3 Architecture (Simplified)", 
            ha='center', va='center', fontsize=20, fontweight='heavy',
            family='sans-serif', color='#2a2a2a')

    # Colors
    c_enc = "#4285F4"  # Google Blue
    c_dec = "#34A853"  # Google Green
    c_bot = "#EA4335"  # Google Red
    c_inout = "#FBBC05" # Google Yellow

    # Nodes
    # 1. Input Image
    draw_rounded_rect(ax, 5, 40, 15, 20, "Input Image\n(Angiogram)", c_inout)

    # 2. Encoder
    draw_rounded_rect(ax, 30, 40, 18, 20, "MobileNetV3\nEncoder\n(Feature Extraction)", c_enc)

    # 3. Decoder
    draw_rounded_rect(ax, 60, 40, 18, 20, "Decoder\n(DSConv +\nCBAM Attention)", c_dec)

    # 4. Output Mask
    draw_rounded_rect(ax, 88, 40, 10, 20, "Output\nMask", c_inout)

    # Edges
    # Input -> Encoder
    draw_arrow(ax, 20, 50, 30, 50, text="")
    
    # Encoder -> Decoder (Bottleneck)
    draw_arrow(ax, 48, 45, 60, 45, text="")
    ax.text(54, 42, "Bottleneck", ha='center', va='top', fontsize=10, fontweight='bold', color=c_bot)

    # Decoder -> Output
    draw_arrow(ax, 78, 50, 88, 50, text="")

    # Skip Connections
    ax.annotate("Skip Connections\n(Multi-scale Context)", xy=(69, 60), xytext=(39, 60),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.4",
                                color="#FF6D00", lw=2.5, linestyle="--"),
                fontsize=11, fontweight='bold', color="#FF6D00", ha="center", va="bottom", zorder=1)

    plt.tight_layout()
    
    os.makedirs("figures", exist_ok=True)
    out_path = os.path.join("figures", "MobileUNetv3_Architecture_Simplified.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', transparent=False, facecolor='white')
    print(f"Saved diagram to: {out_path}")

if __name__ == "__main__":
    generate_diagram()
