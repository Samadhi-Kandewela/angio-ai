import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Create directory
os.makedirs("E:/Research/figures", exist_ok=True)

# Define diagram size and figure
fig, ax = plt.subplots(figsize=(16, 8))
ax.set_xlim(0, 20)
ax.set_ylim(0, 10)
ax.axis('off')

# Professional Font Setting
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 11
})

# Helper function to draw boxes
def draw_box(ax, x, y, width, height, text, facecolor, edgecolor, fontsize=10):
    rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor=edgecolor, facecolor=facecolor, zorder=2)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=fontsize, weight='bold', wrap=True)

# Helper function to draw attention hexagons
def draw_hexagon(ax, x, y, radius, text):
    hexagon = patches.RegularPolygon((x, y), numVertices=6, radius=radius, orientation=0, edgecolor='#92400E', facecolor='#FEF3C7', linewidth=1.8, zorder=2)
    ax.add_patch(hexagon)
    ax.text(x, y, text, ha='center', va='center', fontsize=9, weight='bold', color='#92400E')

# Helper function for arrows
def draw_arrow(ax, x1, y1, x2, y2, style='-', color='black', text=''):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=2, ls=style), zorder=1)
    if text:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.2, text, ha='center', va='bottom', fontsize=9, color=color, style='italic')

# Colors
color_input = '#F3F4F6'
color_encoder = '#D1FAE5'
color_decoder = '#DBEAFE'
color_output = '#FEE2E2'

# Draw Input
draw_box(ax, 0.5, 4.5, 2, 2, 'Input Image\n(512×512×3)', color_input, '#4B5563')

# Draw Encoder Blocks (MobileNetV3)
enc_x, enc_y_start, h, w = 3.5, 8, 1, 2.5
enc_layers = [
    ("MobileNetV3\nLayers 0-1\n[16 Ch]", 8),
    ("MobileNetV3\nLayers 2-3\n[24 Ch]", 6.5),
    ("MobileNetV3\nLayers 4-6\n[40 Ch]", 5),
    ("MobileNetV3\nLayers 7-12\n[112 Ch]", 3.5),
    ("MobileNetV3\nBottleneck\n[960 Ch]", 2)
]

for text, y in enc_layers:
    draw_box(ax, enc_x, y, w, h, text, color_encoder, '#065F46', fontsize=9)

# Draw Decoder Blocks (DSConv)
dec_x, h_dec, w_dec = 11, 1, 3
dec_layers = [
    ("Upsample + DSConv 1", 2),
    ("Upsample + DSConv 2", 3.5),
    ("Upsample + DSConv 3", 5),
    ("Upsample + DSConv 4", 6.5),
    ("Upsample + DSConv 5", 8)
]

for text, y in dec_layers:
    draw_box(ax, dec_x, y, w_dec, h_dec, text, color_decoder, '#1E40AF', fontsize=9)

# Draw Attention Blocks
att_x = 15.5
for text, y in dec_layers:
    draw_hexagon(ax, att_x, y + 0.5, 0.6, "CBAM\nAttention")

# Draw Output
draw_box(ax, 17.5, 4.5, 2, 2, 'Segmentation Mask\n(512×512×1)', color_output, '#991B1B')

# Draw Connections - Encoder Path
draw_arrow(ax, 2.5, 5.5, 3.5, 8.5) # Input to Enc1
for i in range(len(enc_layers)-1):
    draw_arrow(ax, enc_x + w/2, enc_layers[i][1], enc_x + w/2, enc_layers[i+1][1] + h)

# Draw Connections - Decoder Bridge & Path
draw_arrow(ax, enc_x + w, enc_layers[4][1] + h/2, dec_x, dec_layers[0][1] + h_dec/2) # Bottleneck to Dec1
for i in range(len(dec_layers)-1):
    draw_arrow(ax, att_x, dec_layers[i][1] + h_dec, dec_x + w_dec/2, dec_layers[i+1][1]) # Att to Dec

# Draw Connections - Dec to Att
for i in range(len(dec_layers)):
    draw_arrow(ax, dec_x + w_dec, dec_layers[i][1] + h_dec/2, att_x - 0.6, dec_layers[i][1] + h_dec/2)

# Draw Connections - Att to Output (only last one)
draw_arrow(ax, att_x + 0.6, dec_layers[4][1] + h_dec/2, 17.5, 5.5)

# Skip Connections (Dashed)
skip_pairs = [
    (enc_layers[3], dec_layers[0]),
    (enc_layers[2], dec_layers[1]),
    (enc_layers[1], dec_layers[2]),
    (enc_layers[0], dec_layers[3])
]
for enc, dec in skip_pairs:
    draw_arrow(ax, enc_x + w, enc[1] + h/2, dec_x, dec[1] + h_dec/2, style='--', color='#6B7280', text='Skip')

# Input to final decoder skip
draw_arrow(ax, 1.5, 6.5, dec_x, dec_layers[4][1] + h_dec/2, style='--', color='#6B7280', text='Skip')


# Add Title/Labels
plt.title("Figure 2: MobileUNet-v3 Architectural Topology", fontsize=18, weight='bold', pad=20)
plt.text(4.75, 9.5, "Encoder (MobileNetV3-Large)", ha='center', fontsize=14, color='#9CA3AF', weight='bold')
plt.text(12.5, 9.5, "Decoder Pathway (Attention-Guided)", ha='center', fontsize=14, color='#9CA3AF', weight='bold')

plt.tight_layout()
save_path = "E:/Research/figures/Figure_2_MobileUNet_v3_Architecture_Professional.png"
plt.savefig(save_path, dpi=600, bbox_inches='tight')
print(f"Professional architecture diagram successfully generated at {save_path}")
