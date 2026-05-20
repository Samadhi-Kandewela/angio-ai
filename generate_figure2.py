from graphviz import Digraph
import os

# Ensure output directory exists
os.makedirs("E:/Research/figures", exist_ok=True)

# Initialize directed graph
dot = Digraph(comment='MobileUNet-v3 Architecture', format='png')
dot.attr(
    rankdir='LR',
    nodesep='0.8',
    ranksep='1.4',
    splines='ortho',
    fontname='Times New Roman'
)

# Global edge style
dot.attr('edge',
         color='#374151',
         penwidth='1.8',
         arrowsize='0.8')

# --- Professional Color Palette ---
style_input = {
    'shape': 'box',
    'style': 'filled',
    'fillcolor': '#F3F4F6',
    'fontname': 'Times New Roman',
    'fontsize': '12',
    'color': '#4B5563',
    'penwidth': '1.5'
}

style_encoder = {
    'shape': 'rect',
    'style': 'filled,rounded',
    'fillcolor': '#D1FAE5',
    'fontname': 'Times New Roman',
    'fontsize': '12',
    'color': '#065F46',
    'penwidth': '2'
}

style_decoder = {
    'shape': 'rect',
    'style': 'filled,rounded',
    'fillcolor': '#DBEAFE',
    'fontname': 'Times New Roman',
    'fontsize': '12',
    'color': '#1E40AF',
    'penwidth': '2'
}

style_attention = {
    'shape': 'hexagon',
    'style': 'filled',
    'fillcolor': '#FEF3C7',
    'fontname': 'Times New Roman',
    'fontsize': '11',
    'color': '#92400E',
    'penwidth': '1.8'
}

style_output = {
    'shape': 'box',
    'style': 'filled',
    'fillcolor': '#FEE2E2',
    'fontname': 'Times New Roman',
    'fontsize': '12',
    'color': '#991B1B',
    'penwidth': '1.5'
}

# --- Input ---
dot.node('Input', 'Input Image\n(512×512×3)', **style_input)

# --- Encoder Cluster ---
with dot.subgraph(name='cluster_encoder') as c_enc:
    c_enc.attr(
        label='Encoder (MobileNetV3-Large)',
        style='dashed',
        color='#9CA3AF',
        fontname='Times New Roman',
        fontsize='14'
    )

    c_enc.node('Enc1', 'Layer 0–1\n16 Channels', **style_encoder)
    c_enc.node('Enc2', 'Layer 2–3\n24 Channels', **style_encoder)
    c_enc.node('Enc3', 'Layer 4–6\n40 Channels', **style_encoder)
    c_enc.node('Enc4', 'Layer 7–12\n112 Channels', **style_encoder)
    c_enc.node('Enc5', 'Bottleneck\n960 Channels', **style_encoder)

# --- Decoder Cluster ---
with dot.subgraph(name='cluster_decoder') as c_dec:
    c_dec.attr(
        label='Decoder (Attention-Guided)',
        style='dashed',
        color='#9CA3AF',
        fontname='Times New Roman',
        fontsize='14'
    )

    c_dec.node('Dec1', 'Upsample + DSConv Block 1', **style_decoder)
    c_dec.node('Att1', 'CBAM Attention', **style_attention)

    c_dec.node('Dec2', 'Upsample + DSConv Block 2', **style_decoder)
    c_dec.node('Att2', 'CBAM Attention', **style_attention)

    c_dec.node('Dec3', 'Upsample + DSConv Block 3', **style_decoder)
    c_dec.node('Att3', 'CBAM Attention', **style_attention)

    c_dec.node('Dec4', 'Upsample + DSConv Block 4', **style_decoder)
    c_dec.node('Att4', 'CBAM Attention', **style_attention)

    c_dec.node('Dec5', 'Upsample + DSConv Block 5', **style_decoder)
    c_dec.node('Att5', 'CBAM Attention', **style_attention)

# --- Output ---
dot.node('Output', '1×1 Conv\nSegmentation Mask\n(512×512×1)', **style_output)

# --- Main Flow ---
dot.edge('Input', 'Enc1')
dot.edge('Enc1', 'Enc2')
dot.edge('Enc2', 'Enc3')
dot.edge('Enc3', 'Enc4')
dot.edge('Enc4', 'Enc5')

dot.edge('Enc5', 'Dec1')
dot.edge('Dec1', 'Att1')
dot.edge('Att1', 'Dec2')
dot.edge('Dec2', 'Att2')
dot.edge('Att2', 'Dec3')
dot.edge('Dec3', 'Att3')
dot.edge('Att3', 'Dec4')
dot.edge('Dec4', 'Att4')
dot.edge('Att4', 'Dec5')
dot.edge('Dec5', 'Att5')
dot.edge('Att5', 'Output')

# --- Skip Connections ---
skip_style = {
    'style': 'dashed',
    'color': '#6B7280',
    'penwidth': '1.5',
    'fontsize': '10',
    'fontname': 'Times New Roman'
}

dot.edge('Enc4', 'Dec1', label='Skip', **skip_style)
dot.edge('Enc3', 'Dec2', label='Skip', **skip_style)
dot.edge('Enc2', 'Dec3', label='Skip', **skip_style)
dot.edge('Enc1', 'Dec4', label='Skip', **skip_style)
dot.edge('Input', 'Dec5', label='Skip', **skip_style)

# Render high-resolution version
output_path = "E:/Research/figures/Figure_2_MobileUNet_v3_Architecture_Professional"
dot.render(output_path, view=False)

print(f"Professional architecture diagram saved at: {output_path}.png")