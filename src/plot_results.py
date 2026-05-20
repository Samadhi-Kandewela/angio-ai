import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['Baseline', 'DSC-UNet', 'Mobile-v1', 'Mobile-v2', 'DeepLabV3+', 'Mobile-v3']
# Latency (CPU ms)
latency = [2157, 1735, 588, 610, 500, 550] # v3 is faster than v2 due to MobileNetV3 backbone
# Accuracy (Dice)
accuracy = [0.7861, 0.7753, 0.8106, 0.8143, 0.7510, 0.8242] 
# Efficiency (1/FLOPs scaled) - Optional context logic
# flops = [124.5, 27.5, 7.1, 7.2, 6.0, 5.5]

x = np.arange(len(models))
width = 0.25

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Latency (Bar)
color = 'tab:blue'
ax1.set_xlabel('Model')
ax1.set_ylabel('Inference Latency (ms) - Lower is Better', color=color)
bars = ax1.bar(x - width/2, latency, width, color=color, alpha=0.6, label='Latency')
ax1.tick_params(axis='y', labelcolor=color)

# Plot Accuracy (Line)
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Dice Accuracy (0-1) - Higher is Better', color=color)
line = ax2.plot(x + width/2, accuracy, color=color, marker='o', linewidth=2, label='Accuracy')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0.70, 0.85)

# Labels
plt.title('Accuracy vs. Latency Trade-off')
ax1.set_xticks(x)
ax1.set_xticklabels(models)

fig.tight_layout()
plt.grid(True, axis='y', alpha=0.3)
plt.savefig('tradeoff_plot.png', dpi=300)
print("Saved tradeoff_plot.png")
