"""
stage4_classify.py (12-Lead Version)
------------------------------------
Stage IV: Diagnostic classification of a 12-lead ECG signal
using a 1D-ResNet (ResNet-18 style) trained on PTB-XL labels.

Input shape: (Batch, 12, 1000)
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# ─── Constants ─────────────────────────────────────────────────────────────────
SIGNAL_LENGTH = 1000
NUM_CLASSES = 5
CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


# ─── 1D ResNet Architecture (12-Lead) ───────────────────────────────────────
class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 7, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


def make_layer(in_ch, out_ch, blocks, stride=1):
    layers = []
    layers.append(ResidualBlock1D(in_ch, out_ch, stride))
    for _ in range(1, blocks):
        layers.append(ResidualBlock1D(out_ch, out_ch, 1))
    return nn.Sequential(*layers)


class ECGResNet1D(nn.Module):
    """1D-ResNet for 12-lead ECG. Input: (B, 12, SIGNAL_LENGTH)"""
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        # Stem accepts 12 input channels
        self.stem = nn.Sequential(
            nn.Conv1d(12, 32, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32), 
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        # ResNet-18 configuration: 2 blocks per layer
        self.layer1 = make_layer(32,  64,  2, stride=2)
        self.layer2 = make_layer(64,  128, 2, stride=2)
        self.layer3 = make_layer(128, 256, 2, stride=2)
        self.layer4 = make_layer(256, 512, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


# ─── Inference ──────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str, device: str = "cpu") -> nn.Module:
    """Load the 12-lead ResNet model from a checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    
    model = ECGResNet1D(num_classes=NUM_CLASSES)
    # The new model checkpoint is used here
    state_dict = torch.load(checkpoint_path, map_location=device)
    # Sometimes models are saved as dict with 'model_state_dict' or just raw dict
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict(model: nn.Module, signal_12lead_mv: np.ndarray, device: str = "cpu") -> dict:
    """
    Run diagnostic classification on a digitized 12-lead ECG signal.
    
    Args:
        model           : Loaded ECGResNet1D model
        signal_12lead_mv: Numpy array of shape (12, 1000) containing voltages in mV
        device          : Device to run inference on
        
    Returns:
        Dict with predicted_class (int), label (str), confidence (float), and all probabilities
    """
    if signal_12lead_mv.shape != (12, SIGNAL_LENGTH):
        raise ValueError(f"Expected signal shape (12, {SIGNAL_LENGTH}), got {signal_12lead_mv.shape}")
        
    # Standardize each channel independently (mean 0, std 1)
    # as was done during training in ECGDataset
    tensor_input = np.zeros_like(signal_12lead_mv)
    for c in range(12):
        sig = signal_12lead_mv[c]
        std = sig.std()
        if std > 1e-6:
            tensor_input[c] = (sig - sig.mean()) / std
        else:
            # If flatline (zero-padded missing lead), keep it 0
            tensor_input[c] = 0.0

    # Convert to tensor and add batch dimension: (1, 12, 1000)
    tensor_input = torch.from_numpy(tensor_input).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(tensor_input)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        
    pred_idx = int(np.argmax(probs))
    
    return {
        "predicted_class": pred_idx,
        "label": CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {name: float(p) for name, p in zip(CLASS_NAMES, probs)}
    }
