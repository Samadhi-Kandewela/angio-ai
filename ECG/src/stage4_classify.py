"""
stage4_classify.py
------------------
Stage IV: Diagnostic classification of a single-lead ECG signal
using a 1D-ResNet trained on PTB-XL labels.

PTB-XL Diagnostic Classes (5 superclasses):
  0 - NORM : Normal ECG
  1 - MI   : Myocardial Infarction
  2 - STTC : ST/T Change
  3 - CD   : Conduction Disturbance
  4 - HYP  : Hypertrophy
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.signal import resample
from pathlib import Path

# ─── Constants ─────────────────────────────────────────────────────────────────
SIGNAL_LENGTH = 1000
NUM_CLASSES = 5
CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]

SCP_TO_CLASS = {
    "NORM": 0,
    # MI - Myocardial Infarction
    "AMI": 1, "IMI": 1, "LMI": 1, "ALMI": 1, "PMI": 1, "ASMI": 1,
    "ILMI": 1, "IPLMI": 1, "IPMI": 1, "INJAS": 1, "INJAL": 1,
    # STTC - ST/T Change
    "STD_": 2, "STE_": 2, "ISCA": 2, "NST_": 2, "LNGQT": 2, "INVT": 2,
    "NDT": 2, "ISCI": 2, "ISC_": 2, "TAB_": 2, "ISCAL": 2, "ISCIN": 2,
    "ISCLA": 2, "ISCAS": 2, "ISCIL": 2, "ISCIP": 2, "ISCAN": 2,
    # CD - Conduction Disturbance
    "LBBB": 3, "RBBB": 3, "AVB": 3, "AFIB": 3, "AFLT": 3, "WPW": 3,
    "IRBBB": 3, "ILBBB": 3, "CLBBB": 3, "CRBBB": 3,
    "1AVB": 3, "2AVB": 3, "3AVB": 3,
    "LAFB": 3, "LPFB": 3, "IVCD": 3,
    "SVTAC": 3, "PSVT": 3, "TRIGU": 3, "BIGU": 3,
    "PACE": 3, "SVARR": 3, "VCLVH": 3,
    # HYP - Hypertrophy
    "LVH": 4, "RVH": 4, "LAH": 4, "RAH": 4, "SEHYP": 4,
    "LAO/LAE": 4, "RAO/RAE": 4, "LVH_": 4,
}


# ─── 1D ResNet Architecture ─────────────────────────────────────────────────
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


class ECGResNet1D(nn.Module):
    """1D-ResNet for single-lead ECG. Input: (B, 1, SIGNAL_LENGTH)"""
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self.layer1 = ResidualBlock1D(32,  64,  stride=2)
        self.layer2 = ResidualBlock1D(64,  128, stride=2)
        self.layer3 = ResidualBlock1D(128, 256, stride=2)
        self.layer4 = ResidualBlock1D(256, 512, stride=2)
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


# ─── Dataset ────────────────────────────────────────────────────────────────
class ECGDataset(Dataset):
    """Loads pre-extracted .npy signals and integer class labels."""
    def __init__(self, signals_path: str, labels_path: str, augment: bool = False):
        self.signals = np.load(signals_path).astype(np.float32)
        self.labels = np.load(labels_path).astype(np.int64)
        self.augment = augment
        print(f"Loaded {len(self.signals)} samples from {signals_path}")

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        sig = self.signals[idx].copy()
        if self.augment:
            # Amplitude scale +/-10% (was 20%)
            sig = sig * np.random.uniform(0.9, 1.1)
            # Gaussian noise (reduced)
            sig += np.random.normal(0, 0.01 * (sig.std() + 1e-8), sig.shape)
            # Random baseline wander (reduced amplitude)
            t = np.linspace(0, 1, len(sig))
            sig += 0.02 * np.sin(2 * np.pi * np.random.uniform(0.5, 2.0) * t)
            # Random time shift (reduced)
            shift = np.random.randint(-20, 20)
            sig = np.roll(sig, shift)
        sig = (sig - sig.mean()) / (sig.std() + 1e-8)
        return torch.from_numpy(sig).unsqueeze(0), torch.tensor(self.labels[idx])


# ─── Training ───────────────────────────────────────────────────────────────
def train_model(signals_train, labels_train, signals_val, labels_val,
                checkpoint_dir="../checkpoints", epochs=50, batch_size=32, lr=3e-4,
                patience=12):
    """
    Train with four key improvements over v1:
      1. Class-weighted CrossEntropyLoss  -> handles NORM dominance
      2. Stronger augmentation            -> reduces overfitting
      3. ReduceLROnPlateau scheduler      -> adapts to val loss plateau
      4. Early stopping (patience=10)     -> stops before memorization
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    train_ds = ECGDataset(signals_train, labels_train, augment=True)
    val_ds = ECGDataset(signals_val, labels_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0)

    # Fix 1: Compute class weights inversely proportional to class frequency
    labels_arr = train_ds.labels
    class_counts = np.bincount(labels_arr, minlength=NUM_CLASSES).astype(np.float32)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES  # normalize
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Class weights: { {CLASS_NAMES[i]: f'{class_weights[i]:.2f}' for i in range(NUM_CLASSES)} }")

    model = ECGResNet1D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

    # Oversampling: give each sample a weight = its class weight
    # so rare classes (HYP) are sampled ~equally to NORM
    sample_weights = torch.tensor(
        [class_weights[l] for l in train_ds.labels], dtype=torch.float32
    )
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    # Use sampler instead of shuffle=True
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, num_workers=0
    )

    # LR warmup: linearly ramp from lr/10 to lr over first 5 epochs,
    # then ReduceLROnPlateau takes over
    WARMUP_EPOCHS = 5
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS
    )
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    # Standard loss (we removed weight=weight_tensor because 
    # WeightedRandomSampler already balances the classes. Doing both 
    # double-penalizes the majority class and hurts overall accuracy).
    criterion = nn.CrossEntropyLoss()

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    epochs_no_improve = 0
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for sigs, labs in train_loader:
            sigs, labs = sigs.to(device), labs.to(device)
            optimizer.zero_grad()
            loss = criterion(model(sigs), labs)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(sigs)

        model.eval()
        correct = 0
        with torch.no_grad():
            for sigs, labs in val_loader:
                correct += (model(sigs.to(device)).argmax(1) == labs.to(device)).sum().item()

        avg_loss = total_loss / len(train_ds)
        val_acc = correct / len(val_ds)

        # Step warmup scheduler for first WARMUP_EPOCHS, then plateau
        if epoch <= WARMUP_EPOCHS:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch:03d}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.3f} | LR: {current_lr:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            ckpt = os.path.join(checkpoint_dir, "ecg_resnet1d_best.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"  OK Saved -> {ckpt}")
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    json.dump(history, open(os.path.join(checkpoint_dir, "history.json"), "w"), indent=2)
    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    return model


# ─── Inference ──────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str) -> ECGResNet1D:
    model = ECGResNet1D()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    print(f"[Stage IV] Model loaded from {checkpoint_path}")
    return model


def classify(voltage_mv: np.ndarray, model: ECGResNet1D, device: str = "cpu") -> dict:
    sig = resample(voltage_mv, SIGNAL_LENGTH).astype(np.float32)
    sig = (sig - sig.mean()) / (sig.std() + 1e-8)
    tensor = torch.from_numpy(sig).unsqueeze(0).unsqueeze(0).to(device)
    model = model.to(device)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1).squeeze().cpu().numpy()
    pred = int(np.argmax(probs))
    return {
        "predicted_class": pred,
        "label": CLASS_NAMES[pred],
        "confidence": float(probs[pred]),
        "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)},
    }


if __name__ == "__main__":
    model = ECGResNet1D()
    dummy = torch.randn(4, 1, SIGNAL_LENGTH)
    out = model(dummy)
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
