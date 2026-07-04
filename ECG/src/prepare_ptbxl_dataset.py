"""
prepare_ptbxl_dataset.py
------------------------
Downloads the PTB-XL database metadata and ECG records, then prepares
.npy training/validation arrays for Stage IV classification.

What this script does:
  1. Downloads PTB-XL metadata CSVs (ptbxl_database.csv, scp_statements.csv)
  2. Parses diagnostic labels -> maps SCP codes to 5 superclasses
  3. Downloads ECG waveform records (100 Hz version, ~35 MB per 1000 records)
  4. Extracts Lead II (the standard rhythm strip lead)
  5. Resamples all signals to SIGNAL_LENGTH = 1000 samples
  6. Splits 80/20 into train and validation sets
  7. Saves:
       dataset/ptbxl_train_signals.npy  (N_train, 1000)
       dataset/ptbxl_train_labels.npy   (N_train,)
       dataset/ptbxl_val_signals.npy    (N_val, 1000)
       dataset/ptbxl_val_labels.npy     (N_val,)

Usage:
    python prepare_ptbxl_dataset.py
    python prepare_ptbxl_dataset.py --max-records 2000   # faster subset
    python prepare_ptbxl_dataset.py --max-records -1     # full dataset (21k)

PTB-XL reference: Wagner et al. (2020) Nature Scientific Data
"""

import os
import sys
import ast
import argparse
import urllib.request
import zipfile
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
from scipy.signal import resample
from sklearn.model_selection import train_test_split

# ── Add src directory to path so we can import CLASS_NAMES ─────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage4_classify import SCP_TO_CLASS, CLASS_NAMES, SIGNAL_LENGTH

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
DATASET_DIR = ROOT_DIR / "dataset"
PTBXL_DIR = DATASET_DIR / "ptb-xl"
RECORDS_DIR = PTBXL_DIR / "records100"
OUTPUT_DIR = DATASET_DIR

# PhysioNet PTB-XL base URL
PHYSIONET_BASE = "https://physionet.org/files/ptb-xl/1.0.3"

# Lead names in wfdb PTB-XL order (index of Lead II = 1)
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_INDEX = 1  # Lead II


def download_file(url: str, dest: Path, desc: str = ""):
    """Download a file from URL to dest if not already present."""
    if dest.exists():
        print(f"  [skip] Already exists: {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {desc or dest.name}...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  Saved -> {dest}")
    except Exception as e:
        print(f"  ERROR downloading {url}: {e}")
        raise


def download_metadata():
    """Download the PTB-XL metadata CSV files from PhysioNet."""
    print("\n[Step 1] Downloading PTB-XL metadata files...")
    files = {
        "ptbxl_database.csv": f"{PHYSIONET_BASE}/ptbxl_database.csv",
        "scp_statements.csv": f"{PHYSIONET_BASE}/scp_statements.csv",
    }
    for fname, url in files.items():
        dest = PTBXL_DIR / fname
        download_file(url, dest, fname)


def load_metadata(ptbxl_db_path: Path, scp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and return PTB-XL database and SCP statements."""
    print("\n[Step 2] Loading metadata...")
    db = pd.read_csv(ptbxl_db_path, index_col="ecg_id")
    scp = pd.read_csv(scp_path, index_col=0)
    # Filter to only diagnostic SCP codes
    scp_diag = scp[scp["diagnostic"] == 1]
    print(f"  Loaded {len(db)} ECG records")
    print(f"  Diagnostic SCP codes: {len(scp_diag)}")
    return db, scp_diag


def parse_scp_codes(scp_str: str) -> dict:
    """Parse the scp_codes column string into a dict, e.g. \"{'NORM': 100.0}\" -> {'NORM': 100.0}"""
    try:
        return ast.literal_eval(scp_str)
    except Exception:
        return {}


def map_label(scp_codes: dict, scp_diag: pd.DataFrame) -> int | None:
    """
    Map a record's SCP codes to one of the 5 superclass indices.

    Strategy: Filter to only diagnostic codes present in SCP_TO_CLASS.
    Pick the code with the highest confidence score.
    Return None if no matching diagnostic code is found.
    """
    best_class = None
    best_conf = -1.0

    for code, confidence in scp_codes.items():
        if code in SCP_TO_CLASS and confidence > best_conf:
            best_class = SCP_TO_CLASS[code]
            best_conf = confidence

    return best_class


def _find_record(rel_path: str) -> str | None:
    """
    Find where a PTB-XL record actually lives on disk.
    Handles two possible layouts:
      - Correct: PTBXL_DIR/records100/00000/00001_lr.hea
      - Double-nested (old wfdb bug): PTBXL_DIR/records100/00000/records100/00000/00001_lr.hea
    Returns the record path string (without extension) or None if not found.
    """
    # Preferred / correct location
    correct = PTBXL_DIR / (rel_path + ".hea")
    if correct.exists():
        return str(PTBXL_DIR / rel_path)

    # Double-nested fallback (from old downloads)
    fname = Path(rel_path).name       # '00001_lr'
    subfolder = Path(rel_path).parent  # 'records100/00000'
    nested = PTBXL_DIR / subfolder / subfolder / fname
    if (str(nested) + ".hea") and Path(str(nested) + ".hea").exists():
        return str(nested)

    return None


def download_records(ecg_ids: list, db: pd.DataFrame, verbose: bool = True):
    """
    Download missing PTB-XL waveform records directly from PhysioNet via HTTP.
    Uses urllib instead of wfdb.dl_database to avoid Windows path-separator
    issues that cause wfdb to silently hang or download to wrong locations.

    Files are saved to:
      PTBXL_DIR/records100/00000/00001_lr.hea
      PTBXL_DIR/records100/00000/00001_lr.dat
    """
    BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3"
    DAT_MIN_BYTES = 20_000   # valid 100Hz 12-lead .dat is always ~24,000 bytes

    print(f"\n[Step 3] Checking/downloading {len(ecg_ids)} waveform records...")

    def _record_is_valid(rel_path: str) -> bool:
        """True only if BOTH .hea and .dat exist AND .dat is not a partial download."""
        found = _find_record(rel_path)
        if found is None:
            return False
        dat = Path(found + ".dat")
        return dat.exists() and dat.stat().st_size >= DAT_MIN_BYTES

    # Find records that are missing OR have a corrupt/partial .dat file
    missing_ids = [
        ecg_id for ecg_id in ecg_ids
        if not _record_is_valid(db.loc[ecg_id, "filename_lr"])
    ]

    if not missing_ids:
        print("  All records already downloaded.")
        return

    print(f"  {len(missing_ids)} records to download (~{len(missing_ids)*24//1024} MB).")
    print("  Downloading from PhysioNet...")

    success = 0
    failed = 0

    for i, ecg_id in enumerate(missing_ids):
        # Forward-slash path, e.g. 'records100/00000/00001_lr'
        rel_path = db.loc[ecg_id, "filename_lr"].replace("\\", "/")
        local_dir = PTBXL_DIR / Path(rel_path).parent
        local_dir.mkdir(parents=True, exist_ok=True)
        fname = Path(rel_path).name  # '00001_lr'

        record_ok = True
        for ext in (".hea", ".dat"):
            url = f"{BASE_URL}/{rel_path}{ext}"
            dest = local_dir / (fname + ext)
            # Skip if already a valid file (size check for .dat)
            if dest.exists():
                if ext == ".dat" and dest.stat().st_size < DAT_MIN_BYTES:
                    dest.unlink()  # delete corrupt partial download
                else:
                    continue
            # Retry up to 3 times on network errors
            downloaded = False
            for attempt in range(3):
                try:
                    urllib.request.urlretrieve(url, dest)
                    downloaded = True
                    break
                except Exception as e:
                    if attempt == 2 and verbose:
                        print(f"  WARNING: {rel_path}{ext} (attempt {attempt+1}): {e}")
            if not downloaded:
                record_ok = False
                break

        if record_ok:
            success += 1
        else:
            failed += 1

        if verbose and (i + 1) % 200 == 0:
            pct = (i + 1) / len(missing_ids) * 100
            print(f"  [{pct:.0f}%] {i+1}/{len(missing_ids)} processed  "
                  f"(ok={success}, fail={failed})")

    print(f"  Done. Downloaded {success} records, {failed} failed.")


def load_signal(ecg_id: int, db: pd.DataFrame) -> np.ndarray | None:
    """
    Load the Lead II signal from a PTB-XL record.
    Uses _find_record() to locate the file in either valid disk layout.
    Returns a 1D numpy array of voltages in mV, or None on error.
    """
    rel_path = db.loc[ecg_id, "filename_lr"]
    record_path = _find_record(rel_path)

    if record_path is None:
        return None

    try:
        record = wfdb.rdrecord(record_path)
        # record.p_signal shape: (N_samples, 12_leads)
        signal = record.p_signal[:, LEAD_INDEX].astype(np.float32)

        # Remove NaN/Inf values
        if np.any(~np.isfinite(signal)):
            signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

        # Resample to SIGNAL_LENGTH
        signal_resampled = resample(signal, SIGNAL_LENGTH).astype(np.float32)
        return signal_resampled
    except Exception:
        return None


def prepare_dataset(max_records: int = 5000, val_fraction: float = 0.2, seed: int = 42):
    """
    Main data preparation function.

    Args:
        max_records   : maximum number of records to use (-1 = all)
        val_fraction  : fraction of data to use for validation
        seed          : random seed for reproducibility
    """
    # ── Step 1: Download metadata ──────────────────────────────────────────
    PTBXL_DIR.mkdir(parents=True, exist_ok=True)
    download_metadata()

    # ── Step 2: Load metadata ──────────────────────────────────────────────
    db, scp_diag = load_metadata(
        PTBXL_DIR / "ptbxl_database.csv",
        PTBXL_DIR / "scp_statements.csv",
    )

    # ── Step 3: Parse labels and filter records ────────────────────────────
    print("\n[Step 3] Parsing diagnostic labels...")
    labels_raw = {}
    for ecg_id, row in db.iterrows():
        scp_codes = parse_scp_codes(row["scp_codes"])
        label = map_label(scp_codes, scp_diag)
        if label is not None:
            labels_raw[ecg_id] = label

    print(f"  Records with valid labels: {len(labels_raw)}")

    # Print class distribution
    from collections import Counter
    dist = Counter(labels_raw.values())
    for cls_idx, count in sorted(dist.items()):
        print(f"    {CLASS_NAMES[cls_idx]:6s}: {count:5d} records")

    # Apply max_records limit (stratified)
    valid_ids = list(labels_raw.keys())
    valid_labels = [labels_raw[i] for i in valid_ids]

    if max_records > 0 and len(valid_ids) > max_records:
        print(f"\n  Subsampling to {max_records} records (stratified)...")
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=None,
                                     train_size=max_records, random_state=seed)
        idx_keep, _ = next(sss.split(valid_ids, valid_labels))
        valid_ids = [valid_ids[i] for i in idx_keep]
        valid_labels = [valid_labels[i] for i in idx_keep]
        print(f"  After subsampling: {len(valid_ids)} records")

    # ── Step 4: Download waveform records ──────────────────────────────────
    download_records(valid_ids, db)

    # ── Step 5: Load signals ───────────────────────────────────────────────
    print(f"\n[Step 5] Loading {len(valid_ids)} ECG signals (Lead II)...")
    signals = []
    labels = []
    failed = 0

    for i, ecg_id in enumerate(valid_ids):
        if (i + 1) % 200 == 0:
            print(f"  Progress: {i+1}/{len(valid_ids)}")

        sig = load_signal(ecg_id, db)
        if sig is not None:
            signals.append(sig)
            labels.append(labels_raw[ecg_id])
        else:
            failed += 1

    print(f"  Loaded: {len(signals)}, Failed: {failed}")

    signals = np.array(signals, dtype=np.float32)   # (N, SIGNAL_LENGTH)
    labels = np.array(labels, dtype=np.int64)        # (N,)

    # ── Step 6: Train / validation split ───────────────────────────────────
    print(f"\n[Step 6] Splitting dataset ({(1-val_fraction)*100:.0f}% train, "
          f"{val_fraction*100:.0f}% val)...")
    X_train, X_val, y_train, y_val = train_test_split(
        signals, labels,
        test_size=val_fraction,
        random_state=seed,
        stratify=labels,
    )
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}")

    # ── Step 7: Save .npy files ────────────────────────────────────────────
    print(f"\n[Step 7] Saving .npy arrays to {OUTPUT_DIR}...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_DIR / "ptbxl_train_signals.npy", X_train)
    np.save(OUTPUT_DIR / "ptbxl_train_labels.npy", y_train)
    np.save(OUTPUT_DIR / "ptbxl_val_signals.npy", X_val)
    np.save(OUTPUT_DIR / "ptbxl_val_labels.npy", y_val)

    print("  Saved:")
    print(f"    ptbxl_train_signals.npy  shape: {X_train.shape}")
    print(f"    ptbxl_train_labels.npy   shape: {y_train.shape}")
    print(f"    ptbxl_val_signals.npy    shape: {X_val.shape}")
    print(f"    ptbxl_val_labels.npy     shape: {y_val.shape}")

    # ── Print final class distribution ─────────────────────────────────────
    print("\nFinal class distribution (train):")
    train_dist = Counter(y_train.tolist())
    for cls_idx in range(len(CLASS_NAMES)):
        count = train_dist.get(cls_idx, 0)
        pct = count / len(y_train) * 100
        print(f"  {CLASS_NAMES[cls_idx]:6s}: {count:5d} ({pct:.1f}%)")

    print("\nDataset preparation complete!")
    return {
        "train_signals": str(OUTPUT_DIR / "ptbxl_train_signals.npy"),
        "train_labels":  str(OUTPUT_DIR / "ptbxl_train_labels.npy"),
        "val_signals":   str(OUTPUT_DIR / "ptbxl_val_signals.npy"),
        "val_labels":    str(OUTPUT_DIR / "ptbxl_val_labels.npy"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare PTB-XL dataset for Stage IV training")
    parser.add_argument(
        "--max-records", type=int, default=5000,
        help="Max number of records to use. -1 = full dataset (~21k). Default: 5000"
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.2,
        help="Fraction of data to use for validation. Default: 0.2"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility. Default: 42"
    )
    args = parser.parse_args()

    paths = prepare_dataset(
        max_records=args.max_records,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    print("\n" + "=" * 60)
    print("Next step: Train the classifier by running:")
    print()
    print("  python train_classifier.py")
    print()
    print("Or call directly in Python:")
    print("  from stage4_classify import train_model")
    print(f"  train_model('{paths['train_signals']}',")
    print(f"              '{paths['train_labels']}',")
    print(f"              '{paths['val_signals']}',")
    print(f"              '{paths['val_labels']}')")
    print("=" * 60)
