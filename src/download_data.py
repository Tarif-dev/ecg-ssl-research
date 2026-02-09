"""Download ECG datasets from PhysioNet and extract beats/segments."""

import os
import numpy as np
import wfdb


# ============================================================================
# MIT-BIH Record IDs (48 records total)
# ============================================================================
MITBIH_RECORDS = [
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    121,
    122,
    123,
    124,
    200,
    201,
    202,
    203,
    205,
    207,
    208,
    209,
    210,
    212,
    213,
    214,
    215,
    217,
    219,
    220,
    221,
    222,
    223,
    228,
    230,
    231,
    232,
    233,
    234,
]

# Binary classification: Normal (0) vs Arrhythmia (1)
BEAT_LABELS_BINARY = {
    "N": 0,
    "L": 0,
    "R": 0,
    "e": 0,
    "j": 0,  # Normal
    "A": 1,
    "a": 1,
    "J": 1,
    "S": 1,  # Supraventricular
    "V": 1,
    "E": 1,  # Ventricular
    "F": 1,  # Fusion
    "/": 1,
    "f": 1,
    "Q": 1,  # Other abnormal
}

# 5-class AAMI standard
BEAT_LABELS_5CLASS = {
    "N": 0,
    "L": 0,
    "R": 0,
    "e": 0,
    "j": 0,  # Normal (N)
    "A": 1,
    "a": 1,
    "J": 1,
    "S": 1,  # Supraventricular (S)
    "V": 2,
    "E": 2,  # Ventricular (V)
    "F": 3,  # Fusion (F)
    "/": 4,
    "f": 4,
    "Q": 4,  # Unknown (Q)
}


# ============================================================================
# Download Functions
# ============================================================================


def download_mitbih(save_dir):
    """
    Download MIT-BIH Arrhythmia Database from PhysioNet.

    Args:
        save_dir: str, directory to save the database files.
                  Downloads 48 records (~115MB total).
    Returns:
        int, number of records downloaded.
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"Downloading MIT-BIH Arrhythmia Database to {save_dir}...")
    print("This may take a few minutes depending on your connection.\n")
    wfdb.dl_database("mitdb", dl_dir=save_dir)
    downloaded = [f for f in os.listdir(save_dir) if f.endswith(".dat")]
    print(f"\n✓ Download complete! {len(downloaded)}/48 records saved to {save_dir}")
    return len(downloaded)


def download_ptbxl(save_dir):
    """
    Download PTB-XL Database from PhysioNet.

    WARNING: This is ~22GB — only download if you need it.
    For initial development, MIT-BIH alone is sufficient.
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"Downloading PTB-XL Database to {save_dir}...")
    print("⚠ WARNING: This is ~22GB and will take a long time!")
    wfdb.dl_database("ptb-xl", dl_dir=save_dir)
    print(f"✓ Download complete! Files saved to {save_dir}")


# ============================================================================
# Data Loading & Extraction
# ============================================================================


def load_record(record_path, channel=0):
    """
    Load a single MIT-BIH record.

    Args:
        record_path: str, path to record without extension (e.g., 'data/mitdb/100')
        channel: int, 0=Lead II (MLII), 1=Lead V1/V5

    Returns:
        signal: np.ndarray [num_samples], float32
        annotation: wfdb Annotation object
        fs: int, sampling frequency (360 Hz)
    """
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, "atr")
    signal = record.p_signal[:, channel].astype(np.float32)
    return signal, annotation, record.fs


def extract_beats(
    signal, annotation, fs, window_before=0.3, window_after=0.5, label_map=None
):
    """
    Extract individual heartbeats centered on R-peaks.

    Args:
        signal: np.ndarray, full recording signal
        annotation: wfdb Annotation object
        fs: int, sampling frequency
        window_before: float, seconds before R-peak
        window_after: float, seconds after R-peak
        label_map: dict, annotation symbol → class label (default: binary)

    Returns:
        beats: np.ndarray [num_beats, beat_length], float32
        labels: np.ndarray [num_beats], int64
    """
    if label_map is None:
        label_map = BEAT_LABELS_BINARY

    samples_before = int(window_before * fs)
    samples_after = int(window_after * fs)
    beat_length = samples_before + samples_after

    beats, labels = [], []
    for sample, symbol in zip(annotation.sample, annotation.symbol):
        if symbol not in label_map:
            continue
        start = sample - samples_before
        end = sample + samples_after
        if start < 0 or end > len(signal):
            continue
        beat = signal[start:end]
        if len(beat) == beat_length:
            beats.append(beat)
            labels.append(label_map[symbol])

    return np.array(beats, dtype=np.float32), np.array(labels, dtype=np.int64)


def extract_segments(signal, fs, segment_duration=10, overlap=0.75):
    """
    Extract overlapping fixed-length segments (for self-supervised pre-training).

    Args:
        signal: np.ndarray, full recording
        fs: int, sampling frequency
        segment_duration: int, seconds per segment
        overlap: float, overlap fraction between segments

    Returns:
        segments: np.ndarray [num_segments, segment_length], float32
    """
    segment_length = int(segment_duration * fs)
    step = int(segment_length * (1 - overlap))
    segments = []

    for start in range(0, len(signal) - segment_length + 1, step):
        segments.append(signal[start : start + segment_length])

    if len(segments) == 0:
        return np.array([], dtype=np.float32).reshape(0, segment_length)
    return np.array(segments, dtype=np.float32)


def verify_download(data_dir):
    """Check that MIT-BIH data downloaded correctly."""
    dat_files = [f for f in os.listdir(data_dir) if f.endswith(".dat")]
    hea_files = [f for f in os.listdir(data_dir) if f.endswith(".hea")]
    atr_files = [f for f in os.listdir(data_dir) if f.endswith(".atr")]
    print(f"Found: {len(dat_files)} .dat, {len(hea_files)} .hea, {len(atr_files)} .atr")
    if len(dat_files) >= len(MITBIH_RECORDS):
        print("✓ Download verified successfully!")
        return True
    else:
        print("✗ Download incomplete — please re-run download.")
        return False
