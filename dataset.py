"""
Dataset handling: scans data_dir/real and data_dir/fake, extracts (and caches)
MFCC features, and produces stratified train/dev/eval splits.
"""

import os
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from feature_extraction import extract_mfcc

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


def _list_audio_files(folder):
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(AUDIO_EXTS)
    )


def build_file_list(data_dir):
    """
    Returns list of (filepath, label) with label 0 = real, 1 = fake.
    Expects data_dir/real/*.wav and data_dir/fake/*.wav
    """
    real_dir = os.path.join(data_dir, "real")
    fake_dir = os.path.join(data_dir, "fake")
    if not os.path.isdir(real_dir) or not os.path.isdir(fake_dir):
        raise FileNotFoundError(
            f"Expected '{real_dir}' and '{fake_dir}' to exist. "
            "See README.md for the expected data layout."
        )

    files = [(p, 0) for p in _list_audio_files(real_dir)]
    files += [(p, 1) for p in _list_audio_files(fake_dir)]

    if len(files) == 0:
        raise RuntimeError(f"No audio files found under {data_dir} (real/ + fake/).")

    return files


def split_dataset(files, dev_size=0.15, eval_size=0.15, seed=42):
    """Stratified split into train / dev / eval."""
    labels = [lbl for _, lbl in files]
    train_files, temp_files = train_test_split(
        files, test_size=(dev_size + eval_size), stratify=labels, random_state=seed
    )
    temp_labels = [lbl for _, lbl in temp_files]
    relative_eval = eval_size / (dev_size + eval_size)
    dev_files, eval_files = train_test_split(
        temp_files, test_size=relative_eval, stratify=temp_labels, random_state=seed
    )
    return train_files, dev_files, eval_files


class VoiceDataset(Dataset):
    """
    Wraps a list of (filepath, label) pairs, extracting MFCC features on first
    access and caching them to disk (.npy) so repeat epochs / runs are fast.
    """

    def __init__(self, files, cache_dir="outputs/feature_cache"):
        self.files = files
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.files)

    def _cache_path(self, filepath):
        h = hashlib.md5(filepath.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{h}.npy")

    def __getitem__(self, idx):
        filepath, label = self.files[idx]
        cache_path = self._cache_path(filepath)

        if os.path.exists(cache_path):
            feat = np.load(cache_path)
        else:
            feat = extract_mfcc(filepath)
            np.save(cache_path, feat)

        return torch.from_numpy(feat), torch.tensor(label, dtype=torch.long)

    def precompute(self, desc="Extracting features"):
        """Force-extract and cache all features up front (with a progress bar)."""
        for i in tqdm(range(len(self)), desc=desc):
            _ = self[i]
