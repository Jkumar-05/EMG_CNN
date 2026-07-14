"""
EMG CNN trainer for separate training and testing folders (train/test only).

Expected data layout:
    dataset/
        train/
            hand_open/
                trial_1.csv
                trial_2.txt
            hand_close/
                trial_1.csv
                trial_2.txt
        test/
            hand_open/
                trial_7.txt
            hand_close/
                trial_8.txt

Each gesture folder name becomes the class label.

Supported file formats:
    .csv, .txt, .tsv

This version uses only:
    1. Training folder
    2. Testing folder

There is no validation split, no validation accuracy, no validation-based
learning-rate scheduling, and no early stopping. The model trains on the full
training folder, then evaluates once on the testing folder.

Examples:
    python emg_cnn_paper_architecture_v2.py \
        --train-dir ./dataset/train \
        --test-dir ./dataset/test \
        --epochs 200

    python emg_cnn_paper_architecture_v2.py \
        --predict-csv ./dataset/test/hand_open/trial_7.txt \
        --model-path emg_cnn_model.pth
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


TIME_COLUMNS = {
    "time",
    "timestamp",
    "wall_time",
    "elapsed_s",
    "elapsed",
    "seconds",
    "sec",
    "ms",
    "sample",
    "index",
}

SUPPORTED_EXTENSIONS = {".csv", ".txt", ".tsv"}


# ---------------------------------------------------------------------------
# File loading (unchanged from the original script)
# ---------------------------------------------------------------------------

def read_signal_file(file_path: Path) -> pd.DataFrame:
    """Read CSV, TSV, or TXT EMG data into a DataFrame."""
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(file_path)

    if suffix in {".txt", ".tsv"}:
        return pd.read_csv(file_path, sep=None, engine="python")

    raise ValueError(f"Unsupported file type: {file_path}. Use .csv, .txt, or .tsv")


def choose_emg_columns(df: pd.DataFrame, channel_columns: Optional[List[str]] = None) -> List[str]:
    """Choose EMG signal columns from a file."""
    if channel_columns:
        missing = [col for col in channel_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing requested channel columns: {missing}")
        return channel_columns

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    emg_cols = [col for col in numeric_cols if col.lower() not in TIME_COLUMNS]

    if not emg_cols:
        raise ValueError(
            "No EMG columns found. Make sure your file has numeric signal columns, "
            "for example: value, ch1, ch2, ch3, ..."
        )

    return emg_cols


def infer_sampling_rate(df: pd.DataFrame) -> Optional[float]:
    """Infer sampling rate from elapsed_s or timestamp if available."""
    time_values = None

    if "elapsed_s" in df.columns:
        time_values = pd.to_numeric(df["elapsed_s"], errors="coerce").dropna().to_numpy(dtype=float)
    elif "timestamp" in df.columns:
        timedeltas = pd.to_timedelta(df["timestamp"], errors="coerce").dropna()
        if len(timedeltas) > 0:
            time_values = timedeltas.dt.total_seconds().to_numpy(dtype=float)

    if time_values is None or len(time_values) < 3:
        return None

    diffs = np.diff(time_values)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return None

    median_dt = float(np.median(diffs))
    if median_dt <= 0:
        return None

    return 1.0 / median_dt


def load_file_as_channels(file_path: Path, channel_columns: Optional[List[str]] = None) -> np.ndarray:
    """Load one CSV/TXT/TSV and return shape [channels, samples]."""
    df = read_signal_file(file_path)
    emg_cols = choose_emg_columns(df, channel_columns)

    data = df[emg_cols].apply(pd.to_numeric, errors="coerce")
    data = data.interpolate(limit_direction="both").fillna(0.0)

    return data.to_numpy(dtype=np.float32).T


def segment_signal(
    signal: np.ndarray,
    window_samples: int,
    stride_samples: int,
) -> np.ndarray:
    """Convert full signal [channels, samples] into raw (un-normalized) windows.

    CHANGED: normalization used to happen inside this function, per window.
    It has been pulled out entirely so normalization can be applied later,
    consistently, using statistics computed from the training set. See
    compute_channel_stats() / apply_channel_stats() below.
    """
    _, total_samples = signal.shape

    if total_samples < window_samples:
        raise ValueError(
            f"File has only {total_samples} samples, but window requires {window_samples} samples."
        )

    windows = []
    for start in range(0, total_samples - window_samples + 1, stride_samples):
        window = signal[:, start : start + window_samples].copy()
        windows.append(window)

    return np.stack(windows, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# CHANGED: normalization computed once from training data, reused everywhere
# ---------------------------------------------------------------------------

def compute_channel_stats(windows: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std computed across every training window and sample.

    This preserves relative amplitude differences between gestures (a light
    contraction vs. a hard contraction still look different after this),
    unlike per-window z-scoring which erases that information.
    """
    arr = np.stack(windows, axis=0)  # [N, channels, samples]
    mean = arr.mean(axis=(0, 2))
    std = arr.std(axis=(0, 2))
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_channel_stats(windows: List[np.ndarray], mean: np.ndarray, std: np.ndarray) -> None:
    """In-place normalization of a list of windows using fixed stats."""
    mean_r = mean.reshape(-1, 1)
    std_r = std.reshape(-1, 1)
    for i in range(len(windows)):
        windows[i] = (windows[i] - mean_r) / std_r


def normalize_window_per_window(window: np.ndarray) -> np.ndarray:
    """Old-style per-window normalization, kept as an opt-in choice only."""
    mean = window.mean(axis=1, keepdims=True)
    std = window.std(axis=1, keepdims=True)
    std[std < 1e-8] = 1.0
    return (window - mean) / std


# ---------------------------------------------------------------------------
# CHANGED: light EMG-appropriate data augmentation, training split only
# ---------------------------------------------------------------------------

def apply_augmentation(
    window: np.ndarray,
    noise_std_frac: float = 0.03,
    scale_range: Tuple[float, float] = (0.9, 1.1),
    max_shift_frac: float = 0.1,
) -> np.ndarray:
    """Randomly perturb a window to help the model generalize.

    - Per-channel amplitude scaling: simulates electrode placement / contraction
      intensity variation.
    - Additive Gaussian noise scaled to each channel's own std: simulates
      sensor/electrical noise without needing an absolute amplitude reference.
    - Small circular time shift: simulates imperfect windowing/onset detection.
    """
    out = window.copy()

    if np.random.rand() < 0.5:
        scale = np.random.uniform(scale_range[0], scale_range[1], size=(out.shape[0], 1)).astype(np.float32)
        out = out * scale

    if np.random.rand() < 0.5:
        std = out.std(axis=1, keepdims=True)
        noise = np.random.normal(0.0, noise_std_frac, size=out.shape).astype(np.float32) * std
        out = out + noise

    if np.random.rand() < 0.5:
        max_shift = max(1, int(out.shape[1] * max_shift_frac))
        shift = np.random.randint(-max_shift, max_shift + 1)
        out = np.roll(out, shift, axis=1)

    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EMGWindowDataset:
    """Builds raw (un-normalized) windows from gesture folders.

    CHANGED: this class no longer normalizes anything itself. It just loads
    files, segments them into windows, and stores raw values + labels.
    Normalization is applied afterwards by the training pipeline so that the
    exact same train-derived statistics can be reused for test,
    and single-file prediction.
    """

    def __init__(
        self,
        data_dir: Path,
        window_ms: float = 150.0,
        stride_ms: float = 50.0,
        sampling_rate: Optional[float] = None,
        channel_columns: Optional[List[str]] = None,
        label_to_idx: Optional[Dict[str, int]] = None,
        expected_num_channels: Optional[int] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.window_ms = window_ms
        self.stride_ms = stride_ms
        self.sampling_rate = sampling_rate
        self.channel_columns = channel_columns
        self.expected_num_channels = expected_num_channels

        self.windows: List[np.ndarray] = []
        self.labels: List[int] = []
        self.label_to_idx: Dict[str, int] = label_to_idx.copy() if label_to_idx else {}
        self.idx_to_label: Dict[int, str] = {}
        self.num_channels: Optional[int] = expected_num_channels
        self.final_sampling_rate: Optional[float] = None

        self._build()

    def _find_signal_files(self, folder: Path) -> List[Path]:
        return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS])

    def _build(self) -> None:
        gesture_dirs = sorted([p for p in self.data_dir.iterdir() if p.is_dir()])
        if not gesture_dirs:
            raise ValueError(f"No gesture folders found in: {self.data_dir}")

        if self.sampling_rate is None:
            for folder in gesture_dirs:
                signal_files = self._find_signal_files(folder)
                if signal_files:
                    df = read_signal_file(signal_files[0])
                    inferred = infer_sampling_rate(df)
                    if inferred:
                        self.sampling_rate = inferred
                        break

        if self.sampling_rate is None:
            raise ValueError(
                "Could not infer sampling rate. Pass it manually with --sampling-rate, "
                "for example --sampling-rate 1000"
            )

        self.final_sampling_rate = float(self.sampling_rate)
        window_samples = int(round(self.final_sampling_rate * self.window_ms / 1000.0))
        stride_samples = int(round(self.final_sampling_rate * self.stride_ms / 1000.0))

        if window_samples <= 0 or stride_samples <= 0:
            raise ValueError("Window and stride must be positive.")

        if not self.label_to_idx:
            self.label_to_idx = {folder.name: idx for idx, folder in enumerate(gesture_dirs)}

        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        for folder in gesture_dirs:
            label = folder.name
            if label not in self.label_to_idx:
                raise ValueError(
                    f"Testing folder has class '{label}', but that class was not in the training folder."
                )

            label_idx = self.label_to_idx[label]
            signal_files = self._find_signal_files(folder)
            if not signal_files:
                print(f"Warning: no supported signal files found in {folder}")
                continue

            for file_path in signal_files:
                signal = load_file_as_channels(file_path, self.channel_columns)

                if self.num_channels is None:
                    self.num_channels = signal.shape[0]
                elif signal.shape[0] != self.num_channels:
                    raise ValueError(
                        f"Channel mismatch in {file_path}. Expected {self.num_channels}, "
                        f"got {signal.shape[0]}."
                    )

                windows = segment_signal(signal, window_samples, stride_samples)
                self.windows.extend(list(windows))
                self.labels.extend([label_idx] * len(windows))

        if not self.windows:
            raise ValueError(f"No windows were created from: {self.data_dir}")

        self.window_samples = window_samples
        self.stride_samples = stride_samples


def stratified_split_indices(labels: List[int], val_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
    """CHANGED: new function. Stratified train/val split so every class is
    represented proportionally in both splits, using only the training folder
    (the held-out test folder is never touched by this)."""
    rng = np.random.RandomState(seed)
    labels_arr = np.array(labels)
    train_idx: List[int] = []
    val_idx: List[int] = []

    for lbl in np.unique(labels_arr):
        idxs = np.where(labels_arr == lbl)[0].copy()
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_fraction))) if val_fraction > 0 else 0
        val_idx.extend(idxs[:n_val].tolist())
        train_idx.extend(idxs[n_val:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """CHANGED: new function. Inverse-frequency class weights for the loss,
    in case some gestures have more windows than others."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


class WindowSubset(Dataset):
    """CHANGED: new class. Wraps an EMGWindowDataset + a list of indices, and
    optionally applies training-only data augmentation. Using a thin wrapper
    like this means EMGWindowDataset itself stays a simple, dumb container of
    windows/labels, and augmentation/normalization stay easy to reason about."""

    def __init__(self, parent: EMGWindowDataset, indices: List[int], augment: bool = False) -> None:
        self.parent = parent
        self.indices = indices
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        real_idx = self.indices[i]
        window = self.parent.windows[real_idx]
        if self.augment:
            window = apply_augmentation(window)
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # [1, channels, samples]
        y = torch.tensor(self.parent.labels[real_idx], dtype=torch.long)
        return x, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class EMGCNN(nn.Module):
    """CNN matching the paper architecture as closely as possible.

    CHANGED: added a dropout layer before the fully connected classifier
    (paper doesn't mention dropout, but with a modest EMG dataset it's one of
    the cheapest ways to reduce overfitting; set --dropout 0 to disable it and
    match the paper exactly).
    """

    def __init__(self, num_emg_channels: int, window_samples: int, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()

        self.uses_exact_paper_pooling = num_emg_channels >= 6

        if self.uses_exact_paper_pooling:
            pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        else:
            pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
            pool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

        self.conv1_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            pool1,
        )

        self.conv2_block = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            pool2,
        )

        self.conv3_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.features = nn.Sequential(
            self.conv1_block,
            self.conv2_block,
            self.conv3_block,
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_emg_channels, window_samples)
            flattened_size = self.features(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(flattened_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.classifier(x)
        return logits


def print_model_architecture(model: EMGCNN, num_emg_channels: int, window_samples: int) -> None:
    with torch.no_grad():
        x = torch.zeros(1, 1, num_emg_channels, window_samples)
        after_conv1 = model.conv1_block(x)
        after_conv2 = model.conv2_block(after_conv1)
        after_conv3 = model.conv3_block(after_conv2)

    print("CNN architecture")
    print("  Input:", tuple(x.shape), "= [batch, 1, EMG channels, samples]")
    print("  Conv1 block: Conv2d(1 -> 16, 3x3) + BatchNorm + ReLU + Pool")
    print("    Output:", tuple(after_conv1.shape))
    print("  Conv2 block: Conv2d(16 -> 64, 3x3) + BatchNorm + ReLU + Pool")
    print("    Output:", tuple(after_conv2.shape))
    print("  Conv3 block: Conv2d(64 -> 32, 3x3) + BatchNorm + ReLU")
    print("    Output:", tuple(after_conv3.shape))
    print("  Dropout -> Fully connected layer -> class logits")
    if model.uses_exact_paper_pooling:
        print("  Pooling mode: exact paper pooling, 2x2 then 3x3")
    else:
        print("  Pooling mode: time-axis fallback because data has fewer than 6 EMG channels")
    print()



# ---------------------------------------------------------------------------
# Model size, FLOPs, and runtime utilities
# ---------------------------------------------------------------------------

def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable model parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: nn.Module) -> int:
    """Return the total number of model parameters."""
    return sum(p.numel() for p in model.parameters())


def estimate_model_size_mb(model: nn.Module) -> float:
    """Approximate parameter memory assuming float32 weights."""
    total_params = count_total_parameters(model)
    bytes_per_param = 4
    return total_params * bytes_per_param / (1024 ** 2)


def estimate_flops(model: nn.Module, input_shape: Tuple[int, int, int, int], device: torch.device) -> int:
    """Estimate FLOPs for one forward pass.

    Counts Conv2d and Linear layers.
    Uses the common convention:
        1 multiply + 1 add = 2 FLOPs

    This is an approximation, but it is useful for comparing model sizes.
    """
    flops = 0
    hooks = []

    def conv_hook(module: nn.Conv2d, inputs: Tuple[torch.Tensor], output: torch.Tensor) -> None:
        nonlocal flops
        batch_size = output.shape[0]
        out_channels = output.shape[1]
        out_h = output.shape[2]
        out_w = output.shape[3]

        kernel_h, kernel_w = module.kernel_size
        in_channels = module.in_channels
        groups = module.groups

        # Operations per output value:
        # kernel_h * kernel_w * input_channels/groups multiplications
        # and roughly the same number of additions.
        ops_per_output = kernel_h * kernel_w * (in_channels // groups) * 2

        if module.bias is not None:
            ops_per_output += 1

        flops += batch_size * out_channels * out_h * out_w * ops_per_output

    def linear_hook(module: nn.Linear, inputs: Tuple[torch.Tensor], output: torch.Tensor) -> None:
        nonlocal flops
        batch_size = output.shape[0]
        ops_per_output = module.in_features * 2
        if module.bias is not None:
            ops_per_output += 1
        flops += batch_size * module.out_features * ops_per_output

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(input_shape, device=device)
        _ = model(dummy)

    for hook in hooks:
        hook.remove()

    return int(flops)


def format_large_number(value: float) -> str:
    """Format large numbers with K/M/B suffixes."""
    if value >= 1e9:
        return f"{value / 1e9:.3f}B"
    if value >= 1e6:
        return f"{value / 1e6:.3f}M"
    if value >= 1e3:
        return f"{value / 1e3:.3f}K"
    return f"{value:.0f}"

# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return total_loss / total, correct / total


def save_checkpoint(
    model: nn.Module,
    path: Path,
    dataset: EMGWindowDataset,
    args: argparse.Namespace,
    channel_mean: Optional[np.ndarray],
    channel_std: Optional[np.ndarray],
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "num_emg_channels": dataset.num_channels,
        "window_samples": dataset.window_samples,
        "num_classes": len(dataset.label_to_idx),
        "label_to_idx": dataset.label_to_idx,
        "idx_to_label": dataset.idx_to_label,
        "sampling_rate": dataset.final_sampling_rate,
        "window_ms": args.window_ms,
        "stride_ms": args.stride_ms,
        "channel_columns": args.channels,
        "normalize_mode": args.normalize_mode,
        "channel_mean": channel_mean,
        "channel_std": channel_std,
        "dropout": args.dropout,
        "architecture": "paper_cnn_16_64_32_bn_relu_pool_fc_v2",
    }
    torch.save(checkpoint, path)


def run_training_and_testing(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_dataset = EMGWindowDataset(
        data_dir=Path(args.train_dir),
        window_ms=args.window_ms,
        stride_ms=args.stride_ms,
        sampling_rate=args.sampling_rate,
        channel_columns=args.channels,
    )

    test_dataset = EMGWindowDataset(
        data_dir=Path(args.test_dir),
        window_ms=args.window_ms,
        stride_ms=args.stride_ms,
        sampling_rate=train_dataset.final_sampling_rate,
        channel_columns=args.channels,
        label_to_idx=train_dataset.label_to_idx,
        expected_num_channels=train_dataset.num_channels,
    )

    # Train/test only:
    # - All windows from the training folder are used for training.
    # - All windows from the testing folder are used only after training finishes.
    train_idx = list(range(len(train_dataset.windows)))
    test_idx = list(range(len(test_dataset.windows)))

    channel_mean: Optional[np.ndarray] = None
    channel_std: Optional[np.ndarray] = None

    # Normalization statistics are computed from TRAINING data only,
    # then applied to both training and testing data.
    if args.normalize_mode == "global":
        channel_mean, channel_std = compute_channel_stats(train_dataset.windows)
        apply_channel_stats(train_dataset.windows, channel_mean, channel_std)
        apply_channel_stats(test_dataset.windows, channel_mean, channel_std)
    elif args.normalize_mode == "per_window":
        train_dataset.windows = [normalize_window_per_window(w) for w in train_dataset.windows]
        test_dataset.windows = [normalize_window_per_window(w) for w in test_dataset.windows]
    # "none" leaves raw values as-is.

    train_subset = WindowSubset(train_dataset, train_idx, augment=args.augment)
    test_subset = WindowSubset(test_dataset, test_idx, augment=False)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)

    model = EMGCNN(
        num_emg_channels=train_dataset.num_channels,
        window_samples=train_dataset.window_samples,
        num_classes=len(train_dataset.label_to_idx),
        dropout=args.dropout,
    ).to(device)

    print_model_architecture(model, train_dataset.num_channels, train_dataset.window_samples)

    total_params = count_total_parameters(model)
    trainable_params = count_trainable_parameters(model)
    model_size_mb = estimate_model_size_mb(model)
    flops_per_forward = estimate_flops(
        model,
        input_shape=(1, 1, train_dataset.num_channels, train_dataset.window_samples),
        device=device,
    )

    print("Model size")
    print(f"  Total parameters: {total_params:,} ({format_large_number(total_params)})")
    print(f"  Trainable parameters: {trainable_params:,} ({format_large_number(trainable_params)})")
    print(f"  Approx. parameter memory: {model_size_mb:.3f} MB")
    print(f"  Approx. FLOPs per single-window forward pass: {flops_per_forward:,} ({format_large_number(flops_per_forward)})")
    print()

    class_weights = compute_class_weights(train_dataset.labels, len(train_dataset.label_to_idx)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    print("Dataset summary")
    print(f"  Classes: {train_dataset.label_to_idx}")
    print(f"  Training windows: {len(train_dataset.windows)}")
    print(f"  Testing windows: {len(test_dataset.windows)}")
    print(f"  EMG channels: {train_dataset.num_channels}")
    print(f"  Sampling rate: {train_dataset.final_sampling_rate:.2f} Hz")
    print(f"  Window samples: {train_dataset.window_samples}")
    print(f"  Stride samples: {train_dataset.stride_samples}")
    print(f"  Normalization: {args.normalize_mode}")
    print(f"  Augmentation on training data: {args.augment}")
    print(f"  Device: {device}")
    print()

    training_start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)

        if epoch % args.print_every == 0 or epoch == 1 or epoch == args.epochs:
            current_lr = optimizer.param_groups[0]["lr"]
            epoch_runtime = time.perf_counter() - epoch_start_time
            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"lr {current_lr:.2e} | epoch time {epoch_runtime:.2f}s"
            )

    total_training_time = time.perf_counter() - training_start_time

    # Testing happens once after all training is complete.
    testing_start_time = time.perf_counter()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    total_testing_time = time.perf_counter() - testing_start_time

    test_windows = len(test_dataset.windows)
    avg_test_time_per_window_ms = (total_testing_time / max(1, test_windows)) * 1000.0

    save_checkpoint(model, Path(args.model_path), train_dataset, args, channel_mean, channel_std)

    print()
    print("Final test results")
    print(f"  Test loss: {test_loss:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")
    print()
    print("Runtime")
    print(f"  Total training time: {total_training_time:.2f} seconds")
    print(f"  Total testing time: {total_testing_time:.2f} seconds")
    print(f"  Average testing time per window: {avg_test_time_per_window_ms:.4f} ms")
    print(f"Saved model to: {args.model_path}")


def test_dataset_as_subset(dataset: EMGWindowDataset) -> WindowSubset:
    """Small helper so the (already normalized) test dataset can go through
    the same WindowSubset/DataLoader path as train/val, with augmentation off."""
    return WindowSubset(dataset, list(range(len(dataset.windows))), augment=False)


def predict_file(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model_path, map_location=device)

    model = EMGCNN(
        num_emg_channels=checkpoint["num_emg_channels"],
        window_samples=checkpoint["window_samples"],
        num_classes=checkpoint["num_classes"],
        dropout=checkpoint.get("dropout", 0.0),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    idx_to_label = {int(k): v for k, v in checkpoint["idx_to_label"].items()}
    signal = load_file_as_channels(Path(args.predict_csv), checkpoint.get("channel_columns"))

    windows = segment_signal(
        signal,
        window_samples=checkpoint["window_samples"],
        stride_samples=max(1, int(round(checkpoint["sampling_rate"] * checkpoint["stride_ms"] / 1000.0))),
    )
    windows = list(windows)

    # CHANGED: apply the exact same normalization used at training time.
    normalize_mode = checkpoint.get("normalize_mode", "per_window")
    if normalize_mode == "global" and checkpoint.get("channel_mean") is not None:
        apply_channel_stats(windows, checkpoint["channel_mean"], checkpoint["channel_std"])
    elif normalize_mode == "per_window":
        windows = [normalize_window_per_window(w) for w in windows]

    x = torch.tensor(np.stack(windows, axis=0), dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1).cpu().numpy()

    values, counts = np.unique(preds, return_counts=True)
    majority_idx = int(values[np.argmax(counts)])
    majority_label = idx_to_label[majority_idx]
    confidence = float(counts.max() / counts.sum())

    print(f"Predicted gesture: {majority_label}")
    print(f"Majority-vote confidence across windows: {confidence:.3f}")

    print("\nWindow vote counts:")
    for idx, count in zip(values, counts):
        print(f"  {idx_to_label[int(idx)]}: {int(count)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and test an EMG CNN gesture classifier.")

    parser.add_argument("--train-dir", type=str, default=None, help="Training folder containing gesture subfolders.")
    parser.add_argument("--test-dir", type=str, default=None, help="Testing folder containing gesture subfolders.")
    parser.add_argument("--predict-csv", type=str, default=None, help="Run prediction on a single CSV/TXT/TSV file.")
    parser.add_argument("--model-path", type=str, default="emg_cnn_model.pth", help="Path to save/load model.")

    parser.add_argument("--sampling-rate", type=float, default=None, help="Sampling rate in Hz. If omitted, tries elapsed_s or timestamp.")
    parser.add_argument("--window-ms", type=float, default=150.0, help="Window length in milliseconds.")
    parser.add_argument("--stride-ms", type=float, default=50.0, help="Stride between windows in milliseconds.")
    parser.add_argument("--channels", nargs="+", default=None, help="Specific EMG columns to use, e.g. --channels ch1 ch2 ch3")

    # CHANGED: normalization is now a mode, default "global" (fixes amplitude-erasing bug).
    parser.add_argument(
        "--normalize-mode",
        type=str,
        default="global",
        choices=["global", "per_window", "none"],
        help="How to scale windows. 'global' (recommended) uses train-set stats for train/test/inference. "
        "'per_window' matches the old behavior (erases amplitude info). 'none' uses raw values.",
    )

    # Regularization and augmentation options.
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout before the final linear layer. Set 0 to match the paper exactly.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 regularization strength for SGD.")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="Label smoothing for the loss. Set 0 to disable.")
    parser.add_argument("--augment", action="store_true", default=True, help="Apply light data augmentation to the training split.")
    parser.add_argument("--no-augment", dest="augment", action="store_false", help="Disable training-split data augmentation.")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-every", type=int, default=10, help="Print training metrics every N epochs.")

    args = parser.parse_args()

    if args.predict_csv is None and (args.train_dir is None or args.test_dir is None):
        parser.error("Provide --train-dir and --test-dir for training/testing, or --predict-csv for prediction.")

    return args


if __name__ == "__main__":
    args = parse_args()

    if args.predict_csv:
        predict_file(args)
    else:
        run_training_and_testing(args)
