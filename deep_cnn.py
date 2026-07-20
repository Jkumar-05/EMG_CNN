#!/usr/bin/env python3
"""
Deep EMG CNN for hand-gesture recognition.

Key features
------------
- Separate training and testing directories
- Five convolutional modules
- Conv2D + BatchNorm + ReLU
- Residual depthwise-separable blocks
- Squeeze-and-Excitation attention
- Train-set normalization reused for test and prediction
- Final accuracy, precision, recall, F1-score, and confusion matrix
- Parameter count, estimated FLOPs, model size, and runtime
- Single-file prediction

Expected folder layout
----------------------
dataset/
├── Train/
│   ├── Hand_Open/
│   │   ├── trial_1.txt
│   │   └── ...
│   └── Hand_Close/
│       ├── trial_1.txt
│       └── ...
└── Test/
    ├── Hand_Open/
    │   ├── trial_7.txt
    │   └── ...
    └── Hand_Close/
        ├── trial_8.txt
        └── ...

Each file may be .csv, .txt, or .tsv and may include a timestamp column.
Numeric EMG columns such as ch1, ch2, ch3 are used as model inputs.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


SUPPORTED_EXTENSIONS = {".csv", ".txt", ".tsv"}

TIME_COLUMNS = {
    "time",
    "timestamp",
    "wall_time",
    "elapsed",
    "elapsed_s",
    "seconds",
    "sec",
    "ms",
    "sample",
    "sample_index",
    "index",
}


# ---------------------------------------------------------------------------
# Reproducibility and device
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# ---------------------------------------------------------------------------
# EMG file loading
# ---------------------------------------------------------------------------

def read_signal_file(file_path: Path) -> pd.DataFrame:
    """Read CSV, TXT, or TSV into a pandas DataFrame."""
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(file_path)

    if suffix in {".txt", ".tsv"}:
        return pd.read_csv(file_path, sep=None, engine="python")

    raise ValueError(
        f"Unsupported file type: {file_path}. "
        "Use .csv, .txt, or .tsv."
    )


def choose_emg_columns(
    dataframe: pd.DataFrame,
    requested_channels: Optional[List[str]] = None,
) -> List[str]:
    """Choose numeric EMG columns and ignore timestamp/index columns."""
    if requested_channels:
        missing = [
            column
            for column in requested_channels
            if column not in dataframe.columns
        ]

        if missing:
            raise ValueError(
                f"Missing requested channel columns: {missing}"
            )

        return requested_channels

    numeric_columns = (
        dataframe.select_dtypes(include=[np.number]).columns.tolist()
    )

    emg_columns = [
        column
        for column in numeric_columns
        if str(column).strip().lower() not in TIME_COLUMNS
    ]

    if not emg_columns:
        # Try converting string columns to numeric.
        converted = dataframe.apply(pd.to_numeric, errors="coerce")
        emg_columns = [
            column
            for column in converted.columns
            if converted[column].notna().any()
            and str(column).strip().lower() not in TIME_COLUMNS
        ]

    if not emg_columns:
        raise ValueError(
            "No numeric EMG channels were found. Expected columns such as "
            "ch1, ch2, ch3."
        )

    return [str(column) for column in emg_columns]


def infer_sampling_rate(dataframe: pd.DataFrame) -> Optional[float]:
    """Infer sampling rate from elapsed_s or timestamp when possible."""
    time_values: Optional[np.ndarray] = None

    if "elapsed_s" in dataframe.columns:
        time_values = (
            pd.to_numeric(dataframe["elapsed_s"], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )

    elif "timestamp" in dataframe.columns:
        timedeltas = pd.to_timedelta(
            dataframe["timestamp"],
            errors="coerce",
        ).dropna()

        if len(timedeltas) > 0:
            time_values = (
                timedeltas.dt.total_seconds().to_numpy(dtype=float)
            )

    if time_values is None or len(time_values) < 3:
        return None

    differences = np.diff(time_values)
    differences = differences[differences > 0]

    if len(differences) == 0:
        return None

    median_difference = float(np.median(differences))

    if median_difference <= 0:
        return None

    return 1.0 / median_difference


def load_file_as_channels(
    file_path: Path,
    channel_columns: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str], Optional[float]]:
    """
    Load one recording.

    Returns
    -------
    signal:
        Shape [channels, samples]
    channel_names:
        Names of EMG channels
    inferred_sampling_rate:
        Sampling rate inferred from timestamp data, if possible
    """
    dataframe = read_signal_file(file_path)
    emg_columns = choose_emg_columns(dataframe, channel_columns)

    numeric = dataframe[emg_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.interpolate(limit_direction="both").fillna(0.0)

    signal = numeric.to_numpy(dtype=np.float32).T
    sampling_rate = infer_sampling_rate(dataframe)

    return signal, emg_columns, sampling_rate


def segment_signal(
    signal: np.ndarray,
    window_samples: int,
    stride_samples: int,
) -> np.ndarray:
    """
    Convert signal [channels, samples] into windows
    [num_windows, channels, window_samples].
    """
    _, total_samples = signal.shape

    if total_samples < window_samples:
        raise ValueError(
            f"File contains {total_samples} samples, but window size is "
            f"{window_samples}."
        )

    windows = [
        signal[:, start : start + window_samples].copy()
        for start in range(
            0,
            total_samples - window_samples + 1,
            stride_samples,
        )
    ]

    return np.stack(windows).astype(np.float32)


# ---------------------------------------------------------------------------
# Normalization and augmentation
# ---------------------------------------------------------------------------

def compute_channel_stats(
    windows: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean and std from training windows only."""
    stacked = np.stack(windows, axis=0)
    mean = stacked.mean(axis=(0, 2))
    std = stacked.std(axis=(0, 2))
    std[std < 1e-8] = 1.0

    return mean.astype(np.float32), std.astype(np.float32)


def normalize_windows(
    windows: Sequence[np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
) -> List[np.ndarray]:
    """Normalize windows with fixed training statistics."""
    reshaped_mean = mean.reshape(-1, 1)
    reshaped_std = std.reshape(-1, 1)

    return [
        ((window - reshaped_mean) / reshaped_std).astype(np.float32)
        for window in windows
    ]


def normalize_per_window(window: np.ndarray) -> np.ndarray:
    """Optional per-window normalization."""
    mean = window.mean(axis=1, keepdims=True)
    std = window.std(axis=1, keepdims=True)
    std[std < 1e-8] = 1.0

    return ((window - mean) / std).astype(np.float32)


def augment_window(window: np.ndarray) -> np.ndarray:
    """Apply light EMG-appropriate augmentation."""
    output = window.copy()

    if np.random.rand() < 0.5:
        scale = np.random.uniform(
            0.9,
            1.1,
            size=(output.shape[0], 1),
        ).astype(np.float32)
        output *= scale

    if np.random.rand() < 0.5:
        channel_std = output.std(axis=1, keepdims=True)
        noise = np.random.normal(
            0.0,
            0.03,
            size=output.shape,
        ).astype(np.float32)
        output += noise * channel_std

    if np.random.rand() < 0.5:
        max_shift = max(1, int(output.shape[1] * 0.1))
        shift = np.random.randint(-max_shift, max_shift + 1)
        output = np.roll(output, shift, axis=1)

    return output.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EMGDataset:
    """Load gesture folders and create fixed-size EMG windows."""

    def __init__(
        self,
        data_dir: Path,
        window_ms: float,
        stride_ms: float,
        sampling_rate: Optional[float] = None,
        channel_columns: Optional[List[str]] = None,
        label_to_idx: Optional[Dict[str, int]] = None,
        expected_num_channels: Optional[int] = None,
        expected_class_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.window_ms = window_ms
        self.stride_ms = stride_ms
        self.sampling_rate = sampling_rate
        self.channel_columns = channel_columns

        self.windows: List[np.ndarray] = []
        self.labels: List[int] = []
        self.files: List[Path] = []

        self.label_to_idx = (
            dict(label_to_idx)
            if label_to_idx is not None
            else {}
        )

        self.idx_to_label: Dict[int, str] = {}
        self.num_channels = expected_num_channels
        self.channel_names: Optional[List[str]] = None
        self.final_sampling_rate: Optional[float] = None

        self._build(expected_class_names)

    def _find_signal_files(self, folder: Path) -> List[Path]:
        return sorted(
            path
            for path in folder.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    def _build(
        self,
        expected_class_names: Optional[Sequence[str]],
    ) -> None:
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory does not exist: {self.data_dir}"
            )

        gesture_directories = sorted(
            path
            for path in self.data_dir.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        )

        if not gesture_directories:
            raise ValueError(
                f"No gesture folders found in {self.data_dir}"
            )

        class_names = [folder.name for folder in gesture_directories]

        if expected_class_names is not None:
            if set(class_names) != set(expected_class_names):
                missing = sorted(
                    set(expected_class_names) - set(class_names)
                )
                extra = sorted(
                    set(class_names) - set(expected_class_names)
                )

                raise ValueError(
                    "Training and testing class folders must match exactly.\n"
                    f"Missing from this directory: {missing}\n"
                    f"Unexpected classes: {extra}"
                )

        if not self.label_to_idx:
            self.label_to_idx = {
                class_name: index
                for index, class_name in enumerate(class_names)
            }

        self.idx_to_label = {
            index: label
            for label, index in self.label_to_idx.items()
        }

        first_valid_sampling_rate = self.sampling_rate

        if first_valid_sampling_rate is None:
            for folder in gesture_directories:
                files = self._find_signal_files(folder)

                if not files:
                    continue

                dataframe = read_signal_file(files[0])
                inferred = infer_sampling_rate(dataframe)

                if inferred is not None:
                    first_valid_sampling_rate = inferred
                    break

        if first_valid_sampling_rate is None:
            raise ValueError(
                "Could not infer the sampling rate. Pass it manually using "
                "--sampling-rate, for example --sampling-rate 1000."
            )

        self.final_sampling_rate = float(first_valid_sampling_rate)

        window_samples = int(
            round(
                self.final_sampling_rate
                * self.window_ms
                / 1000.0
            )
        )

        stride_samples = int(
            round(
                self.final_sampling_rate
                * self.stride_ms
                / 1000.0
            )
        )

        if window_samples <= 0 or stride_samples <= 0:
            raise ValueError(
                "Window and stride must both be positive."
            )

        self.window_samples = window_samples
        self.stride_samples = stride_samples

        for folder in gesture_directories:
            label_name = folder.name

            if label_name not in self.label_to_idx:
                raise ValueError(
                    f"Class '{label_name}' was not found in training labels."
                )

            label_index = self.label_to_idx[label_name]
            signal_files = self._find_signal_files(folder)

            if not signal_files:
                raise ValueError(
                    f"No supported files found in {folder}"
                )

            for file_path in signal_files:
                signal, channel_names, _ = load_file_as_channels(
                    file_path,
                    self.channel_columns,
                )

                if self.channel_names is None:
                    self.channel_names = channel_names
                elif channel_names != self.channel_names:
                    raise ValueError(
                        f"Channel mismatch in {file_path}\n"
                        f"Expected: {self.channel_names}\n"
                        f"Found:    {channel_names}"
                    )

                if self.num_channels is None:
                    self.num_channels = signal.shape[0]
                elif signal.shape[0] != self.num_channels:
                    raise ValueError(
                        f"Channel count mismatch in {file_path}. "
                        f"Expected {self.num_channels}, got {signal.shape[0]}."
                    )

                windows = segment_signal(
                    signal,
                    self.window_samples,
                    self.stride_samples,
                )

                self.windows.extend(list(windows))
                self.labels.extend(
                    [label_index] * len(windows)
                )
                self.files.extend(
                    [file_path] * len(windows)
                )

        if not self.windows:
            raise ValueError(
                f"No windows were created from {self.data_dir}"
            )


class WindowDataset(Dataset[Tuple[Tensor, Tensor]]):
    """PyTorch wrapper around already prepared windows."""

    def __init__(
        self,
        windows: Sequence[np.ndarray],
        labels: Sequence[int],
        augment: bool = False,
    ) -> None:
        self.windows = list(windows)
        self.labels = list(labels)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        window = self.windows[index]

        if self.augment:
            window = augment_window(window)

        # [channels, samples] -> [1, channels, samples]
        input_tensor = torch.tensor(
            window,
            dtype=torch.float32,
        ).unsqueeze(0)

        label_tensor = torch.tensor(
            self.labels[index],
            dtype=torch.long,
        )

        return input_tensor, label_tensor


# ---------------------------------------------------------------------------
# Five-module CNN
# ---------------------------------------------------------------------------

class ConvBNReLU(nn.Module):
    """Conv2D -> BatchNorm -> ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        groups: int = 1,
    ) -> None:
        super().__init__()

        padding = (
            kernel_size[0] // 2,
            kernel_size[1] // 2,
        )

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.block(inputs)


class SqueezeExcitation(nn.Module):
    """Channel attention over learned feature maps."""

    def __init__(
        self,
        channels: int,
        reduction: int = 8,
    ) -> None:
        super().__init__()

        hidden_channels = max(channels // reduction, 8)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.scale = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden_channels,
                kernel_size=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                channels,
                kernel_size=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * self.scale(self.pool(inputs))


class ResidualSeparableModule(nn.Module):
    """
    Depthwise temporal convolution + pointwise convolution +
    BatchNorm + ReLU + SE attention + residual connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_kernel: int,
        downsample_time: bool,
    ) -> None:
        super().__init__()

        stride = (1, 2) if downsample_time else (1, 1)

        self.depthwise = ConvBNReLU(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(1, temporal_kernel),
            stride=stride,
            groups=in_channels,
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.attention = SqueezeExcitation(out_channels)

        if in_channels != out_channels or downsample_time:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        residual = self.residual(inputs)

        output = self.depthwise(inputs)
        output = self.pointwise(output)
        output = self.attention(output)

        return self.activation(output + residual)


class DeepEMGCNN(nn.Module):
    """
    Five-module deep 2D CNN.

    Input shape:
        [batch, 1, EMG_channels, time_samples]
    """

    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()

        # Module 1: broad temporal patterns
        self.module_1 = ConvBNReLU(
            in_channels=1,
            out_channels=32,
            kernel_size=(1, 15),
            stride=(1, 2),
        )

        # Module 2: neighboring-channel and temporal patterns
        self.module_2 = ConvBNReLU(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 11),
            stride=(1, 2),
        )

        # Modules 3-5: efficient deep residual temporal processing
        self.module_3 = ResidualSeparableModule(
            in_channels=64,
            out_channels=96,
            temporal_kernel=9,
            downsample_time=True,
        )

        self.module_4 = ResidualSeparableModule(
            in_channels=96,
            out_channels=128,
            temporal_kernel=7,
            downsample_time=True,
        )

        self.module_5 = ResidualSeparableModule(
            in_channels=128,
            out_channels=160,
            temporal_kernel=5,
            downsample_time=False,
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(160, num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(
                    layer.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(1)

        if inputs.ndim != 4:
            raise ValueError(
                "Expected input shape [batch, channels, time] or "
                "[batch, 1, channels, time]."
            )

        output = self.module_1(inputs)
        output = self.module_2(output)
        output = self.module_3(output)
        output = self.module_4(output)
        output = self.module_5(output)

        output = self.global_pool(output)
        output = torch.flatten(output, start_dim=1)
        output = self.dropout(output)

        return self.classifier(output)


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------

def count_total_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def estimate_model_size_mb(model: nn.Module) -> float:
    bytes_used = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )

    return bytes_used / (1024 ** 2)


def estimate_flops(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int],
    device: torch.device,
) -> int:
    """
    Approximate Conv2D and Linear FLOPs for one forward pass.

    Convention:
        one multiplication + one addition = two FLOPs
    """
    total_flops = 0
    hooks = []

    def conv_hook(
        module: nn.Conv2d,
        inputs: Tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        nonlocal total_flops

        output_elements = output.numel()
        kernel_operations = (
            module.kernel_size[0]
            * module.kernel_size[1]
            * (module.in_channels // module.groups)
            * 2
        )

        total_flops += output_elements * kernel_operations

    def linear_hook(
        module: nn.Linear,
        inputs: Tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        nonlocal total_flops

        batch_size = output.shape[0]
        total_flops += (
            batch_size
            * module.in_features
            * module.out_features
            * 2
        )

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(
                module.register_forward_hook(conv_hook)
            )

        elif isinstance(module, nn.Linear):
            hooks.append(
                module.register_forward_hook(linear_hook)
            )

    model.eval()

    with torch.no_grad():
        dummy = torch.zeros(input_shape, device=device)
        model(dummy)

    for hook in hooks:
        hook.remove()

    return int(total_flops)


def print_model_architecture(
    model: DeepEMGCNN,
    num_channels: int,
    window_samples: int,
    device: torch.device,
) -> None:
    """Print the output shape after every module."""
    model.eval()

    with torch.no_grad():
        x = torch.zeros(
            1,
            1,
            num_channels,
            window_samples,
            device=device,
        )

        output_1 = model.module_1(x)
        output_2 = model.module_2(output_1)
        output_3 = model.module_3(output_2)
        output_4 = model.module_4(output_3)
        output_5 = model.module_5(output_4)

    print("\nCNN architecture")
    print(f"  Input:    {tuple(x.shape)}")
    print(f"  Module 1: {tuple(output_1.shape)}")
    print(f"  Module 2: {tuple(output_2.shape)}")
    print(f"  Module 3: {tuple(output_3.shape)}")
    print(f"  Module 4: {tuple(output_4.shape)}")
    print(f"  Module 5: {tuple(output_5.shape)}")
    print("  Global average pool -> Dropout -> Linear classifier")


# ---------------------------------------------------------------------------
# Training and metrics
# ---------------------------------------------------------------------------

@dataclass
class EpochMetrics:
    loss: float
    accuracy: float


@dataclass
class ClassificationMetrics:
    confusion_matrix: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    support: np.ndarray
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float


def compute_class_weights(
    labels: Sequence[int],
    num_classes: int,
) -> Tensor:
    counts = np.bincount(
        np.asarray(labels),
        minlength=num_classes,
    ).astype(np.float32)

    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)

    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> EpochMetrics:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(inputs)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (
            logits.argmax(dim=1) == labels
        ).sum().item()
        total_examples += batch_size

    return EpochMetrics(
        loss=total_loss / total_examples,
        accuracy=total_correct / total_examples,
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[EpochMetrics, np.ndarray, np.ndarray, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    all_labels: List[np.ndarray] = []
    all_predictions: List[np.ndarray] = []

    start_time = time.perf_counter()

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_correct += (
                predictions == labels
            ).sum().item()
            total_examples += batch_size

            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())

    runtime = time.perf_counter() - start_time

    return (
        EpochMetrics(
            loss=total_loss / total_examples,
            accuracy=total_correct / total_examples,
        ),
        np.concatenate(all_labels),
        np.concatenate(all_predictions),
        runtime,
    )


def calculate_classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
) -> ClassificationMetrics:
    confusion_matrix = np.zeros(
        (num_classes, num_classes),
        dtype=np.int64,
    )

    for true_label, predicted_label in zip(
        labels,
        predictions,
    ):
        confusion_matrix[
            int(true_label),
            int(predicted_label),
        ] += 1

    true_positive = np.diag(confusion_matrix).astype(np.float64)
    predicted_count = confusion_matrix.sum(axis=0).astype(np.float64)
    actual_count = confusion_matrix.sum(axis=1).astype(np.float64)

    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros_like(true_positive),
        where=predicted_count != 0,
    )

    recall = np.divide(
        true_positive,
        actual_count,
        out=np.zeros_like(true_positive),
        where=actual_count != 0,
    )

    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    total_support = actual_count.sum()

    weighted_f1 = (
        float(np.sum(f1 * actual_count) / total_support)
        if total_support > 0
        else 0.0
    )

    return ClassificationMetrics(
        confusion_matrix=confusion_matrix,
        precision=precision,
        recall=recall,
        f1=f1,
        support=actual_count.astype(np.int64),
        macro_precision=float(precision.mean()),
        macro_recall=float(recall.mean()),
        macro_f1=float(f1.mean()),
        weighted_f1=weighted_f1,
    )


def print_final_report(
    train_metrics: EpochMetrics,
    test_metrics: EpochMetrics,
    classification_metrics: ClassificationMetrics,
    class_names: Sequence[str],
    training_runtime: float,
    testing_runtime: float,
    test_window_count: int,
    total_parameters: int,
    trainable_parameters: int,
    model_size_mb: float,
    flops: int,
) -> None:
    print("\n" + "=" * 76)
    print("FINAL RESULTS")
    print("=" * 76)

    print(f"Final training loss:       {train_metrics.loss:.4f}")
    print(f"Final training accuracy:   {train_metrics.accuracy * 100:.2f}%")
    print(f"Test loss:                 {test_metrics.loss:.4f}")
    print(f"Test accuracy:             {test_metrics.accuracy * 100:.2f}%")
    print(f"Macro precision:           {classification_metrics.macro_precision * 100:.2f}%")
    print(f"Macro recall:              {classification_metrics.macro_recall * 100:.2f}%")
    print(f"Macro F1-score:            {classification_metrics.macro_f1 * 100:.2f}%")
    print(f"Weighted F1-score:         {classification_metrics.weighted_f1 * 100:.2f}%")

    print("\nPer-class metrics")
    print(
        f"{'Class':<24}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'Support':>10}"
    )
    print("-" * 70)

    for index, class_name in enumerate(class_names):
        print(
            f"{class_name:<24}"
            f"{classification_metrics.precision[index] * 100:>11.2f}%"
            f"{classification_metrics.recall[index] * 100:>11.2f}%"
            f"{classification_metrics.f1[index] * 100:>11.2f}%"
            f"{classification_metrics.support[index]:>10d}"
        )

    print("\nConfusion matrix")
    print("Rows = actual class; columns = predicted class")

    class_width = max(
        14,
        max(len(class_name) for class_name in class_names) + 2,
    )

    header = (
        " " * class_width
        + "".join(
            f"{class_name[:10]:>11}"
            for class_name in class_names
        )
    )

    print(header)

    for row_index, class_name in enumerate(class_names):
        row_values = "".join(
            f"{value:>11d}"
            for value in classification_metrics.confusion_matrix[row_index]
        )

        print(
            f"{class_name:<{class_width}}"
            f"{row_values}"
        )

    average_test_time_ms = (
        testing_runtime / max(1, test_window_count)
    ) * 1000.0

    print("\nRuntime and model size")
    print(f"Total training time:       {training_runtime:.2f} seconds")
    print(f"Total testing time:        {testing_runtime:.4f} seconds")
    print(f"Average time per window:   {average_test_time_ms:.4f} ms")
    print(f"Total parameters:          {total_parameters:,}")
    print(f"Trainable parameters:      {trainable_parameters:,}")
    print(f"Approx. model size:        {model_size_mb:.3f} MB")
    print(f"Approx. FLOPs per window:  {flops:,}")
    print("=" * 76)


# ---------------------------------------------------------------------------
# Train / test pipeline
# ---------------------------------------------------------------------------

def run_training(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = choose_device()

    train_dataset = EMGDataset(
        data_dir=args.train_dir,
        window_ms=args.window_ms,
        stride_ms=args.stride_ms,
        sampling_rate=args.sampling_rate,
        channel_columns=args.channels,
    )

    class_names = [
        train_dataset.idx_to_label[index]
        for index in range(len(train_dataset.idx_to_label))
    ]

    test_dataset = EMGDataset(
        data_dir=args.test_dir,
        window_ms=args.window_ms,
        stride_ms=args.stride_ms,
        sampling_rate=train_dataset.final_sampling_rate,
        channel_columns=args.channels,
        label_to_idx=train_dataset.label_to_idx,
        expected_num_channels=train_dataset.num_channels,
        expected_class_names=class_names,
    )

    channel_mean: Optional[np.ndarray] = None
    channel_std: Optional[np.ndarray] = None

    if args.normalize_mode == "global":
        channel_mean, channel_std = compute_channel_stats(
            train_dataset.windows
        )

        train_windows = normalize_windows(
            train_dataset.windows,
            channel_mean,
            channel_std,
        )

        test_windows = normalize_windows(
            test_dataset.windows,
            channel_mean,
            channel_std,
        )

    elif args.normalize_mode == "per_window":
        train_windows = [
            normalize_per_window(window)
            for window in train_dataset.windows
        ]

        test_windows = [
            normalize_per_window(window)
            for window in test_dataset.windows
        ]

    else:
        train_windows = list(train_dataset.windows)
        test_windows = list(test_dataset.windows)

    train_torch_dataset = WindowDataset(
        train_windows,
        train_dataset.labels,
        augment=args.augment,
    )

    test_torch_dataset = WindowDataset(
        test_windows,
        test_dataset.labels,
        augment=False,
    )

    train_loader = DataLoader(
        train_torch_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    test_loader = DataLoader(
        test_torch_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = DeepEMGCNN(
        num_classes=len(class_names),
        dropout=args.dropout,
    ).to(device)

    print_model_architecture(
        model=model,
        num_channels=int(train_dataset.num_channels),
        window_samples=train_dataset.window_samples,
        device=device,
    )

    total_parameters = count_total_parameters(model)
    trainable_parameters = count_trainable_parameters(model)
    model_size_mb = estimate_model_size_mb(model)

    flops = estimate_flops(
        model=model,
        input_shape=(
            1,
            1,
            int(train_dataset.num_channels),
            train_dataset.window_samples,
        ),
        device=device,
    )

    class_weights = compute_class_weights(
        train_dataset.labels,
        len(class_names),
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print("\nDataset summary")
    print(f"  Training directory:      {args.train_dir}")
    print(f"  Testing directory:       {args.test_dir}")
    print(f"  Classes:                 {train_dataset.label_to_idx}")
    print(f"  Training windows:        {len(train_dataset.windows)}")
    print(f"  Testing windows:         {len(test_dataset.windows)}")
    print(f"  EMG channels:            {train_dataset.channel_names}")
    print(f"  Sampling rate:           {train_dataset.final_sampling_rate:.2f} Hz")
    print(f"  Window samples:          {train_dataset.window_samples}")
    print(f"  Stride samples:          {train_dataset.stride_samples}")
    print(f"  Normalization:           {args.normalize_mode}")
    print(f"  Training augmentation:   {args.augment}")
    print(f"  Device:                  {device}")

    print("\nModel summary")
    print(f"  Total parameters:        {total_parameters:,}")
    print(f"  Trainable parameters:    {trainable_parameters:,}")
    print(f"  Approx. model size:      {model_size_mb:.3f} MB")
    print(f"  Approx. FLOPs/window:    {flops:,}")

    training_start = time.perf_counter()
    final_train_metrics: Optional[EpochMetrics] = None

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()

        final_train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        epoch_runtime = time.perf_counter() - epoch_start

        if (
            epoch == 1
            or epoch == args.epochs
            or epoch % args.print_every == 0
        ):
            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"train loss {final_train_metrics.loss:.4f} | "
                f"train acc {final_train_metrics.accuracy * 100:6.2f}% | "
                f"lr {optimizer.param_groups[0]['lr']:.2e} | "
                f"{epoch_runtime:.2f}s"
            )

    training_runtime = time.perf_counter() - training_start

    test_metrics, labels, predictions, testing_runtime = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    classification_metrics = calculate_classification_metrics(
        labels=labels,
        predictions=predictions,
        num_classes=len(class_names),
    )

    if final_train_metrics is None:
        raise RuntimeError("Training did not run any epochs.")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "architecture": "deep_emg_cnn_5_module_residual_se_attention",
        "num_classes": len(class_names),
        "class_names": class_names,
        "label_to_idx": train_dataset.label_to_idx,
        "idx_to_label": train_dataset.idx_to_label,
        "num_emg_channels": train_dataset.num_channels,
        "channel_names": train_dataset.channel_names,
        "sampling_rate": train_dataset.final_sampling_rate,
        "window_ms": args.window_ms,
        "stride_ms": args.stride_ms,
        "window_samples": train_dataset.window_samples,
        "stride_samples": train_dataset.stride_samples,
        "normalize_mode": args.normalize_mode,
        "channel_mean": channel_mean,
        "channel_std": channel_std,
        "dropout": args.dropout,
        "test_accuracy": test_metrics.accuracy,
        "macro_f1": classification_metrics.macro_f1,
    }

    args.model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(checkpoint, args.model_path)

    print_final_report(
        train_metrics=final_train_metrics,
        test_metrics=test_metrics,
        classification_metrics=classification_metrics,
        class_names=class_names,
        training_runtime=training_runtime,
        testing_runtime=testing_runtime,
        test_window_count=len(test_dataset.windows),
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
        model_size_mb=model_size_mb,
        flops=flops,
    )

    print(f"Saved model to: {args.model_path}")


# ---------------------------------------------------------------------------
# Single-file prediction
# ---------------------------------------------------------------------------

def predict_file(args: argparse.Namespace) -> None:
    device = choose_device()

    checkpoint = torch.load(
        args.model_path,
        map_location=device,
        weights_only=False,
    )

    class_names: List[str] = checkpoint["class_names"]
    channel_names: List[str] = checkpoint["channel_names"]

    model = DeepEMGCNN(
        num_classes=checkpoint["num_classes"],
        dropout=checkpoint.get("dropout", 0.4),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    signal, found_channel_names, _ = load_file_as_channels(
        args.predict_file,
        channel_names,
    )

    if found_channel_names != channel_names:
        raise ValueError(
            f"Prediction channels do not match training channels.\n"
            f"Expected: {channel_names}\n"
            f"Found:    {found_channel_names}"
        )

    windows_array = segment_signal(
        signal=signal,
        window_samples=checkpoint["window_samples"],
        stride_samples=checkpoint["stride_samples"],
    )

    windows = list(windows_array)

    normalize_mode = checkpoint.get(
        "normalize_mode",
        "global",
    )

    if (
        normalize_mode == "global"
        and checkpoint.get("channel_mean") is not None
    ):
        windows = normalize_windows(
            windows,
            checkpoint["channel_mean"],
            checkpoint["channel_std"],
        )

    elif normalize_mode == "per_window":
        windows = [
            normalize_per_window(window)
            for window in windows
        ]

    batch = torch.tensor(
        np.stack(windows),
        dtype=torch.float32,
    ).unsqueeze(1).to(device)

    start_time = time.perf_counter()

    with torch.no_grad():
        logits = model(batch)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

    runtime = time.perf_counter() - start_time

    prediction_values, prediction_counts = np.unique(
        predictions.cpu().numpy(),
        return_counts=True,
    )

    majority_index = int(
        prediction_values[np.argmax(prediction_counts)]
    )

    majority_label = class_names[majority_index]
    vote_confidence = float(
        prediction_counts.max() / prediction_counts.sum()
    )

    average_probabilities = probabilities.mean(dim=0).cpu().numpy()

    print(f"\nPredicted gesture: {majority_label}")
    print(
        "Majority-vote confidence: "
        f"{vote_confidence * 100:.2f}%"
    )

    print("\nAverage class probabilities")

    for index, class_name in enumerate(class_names):
        print(
            f"  {class_name}: "
            f"{average_probabilities[index] * 100:.2f}%"
        )

    print("\nWindow vote counts")

    for index, count in zip(
        prediction_values,
        prediction_counts,
    ):
        print(
            f"  {class_names[int(index)]}: {int(count)}"
        )

    print(f"\nWindows evaluated: {len(windows)}")
    print(f"Total inference time: {runtime * 1000:.3f} ms")
    print(
        "Average inference time per window: "
        f"{runtime * 1000 / len(windows):.4f} ms"
    )


# ---------------------------------------------------------------------------
# File inspection
# ---------------------------------------------------------------------------

def inspect_file(args: argparse.Namespace) -> None:
    signal, channel_names, inferred_rate = load_file_as_channels(
        args.inspect_file,
        args.channels,
    )

    summary = {
        "file": str(args.inspect_file),
        "channels": channel_names,
        "num_channels": int(signal.shape[0]),
        "samples": int(signal.shape[1]),
        "inferred_sampling_rate": inferred_rate,
    }

    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train, test, inspect, or run prediction with the "
            "five-module Deep EMG CNN."
        )
    )

    parser.add_argument(
        "--train-dir",
        type=Path,
        default=None,
        help="Training directory containing gesture subfolders.",
    )

    parser.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help="Testing directory containing matching gesture subfolders.",
    )

    parser.add_argument(
        "--predict-file",
        type=Path,
        default=None,
        help="Run prediction on one CSV/TXT/TSV recording.",
    )

    parser.add_argument(
        "--inspect-file",
        type=Path,
        default=None,
        help="Inspect one CSV/TXT/TSV recording.",
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("deep_emg_cnn_model.pth"),
        help="Path used to save or load the model.",
    )

    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=None,
        help=(
            "Sampling rate in Hz. If omitted, the script tries to infer "
            "it from elapsed_s or timestamp."
        ),
    )

    parser.add_argument(
        "--window-ms",
        type=float,
        default=256.0,
        help="Window length in milliseconds.",
    )

    parser.add_argument(
        "--stride-ms",
        type=float,
        default=128.0,
        help="Stride in milliseconds.",
    )

    parser.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help=(
            "Specific EMG columns to use, for example "
            "--channels ch1 ch2 ch3"
        ),
    )

    parser.add_argument(
        "--normalize-mode",
        choices=["global", "per_window", "none"],
        default="global",
        help=(
            "global uses training-set channel statistics for train/test/"
            "prediction; per_window normalizes each window independently."
        ),
    )

    parser.add_argument(
        "--augment",
        action="store_true",
        default=True,
        help="Enable training-only EMG augmentation.",
    )

    parser.add_argument(
        "--no-augment",
        dest="augment",
        action="store_false",
        help="Disable training augmentation.",
    )

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    training_requested = (
        args.train_dir is not None
        or args.test_dir is not None
    )

    if training_requested:
        if args.train_dir is None or args.test_dir is None:
            parser.error(
                "Training requires both --train-dir and --test-dir."
            )

        if (
            args.predict_file is not None
            or args.inspect_file is not None
        ):
            parser.error(
                "Do not combine training with prediction or inspection."
            )

    elif args.predict_file is not None:
        if args.inspect_file is not None:
            parser.error(
                "Choose either --predict-file or --inspect-file."
            )

    elif args.inspect_file is None:
        parser.error(
            "Provide --train-dir and --test-dir, --predict-file, "
            "or --inspect-file."
        )

    return args


def main() -> None:
    args = parse_args()

    if args.train_dir is not None:
        run_training(args)

    elif args.predict_file is not None:
        predict_file(args)

    else:
        inspect_file(args)


if __name__ == "__main__":
    main()
