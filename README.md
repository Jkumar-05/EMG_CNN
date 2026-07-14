# EMG Gesture Recognition using Convolutional Neural Networks

A PyTorch implementation of a Convolutional Neural Network (CNN) for classifying hand gestures from electromyography (EMG) signals.

## Overview

This project trains and evaluates a CNN capable of recognizing hand gestures from multi-channel EMG data. The model automatically preprocesses raw EMG recordings by segmenting them into windows before performing classification.

The repository supports both model training and real-time prediction using previously trained models.

## Features

- Multi-channel EMG signal support
- Automatic sampling rate detection
- Sliding window segmentation
- Per-window normalization
- CNN-based gesture classification
- Majority-vote prediction
- Support for CSV, TXT, and TSV datasets
- Separate training and testing datasets

## Project Structure

```text
EMG_CNN/
├── main_cnn_emg.py
├── README.md
├── .gitignore
├── dataset/
│   ├── train/
│   └── test/
└── emg_cnn_model.pth
```

## Requirements

- Python 3.10+
- PyTorch
- NumPy
- Pandas

## Installation

Clone the repository:

```bash
git clone https://github.com/Jkumar-05/EMG_CNN.git
cd EMG_CNN
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Training

```bash
python main_cnn_emg.py \
    --train-dir dataset/train \
    --test-dir dataset/test
```

## Prediction

```bash
python main_cnn_emg.py \
    --predict-csv sample.txt \
    --model-path emg_cnn_model.pth
```

## Future Improvements

- Real-time Raspberry Pi deployment
- Live EMG streaming
- Additional gesture classes
- Hyperparameter optimization
- Lightweight model for embedded systems

## Author

**Jatin Kumar**
