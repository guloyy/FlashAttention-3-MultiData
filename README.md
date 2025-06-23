# FlashAttention-3-MultiData

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](#license)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-lightgrey)](#requirements)

## Transformer Self-Attention with FlashAttention3

An implementation of FlashAttention3 in a Transformer-style attention module, benchmarked against PyTorch's default scaled dot-product attention (including FlashAttention2 optimizations).

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Requirements](#requirements)
3. [Installation](#installation)

   * [Docker (Recommended)](#docker)
   * [Manual Setup](#manual-setup)
4. [Usage](#usage)

   * [Training](#training)
   * [Testing](#testing)
5. [Performance Summary](#performance-summary)
6. [Report & Logs](#report--logs)
7. [License](#license)

---

## Project Structure

```
📦
├── dataset.py                       # Load Parquet-format training data
├── model.py                         # Transformer model w/ optional FlashAttention3
├── train.py                         # Training script + benchmarking metrics
├── test_flash_attention.py          # Correctness tests for attention modes
├── utils.py                         # Training & data utility functions
├── Dockerfile                       # Container for reproducible environment
├── flash_attention_3.pdf            # Report & performance analysis
├── output_flash_attention_final.log # Training log output
└── output_flash_attention_final.err # Error logs (if any)
```

---

## Requirements

* Python 3.8 or higher
* GPUs with CUDA compatibility (for FlashAttention3)

**Python packages:**

```
pip install torch flash-attn pandas pyarrow numpy
```

---

## Installation

### Docker (Recommended)

Build and run the container with GPU support:

```bash
docker build -t flashattn3 .
docker run --gpus all flashattn3
```

### Manual Setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv venv && source venv/bin/activate
   ```
2. Install dependencies:

   ```bash
   pip install torch flash-attn pandas pyarrow numpy
   ```

---

## Usage

### Training

* **Baseline (FlashAttention2)**

  ```bash
  python train.py --use_flash_attn=False
  ```

* **FlashAttention3-Enhanced**

  ```bash
  python train.py --use_flash_attn=True
  ```

Training metrics (tokens/sec, MFU, TFLOPs) will be printed to the console and saved in `output_flash_attention_final.log`.

### Testing

Run unit tests to verify correctness across attention modes:

```bash
pytest test_flash_attention.py
```

---

## Performance Summary

| Model           | Tokens/sec | Train Token % | MFU %  | TFLOPs | Inference Latency (ms) |
| --------------- | ---------- | ------------- | ------ | ------ | ---------------------- |
| Baseline        | 7,893.10   | 22.59%        | 41.13% | 406.82 | 150.28                 |
| FlashAttention3 | 8,127.71   | 22.59%        | 42.36% | 418.91 | 141.08                 |

---

## Report & Logs

* Project report (PDF): [flash\_attention\_3.pdf](flash_attention_3.pdf)
* Training log: `output_flash_attention_final.log`
* Error log: `output_flash_attention_final.err`

---

## License

This project is licensed under the [MIT License](LICENSE).




