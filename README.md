# FlashAttention-3-MultiData
FlashAttention3 Integration in Transformer Self-Attention

This repository presents an implementation of FlashAttention3 within a Transformer-style attention module, comparing it against the default PyTorch scaled dot-product attention (which includes FlashAttention2 optimizations).
Project Structure

.
├── dataset.py                          # Loads Parquet-format training data
├── model.py                            # Transformer model with optional FlashAttention3
├── train.py                            # Trains model with benchmarking metrics
├── test_flash_attention.py             # Tests correctness across attention modes
├── utils.py                            # Utility functions for training and data
├── Dockerfile                          # Docker container for reproducibility
├── flash_attention_3.pdf               # Project report and performance analysis
├── output_flash_attention_final450919.log  # Training log output
├── output_flash_attention_final450919.err  # Error logs (if any)

Setup

Docker (recommended):
docker build -t flashattn3 .
docker run --gpus all flashattn3

Manual Setup:
pip install torch flash-attn pandas pyarrow numpy

Training

Run baseline (FlashAttention2) training:
python train.py --use_flash_attn=False

Run FlashAttention3-enhanced training:
python train.py --use_flash_attn=True

Testing

Run correctness tests:
pytest test_flash_attention.py

Performance Summary:
| Model            | Tokens/sec | Train Token % | MFU % | TFLOPs | Inference Latency (ms) |
|------------------|------------|----------------|--------|---------|-------------------------|
| Baseline         | 7893.10    | 22.59%         | 41.13% | 406.82  | 150.28                  |
| FlashAttention3  | 8127.71    | 22.59%         | 42.36% | 418.91  | 141.08                  |



