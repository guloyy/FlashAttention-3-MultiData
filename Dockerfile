# Base NVIDIA PyTorch image (CUDA 12.x, PyTorch 2.x)
FROM nvcr.io/nvidia/pytorch:24.11-py3

# Install system packages and Python tools
RUN apt-get update && \
    apt-get install -y python3-pip python3-venv git build-essential && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install --no-cache-dir --upgrade pip setuptools==69.5.1

# Install essential Python libraries for AI training
RUN python3 -m pip install --no-cache-dir packaging ninja \
    datasets transformers accelerate wandb pyyaml numpy \
    sentencepiece tensorboard jupyter deepspeed seaborn pandas tqdm dacite \
    safetensors

# Set environment variables to optimize FlashAttention 3 build
ENV MAX_JOBS=8 \
    FLASH_ATTENTION_DISABLE_BACKWARD=FALSE \
    FLASH_ATTENTION_DISABLE_SPLIT=TRUE \
    FLASH_ATTENTION_DISABLE_LOCAL=TRUE \
    FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE \
    FLASH_ATTENTION_DISABLE_FP16=TRUE \
    FLASH_ATTENTION_DISABLE_FP8=TRUE \
    FLASH_ATTENTION_DISABLE_APPENDKV=TRUE \
    FLASH_ATTENTION_DISABLE_VARLEN=TRUE \
    FLASH_ATTENTION_DISABLE_CLUSTER=FALSE \
    FLASH_ATTENTION_DISABLE_PACKGQA=TRUE \
    FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE \
    FLASH_ATTENTION_DISABLE_HDIM64=TRUE \
    FLASH_ATTENTION_DISABLE_HDIM96=TRUE \
    FLASH_ATTENTION_DISABLE_HDIM128=FALSE \
    FLASH_ATTENTION_DISABLE_HDIM192=TRUE \
    FLASH_ATTENTION_DISABLE_HDIM256=TRUE

# Clone FlashAttention repository and build FlashAttention-3 (Hopper) 
RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention/hopper && \
    python setup.py install

# Create workspace directory
RUN mkdir -p /workspace




