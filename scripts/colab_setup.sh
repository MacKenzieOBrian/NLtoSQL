#!/usr/bin/env bash
set -euo pipefail

export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"

# remove conflicting preinstalled versions
pip uninstall -y \
  torch torchvision torchaudio bitsandbytes triton \
  transformers accelerate peft trl datasets \
  numpy pandas fsspec requests google-auth \
  scipy scikit-learn || true

# base deps
pip install -q --no-cache-dir --force-reinstall \
  numpy==1.26.4 pandas==2.2.1 scipy scikit-learn \
  fsspec==2024.5.0 requests==2.31.0 google-auth==2.43.0

# torch + cuda 12.1
pip install -q --no-cache-dir --force-reinstall \
  torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

# bitsandbytes + triton + hf stack
pip install -q --no-cache-dir --force-reinstall \
  bitsandbytes==0.43.3 triton==2.3.1 \
  transformers==4.44.2 accelerate==0.33.0 peft==0.17.0 trl==0.9.6 datasets==2.20.0

echo "Setup complete. Restart runtime once, then run the rest of the notebook."

