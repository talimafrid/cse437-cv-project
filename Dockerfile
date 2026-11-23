# Self-contained demo image: code, pinned packages, trained detector, gallery index, SigLIP2 weights
# and the GroZi-3.2k shelf images. Runs on CPU anywhere, and on an NVIDIA GPU with --gpus all.
FROM python:3.13-slim

# opencv-python needs libGL and glib at import time
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf YOLO_CONFIG_DIR=/tmp/ultralytics
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126

# Bake the frozen backbone into the image so the container never needs the internet
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('google/siglip2-base-patch16-384')"
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Data and models change less often than code, so they go first for layer caching
COPY data/grozi/bb data/grozi/bb
COPY data/grozi/Testing data/grozi/Testing
COPY outputs/index/siglip2_b_v8 outputs/index/siglip2_b_v8
COPY outputs/planograms outputs/planograms
COPY models/sku110k_yolo11s.pt models/
COPY configs configs
COPY splits splits
COPY src src

ENV GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=7860
EXPOSE 7860
CMD ["python", "-m", "src.app"]
