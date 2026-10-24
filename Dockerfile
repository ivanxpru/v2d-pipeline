# Docker-образ для video→3D pipeline (VR) — БАЗОВЫЙ, без conda/ML-стека.
# Стек (conda env v2d, torch и пр.) ставится на инстансе через scripts/setup-server.sh.
# База: NVIDIA CUDA 12.1 runtime на Ubuntu 22.04 (совместимо с RTX 3090, driver >= 530).
# Сборка:  docker build -t ivanxphub/v2d-pipeline:latest -f docker/Dockerfile .
# Далее:   qudata create_template { image: "ivanxphub/v2d-pipeline", tag: "latest",
#          ssh_enabled: true, volume_mount_path: "/workspace", min_storage_gb: 50 }
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

# Системные пакеты + ffmpeg (в Ubuntu 22.04 собран с NVENC)
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
        git wget unzip ca-certificates ffmpeg rsync htop tmux openssh-client \
        libgl1 libglib2.0-0 python3 python3-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Прогрев весов моделей при сборке — на инстансе качать не нужно (счёт GPU идёт).
# Depth Anything V2 (правильный -hf репозиторий) + Real-ESRGAN. wget -c с ретраями
# (HF CDN xet рвёт соединения; snapshot_download ненадёжен — см. memory).
RUN mkdir -p /workspace/models/Depth-Anything-V2-Large-hf /workspace/models && \
    cd /workspace/models/Depth-Anything-V2-Large-hf && \
    for f in config.json preprocessor_config.json model.safetensors; do \
        wget -c -q --tries=0 --timeout=60 "https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf/resolve/main/$f" || exit 1; \
    done && \
    cd /workspace/models && \
    wget -c -q --tries=0 --timeout=60 -O RealESRGAN_x4plus.pth \
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

# Структура workspace (volume монтируется в /workspace на хостинге)
RUN mkdir -p /workspace/input /workspace/output /workspace/models /workspace/logs
WORKDIR /workspace

CMD ["sleep", "infinity"]
