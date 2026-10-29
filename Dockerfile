# Docker-образ video→3D pipeline (VR) — ПОЛНЫЙ, без conda.
# Вся ML-инфраструктура (torch cu121, DA3, зависимости, веса) запечена в образ —
# инстанс после аренды готов к работе сразу, setup-server.sh не нужен.
#
# База: NVIDIA CUDA 12.1 runtime, Ubuntu 22.04 (RTX 3090, driver >= 530).
# Сборка: docker build -t ivanxphub/v2d-pipeline:latest -f docker/Dockerfile .
# (локально/на CI — сборка CUDA-образа НЕ требует GPU, torch = готовые wheels)
#
# ВАЖНО (урок memory/docker-lessons.md): хостинг Qudata монтирует volume ПОВЕРХ
# /workspace — файлы из слоя скрываются пустым volume. Поэтому все веса запечены
# в /opt/models, а entrypoint.sh при старте докладывает их в /workspace/models.

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ---------- Системные пакеты ----------
# ffmpeg из Ubuntu 22.04 собран с libx264/libx265 (NVENC-энкодер системного
# ffmpeg несовместим с драйверами >= 580 — «unsupported device»; энкодим x264/x265,
# декод/кодирование GPU-пути в ffmpeg-контейнере не требуются).
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
        git wget unzip ca-certificates ffmpeg rsync htop tmux openssh-client \
        libgl1 libglib2.0-0 python3 python3-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

# ---------- Python-стек (PyPI из CI-сети — быстро и без прокси) ----------
# PyTorch 2.4.1 + cu121 (готовые wheels, бандлят CUDA-библиотеки внутри).
# Порядок важен: torch ПЕРВЫМ (урок memory/pip-cuda-lessons.md).
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
        torch==2.4.1+cu121 torchvision==0.19.1+cu121 \
        --index-url https://download.pytorch.org/whl/cu121

RUN python3 -m pip install --no-cache-dir \
        numpy==2.2.6 \
        opencv-python-headless scipy pillow tqdm \
        safetensors huggingface_hub transformers accelerate \
        einops omegaconf addict easydict \
        "moviepy==1.0.3" \
        plyfile matplotlib evo \
        pandas prettytable numba pypose \
        faiss-cpu

# DA3-пакет ставим без зависимостей (потянул бы gsplat/pycolmap — не нужны).
RUN python3 -m pip install --no-cache-dir --no-deps depth-anything-3==0.1.1

# Патч api.py: делаем optional-импорты (export/pose_align) безопасными.
COPY patches/patch_da3.py /tmp/patch_da3.py
RUN python3 /tmp/patch_da3.py && rm -f /tmp/patch_da3.py

# ---------- Веса моделей (запечены в /opt/models, entrypoint переносит в /workspace) ----------
# wget -c --tries=0: HF CDN xet рвёт соединения (урок memory/hf-download-lessons.md).
RUN mkdir -p /opt/models/Depth-Anything-V2-Large-hf \
             /opt/models/da3mono-large && cd /opt/models && \
    for f in config.json preprocessor_config.json model.safetensors; do \
        wget -c -q --tries=0 --timeout=60 \
            "https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf/resolve/main/$f" \
            -O "Depth-Anything-V2-Large-hf/$f" || exit 1; \
    done && \
    for f in config.json model.safetensors; do \
        wget -c -q --tries=0 --timeout=60 \
            "https://huggingface.co/depth-anything/DA3MONO-LARGE/resolve/main/$f" \
            -O "da3mono-large/$f" || exit 1; \
    done && \
    wget -c -q --tries=0 --timeout=60 -O RealESRGAN_x4plus.pth \
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" && \
    mkdir -p salad && cd salad && \
    wget -c -q --tries=0 --timeout=60 -O dino_salad.ckpt \
        "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt" && \
    cd .. && du -sh /opt/models

# ---------- Скрипты пайплайна ----------
COPY scripts/ /workspace/scripts/

# ---------- Структура workspace ----------
RUN mkdir -p /workspace/input /workspace/output /workspace/models /workspace/logs
WORKDIR /workspace

# Entrypoint: переносит запечённые веса в /workspace/models (если volume пуст)
# и оставляет контейнер живым (ssh-шаблон Qudata подключается внутрь).
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["sleep", "infinity"]
