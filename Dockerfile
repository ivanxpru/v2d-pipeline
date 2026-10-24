# Docker-образ для video→3D pipeline (VR).
# База: NVIDIA CUDA 12.1 runtime на Ubuntu 22.04 (совместимо с RTX 3090, driver >= 530).
# Сборка:  docker build -t <registry>/v2d-pipeline:latest -f docker/Dockerfile .
# Далее:   qudata create_template { image: "<registry>/v2d-pipeline", tag: "latest",
#          ssh_enabled: true, volume_mount_path: "/workspace", min_storage_gb: 50 }
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

# Системные пакеты + ffmpeg (в Ubuntu 22.04 собран с NVENC)
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
        git wget unzip ca-certificates ffmpeg rsync htop tmux openssh-client \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Miniconda + окружение v2d
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh \
    && bash /tmp/mc.sh -b -p /opt/miniconda3 && rm /tmp/mc.sh
ENV PATH=/opt/miniconda3/bin:$PATH

RUN conda create -y -n v2d python=3.10 \
    && /opt/miniconda3/envs/v2d/bin/pip install --no-cache-dir --upgrade pip \
    # ВАЖНО: torch pin + строго первым (иначе basicsr при sdist-сборке тянет
    # свежий torch 2.9 + CUDA 12.6/12.9 — +2.6 ГБ и десятки минут сборки).
    && /opt/miniconda3/envs/v2d/bin/pip install --no-cache-dir \
        torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 \
    # Пин diffusers: новее требует torch.accelerator (нет в 2.5.1)
    && /opt/miniconda3/envs/v2d/bin/pip install --no-cache-dir \
        opencv-python-headless numpy scipy pillow tqdm \
        "diffusers==0.31.0" "transformers==4.46.3" "accelerate==1.1.1" \
        "huggingface_hub==0.26.5" safetensors \
    # basicsr/realesrgan: sdist => обязательно --no-build-isolation,
    # иначе изолятор сборки ставит СВЕЖИЙ torch с полным CUDA-стеком
    && /opt/miniconda3/envs/v2d/bin/pip install --no-cache-dir --no-build-isolation \
        "basicsr==1.4.2" "realesrgan==0.3.0"

# Патч basicsr 1.4.2 под torchvision >= 0.17 (functional_tensor удалён)
RUN sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' \
    /opt/miniconda3/envs/v2d/lib/python3.10/site-packages/basicsr/data/degradations.py

# Активация v2d по умолчанию
ENV CONDA_DEFAULT_ENV=v2d
ENV PATH=/opt/miniconda3/envs/v2d/bin:$PATH

# Структура workspace (volume монтируется в /workspace на хостинге)
RUN mkdir -p /workspace/input /workspace/output /workspace/models /workspace/logs
WORKDIR /workspace

# Прогрев кэша модели при сборке (по желанию — раскомментировать):
# RUN python -c "from transformers import AutoImageProcessor, AutoModelForDepthEstimation; \
#     AutoImageProcessor.from_pretrained('depth-anything/Depth-Anything-V2-Large'); \
#     AutoModelForDepthEstimation.from_pretrained('depth-anything/Depth-Anything-V2-Large')"

CMD ["sleep", "infinity"]
