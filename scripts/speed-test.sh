#!/usr/bin/env bash
# Быстрый тест канала и GPU сразу после аренды инстанса.
# Мотивация (заезд 2026-10-29, Калифорния): оффер заявлял 3 Гбит/с, реально
# сервер→CDN/S3 отдавал 0.13–0.9 МБ/с → только докачка torch заняла 4 часа.
# Правило: скорость ниже порога → инстанс гасить сразу, потеря ~2-3 мин аренды
# вместо часов простоя.
#
# Запуск на свежем инстансе (в образе v2d-pipeline-full или через scp):
#   bash /workspace/scripts/speed-test.sh [MIN_MBPS]
#   MIN_MBPS — порог приёма, по умолчанию 5 МБ/с (гигабит-класс).
#   Коды выхода: 0 = канал OK, 1 = медленно (ГАСИТЬ инстанс), 2 = GPU/CUDA не ОК.
#   Флаг-опция до порога: REAL=1 bash speed-test.sh — только факт-конфиг,
#   без замера сети (для оценки «стоит ли вообще работать»).
#
# Платформа Qudata показывает в оффере конфиг ФИЗИЧЕСКОГО ХОСТА, а не контейнера
# (проверено 2026-10-30: «35 ГБ RAM» в оффере = 503 ГБ по факту; «192 ядра» = 13
# vCPU по cgroup). Поэтому все параметры меряем изнутри: cgroup-квоты, df, /proc.
#
# Что меряется:
#   [ФАКТ] vCPU (cgroup), RAM (cgroup+free), диск (df), GPU (nvidia-smi)
#   [КАНАЛ] по 20с каждый: pytorch CDN, HF, PyPI, DagsHub S3
# Решение «работать / гасить» — по каналу и факт-конфигу.
set -euo pipefail

MIN_MBPS="${1:-5}"
TEST_SECONDS=20
FAIL=0

measure() {
    # measure <url> <label> — качает TEST_SECONDS, печатает МБ/с
    local url="$1" label="$2"
    local bytes
    bytes=$(timeout $((TEST_SECONDS + 5)) curl -sL -o /dev/null \
        --max-time "$TEST_SECONDS" -w '%{size_download}' "$url" 2>/dev/null || echo 0)
    python3 -c "print(f'{${bytes}/${TEST_SECONDS}/1048576:.2f}')"
}

echo "=== Порог приёма: ${MIN_MBPS} МБ/с, замер по ${TEST_SECONDS}с ==="

# ---------- 0. ФАКТ-конфигурация контейнера (платформа показывает хост!) ----------
echo "=== Факт-конфигурация контейнера ==="
# vCPU: cgroup v2/v1 — единственная правда
VCPU=unknown
if [ -s /sys/fs/cgroup/cpu.max ]; then
    read -r QUA PER < /sys/fs/cgroup/cpu.max
    [ "$QUA" != "max" ] && VCPU=$(( QUA / PER ))
fi
[ "$VCPU" = unknown ] && {
    QUA=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo -1)
    PER=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo 100000)
    [ "${QUA:-0}" -gt 0 ] 2>/dev/null && VCPU=$(( QUA / PER ))
}
[ "$VCPU" = unknown ] && VCPU=$(nproc)
echo "vCPU (cgroup):  ${VCPU}"   # оффер обычно показывает ядра хоста — не верить

# RAM: cgroup лимит vs host
RAM_CGROUP=unknown
if [ -s /sys/fs/cgroup/memory.max ]; then
    M=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo max)
    [ "$M" != "max" ] && RAM_CGROUP=$(( M / 1073741824 ))
fi
[ "$RAM_CGROUP" = unknown ] && {
    M=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo max)
    [ "${M:-max}" != "max" ] && [ "$M" -lt 9000000000000 ] 2>/dev/null && RAM_CGROUP=$(( M / 1073741824 ))
}
RAM_FREE=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')
echo "RAM cgroup:     ${RAM_CGROUP} ГБ (host по free: ${RAM_FREE} ГБ)"  # LESSER из двух = реальный лимит

# Диск: реальные маунты (overlay = контейнер, /workspace может быть volume)
df -h / /workspace 2>/dev/null | awk 'NR==1 || /overlay|workspace/'
DISK_FREE=$(df -BG --output=avail /workspace 2>/dev/null | tail -1 | tr -dc 0-9)
echo "Диск /workspace свободно: ${DISK_FREE:-?} ГБ"

# GPU
if nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "ОШИБКА: nvidia-smi недоступен"
    exit 2
fi

# Оценка «стоит ли работать»:
#   диск < 40 ГБ свободно — впритык (веса уже в образе, но depth-карты жрут много:
#   2K x 6min = ~50 ГБ raw+smoothed), RAM < 8 ГБ — стек карт в память не влезет.
if [ "${DISK_FREE:-0}" -lt 40 ]; then
    echo "ПРЕДУПРЕЖДЕНИЕ: < 40 ГБ на /workspace — картам глубины может не хватить"
fi

# ---------- Быстрый режим: только факт, без замера канала ----------
if [ "${REAL:-0}" = "1" ]; then
    echo "=== REAL-режим: факт-конфигурация выведена, сеть не мерилась ==="
    exit 0
fi

# ---------- 1. PyTorch CDN (torch cu121 ~800 МБ) ----------
S=$(measure "https://download.pytorch.org/whl/cu121/torch-2.4.1%2Bcu121-cp310-cp310-linux_x86_64.whl" "torch")
echo "pytorch CDN:        ${S} МБ/с"
awk -v s="$S" -v m="$MIN_MBPS" 'BEGIN { exit (s+0 < m) ? 1 : 0 }' || FAIL=1

# ---------- 2. HuggingFace CDN (веса) ----------
S=$(measure "https://huggingface.co/depth-anything/DA3MONO-LARGE/resolve/main/model.safetensors" "hf")
echo "huggingface CDN:    ${S} МБ/с"
awk -v s="$S" -v m="$MIN_MBPS" 'BEGIN { exit (s+0 < m) ? 1 : 0 }' || FAIL=1

# ---------- 3. PyPI ----------
S=$(measure "https://files.pythonhosted.org/packages/source/n/numpy/numpy-1.26.4.tar.gz" "pypi")
echo "files.pythonhosted: ${S} МБ/с"
awk -v s="$S" -v m="$MIN_MBPS" 'BEGIN { exit (s+0 < m) ? 1 : 0 }' || FAIL=1

# ---------- 4. DagsHub S3 (опционально, если ключи доставлены) ----------
if [ -s /workspace/.s3env ]; then
    # shellcheck disable=SC1091
    source /workspace/.s3env
    export AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_DEFAULT_REGION
    B=$(aws --region "$AWS_DEFAULT_REGION" --endpoint-url "$S3_ENDPOINT" \
        s3api head-object --bucket "$S3_BUCKET" \
        --key wheels/torch-2.4.1+cu121-cp310-cp310-linux_x86_64.whl \
        --query ContentLength --output text 2>/dev/null || echo 0)
    if [ "${B:-0}" -gt 0 ]; then
        O=$((RANDOM % (B - 8388608)))
        bytes=$(timeout $((TEST_SECONDS + 5)) aws --region "$AWS_DEFAULT_REGION" \
            --endpoint-url "$S3_ENDPOINT" s3api get-object \
            --bucket "$S3_BUCKET" --key "wheels/torch-2.4.1+cu121-cp310-cp310-linux_x86_64.whl" \
            --range "bytes=$O-$((O + 8388607))" /dev/stdout 2>/dev/null | wc -c)
        S=$(python3 -c "print(f'{${bytes}/8/1048576:.2f}')")
        echo "DagsHub S3 (8 МБ):  ${S} МБ/с"
        awk -v s="$S" -v m="$MIN_MBPS" 'BEGIN { exit (s+0 < m) ? 1 : 0 }' || FAIL=1
    else
        echo "DagsHub S3: ключи есть, wheels/ в бакете нет — пропущено"
    fi
else
    echo "DagsHub S3: /workspace/.s3env нет — пропущено (крайне рекомендую прогнать env-up ДО решения о заезде)"
fi

# ---------- 5. GPU sanity уже проверен выше (0) ----------

echo
# Полный образ (v2d-pipeline-full) не качает torch/pip на инстансе — провал PyPI
# не блокирует заезд. Считаем блокирующим только FAIL=2 (GPU); канал — только
# если провалились ВСЕ каналы, из которых хоть что-то реально нужно.
if [ "$FAIL" -eq 2 ]; then
    echo "=== GPU/CUDA ПРОБЛЕМА — гасить инстанс ==="
    exit 2
fi
if [ "$FAIL" -eq 1 ]; then
    if [ "${PIPELINE_IMAGE:-1}" = "1" ]; then
        # в полном образе torch/веса запечены: PyPI не нужен, torch/HF не нужны.
        # Критичен только аплоад входного видео (rsync/scp локаль->сервер).
        echo "=== Часть каналов ниже порога, но в полном образе зависимости запечены ==="
        echo "=== Блокирующий канал один: локаль->сервер (rsync). Проверяется отдельно ==="
        echo "=== (30 МБ < 20 c = ок; иначе гасить) ==="
        exit 0
    fi
    echo "=== КАНАЛ МЕДЛЕННЫЙ (< ${MIN_MBPS} МБ/с) — ГАСИТЬ ИНСТАНС И ИСКАТЬ ДРУГОЙ ==="
    echo "Урок 2026-10-29: заявленные 3 Гбит/с <> реальные 0.2 МБ/с; встречается часто."
    exit 1
fi
echo "=== КАНАЛ И GPU В НОРМЕ — заезд можно продолжать ==="
exit 0
