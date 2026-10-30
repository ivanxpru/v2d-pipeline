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
#
# Что меряется (по 20 с каждый, параллельно не гоняем — нужен честный канал):
#   1) download.pytorch.org — откуда качается torch (~800 МБ, главный груз)
#   2) huggingface.co       — веса моделей (HF CDN, xet)
#   3) files.pythonhosted.org — pip-пакеты
#   4) DagsHub S3 (если есть /workspace/.s3env) — промежуточное хранилище
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

# ---------- 5. GPU sanity ----------
if nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    python3 -c "import torch; assert torch.cuda.is_available(); print('CUDA OK:', torch.cuda.get_device_name(0))" || FAIL=2
else
    echo "ОШИБКА: nvidia-smi недоступен"
    FAIL=2
fi

echo
if [ "$FAIL" -eq 0 ]; then
    echo "=== КАНАЛ И GPU В НОРМЕ — заезд можно продолжать ==="
elif [ "$FAIL" -eq 2 ]; then
    echo "=== GPU/CUDA ПРОБЛЕМА — гасить инстанс ==="
else
    echo "=== КАНАЛ МЕДЛЕННЫЙ (< ${MIN_MBPS} МБ/с) — ГАСИТЬ ИНСТАНС И ИСКАТЬ ДРУГОЙ ==="
    echo "Урок 2026-10-29: заявленные 3 Гбит/с <> реальные 0.2 МБ/с; встречается часто."
fi
exit "$FAIL"
