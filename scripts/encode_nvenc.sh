#!/usr/bin/env bash
# Аппаратное перекодирование видео через NVENC (RTX 3090).
# Запускается НА СЕРВЕРЕ: ./scripts/encode_nvenc.sh input.mp4 [output.mp4] [quality]
#   quality: fast | balanced | quality (по умолчанию balanced)
set -euo pipefail

INPUT="${1:?Использование: $0 input.mp4 [output.mp4] [fast|balanced|quality]}"
OUTPUT="${2:-}"
QUALITY="${3:-balanced}"

[[ -f "$INPUT" ]] || { echo "Файл не найден: $INPUT"; exit 1; }
if [[ -z "$OUTPUT" ]]; then
  dir=$(dirname "$INPUT"); base=$(basename "$INPUT"); base="${base%.*}"
  OUTPUT="$dir/${base}_h265.mp4"
fi

# Проверка NVENC. grep -q в pipe при pipefail даёт SIGPIPE для ffmpeg —
# поэтому сначала читаем вывод целиком.
FF_ENCODERS=$(ffmpeg -hide_banner -encoders 2>/dev/null || true)
echo "$FF_ENCODERS" | grep -q hevc_nvenc || {
  echo "ОШИБКА: hevc_nvenc недоступен (нет GPU/NVENC в этой сборке ffmpeg)"; exit 1;
}

# Настройки качества: bitrate multiplier от 1080p, preset NVENC
case "$QUALITY" in
  fast)     PRESET=p1; CQ=30 ;;
  balanced) PRESET=p4; CQ=25 ;;
  quality)  PRESET=p7; CQ=20 ;;
  *) echo "quality: fast|balanced|quality"; exit 1;;
esac

echo "==> NVENC H.265: $INPUT → $OUTPUT (preset $PRESET, cq $CQ)"
ffmpeg -y -i "$INPUT" \
  -c:v hevc_nvenc -preset "$PRESET" -rc vbr -cq "$CQ" \
  -b:v 0 -maxrate 40M -bufsize 80M \
  -pix_fmt p010le -profile:v main10 \
  -c:a copy \
  "$OUTPUT"

echo "==> Готово: $OUTPUT ($(du -h "$OUTPUT" | cut -f1))"

# Пример стерео-склейки для VR (если есть left/right):
#   ffmpeg -i left.mp4 -i right.mp4 -filter_complex hstack \
#     -c:v hevc_nvenc -preset p4 -cq 25 -c:a aac out_sbs.mp4
