#!/usr/bin/env bash
# Пресейкал входного видео до 2K (2560x1440) на сервере: аппаратный декод
# (NVDEC, включая AV1 на Ada/40xx) → lanczos → NVENC HEVC main10.
# Зачем: 4K AV1 декодировать CPU дорого, а для depth+DIBR 2K достаточно,
# а SBS получится 5120x1440 — в лимитах Pico 4 Ultra.
#
# Запуск НА СЕРВЕРЕ: ./scripts/prescale_2k.sh input.mp4 [output.mp4]
set -euo pipefail

INPUT="${1:?Использование: $0 input.mp4 [output.mp4]}"
OUTPUT="${2:-}"
TARGET_W=2560
TARGET_H=1440

[[ -f "$INPUT" ]] || { echo "Файл не найден: $INPUT"; exit 1; }
if [[ -z "$OUTPUT" ]]; then
  dir=$(dirname "$INPUT"); base=$(basename "$INPUT"); base="${base%.*}"
  OUTPUT="$dir/${base}_2k.mp4"
fi

# Выбор аппаратного декодера по кодеку входа (av1_cuvid есть только на Ada+)
CODEC=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name \
  -of csv=p=0 "$INPUT")
HWDEC=""
AV1_DECODERS=$(ffmpeg -hide_banner -decoders 2>/dev/null || true)
if echo "$AV1_DECODERS" | grep -q "av1_cuvid"; then
  [[ "$CODEC" == "av1" ]] && HWDEC="-hwaccel cuda -c:v av1_cuvid"
fi
[[ -z "$HWDEC" ]] && HWDEC="-hwaccel cuda"   # h264/hevc/vp9 — универсально

# Проверка NVENC. Внимание: нельзя `ffmpeg | grep -q` при set -o pipefail —
# grep -q закрывает pipe рано, ffmpeg получает SIGPIPE (141), pipefail решает,
# что команда упала. Поэтому сначала читаем весь вывод.
FF_ENCODERS=$(ffmpeg -hide_banner -encoders 2>/dev/null || true)
echo "$FF_ENCODERS" | grep -q hevc_nvenc || {
  echo "ОШИБКА: hevc_nvenc недоступен"; exit 1; }

echo "==> Пресейкал $INPUT ($CODEC) → ${TARGET_W}x${TARGET_H} → $OUTPUT"
# -noautorotate: если видео повернуто метаданными — сначала проверить!
ffmpeg -y -loglevel error $HWDEC -i "$INPUT" \
  -vf "scale=${TARGET_W}:${TARGET_H}:flags=lanczos" \
  -c:v hevc_nvenc -preset p4 -rc vbr -cq 20 -b:v 0 -maxrate 40M -bufsize 80M \
  -pix_fmt p010le -profile:v main10 \
  -c:a aac -b:a 192k -ar 48000 \
  -movflags +faststart \
  "$OUTPUT"

echo "==> Готово: $OUTPUT ($(du -h "$OUTPUT" | cut -f1))"
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,codec_name \
  -of default=noprint_wrappers=1 "$OUTPUT"
