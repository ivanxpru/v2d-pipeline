#!/bin/bash
# Entrypoint v2d-pipeline.
# Урок memory/docker-lessons.md: хостинг Qudata монтирует volume ПОВЕРХ /workspace,
# поэтому запечённые в образе веса скрываются пустым volume. Переносим их из
# слоя (/opt/models) в /workspace/models идемпотентно, без перезаписи существующего.
set -euo pipefail

if [ -d /opt/models ]; then
    mkdir -p /workspace/models
    # -n (no-clobber): не трогаем то, что уже есть в volume (например, докачано с S3)
    cp -rn /opt/models/. /workspace/models/ 2>/dev/null || true
    # wget-обрывки (.part / пустые файлы) от прошлых сессий — подчистить, чтобы
    # пустой файл не считался «готовым» весом
    find /workspace/models -type f -size 0 -delete 2>/dev/null || true
fi

mkdir -p /workspace/input /workspace/output /workspace/logs

exec "$@"
