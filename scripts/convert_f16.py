#!/usr/bin/env python3
"""Однократная конвертация raw depth-карт: float32 → float16.

Каждая карта нормализуется в [0, 1] по СВОЕМУ локальному min/max перед
приведением к float16 — иначе точности f16 (диапазон ~±65504) может не
хватить для карт с большим динамическим диапазоном. Относительная
глубина сохраняется, метрики карт не меняются.

Запуск НА СЕРВЕРЕ:
  /opt/miniconda3/envs/v2d/bin/python scripts/convert_f16.py
"""
import os
import sys

import numpy as np

RAW_DIR = "/workspace/output/depth/raw"


def main():
    files = sorted(
        os.path.join(RAW_DIR, f) for f in os.listdir(RAW_DIR)
        if f.endswith(".npy"))
    done = 0
    for i, path in enumerate(files):
        try:
            d = np.load(path)
        except Exception as e:
            print(f"SKIP {path}: {e}", flush=True)
            continue
        if d.dtype != np.float32:
            done += 1
            continue
        mn, mx = float(d.min()), float(d.max())
        np.save(path, ((d - mn) / (mx - mn + 1e-8)).astype(np.float16))
        done += 1
        if done % 200 == 0:
            print(f"{done}/{len(files)}", flush=True)
    print(f"Готово: {done}/{len(files)} карт в float16", flush=True)


if __name__ == "__main__":
    sys.exit(main())
