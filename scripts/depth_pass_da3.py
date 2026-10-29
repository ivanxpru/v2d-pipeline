#!/usr/bin/env python3
"""Проход 1 для DA3: карта глубин для видео через Depth Anything 3 (Streaming/Batch).

Сохраняет:
- output_dir/raw/<frame_idx:06d>.npy (float16 или float32)
- output_dir/meta.json (совместим с stereo_pass.py)
"""
import argparse
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import safetensors.torch
import torch
from depth_anything_3.api import DepthAnything3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="входное видео")
    ap.add_argument("output_dir", help="куда сохранять карты глубины")
    ap.add_argument("--model-weights",
                    default="/workspace/models/da3mono-large/model.safetensors")
    ap.add_argument("--model-name", default="da3mono-large")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--process-res", type=int, default=504,
                    help="длинная сторона для DA3 (504 дефолт, 1008 для 2K)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--f16", action="store_true", default=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    raw_dir = os.path.join(args.output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # Параметры видео через ffprobe
    probe = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
        "-of", "json", args.input
    ])
    info = json.loads(probe)["streams"][0]
    w, h = int(info["width"]), int(info["height"])
    num, den = info["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    nb_frames = int(info.get("nb_frames") or 0)
    print(f"Вход: {w}x{h} @ {fps:.3f} fps, frames ~{nb_frames}")

    print(f"Загрузка модели DA3 ({args.model_name})...")
    model = DepthAnything3(model_name=args.model_name)
    weights = safetensors.torch.load_file(args.model_weights)
    model.load_state_dict(weights, strict=False)
    model = model.to("cuda").eval()
    print("Модель готова на GPU.")

    # Декод кадров через ffmpeg pipe
    ffmpeg_cmd = [
        "ffmpeg", "-nostdin", "-v", "error",
        "-i", args.input,
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, bufsize=10**8)
    frame_size = w * h * 3

    batch_frames = []
    batch_indices = []
    raw_maps = []
    frame_idx = 0
    t0 = time.time()

    def process_batch(frames, indices):
        nonlocal raw_maps
        if not frames:
            return
        # model.inference принимает список numpy RGB кадров
        # process_res задает длинную сторону (504 или 1008)
        pred = model.inference(frames, process_res=args.process_res)
        # pred.depth: shape (N, H_proc, W_proc), где меньшее число = ближе к камере (метры/дистанция)
        # Инвертируем: в DIBR пайплайне БОЛЬШЕ = БЛИЖЕ (диспаритет).
        # Нормализуем по батчу / пер-карте в единый диапазон
        depths_proc = pred.depth  # numpy array (N, H', W')
        for i, idx in enumerate(indices):
            d_p = depths_proc[i]
            # ресайз к полному разрешению видео (w, h)
            d_full = cv2.resize(d_p, (w, h), interpolation=cv2.INTER_CUBIC)
            # DA3: дистанция (depth) -> инвертируем в близость/диспаритет
            # disparity ~ 1 / (depth + eps)
            disp = (1.0 / (np.maximum(d_full, 1e-3))).astype(np.float32)
            out_file = os.path.join(raw_dir, f"{idx:06d}.npy")
            if args.f16:
                np.save(out_file, disp.astype(np.float16))
            else:
                np.save(out_file, disp)
            raw_maps.append(idx)

    print("Начало инференса глубины DA3...")
    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break
        if frame_idx % args.stride == 0:
            rgb = np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy()
            batch_frames.append(rgb)
            batch_indices.append(frame_idx)
            if len(batch_frames) >= args.batch_size:
                process_batch(batch_frames, batch_indices)
                batch_frames.clear()
                batch_indices.clear()
                elapsed = time.time() - t0
                speed = len(raw_maps) / max(elapsed, 1e-3)
                print(f"  обработано {len(raw_maps)} кадров ({speed:.2f} fps)...", flush=True)
        frame_idx += 1

    if batch_frames:
        process_batch(batch_frames, batch_indices)

    proc.stdout.close()
    proc.wait()

    # Глобальный min/max для метаданных
    print("Расчёт глобальных min/max для метаданных...")
    gmin, gmax = float("inf"), float("-inf")
    for idx in raw_maps:
        d = np.load(os.path.join(raw_dir, f"{idx:06d}.npy")).astype(np.float32)
        mn, mx = float(d.min()), float(d.max())
        if mn < gmin:
            gmin = mn
        if mx > gmax:
            gmax = mx

    meta = {
        "width": w,
        "height": h,
        "fps": fps,
        "frames_total": frame_idx,
        "stride": args.stride,
        "model": args.model_name,
        "process_res": args.process_res,
        "global_min": gmin,
        "global_max": gmax,
        "f16": args.f16
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    total_time = time.time() - t0
    print(f"Готово! Обработано {len(raw_maps)}/{frame_idx} кадров за {total_time:.1f}с ({len(raw_maps)/total_time:.2f} fps)")
    print(f"Meta: global_min={gmin:.4f}, global_max={gmax:.4f}")


if __name__ == "__main__":
    main()
