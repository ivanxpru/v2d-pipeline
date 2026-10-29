#!/usr/bin/env python3
"""Проход 2: стерео из готовых карт глубин (DIBR).

Работает с ПОСЛЕДОВАТЕЛЬНОСТЬЮ, а не с отдельными кадрами:
- temporal smoothing depth-карт (экспоненциальное сглаживание + медиана),
  убирает дрожание между кадрами;
- интерполяция depth для кадров, пропущенных stride'ом на проходе 1;
- варпинг левый/правый с inpainting дыр (foreground extension).

Использование:
  python stereo_pass.py <input_video> <depth_dir> <output_dir> [--baseline 2.5]
    [--workers N] [--encoder nvenc|x264] [--no-median]

ПАРАЛЛЕЛИЗМ (--workers): CPU warp/inpaint однопоточные — чанки кадров
обрабатываются параллельными процессами (multiprocessing fork).
Сглаживание depth выполняется ОДИН раз в main (→ depth_dir/smoothed/),
воркеры читают готовые карты и только варпят. Сборка — concat -c:v copy.
КОДИРОВАНИЕ: кадры pipe в ffmpeg stdin (NVENC HEVC main10 или x264) —
без промежуточного mp4v-файла и двойного перекодирования.

ПАМЯТЬ: обработка ПОТОКОВАЯ, со скользящим окном из ~5 depth-карт
(~40 МБ на 1080p) вместо всего стека — на 5014-кадровом видео это
разница между 41 ГБ (OOM-kill) и <1 ГБ.
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from collections import deque

import cv2
import numpy as np


# ---------- Кодер вывода: pipe кадров → ffmpeg (NVENC/x264) ----------
def open_encoder(encoder: str, path: str, w: int, h: int, fps: float,
                 comment: str = ""):
    """Возвращает Popen ffmpeg, пишущего кадры bgr24 в stdin. Без
    промежуточного mp4v-файла — минус один полный рендер и декод."""
    if encoder == "nvenc":
        vcodec = ["-c:v", "hevc_nvenc", "-preset", "p4", "-rc", "vbr",
                  "-cq", "21", "-b:v", "0",
                  "-vf", "format=yuv420p10le", "-pix_fmt", "p010le",
                  "-profile:v", "main10"]
    else:
        vcodec = ["-c:v", "libx264", "-crf", "18", "-preset", "fast",
                  "-pix_fmt", "yuv420p"]
    meta_args = []
    if comment:
        meta_args = ["-metadata", f"comment={comment}",
                     "-metadata", "artist=v2d-pipeline",
                     "-metadata", f"encoder=v2d stereo_pass {encoder}"]
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
           "-r", f"{fps:.6f}", "-i", "-",
           "-an"] + meta_args + vcodec + ["-movflags", "+faststart", path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def run_chunk(args, rng, stereo_dir):
    """Воркер: обработка диапазона кадров [start, end) → warp+inpaint →
    NVENC/x264. Пишет left_<start>.mp4 и right_<start>.mp4."""
    start, end = rng
    tag = f"[{start:05d}]"
    try:
        meta = json.load(open(os.path.join(args.depth_dir, "meta.json")))
        w, h = meta["width"], meta["height"]
        fps, total = meta["fps"], meta["frames_total"]
        gmin, gmax = meta["global_min"], meta["global_max"]

        raw_dir = os.path.join(args.depth_dir, "smoothed")
        key_indices = sorted(
            int(f[:6]) for f in os.listdir(raw_dir) if f.endswith(".npy"))
        # ключи с запасом на 1 кадр назад (temporal inpaint на стыке чанков)
        keys = [k for k in key_indices if start - 1 <= k < end]

        def emit_local():
            prev = None
            for k in keys:
                d = np.clip(
                    (np.load(os.path.join(raw_dir, f"{k:06d}.npy")) - gmin) /
                    (gmax - gmin + 1e-8), 0.0, 1.0).astype(np.float32)
                if prev is None:
                    prev = (k, d)
                    # первая карта может начинаться позже start (медианное
                    # окно сдвигает выход сглаживания) — добираем начало
                    # константной глубиной, иначе рассинхрон на кадре start
                    for t in range(start, k):
                        yield t, d
                    continue
                i0, d0 = prev
                for t in range(max(i0, start), k):
                    wgt = (t - i0) / max(k - i0, 1)
                    yield t, d0 * (1.0 - wgt) + d * wgt
                prev = (k, d)
            if prev is not None:
                for t in range(max(prev[0], start), end):
                    yield t, prev[1]

        ffmpeg_in = ["ffmpeg", "-nostdin", "-v", "error", "-i", args.input,
                     "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        proc = subprocess.Popen(ffmpeg_in, stdout=subprocess.PIPE)

        f_left = os.path.join(stereo_dir, f"left_{start:05d}.mp4")
        f_right = os.path.join(stereo_dir, f"right_{start:05d}.mp4")
        vw_l = open_encoder(args.encoder, f_left, w, h, fps,
                            getattr(args, "comment", ""))
        vw_r = open_encoder(args.encoder, f_right, w, h, fps,
                            getattr(args, "comment", ""))

        xs = np.tile(np.arange(w), (h, 1)).astype(np.float32)
        ys = np.repeat(np.arange(h, dtype=np.float32)[:, None], w, axis=1)
        max_shift = np.float32(w * args.baseline / 100.0)

        depth_iter = emit_local()
        frame_idx = 0
        prev_result = None   # (left, right, hole_l, hole_r) temporal inpaint
        written = 0
        while frame_idx < end:
            raw = proc.stdout.read(w * h * 3)
            if len(raw) < w * h * 3:
                break
            if frame_idx < start:   # кадры до начала чанка — пропустить
                frame_idx += 1
                continue
            try:
                t, depth = next(depth_iter)
            except StopIteration:
                depth = prev_map
                t = frame_idx
            if t != frame_idx:
                print(f"{tag} ! рассинхрон: кадр {frame_idx}, depth {t}")
                break
            prev_map = depth
            frame = np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy()

            # EDGE-AWARE сглаживание: guided filter. ВАЖНО: guide и src
            # оба float32, иначе guidedFilter молча возвращает NaN (см.
            # урок: чёрные прямоугольные зоны в left/right)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY
                                ).astype(np.float32) / 255.0
            if args.gf_radius > 0 and hasattr(cv2, "ximgproc"):
                depth = cv2.ximgproc.guidedFilter(
                    gray, depth, args.gf_radius, 1e-3)
            depth = np.nan_to_num(depth, nan=0.0, posinf=1.0, neginf=0.0)

            # DA2/MiDaS: БОЛЬШЕ = БЛИЖЕ. Конвергенция: вычитаем C —
            # depth < C уходит за экран, depth > C вылетает вперёд.
            # ЗНАК (исправлено 2026-10-28 по свип-тестам на Pico):
            # near в ЛЕВОМ глазу смещается ВПРАВО → left = xs + shift.
            shift = (depth - args.convergence) * max_shift
            if args.fwd:
                # FORWARD-warping с z-буфером: копий краёв не бывает,
                # вместо них щели → onion-peel (см. свип 2026-10-28)
                left, right, hole_l, hole_r = forward_warp(frame, shift)
                if args.hole_dilate > 0:
                    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                        (2 * args.hole_dilate + 1, 2 * args.hole_dilate + 1))
                    hole_l = cv2.dilate(hole_l.astype(np.uint8), kd) > 0
                    hole_r = cv2.dilate(hole_r.astype(np.uint8), kd) > 0
            else:
                map_left_x = xs + shift
                map_right_x = xs - shift
                left = cv2.remap(frame, map_left_x, ys, cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
                right = cv2.remap(frame, map_right_x, ys, cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)

            # Дыры: при --fwd их даёт сам forward-warp (окклюзии + выход
            # за кадр). В backward-режиме — градиент сдвига, как раньше.
            if args.fwd:
                pass  # hole_l/hole_r уже готовы из forward_warp
            else:
                eps = args.hole_eps
                grad = np.abs(np.gradient(shift, axis=1))
                hole_l = (map_left_x < 0) | (map_left_x > w - 1) | (grad > eps)
                hole_r = (map_right_x < 0) | (map_right_x > w - 1) | (grad > eps)

            # Дилатация маски на N px: инпейнт захватывает полупрозрачную
            # кромку растянутых пикселей (остаточный ореол) вокруг дыр.
            if args.hole_dilate > 0:
                k = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (2 * args.hole_dilate + 1, 2 * args.hole_dilate + 1))
                hole_l = cv2.dilate(hole_l.astype(np.uint8), k) > 0
                hole_r = cv2.dilate(hole_r.astype(np.uint8), k) > 0

            if args.fwd:
                # temporal: сначала содержимое из прошлого кадра в дырах,
                # затем onion-peel для новых областей
                if prev_result is not None:
                    pl, pr, phl, phr = prev_result
                    reuse = hole_l & phl
                    left[reuse] = pl[reuse]
                    right[reuse] = pr[reuse]
                left = onion_fill(left, hole_l)
                right = onion_fill(right, hole_r)
            elif prev_result is None:
                left = inpaint_holes(left, hole_l)
                right = inpaint_holes(right, hole_r)
            else:
                pl, pr, phl, phr = prev_result
                left = inpaint_holes(left, hole_l, prev_valid=pl,
                                     prev_holes=phl)
                right = inpaint_holes(right, hole_r, prev_valid=pr,
                                      prev_holes=phr)
            prev_result = (left.copy(), right.copy(), hole_l, hole_r)

            # Soft edges (Real Realm style): линейное затухание крайних
            # полос каждого глаза к чёрному. Мозг не видит жёсткой рамки
            # → стерео-окно не «режет» вылетающие объекты.
            if args.edge_feather > 0:
                f = args.edge_feather
                ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
                fade = np.ones(w, dtype=np.float32)
                fade[:f] = ramp
                fade[-f:] = ramp[::-1]
                fade = fade[None, :]  # broadcast по высоте
                left *= fade
                right *= fade

            vw_l.stdin.write(left.tobytes())
            vw_r.stdin.write(right.tobytes())
            written += 1
            frame_idx += 1
            if written % 500 == 0:
                print(f"{tag} стерео: {written}/{end - start}", flush=True)

        proc.stdout.close(); proc.wait()
        vw_l.stdin.close(); vw_l.wait()
        vw_r.stdin.close(); vw_r.wait()
        rc = max(proc.returncode or 0, vw_l.returncode or 0,
                 vw_r.returncode or 0)
        if written != end - start:
            sys.exit(f"{tag} ОШИБКА: {written} из {end - start} кадров, "
                     f"ffmpeg rc={rc}")
        print(f"{tag} готово: {written} кадров", flush=True)
    except SystemExit:
        raise
    except Exception as e:
        print(f"{tag} ОШИБКА: {e}", flush=True)
        sys.exit(1)


def onion_fill(img: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Послойное заполнение дыры (см. свип 2026-10-28): каждый слой —
    среднее известных соседей 3x3; фон «продолжается» без затягивания
    чужой текстуры."""
    if not hole.any():
        return img
    out = img.astype(np.float32)
    known = ~hole
    yy, xx = np.where(hole)
    y0, y1, x0, x1 = yy.min(), yy.max() + 1, xx.min(), xx.max() + 1
    k = known[y0:y1, x0:x1].copy()
    o = out[y0:y1, x0:x1]
    kern = np.ones((3, 3), np.float32) / 9.0
    while not k.all():
        cnt = cv2.filter2D(k.astype(np.float32), -1, kern)
        acc = cv2.filter2D(o * k[..., None], -1, kern)
        border = (~k) & (cnt > 0)
        o[border] = (acc[border] / cnt[border, None])
        k |= border
    out[y0:y1, x0:x1] = o
    return out.astype(np.uint8)


def forward_warp(frame: np.ndarray, shift: np.ndarray):
    """Forward-warping с z-буфером (см. свип 2026-10-28): пиксель источника
    летит в target=x±shift, ближние побеждают. Копий краёв не бывает;
    возвращает (left, right, hole_l, hole_r). Знак: left = xs+shift."""
    h, w = shift.shape
    xi = np.tile(np.arange(w, dtype=np.int32), (h, 1))
    yi = np.repeat(np.arange(h, dtype=np.int32)[:, None], w, axis=1)
    s = shift.ravel()
    xif = xi.ravel(); yif = yi.ravel()
    txl = np.clip(xif + np.rint(s).astype(np.int32), 0, w - 1)
    txr = np.clip(xif - np.rint(s).astype(np.int32), 0, w - 1)
    out_l = np.zeros_like(frame); out_r = np.zeros_like(frame)
    hole_l = np.ones((h, w), bool); hole_r = np.ones((h, w), bool)
    ordl = np.lexsort((txl, s))       # ближние (большие s) пишутся последними
    out_l[yif[ordl], txl[ordl]] = frame[yif[ordl], xif[ordl]]
    hole_l[yif[ordl], txl[ordl]] = False
    ordr = np.lexsort((txr, -s))
    out_r[yif[ordr], txr[ordr]] = frame[yif[ordr], xif[ordr]]
    hole_r[yif[ordr], txr[ordr]] = False
    return out_l, out_r, hole_l, hole_r


def inpaint_holes(img: np.ndarray, hole_mask: np.ndarray,
                  prev_valid: np.ndarray | None = None,
                  prev_holes: np.ndarray | None = None) -> np.ndarray:
    """Заливка дыр после варпа: СНАЧАЛА пиксели из предыдущего кадра
    (temporal propagation — фон не «плывёт», т.к. переиспользуем уже
    согласованный во времени результат), остаток — TELEA в bbox."""
    if not hole_mask.any():
        return img
    ys, xs = np.where(hole_mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    msub = hole_mask[y0:y1, x0:x1].astype(np.uint8) * 255
    sub = img[y0:y1, x0:x1]
    if prev_valid is not None:
        pv = prev_valid[y0:y1, x0:x1]
        # переносим только туда, где в прошлом кадре есть валидные пиксели
        sub[msub > 0] = pv[msub > 0]
    else:
        img[y0:y1, x0:x1] = cv2.inpaint(sub, msub, 5, cv2.INPAINT_TELEA)
        return img
    # остаточные дыры (были дырами и в прошлом кадре) — TELEA
    rest = msub.copy()
    if prev_holes is not None:
        rest[prev_holes[y0:y1, x0:x1] & (msub > 0)] = 255
    if (rest > 0).any():
        img[y0:y1, x0:x1] = cv2.inpaint(img[y0:y1, x0:x1], rest, 5,
                                        cv2.INPAINT_TELEA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("depth_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--baseline", type=float, default=2.5,
                    help="сила стерео-сдвига, %% ширины кадра")
    ap.add_argument("--convergence", type=float, default=0.0,
                    help="глубина плоскости экрана 0..1 (0 = фон в экране, "
                         "всё вылетает вперёд; 0.2-0.3 = объекты ближе этой "
                         "глубины вылетают из экрана, дальние уходят за него)")
    ap.add_argument("--edge-feather", type=int, default=0,
                    help="размытие крайних N px каждого глаза (стиль Real "
                         "Realm: мягкая рамка скрывает стерео-окно, объекты "
                         "могут вылетать без дискомфорта; ~5%% ширины)")
    ap.add_argument("--gf-radius", type=int, default=8,
                    help="радиус guidedFilter по кадру: больше = стабильнее, "
                         "меньше = детальнее лица (0 = выключить фильтр)")
    ap.add_argument("--hole-eps", type=float, default=2.0,
                    help="порог градиента сдвига для дыр: ниже = агрессивнее "
                         "инпейнт краёв (убирает ореол-дублирование, 1.0-1.5 "
                         "рекомендуется при заметном гало)")
    ap.add_argument("--hole-dilate", type=int, default=0,
                    help="дилатация hole-маски на N px: инпейнт захватывает "
                         "кромку растянутых пикселей (1-2 при остаточном "
                         "ореоле после снижения hole-eps)")
    ap.add_argument("--fwd", action="store_true",
                    help="forward-warping с z-буфером + onion-peel (конфиг "
                         "2026-10-28: убирает двоение краёв; рекомендуем: "
                         "--gf-radius 2 --baseline 1.5)")
    ap.add_argument("--alpha", type=float, default=0.6,
                    help="вес текущего кадра в экспоненциальном сглаживании")
    ap.add_argument("--no-median", action="store_true")
    ap.add_argument("--workers", type=int, default=1,
                    help="N параллельных процессов warp/inpaint по чанкам "
                         "кадров (CPU-этап однопоточен; 0 = все ядра)")
    ap.add_argument("--encoder", choices=["nvenc", "x264"], default="nvenc",
                    help="NVENC ~в 10 раз быстрее; x264 — fallback без GPU")
    args = ap.parse_args()

    with open(os.path.join(args.depth_dir, "meta.json")) as f:
        meta = json.load(f)

    # метаданные о пресете в MP4 (udta ©cmt) — видны в ffprobe/MediaInfo/VLC
    args.comment = (
        f"v2d stereo | model={os.path.basename(meta.get('model', 'unknown'))} "
        f"depth-res={meta.get('depth_res', 'auto')} stride={meta['stride']} "
        f"| baseline={args.baseline}% convergence={args.convergence} "
        f"gf-radius={args.gf_radius} hole-eps={args.hole_eps} "
        f"hole-dilate={args.hole_dilate} fwd={'on' if args.fwd else 'off'} "
        f"| src={os.path.basename(args.input)}")
    w, h = meta["width"], meta["height"]
    fps, total = meta["fps"], meta["frames_total"]
    stride = meta["stride"]
    gmin, gmax = meta["global_min"], meta["global_max"]
    print(f"Видео: {w}x{h} @ {fps:.3f}, {total} кадров; stride={stride}")

    raw_dir = os.path.join(args.depth_dir, "raw")
    key_indices = sorted(
        int(f[:6]) for f in os.listdir(raw_dir) if f.endswith(".npy"))
    n_keys = len(key_indices)
    print(f"Ключевых depth-карт: {n_keys} (потоковая обработка)")

    alpha = args.alpha
    median_k = 1 if args.no_median else 5
    half = median_k // 2

    # ---------- Шаг 1 (однократно): временное сглаживание ключевых карт ----
    # exp-сглаживание каузально + медиана окном 2*half+1 с edge-паддингом.
    # Результат пишется в depth_dir/smoothed/ — воркеры грузят готовые
    # карты, не повторяя сглаживание (оно быстрое, но даёт одинаковую
    # каузальную историю для всех чанков).
    smooth_dir = os.path.join(args.depth_dir, "smoothed")
    os.makedirs(smooth_dir, exist_ok=True)
    if len(os.listdir(smooth_dir)) >= n_keys:
        print(f"Сглаженные карты уже есть: {smooth_dir} ({n_keys})")
    else:
        window = deque(maxlen=2 * half + 1)
        acc = None
        yielded = []
        for pos, idx in enumerate(key_indices):
            d = np.clip(
                (np.load(os.path.join(raw_dir, f"{idx:06d}.npy")) - gmin) /
                (gmax - gmin + 1e-8), 0.0, 1.0).astype(np.float32)
            acc = d.copy() if acc is None else alpha * d + (1.0 - alpha) * acc
            window.append(acc)
            if len(window) == window.maxlen:
                # медиана окна; центр окна = pos - half
                out_idx = idx - half
                np.save(os.path.join(smooth_dir, f"{out_idx:06d}.npy"),
                        np.median(np.stack(window), axis=0).astype(np.float16))
                yielded.append(out_idx)
            if pos % 500 == 0:
                print(f"  сглаживание: {pos + 1}/{n_keys}", flush=True)
        # хвост: окна выходят за конец — edge-pad
        buf = list(window)
        base = n_keys - len(buf)  # ключ, соответствующий buf[0]
        for k in range(half, 0, -1):
            centre = n_keys - k
            sel = []
            for t in range(centre - half, centre + half + 1):
                j = min(max(t - base, 0), len(buf) - 1)  # edge-pad
                sel.append(buf[j])
            out_idx = key_indices[centre]
            np.save(os.path.join(smooth_dir, f"{out_idx:06d}.npy"),
                    np.median(np.stack(sel), axis=0).astype(np.float16))
            yielded.append(out_idx)
        print(f"  сглаживание: {len(yielded)} карт → {smooth_dir}")

    # ---------- Шаг 2: варпинг — чанки кадров в параллельных воркерах ------
    stereo_dir = os.path.join(args.output_dir, "stereo")
    os.makedirs(stereo_dir, exist_ok=True)

    workers = args.workers if args.workers > 0 else os.cpu_count() or 1
    workers = max(1, min(workers, total))  # не больше, чем кадров
    chunk_len = (total + workers - 1) // workers
    ranges = [(s, min(s + chunk_len, total))
              for s in range(0, total, chunk_len)]
    print(f"Воркеров: {workers}, чанков: {len(ranges)}")

    if workers == 1:
        run_chunk(args, ranges[0], stereo_dir)
    else:
        ctx = mp.get_context("fork")   # args копируются форком — дёшево
        procs = []
        todo = list(ranges)
        free = min(workers, len(todo))
        for _ in range(free):
            p = ctx.Process(target=run_chunk, args=(args, todo.pop(0),
                                                    stereo_dir))
            p.start(); procs.append(p)
        failed = []
        while todo or procs:
            alive = False
            for i, p in enumerate(procs):
                if p.is_alive():
                    alive = True
                    continue
                p.join()
                if p.exitcode != 0:
                    failed.append(i)
                procs[i] = None
            procs = [p for p in procs if p is not None]
            if todo and len(procs) < free:
                p = ctx.Process(target=run_chunk,
                                args=(args, todo.pop(0), stereo_dir))
                p.start(); procs.append(p)
            elif alive:
                time.sleep(0.5)
            elif not todo:
                break
        if failed:
            sys.exit(f"ОШИБКА: упало чанков: {len(failed)}")

    # ---------- Шаг 3: сборка чанков + аудио ----------
    has_audio = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", args.input],
        capture_output=True, text=True).stdout.strip() == "audio"

    for side in ("left", "right"):
        parts = sorted(glob.glob(os.path.join(stereo_dir, f"{side}_*.mp4")))
        if not parts:
            sys.exit(f"ОШИБКА: чанки {side} не найдены — воркер упал?")
        if len(parts) == 1:
            shutil.move(parts[0], os.path.join(stereo_dir, f"{side}.mp4"))
        else:
            concat = os.path.join(stereo_dir, f"concat_{side}.txt")
            with open(concat, "w") as f:
                for p in parts:
                    f.write(f"file '{os.path.abspath(p)}'\n")
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat",
                   "-safe", "0", "-i", concat, "-c:v", "copy",
                   "-movflags", "+faststart",
                   os.path.join(stereo_dir, f"{side}.mp4")]
            subprocess.run(cmd, check=True)
            os.remove(concat)
            for p in parts:
                os.remove(p)
        # аудио приклеиваем одним mux к собранному файлу (copy видео)
        if has_audio:
            tmp = os.path.join(stereo_dir, f"{side}_mux.mp4")
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-i", os.path.join(stereo_dir, f"{side}.mp4"),
                 "-i", args.input,
                 "-map", "0:v:0", "-map", "1:a:0",
                 "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                 "-movflags", "+faststart", tmp], check=True)
            os.replace(tmp, os.path.join(stereo_dir, f"{side}.mp4"))

    # ---------- Шаг 4: SBS — целевой файл для Pico ----------
    # Видео чанков уже закодировано NVENC/x264 воркерами; здесь один
    # hstack-проход. Pico 4 Ultra играет HEVC main10 → кодируем в него.
    sbs_cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-i", os.path.join(stereo_dir, "left.mp4"),
               "-i", os.path.join(stereo_dir, "right.mp4"),
               "-filter_complex", "[0:v][1:v]hstack"]
    if args.encoder == "nvenc":
        sbs_cmd += ["-c:v", "hevc_nvenc", "-preset", "p4", "-rc", "vbr",
                    "-cq", "23", "-b:v", "0", "-pix_fmt", "p010le",
                    "-profile:v", "main10"]
    else:
        sbs_cmd += ["-c:v", "libx264", "-crf", "23", "-preset", "medium",
                    "-pix_fmt", "yuv420p"]
    if has_audio:
        sbs_cmd += ["-map", "0:a:0", "-c:a", "copy"]
    sbs_cmd += ["-movflags", "+faststart",
                os.path.join(stereo_dir, "side_by_side_preview.mp4")]
    subprocess.run(sbs_cmd, check=True)

    print(f"Готово: {stereo_dir}/{{left,right}}.mp4"
          + (" (+аудио)" if has_audio else ""))


if __name__ == "__main__":
    main()
