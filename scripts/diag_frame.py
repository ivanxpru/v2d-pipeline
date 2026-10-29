#!/usr/bin/env python
"""Диагностика чёрного прямоугольника на кадре 240: воспроизводим
pipeline stereo_pass по шагам и сохраняем каждую стадию."""
import json
import subprocess
import sys

import cv2
import numpy as np

FRAME = int(sys.argv[1]) if len(sys.argv) > 1 else 240
OUT = "/tmp/diag"
import os
os.makedirs(OUT, exist_ok=True)

meta = json.load(open("/workspace/output/depth/meta.json"))
gmin, gmax = meta["global_min"], meta["global_max"]
print("meta:", gmin, gmax)

# 1) исходный кадр
subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                "-i", "/workspace/output/restored.mp4",
                "-vf", f"select=eq(n\\,{FRAME})",
                "-frames:v", "1", f"{OUT}/src.png"], check=True)
frame = cv2.imread(f"{OUT}/src.png")
h, w = frame.shape[:2]
print("frame:", frame.shape)

# 2) глубина
depth = np.load(f"/workspace/output/depth/raw/{FRAME:06d}.npy").astype(np.float32)
d_hat = (depth - gmin) / (gmax - gmin)
d_hat = np.clip(d_hat, 0.0, 1.0)
print("d_hat min/max/mean:", d_hat.min(), d_hat.max(), d_hat.mean())

# 3) guided filter как в stereo_pass
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
d_gf = cv2.ximgproc.guidedFilter(gray, d_hat, 8, 1e-3)
print("guided delta:", np.abs(d_gf - d_hat).max())

# 4) варп
xs = np.tile(np.arange(w), (h, 1)).astype(np.float32)
ys = np.repeat(np.arange(h, dtype=np.float32)[:, None], w, axis=1)
max_shift = np.float32(w * 2.5 / 100.0)
shift = d_gf * max_shift
map_lx = xs - shift
left = cv2.remap(frame, map_lx, ys, cv2.INTER_LINEAR,
                 borderMode=cv2.BORDER_REPLICATE)
cv2.imwrite(f"{OUT}/warped_pre_inpaint.png", left)

# 5) маска дыр
eps = 2.0
grad = np.abs(np.gradient(shift, axis=1))
hole_l = (map_lx < 0) | (map_lx > w - 1) | (grad > eps)
print("holes:", int(hole_l.sum()), "=", round(100 * hole_l.mean(), 2), "%")
vis = frame.copy()
vis[hole_l] = (0, 0, 255)
cv2.imwrite(f"{OUT}/hole_mask.png", vis)

# статистика градиента: где именно превышение
print("grad>eps rows: min", int(np.where(hole_l.any(axis=1))[0].min()) if hole_l.any() else -1,
      "max", int(np.where(hole_l.any(axis=1))[0].max()) if hole_l.any() else -1)

# 6) разложение маски на компоненты
grad_only = grad > eps
oob_only = (map_lx < 0) | (map_lx > w - 1)
print("grad-only:", int(grad_only.sum()), " oob:", int(oob_only.sum()))
vis2 = frame.copy()
vis2[grad_only] = (0, 0, 255)
vis2[oob_only] = (0, 255, 0)
cv2.imwrite(f"{OUT}/hole_components.png", vis2)

# 7) сдвиг визуально
cv2.imwrite(f"{OUT}/shift_norm.png",
            np.clip(shift / max_shift * 255, 0, 255).astype(np.uint8))

# 8) inpaint как в скрипте (одиночный кадр, без prev)
m = (hole_l.astype(np.uint8)) * 255
if m.any():
    ysa, xsa = np.where(hole_l)
    y0, y1 = ysa.min(), ysa.max() + 1
    x0, x1 = xsa.min(), xsa.max() + 1
    print("hole bbox:", y0, y1, x0, x1)
    msub = m[y0:y1, x0:x1]
    filled = cv2.inpaint(left[y0:y1, x0:x1], msub, 5, cv2.INPAINT_TELEA)
    left[y0:y1, x0:x1] = filled
cv2.imwrite(f"{OUT}/warped_post_inpaint.png", left)
print("done ->", OUT)
