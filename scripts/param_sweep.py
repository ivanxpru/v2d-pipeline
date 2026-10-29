#!/usr/bin/env python3
"""Свип параметров стерео на одном стоп-кадре.

Берёт кадр (jpg/png), считает (или берёт готовую) карту глубины, прогоняет
ЧЕРЕЗ ТУ ЖЕ математику, что stereo_pass.py (guidedFilter → convergence-shift →
remap → hole-mask → inpaint → SBS), для набора пресетов от лайтового до
ультра. Каждый вариант — отдельный файл data/test/sweep/<name>.jpg.

Использование (на сервере, env v2d):
  python scripts/param_sweep.py data/test/frames/frame_mid_2k.jpg \
      [--depth <npy>]            # готовая карта (float32, HxW, больше=ближе)
      [--preset light|medium|strong|ultra|all]   # default: all
      [--out data/test/sweep]
      [--depth-res 518]          # разрешение инференса Depth Anything V2
                                 # (кратно 14); 0 = родное разрешение кадра

Пресеты (baseline % ширины, convergence, gf-radius, hole-eps, hole-dilate):
  light    1.5  0.50  8  2.0  0   — без вылета, лёгкий объём
  medium   2.5  0.40  8  2.0  0   — умеренный объём, объекты у плоскости
  strong   4.0  0.25  8  1.5  1   — рабочий конфиг Gary Moore v2
  ultra    5.5  0.15  6  1.2  2   — максимальный поп-аут (на грани артефактов)
Плюс вариант "source" — исходный кадр для сравнения (не варпится).
"""
import argparse
import json
import os

import cv2
import numpy as np

PRESETS = {
    "light":  dict(baseline=1.5, convergence=0.50, gf_radius=8, hole_eps=2.0, hole_dilate=0),
    "medium": dict(baseline=2.5, convergence=0.40, gf_radius=8, hole_eps=2.0, hole_dilate=0),
    "strong": dict(baseline=4.0, convergence=0.25, gf_radius=8, hole_eps=1.5, hole_dilate=1),
    "ultra":  dict(baseline=5.5, convergence=0.15, gf_radius=6, hole_eps=1.2, hole_dilate=2),
}


def compute_depth(frame: np.ndarray, res: int) -> np.ndarray:
    """Depth Anything V2 → float32 HxW, БОЛЬШЕ = БЛИЖЕ, без нормализации
    0..1 (нормализация не влияет на свип: shift линейна по depth, а
    convergence задан в нормализованных единицах — поэтому приводим к 0..1
    per-frame min-max, для стоп-кадра это эквивалентно global-нормализации
    полного прогона)."""
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if res > 0:
        scale = res / max(h, w)
        nh, nw = max(14, round(h * scale / 14) * 14), max(14, round(w * scale / 14) * 14)
        small = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    else:
        small = rgb

    repo = os.environ.get("DA2_REPO", "depth-anything/Depth-Anything-V2-Large-hf")
    proc = AutoImageProcessor.from_pretrained(repo)
    model = AutoModelForDepthEstimation.from_pretrained(repo).cuda().eval()

    with torch.no_grad():
        inputs = proc(images=small, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(**inputs)
    d = out.predicted_depth[0].float().cpu().numpy()
    d = cv2.resize(d, (w, h), interpolation=cv2.INTER_CUBIC)
    lo, hi = float(d.min()), float(d.max())
    return ((d - lo) / max(hi - lo, 1e-8)).astype(np.float32)


def compute_depth_da3(frame: np.ndarray, res: int):
    """DA3MONO-LARGE через официальный model.inference() (эталонный препроцессинг
    из оригинального репо). Возвращает (depth 0..1, conf HxW float|None).
    Эталонный process_res оригинала = 504 upper_bound_resize; для 2K пробовать 1008."""
    import torch
    from depth_anything_3.api import DepthAnything3

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    net = DepthAnything3(model_name="da3mono-large")
    sd = os.path.join(os.environ.get("DA3_DIR", "/workspace/models/da3mono-large"),
                      "model.safetensors")
    if os.path.exists(sd):
        from safetensors.torch import load_file
        net.load_state_dict(load_file(sd), strict=False)
    net = net.cuda().eval()
    with torch.no_grad():
        pred = net.inference([rgb], process_res=res if res > 0 else 504,
                             process_res_method="upper_bound_resize",
                             export_feat_layers=[])
    d = np.asarray(pred.depth).squeeze().astype(np.float32)
    conf = None
    if getattr(pred, "conf", None) is not None:
        conf = np.asarray(pred.conf).squeeze().astype(np.float32)
        conf = cv2.resize(conf, (w, h), interpolation=cv2.INTER_NEAREST)
    d = cv2.resize(d, (w, h), interpolation=cv2.INTER_CUBIC)
    lo, hi = float(d.min()), float(d.max())
    # DA3MONO предсказывает depth напрямую (не disparity); если карта
    # инвертирована (фон светлее объекта) — смотреть depthvis и флагать вручную
    return ((d - lo) / max(hi - lo, 1e-8)).astype(np.float32), conf


def harden_depth(d: np.ndarray, k: float) -> np.ndarray:
    """S-кривая: давит мягкий ореол вокруг объектов (k=0 off, 1..3 сильнее)."""
    if k <= 0:
        return d
    x = np.clip(d, 0, 1)
    return (x + (x * x * (3 - 2 * x) - x) * min(k, 3) / 3).astype(np.float32)


def quantize_depth(d: np.ndarray, n: int) -> np.ndarray:
    """Слоёная глубина как у Pico-конвертера: N плоских уровней со ступенями.
    Убирает пандусы — источник промежуточного сдвига и двоения краёв."""
    if n <= 0:
        return d
    return (np.round(d * n) / n).astype(np.float32)


def snap_depth(d: np.ndarray, k: int) -> np.ndarray:
    """Убрать переходный пандус на границах объектов: max-pool по окну k —
    граничные пиксели принимают значение БЛИЖНЕЙ стороны. Пандус и есть
    источник двоения краёв (глаз на границе лица). k=0 → off."""
    if k <= 0:
        return d
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * k + 1, 2 * k + 1))
    return np.maximum(d, cv2.dilate(d, ker)).astype(np.float32)


def forward_warp(frame: np.ndarray, shift: np.ndarray):
    """Forward-warping с z-буфером: пиксель источника летит в target=x+shift,
    БЛИЖНИЕ (больший |shift|) побеждают. Копий краёв не бывает по построению;
    вместо них щели (hole) — их закрывает inpaint.
    Возвращает (left, right, hole_l, hole_r). Знак: left = xs+shift,
    right = xs-shift (проверенный swaptest-знак: near в левом глазу вправо)."""
    h, w = shift.shape
    xs = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    ys = np.repeat(np.arange(h, dtype=np.float32)[:, None], w, axis=1)
    out_l = np.zeros_like(frame); out_r = np.zeros_like(frame)
    zbuf_l = np.full((h, w), -1e9, np.float32)
    zbuf_r = np.full((h, w), -1e9, np.float32)
    hole_l = np.ones((h, w), bool); hole_r = np.ones((h, w), bool)
    s = shift.ravel(); order = np.argsort(s)  # дальние (меньшие) первыми
    xi = xs.ravel().astype(np.int32); yi = ys.ravel().astype(np.int32)
    flat_s = s
    # left: target = x + s (ближние дальше вправо)
    txl = np.clip(xi + np.rint(flat_s).astype(np.int32), 0, w - 1)
    # right: target = x - s
    txr = np.clip(xi - np.rint(flat_s).astype(np.int32), 0, w - 1)
    yf = yi
    for idx in order:  # цикл по пикселям недопустимо медленен — векторизуем:
        pass
    # векторная версия: сортировка по shift, присвоение «последним пишется»
    ordl = np.lexsort((txl, s))   # возрастание s → ближние пишутся последними
    out_l[yf[ordl], txl[ordl]] = frame[yf[ordl], xi[ordl]]
    zbuf_l[:] = 1; hole_l[yf[ordl], txl[ordl]] = False
    ordr = np.lexsort((txr, -s))  # убывание s → ближние пишутся последними
    out_r[yf[ordr], txr[ordr]] = frame[yf[ordr], xi[ordr]]
    zbuf_r[:] = 1; hole_r[yf[ordr], txr[ordr]] = False
    return out_l, out_r, hole_l, hole_r


def warp_stereo(frame: np.ndarray, depth: np.ndarray, p: dict, conf: np.ndarray = None):
    """Точная копия математики stereo_pass.py (без temporal-части).
    p['hole_grad_before_snap']=1 → маска дыр считается по ДО-snap градиенту
    (snap не плодит лишних дыр, закрываются только реальные окклюзии)."""
    p.setdefault("inpaint_method", "telea")
    p.setdefault("hole_grad_before_snap", 0)
    p.setdefault("snap", 0)
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    d = depth
    if p["gf_radius"] > 0 and hasattr(cv2, "ximgproc"):
        d = cv2.ximgproc.guidedFilter(gray, d, p["gf_radius"], 1e-3)
    d = np.nan_to_num(d, nan=0.0, posinf=1.0, neginf=0.0)
    d = quantize_depth(d, p.get("quantize", 0))
    # маска дыр по ДО-snap градиенту: snap не плодит лишние дыры
    if p["hole_grad_before_snap"]:
        shift_pre = (d - np.float32(p["convergence"])) * np.float32(w * p["baseline"] / 100.0)
        grad_pre = np.abs(np.gradient(shift_pre, axis=1))
        d = snap_depth(d, p["snap"])

    xs = np.tile(np.arange(w), (h, 1)).astype(np.float32)
    ys = np.repeat(np.arange(h, dtype=np.float32)[:, None], w, axis=1)
    max_shift = np.float32(w * p["baseline"] / 100.0)
    shift = (d - np.float32(p["convergence"])) * max_shift
    map_lx = xs - shift
    map_rx = xs + shift
    left = cv2.remap(frame, map_lx, ys, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)
    right = cv2.remap(frame, map_rx, ys, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REPLICATE)

    eps = p["hole_eps"]
    if p.get("fwd"):
        left, right, hole_l, hole_r = forward_warp(frame, shift)
        if p["hole_dilate"] > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * p["hole_dilate"] + 1, 2 * p["hole_dilate"] + 1))
            hole_l = cv2.dilate(hole_l.astype(np.uint8), k) > 0
            hole_r = cv2.dilate(hole_r.astype(np.uint8), k) > 0
    else:
        grad = grad_pre if p["hole_grad_before_snap"] else np.abs(np.gradient(shift, axis=1))
        hole_l = (map_lx < 0) | (map_lx > w - 1) | (grad > eps)
        hole_r = (map_rx < 0) | (map_rx > w - 1) | (grad > eps)
        if p["hole_dilate"] > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * p["hole_dilate"] + 1, 2 * p["hole_dilate"] + 1))
            hole_l = cv2.dilate(hole_l.astype(np.uint8), k) > 0
            hole_r = cv2.dilate(hole_r.astype(np.uint8), k) > 0
        left = inpaint(left, hole_l, p["inpaint_method"])
        right = inpaint(right, hole_r, p["inpaint_method"])
    if conf is not None:
        # эталонный механизм DA3: низкая уверенность → дыра (маскирует ореол)
        cmask = conf < p["conf_pct"]
        hole_l |= cmask
        hole_r |= cmask
    if p.get("fwd"):
        fill = onion_fill if p.get("hole_method") == "onion" else (lambda i, h: inpaint(i, h, p["inpaint_method"]))
        left = fill(left, hole_l)
        right = fill(right, hole_r)

    sbs = np.concatenate([left, right], axis=1)
    stats = (int(hole_l.sum()), 100 * hole_l.mean(),
             int(hole_r.sum()), 100 * hole_r.mean())
    return sbs, stats


def onion_fill(img: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Послойное заполнение дыры: каждый слой — среднее известных соседей 3x3.
    Фон «продолжается» в дыру без затягивания чужой текстуры (лечит шлейф
    у краёв при больших окклюзионных полосах forward-warp)."""
    if not hole.any():
        return img
    out = img.astype(np.float32)
    known = ~hole
    # работаем только в bbox дыры
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


def inpaint(img: np.ndarray, hole: np.ndarray, method: str = "telea") -> np.ndarray:
    """Первый кадр → inpaint в bbox. method: telea (дефолт) | ns."""
    if not hole.any():
        return img
    yy, xx = np.where(hole)
    y0, y1, x0, x1 = yy.min(), yy.max() + 1, xx.min(), xx.max() + 1
    m = (hole[y0:y1, x0:x1].astype(np.uint8)) * 255
    mcv = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    img[y0:y1, x0:x1] = cv2.inpaint(img[y0:y1, x0:x1], m, 3, mcv)
    return img


def label(img: np.ndarray, text: str) -> np.ndarray:
    """Текст в центре ПОЛНОГО SBS-изображения: каждый глаз видит свою половину
    надписи, при сведении в VR складывается в целую строку в центре."""
    out = img.copy()
    h, w2 = out.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    x, y = (w2 - tw) // 2, (h + th) // 2
    cv2.rectangle(out, (x - 12, y - th - 12), (x + tw + 12, y + 12), (0, 0, 0), -1)
    cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frame")
    ap.add_argument("--depth", help="готовая .npy карта (HxW float32, больше=ближе); "
                                    "иначе инференс Depth Anything V2")
    ap.add_argument("--preset", default="all", help="light|medium|strong|ultra|all")
    ap.add_argument("--out", default="data/test/sweep")
    ap.add_argument("--depth-res", type=int, default=518,
                    help="разрешение инференса DA2, кратно 14 (0 = родное)")
    ap.add_argument("--model", default="da2", choices=["da2", "da3"],
                    help="модель глубины: DA2-Large (transformers) или DA3-mono")
    ap.add_argument("--harden", type=float, default=0.0,
                    help="S-кривая глубины против ореола (0=off, 1..3)")
    ap.add_argument("--tag", default="",
                    help="суффикс имени файла (иначе auto)")
    ap.add_argument("--conf-mask", type=float, default=0.0,
                    help="порог уверенности DA3 0..1: ниже → дыра (0=off, напр. 0.3)")
    ap.add_argument("--snap", type=int, default=0,
                    help="резкий край: max-pool глубины по окну k (0=off, 1..3)")
    ap.add_argument("--inpaint", default="telea", choices=["telea", "ns"],
                    help="метод inpaint дыр")
    ap.add_argument("--gf-radius", type=int, default=0,
                    help="переопределить радиус guidedFilter пресета (0 = как в пресете)")
    ap.add_argument("--baseline", type=float, default=0,
                    help="переопределить baseline пресета в % ширины (0 = как в пресете)")
    ap.add_argument("--swap-eyes", action="store_true",
                    help="поменять L/R местами (тест знака параллакса/псевдостерео)")
    ap.add_argument("--hole-pre-snap", action="store_true",
                    help="маску дыр считать по до-snap градиенту (меньше лишних дыр)")
    ap.add_argument("--quantize", type=int, default=0,
                    help="слои: квантовать глубину на N уровней (0=off, 6..16)")
    ap.add_argument("--fwd", action="store_true",
                    help="forward-warping с z-буфером (без копий краёв, дыры инпейнтятся)")
    ap.add_argument("--hole-fill", default="inpaint", choices=["inpaint", "onion"],
                    help="заполнение дыр при --fwd: inpaint (TELEA/NS) или onion-peel")
    args = ap.parse_args()

    frame = cv2.imread(args.frame)
    if frame is None:
        raise SystemExit(f"не читается кадр: {args.frame}")
    h, w = frame.shape[:2]
    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.frame))[0]

    if args.depth:
        depth = np.load(args.depth).astype(np.float32)
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)
        lo, hi = float(depth.min()), float(depth.max())
        depth = ((depth - lo) / max(hi - lo, 1e-8)).astype(np.float32)
        print(f"depth: {args.depth} → 0..1 [{lo:.4f}..{hi:.4f}]")
    else:
        conf = None
        print(f"инференс {args.model.upper()} ...")
        if args.model == "da2":
            depth = compute_depth(frame, args.depth_res)
        else:
            depth, conf = compute_depth_da3(frame, args.depth_res)
        depth = harden_depth(depth, args.harden)
        tag0 = args.tag or f"{args.model}{'_h' + str(args.harden) if args.harden else ''}"
        np.save(os.path.join(args.out, f"{stem}_depth_{tag0}.npy"), depth)
        cv2.imwrite(os.path.join(args.out, f"{stem}_depthvis_{tag0}.jpg"),
                    np.clip(depth * 255, 0, 255).astype(np.uint8))
        if conf is not None:
            np.save(os.path.join(args.out, f"{stem}_conf_{tag0}.npy"), conf)
            cv2.imwrite(os.path.join(args.out, f"{stem}_confvis_{tag0}.jpg"),
                        np.clip(conf * 255, 0, 255).astype(np.uint8))
        print("depth сохранён (npy + vis)")

    tag0 = args.tag or ("da2" if not args.depth else "custom")
    names = list(PRESETS) if args.preset == "all" else [args.preset]
    cv2.imwrite(os.path.join(args.out, f"{stem}_00_source.jpg"),
                label(frame, "SOURCE (original)"))

    print(f"\n{'пресет':8} {'baseline':>8} {'conv':>5} {'gf':>3} "
          f"{'eps':>4} {'dil':>3} | {'дыры L':>12} | {'дыры R':>12}")
    manifest = []
    for i, name in enumerate(names, 1):
        p = dict(PRESETS[name])
        if args.gf_radius:
            p["gf_radius"] = args.gf_radius
        if args.baseline:
            p["baseline"] = args.baseline
        p["conf_pct"] = args.conf_mask
        p["snap"] = args.snap
        p["inpaint_method"] = args.inpaint
        p["quantize"] = args.quantize
        p["fwd"] = bool(args.fwd)
        p["hole_method"] = args.hole_fill
        p["hole_grad_before_snap"] = 1 if args.hole_pre_snap else 0
        sbs, st = warp_stereo(frame, depth, p, conf if args.conf_mask > 0 else None)
        if args.swap_eyes:
            w2 = sbs.shape[1] // 2
            sbs = np.concatenate([sbs[:, w2:], sbs[:, :w2]], axis=1)
        tag = (f"{name}  bl={p['baseline']}% conv={p['convergence']} "
               f"gf={p['gf_radius']} eps={p['hole_eps']} dil={p['hole_dilate']}")
        fname = f"{stem}_{tag0}_{i:02d}_{name}.jpg"
        # подпись = имя файла в центре полного SBS (в VR сложится в целую строку);
        # тег настроек — в манифесте
        cv2.imwrite(os.path.join(args.out, fname),
                    label(sbs, fname), [cv2.IMWRITE_JPEG_QUALITY, 92])
        pm = {k: v for k, v in p.items() if k not in ("conf_pct", "snap")}
        manifest.append(dict(file=fname, model=args.model, depth=args.depth or "infer",
                             depth_res=args.depth_res, harden=args.harden,
                             conf_mask=args.conf_mask, snap=args.snap,
                             inpaint=args.inpaint, hole_fill=args.hole_fill,
                             hole_pre_snap=args.hole_pre_snap,
                             preset=name, **pm))
        print(f"{name:8} {p['baseline']:>7.1f}% {p['convergence']:>5.2f} "
              f"{p['gf_radius']:>3} {p['hole_eps']:>4.1f} {p['hole_dilate']:>3} | "
              f"{st[0]:>6} ({st[1]:.2f}%) | {st[2]:>6} ({st[3]:.2f}%)")

    mpath = os.path.join(args.out, "manifest.jsonl")
    with open(mpath, "a") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"настройки дописаны: {mpath}")

    print(f"\nГотово: {args.out}/ — смотреть на Pico (SBS, каждый файл = стереопара)")


if __name__ == "__main__":
    main()
