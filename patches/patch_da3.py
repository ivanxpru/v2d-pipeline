#!/usr/bin/env python3
"""Патч depth_anything_3.api: делаем необязательные импорты безопасными.

Боевой пайплайн (depth-проход, DIBR) не использует экспорт 3DGS/COLMAP и
выравнивание поз — их импорты тянут тяжёлые/капризные зависимости (gsplat,
pycolmap, evo). Патч проверен вручную на инстансе 3090 (2026-10-29),
здесь — воспроизведение на этапе сборки.
"""
import importlib.util


def patch(pattern: str, replacement: str) -> None:
    path = importlib.util.find_spec("depth_anything_3.api").origin
    content = open(path).read()
    if replacement in content:
        return  # уже пропатчен
    if pattern not in content:
        raise SystemExit(f"Паттерн не найден в {path}: {pattern[:60]!r}")
    open(path, "w").write(content.replace(pattern, replacement))


patch(
    "from depth_anything_3.utils.export import export",
    "try:\n    from depth_anything_3.utils.export import export\n"
    "except ImportError:\n    export = None",
)
patch(
    "from depth_anything_3.utils.pose_align import align_poses_umeyama",
    "try:\n    from depth_anything_3.utils.pose_align import align_poses_umeyama\n"
    "except ImportError:\n    align_poses_umeyama = None",
)
print("patch_da3: OK")
