#!/usr/bin/env python3
"""Читает пресет v2d из ©cmt mp4 (см. stereo_pass.py) в json/dict.
Использование: read_preset.py video.mp4 [--json] [--all]"""
import argparse, json, subprocess, sys

def read_tags(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format_tags",
         "-of", "json", path], capture_output=True, text=True).stdout
    return json.loads(out).get("format", {}).get("tags", {})

def parse_comment(cmt):
    if not cmt or not cmt.startswith("v2d stereo"):
        return None
    d = {}
    src = None
    for part in cmt.replace("v2d stereo |", "").split("|"):
        part = part.strip()
        if part.startswith("src="):
            src = part[4:]
            continue
        for kv in part.split():
            if "=" in kv:
                k, v = kv.split("=", 1)
                try:
                    d[k] = float(v) if "." in v else int(v)
                except ValueError:
                    d[k] = v
    d["src"] = src
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    rows = []
    for f in args.files:
        tags = read_tags(f)
        p = parse_comment(tags.get("comment", ""))
        row = {"file": f, "preset": p or "нет v2d-тега",
               "artist": tags.get("artist", "")}
        rows.append(row)
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        for r in rows:
            print(r["file"])
            print("  ", r["preset"])
            print("   artist:", r["artist"])

if __name__ == "__main__":
    main()
