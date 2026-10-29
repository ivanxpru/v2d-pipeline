#!/usr/bin/env python3
"""Добавляет Pico-совместимые 3D-боксы в SBS HEVC mp4 БЕЗ перекодирования.

Вставляет в sample entry видео-трека (после hvcC):
  - vexu -> eyes {stri=3(SBS), cams, blin(baseline), cmfy, dadj, proj/prji/rect}
    (байты vexu копируются из референса камеры Pico дословно)
  - hfov (FOV в тысячных градуса)
  - colr nclx
Дополнительно moov/meta (mdta): com.android.manufacturer/model/version —
источник полей "Made"/модели в инфо Pico (можно задать через ffmpeg -metadata
ДО этого скрипта, либо --made).

Использование:
  python3 mp4_add_3d.py input.mp4 output.mp4 [--baseline-mm 64] [--fov-deg 76.35]
Порядок применения с ffmpeg-метаданными: СНАЧАЛА ffmpeg -c copy -metadata ...,
ЗАТЕМ этот скрипт (ffmpeg при ремаксе выбрасывает неизвестные боксы из stsd).
"""
import struct
import sys


def boxes(data, off, end):
    p = off
    res = []
    while p < end - 7:
        size = struct.unpack(">I", data[p:p + 4])[0]
        typ = data[p + 4:p + 8].decode("latin1")
        hdr = 8
        if size == 1:
            size = struct.unpack(">Q", data[p + 8:p + 16])[0]
            hdr = 16
        elif size == 0:
            size = end - p
        res.append((typ, p, size, hdr))
        p += size
    return res


def find_video_hvcc(buf):
    """(moov, trak, mdia, minf, stbl, stsd, entry, hvcC_end) видео-трека."""
    for typ, p, size, hdr in boxes(buf, 0, len(buf)):
        if typ != "moov":
            continue
        for t2, p2, s2, h2 in boxes(buf, p + hdr, p + size):
            if t2 != "trak":
                continue
            # hdlr 'vide' лежит внутри mdia
            is_video = False
            for t3, p3, s3, h3 in boxes(buf, p2 + h2, p2 + s2):
                if t3 != "mdia":
                    continue
                for t4, p4, s4, h4 in boxes(buf, p3 + h3, p3 + s3):
                    if t4 == "hdlr" and b"vide" in buf[p4 + h4:p4 + s4]:
                        is_video = True
            if not is_video:
                continue
            for t3, p3, s3, h3 in boxes(buf, p2 + h2, p2 + s2):
                if t3 != "mdia":
                    continue
                for t4, p4, s4, h4 in boxes(buf, p3 + h3, p3 + s3):
                    if t4 != "minf":
                        continue
                    for t5, p5, s5, h5 in boxes(buf, p4 + h4, p4 + s4):
                        if t5 != "stbl":
                            continue
                        for t6, p6, s6, h6 in boxes(buf, p5 + h5, p5 + s5):
                            if t6 != "stsd":
                                continue
                            ep = p6 + 8 + 4 + 4
                            esz = struct.unpack(">I", buf[ep:ep + 4])[0]
                            if buf[ep + 4:ep + 8].decode("latin1") not in ("hvc1", "hev1"):
                                continue
                            for t7, p7, s7, h7 in boxes(buf, ep + 8 + 78, ep + esz):
                                if t7 == "hvcC":
                                    return (p, p2, p3, p4, p5, p6, ep, esz, p7 + s7)
    return None


def extract_vexu(ref_path):
    """Копирует бокс vexu дословно из референсного файла камеры Pico."""
    ref = open(ref_path, "rb").read()
    idx = ref.find(b"vexu")
    if idx == -1:
        return None
    sz = struct.unpack(">I", ref[idx - 4:idx])[0]
    return ref[idx - 4:idx - 4 + sz]


def main():
    argv = sys.argv[1:]
    baseline_mm = 64.0
    fov_deg = 76.35
    ref_path = None
    args = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--baseline-mm":
            baseline_mm = float(argv[i + 1]); i += 2
        elif a == "--fov-deg":
            fov_deg = float(argv[i + 1]); i += 2
        elif a == "--vexu-ref":
            ref_path = argv[i + 1]; i += 2
        else:
            args.append(a); i += 1
    inp, outp = args[0], args[1]

    buf = bytearray(open(inp, "rb").read())
    r = find_video_hvcc(bytes(buf))
    if r is None:
        print("ERROR: hvc1/hvcC видео-трека не найден")
        sys.exit(1)
    moov, trak, mdia, minf, stbl, stsd, ep, esz, hvcC_end = r

    parts = []
    if ref_path:
        vexu = extract_vexu(ref_path)
        if vexu:
            parts.append(vexu)
        else:
            print("WARN: vexu в референсе не найден")
    else:
        # vexu из дефолтного референса Pico (stri=3, cams, blin=64мм, cmfy, dadj, proj)
        parts.append(bytes.fromhex(
            "00000065 76657875 00000045 65796573"
            "0000000d 73747269 00000000 03"
            "00000018 63616d73 00000010 626c696e 000000000000fa00"
            "00000018 636d6679 00000010 6461646a 0000000000000000"
            "00000018 70726f6a 00000010 70726a69 00000000 72656374"
            .replace(" ", "")))
    parts.append(struct.pack(">I", 12) + b"hfov" +
                 struct.pack(">I", int(round(fov_deg * 1000))))
    parts.append(struct.pack(">I", 19) + b"colr" + b"nclx" +
                 bytes([0x00, 0x05, 0x00, 0x01, 0x00, 0x06, 0x00]))
    ins = b"".join(parts)
    add = len(ins)

    out = bytearray(buf[:hvcC_end]) + bytearray(ins) + bytearray(buf[hvcC_end:])
    struct.pack_into(">I", out, ep, esz + add)   # sample entry
    for pos in (stsd, stbl, minf, mdia, trak, moov):
        old = struct.unpack(">I", out[pos:pos + 4])[0]
        struct.pack_into(">I", out, pos, old + add)

    # если moov стоит ДО mdat (faststart) — chunk offsets сдвинулись
    top = boxes(out, 0, len(out))
    moov_p = [p for t, p, _, _ in top if t == "moov"][0]
    mdat_p = [p for t, p, _, _ in top if t == "mdat"][0]
    if mdat_p > moov_p:
        cnt = [0]

        def patch_offsets(data, off, end):
            for typ, p, size, hdr in boxes(data, off, end):
                if typ == "stco":
                    q = p + hdr + 4
                    k = struct.unpack(">I", data[q:q + 4])[0]
                    for i in range(k):
                        o = q + 4 + i * 4
                        v = struct.unpack(">I", data[o:o + 4])[0]
                        struct.pack_into(">I", data, o, v + add)
                    cnt[0] += k
                elif typ in ("moov", "trak", "mdia", "minf", "stbl"):
                    patch_offsets(data, p + hdr, p + size)

        patch_offsets(out, moov_p, len(out))
        print(f"patched {cnt[0]} chunk offsets (+{add})")

    open(outp, "wb").write(out)
    print(f"OK: вставлено {add} байт (vexu+hfov+colr) -> {outp}")


if __name__ == "__main__":
    main()
