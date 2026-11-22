"""Pull complete files out of a (possibly still downloading) uncompressed tar-wrapped zip.

Usage: python -I extract_partial.py ARCHIVE OUTDIR
Each zip entry is stored (method 0), so a file can be copied out as soon as its
last byte has arrived. Already-extracted files are skipped.
"""
import os
import struct
import sys

archive, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)
have = os.path.getsize(archive)
with open(archive, "rb") as f:
    off = 512  # skip the tar header
    while True:
        f.seek(off)
        lh = f.read(30)
        if len(lh) < 30 or lh[:4] != b"PK\x03\x04":
            break
        _, method = struct.unpack("<HH", lh[6:10])
        csz = struct.unpack("<I", lh[18:22])[0]
        n, e = struct.unpack("<HH", lh[26:30])
        name = f.read(n).decode()
        start = off + 30 + n + e
        off = start + csz
        if csz == 0 or method != 0:
            continue
        base = os.path.basename(name)
        dest = os.path.join(outdir, base)
        if off > have:
            print(f"incomplete: {base} ({(have - start) / csz:.0%})")
            break
        if os.path.exists(dest) and os.path.getsize(dest) == csz:
            continue
        f.seek(start)
        with open(dest, "wb") as g:
            left = csz
            while left:
                chunk = f.read(min(left, 1 << 24))
                g.write(chunk)
                left -= len(chunk)
        print("extracted:", base, f"{csz / 1e6:.1f} MB")
