"""List the files inside a remote uncompressed tar-wrapped zip by hopping between
local headers with small HTTP range requests. Prints name, start byte, size.

Usage: python -I list_remote.py URL
"""
import struct
import subprocess
import sys

url = sys.argv[1]


def get(a, b):
    return subprocess.run(["curl", "-s", "-m", "120", "-r", f"{a}-{b}", url], capture_output=True, check=True).stdout


off = 512
while True:
    lh = get(off, off + 299)
    if lh[:4] != b"PK\x03\x04":
        break
    csz = struct.unpack("<I", lh[18:22])[0]
    n, e = struct.unpack("<HH", lh[26:30])
    name = lh[30 : 30 + n].decode()
    start = off + 30 + n + e
    print(name, start, csz, flush=True)
    off = start + csz
