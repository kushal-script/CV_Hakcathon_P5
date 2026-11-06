import io
import sys
import threading
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"


class RangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
        self.pos = 0
        self.cache = {}

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def get(self, start, n):
        end = min(self.size, start + n) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return r.read()
            except Exception:
                if attempt == 4:
                    raise

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        data = self.get(self.pos, n)
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def member_bytes(url, info):
    rf = RangeFile.__new__(RangeFile)
    rf.url = url
    rf.size = 1 << 62
    head = rf.get(info.header_offset, 30)
    name_len = int.from_bytes(head[26:28], "little")
    extra_len = int.from_bytes(head[28:30], "little")
    start = info.header_offset + 30 + name_len + extra_len
    raw = rf.get(start, info.compress_size)
    if info.compress_type == zipfile.ZIP_STORED:
        return raw
    import zlib
    return zlib.decompress(raw, -15)


def fetch(drive, out_root, every=2, cams=("image_00",), workers=16):
    date = drive[:10]
    url = f"{BASE}/{drive}/{drive}_sync.zip"
    zf = zipfile.ZipFile(RangeFile(url))
    infos = zf.infolist()
    want = []
    for i in infos:
        parts = i.filename.split("/")
        if len(parts) < 4 or i.is_dir():
            continue
        kind = parts[2]
        if kind == "oxts" or (kind in cams and parts[3] == "data" and int(Path(parts[-1]).stem) % every == 0) or (kind in cams and parts[-1] == "timestamps.txt"):
            want.append(i)
    out = Path(out_root)
    todo = [i for i in want if not (out / i.filename).exists()]
    print(drive, "members", len(want), "todo", len(todo), "MB", round(sum(i.compress_size for i in todo) / 1e6, 1), flush=True)
    lock = threading.Lock()
    done = [0]

    def job(i):
        p = out / i.filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(member_bytes(url, i))
        with lock:
            done[0] += 1
            if done[0] % 200 == 0:
                print(drive, done[0], "/", len(todo), flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(job, todo))
    calib = out / date / "calib_cam_to_cam.txt"
    if not calib.exists():
        cz = zipfile.ZipFile(RangeFile(f"{BASE}/{date}_calib.zip"))
        for i in cz.infolist():
            if not i.is_dir():
                p = out / i.filename
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(cz.read(i))
    print(drive, "done", flush=True)


if __name__ == "__main__":
    for d in sys.argv[2:]:
        fetch(d, sys.argv[1])
