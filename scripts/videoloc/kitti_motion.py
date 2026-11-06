import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from p5.geo.videoloc.motion import MotionConfig, measure


def stamps(path):
    out = []
    for line in Path(path).read_text().split("\n"):
        line = line.strip()
        if line:
            d, t = line.split(" ")
            hh, mm, ss = t.split(":")
            out.append(int(hh) * 3600 + int(mm) * 60 + float(ss))
    return np.array(out)


def focal(calib_file):
    for line in Path(calib_file).read_text().split("\n"):
        if line.startswith("P_rect_00:"):
            return float(line.split()[1])
    raise ValueError("no P_rect_00")


def main(root, drive, every, out, use_focal):
    date = drive[:10]
    base = Path(root) / date / f"{drive}_sync"
    imgs = sorted((base / "image_00" / "data").glob("*.png"))
    ts = stamps(base / "image_00" / "timestamps.txt")
    ox = sorted((base / "oxts" / "data").glob("*.txt"))
    gt = np.array([[float(v) for v in f.read_text().split()[:25]] for f in ox])
    f = focal(Path(root) / date / "calib_cam_to_cam.txt") if use_focal else None
    sel = list(range(0, len(imgs), every))

    def gen():
        for i in sel:
            yield ts[i] - ts[0], cv2.imread(str(imgs[i]))

    recs, extra = measure(gen(), MotionConfig(target_fps=10.0 / every, max_side=640), focal_px=f)
    with open(out, "w") as fh:
        for i, r in zip(sel, recs):
            g = gt[min(i, len(gt) - 1)]
            r.update(frame=i, lat=g[0], lon=g[1], yaw=g[5], vf=g[8], wz=g[19] if len(g) > 19 else None)
            fh.write(json.dumps(r) + "\n")
    print(drive, "frames", len(recs), "focal", extra["K"][0][0], flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5] == "1")
