import json
import math
import sys
from collections import defaultdict


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((c - h) / d, 3), round((c + h) / d, 3))


def summarize(rows, tau=0.9, ok_m=50.0):
    by = defaultdict(list)
    for r in rows:
        by[int(r["L"])].append(r)
    by["all"] = rows
    out = {}
    for L, rs in by.items():
        n = len(rs)
        ans = [r for r in rs if r["conf"] >= tau]
        right = [r for r in ans if r["err"] <= ok_m]
        wrong = [r for r in ans if r["err"] > ok_m]
        top1 = sum(1 for r in rs if r["err"] <= ok_m)
        out[str(L)] = {"clips": n, "answered": len(ans), "coverage": round(len(ans) / n, 3) if n else None,
                       "correct_when_answered": len(right), "precision": round(len(right) / len(ans), 3) if ans else None,
                       "precision_ci95": wilson(len(right), len(ans)), "confident_wrong": len(wrong),
                       "confident_wrong_rate": round(len(wrong) / n, 3) if n else None,
                       "best_guess_within_50m": top1, "median_err_answered_m": sorted(r["err"] for r in ans)[len(ans) // 2] if ans else None}
    bins = [(0, .5), (.5, .8), (.8, .9), (.9, .99), (.99, 1.01)]
    rel = []
    for lo, hi in bins:
        b = [r for r in rows if lo <= r["conf"] < hi]
        if b:
            rel.append({"range": [lo, hi], "n": len(b), "mean_conf": round(sum(r["conf"] for r in b) / len(b), 3),
                        "accuracy": round(sum(r["err"] <= ok_m for r in b) / len(b), 3)})
    return {"by_length": out, "reliability": rel}


if __name__ == "__main__":
    rows = json.load(open(sys.argv[1])) if sys.argv[1].endswith(".json") else [json.loads(l) for l in open(sys.argv[1]) if l.startswith("{")]
    s = summarize(rows)
    print(json.dumps(s, indent=1))
    for r in sorted(rows, key=lambda r: (r["file"], r["L"], r["t0"])):
        flag = "ANSWER" if r["conf"] >= 0.9 else "abstain"
        print(f"{r['file'][-15:]} L={r['L']:.0f} t0={r['t0']:.0f} conf={r['conf']:.3f} p_best={r['p_best']:.3f} err={r['err']:.0f} moved={r['moved_m']:.0f} turn={r['abs_turn_deg']:.0f} {flag}")
