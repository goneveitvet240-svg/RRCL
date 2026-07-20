"""Domain-incremental counting with an auxiliary image-level RLS head.

The original run_real.py predicts per-patch counts and sums them. This script
keeps that path intact and adds a second closed-form head:

    mean pooled patch features -> image total count

At test time it evaluates

    count = alpha * patch_sum + (1 - alpha) * image_head_count

This is a low-risk accuracy probe: it keeps the frozen backbone, closed-form
ridge heads, streaming accumulation, and exemplar-free setting.
"""
import argparse
import csv
import json
import os
import time
import numpy as np
import re

from rls_head import ForgettingRidgeRLS
from metrics import mae, forgetting_matrix_stats
from run_real import _SelfTest, _first_dim


_DCT_CACHE = {}


def _dct_matrix(g, k):
    key = (g, k)
    if key in _DCT_CACHE:
        return _DCT_CACHE[key]
    x = np.arange(g, dtype=np.float64)[:, None]
    u = np.arange(k, dtype=np.float64)[None, :]
    C = np.cos(np.pi * (x + 0.5) * u / g)
    C[:, 0] *= np.sqrt(1.0 / g)
    if k > 1:
        C[:, 1:] *= np.sqrt(2.0 / g)
    _DCT_CACHE[key] = C
    return C


def _preserve_sum(y2, target_sum):
    s = float(y2.sum())
    if abs(s) > 1e-12:
        y2 = y2 * (float(target_sum) / s)
    return y2


def _dct_lowpass(y2, k):
    g = y2.shape[0]
    k = max(1, min(int(k), g))
    C = _dct_matrix(g, k)
    coeff = C.T @ y2 @ C
    out = C @ coeff @ C.T
    out = np.clip(out, 0.0, None)
    return _preserve_sum(out, y2.sum())


def _gaussian_smooth(y2, sigma):
    sigma = float(sigma)
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / max(sigma, 1e-12)) ** 2)
    k /= k.sum()
    padded = np.pad(y2, ((radius, radius), (radius, radius)), mode="constant")
    tmp = np.zeros_like(padded, dtype=np.float64)
    for j, w in enumerate(k):
        tmp[:, radius:-radius] += w * padded[:, j:j + y2.shape[1]]
    out_full = np.zeros_like(tmp, dtype=np.float64)
    for i, w in enumerate(k):
        out_full[radius:-radius, :] += w * tmp[i:i + y2.shape[0], :]
    out = out_full[radius:-radius, radius:-radius]
    out = np.clip(out, 0.0, None)
    return _preserve_sum(out, y2.sum())


def transform_patch_target(Y, mode):
    Y = np.asarray(Y, dtype=np.float64).reshape(-1, 1)
    if mode == "raw":
        return Y
    P = Y.shape[0]
    g = int(round(np.sqrt(P)))
    if g * g != P:
        raise ValueError(f"target length {P} is not a square grid")
    y2 = Y[:, 0].reshape(g, g)
    if mode.startswith("dct"):
        m = re.fullmatch(r"dct(\d+)", mode)
        if not m:
            raise ValueError(f"bad dct mode: {mode}; use e.g. dct3")
        out = _dct_lowpass(y2, int(m.group(1)))
    elif mode.startswith("gauss"):
        m = re.fullmatch(r"gauss([0-9.]+)", mode)
        if not m:
            raise ValueError(f"bad gaussian mode: {mode}; use e.g. gauss1.5")
        out = _gaussian_smooth(y2, float(m.group(1)))
    else:
        raise ValueError(f"unknown patch target: {mode}")
    return out.reshape(-1, 1)


def _image_feature(X):
    """One image descriptor from patch tokens."""
    return np.mean(np.asarray(X, dtype=np.float64), axis=0, keepdims=True)


def _image_target(Y):
    return np.asarray([[float(np.asarray(Y, dtype=np.float64).sum())]])


def _eval_domain(domains, d, patch_head, image_head, alpha):
    preds, gts = [], []
    for X, Y, _ in domains.stream("test", d):
        patch_count = float(patch_head.predict(X)[:, 0].sum())
        image_count = float(image_head.predict(_image_feature(X))[0, 0])
        pred = alpha * patch_count + (1.0 - alpha) * image_count
        preds.append(max(pred, 0.0))
        gts.append(float(np.asarray(Y).sum()))
    return mae(preds, gts)


def run(domains, patch_targets, forgets, alphas, lam, out):
    T = domains.n_domains()
    d_in = _first_dim(domains)
    results = {}
    for patch_target in patch_targets:
        for forget in forgets:
            patch_head = ForgettingRidgeRLS(d_in=d_in, d_out=1, lam=lam, forget=forget)
            image_head = ForgettingRidgeRLS(d_in=d_in, d_out=1, lam=lam, forget=forget)
            matrices = {alpha: np.full((T, T), np.nan) for alpha in alphas}
            for t in range(T):
                patch_head.begin_task()
                image_head.begin_task()
                n = 0
                for X, Y, _ in domains.stream("train", t):
                    patch_head.accumulate(X, transform_patch_target(Y, patch_target))
                    image_head.accumulate(_image_feature(X), _image_target(Y))
                    n += 1
                patch_head.solve()
                image_head.solve()
                for i in range(t + 1):
                    for alpha in alphas:
                        matrices[alpha][t, i] = _eval_domain(
                            domains, i, patch_head, image_head, alpha
                        )
                best_seen = min(
                    (matrices[a][t, : t + 1].mean(), a) for a in alphas
                )
                print(
                    f"  [target={patch_target}, forget={forget:g}] learned domain {t} "
                    f"({n} imgs); best seen alpha={best_seen[1]:g}, avg={best_seen[0]:.2f}"
                )
            for alpha, M in matrices.items():
                label = f"target={patch_target},forget={forget:g},alpha={alpha:g}"
                results[label] = {
                    "matrix": M.tolist(),
                    "patch_target": patch_target,
                    "forget": float(forget),
                    "alpha": float(alpha),
                    **forgetting_matrix_stats(M),
                }

    os.makedirs(out, exist_ok=True)
    payload = {
        "forgets": forgets,
        "alphas": alphas,
        "patch_targets": patch_targets,
        "lam": lam,
        "d_in": d_in,
        "results": results,
    }
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    write_report(results, out)
    return results


def write_report(results, out):
    rows = []
    for label, r in results.items():
        rows.append(
            (
                label,
                r["patch_target"],
                r["forget"],
                r["alpha"],
                r["final_avg_mae"],
                r["nBwT"],
                r["final_oldest_mae"],
                r["final_newest_mae"],
            )
        )
    rows.sort(key=lambda x: x[4])
    hdr = [
        "setting",
        "patch_target",
        "forget",
        "alpha_patch",
        "final_avg_MAE",
        "nBwT",
        "oldest_MAE",
        "newest_MAE",
    ]
    with open(os.path.join(out, "report.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(rows)
    lines = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    for r in rows:
        lines.append(
            f"| {r[0]} | {r[1]} | {r[2]:g} | {r[3]:g} | {r[4]:.3f} | "
            f"{r[5]:.3f} | {r[6]:.3f} | {r[7]:.3f} |"
        )
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n===== Image-level auxiliary head report =====")
    print("\n".join(lines[:22]))
    print(f"\nreport -> {os.path.join(out, 'report.md')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--forgets", type=str, default="1.0,0.4,0.2,0.15,0.1")
    ap.add_argument("--alphas", type=str, default="1.0,0.75,0.5,0.25,0.0")
    ap.add_argument("--patch-targets", type=str, default="raw",
                    help="comma list: raw,dct3,dct5,gauss1.0,...")
    ap.add_argument("--lam", type=float, default=1e2)
    ap.add_argument("--img-size", type=int, default=518)
    ap.add_argument("--backbone", type=str, default="dinov2_vitb14")
    ap.add_argument("--max-per-domain", type=int, default=None)
    ap.add_argument("--out", type=str, default="runs_real_image_aux")
    a = ap.parse_args()

    forgets = [float(x) for x in a.forgets.split(",")]
    alphas = [float(x) for x in a.alphas.split(",")]
    patch_targets = [x.strip() for x in a.patch_targets.split(",") if x.strip()]

    if a.selftest:
        domains = _SelfTest()
    elif a.config:
        from datasets_real import RealCountingDomains, build_from_config

        with open(a.config) as f:
            cfg = json.load(f)
        domains = RealCountingDomains(
            build_from_config(cfg),
            backbone=a.backbone,
            img_size=a.img_size,
            max_per_domain=a.max_per_domain,
        )
    else:
        ap.error("pass --selftest or --config domains.json")

    t0 = time.time()
    run(domains, patch_targets, forgets, alphas, a.lam, a.out)
    print(f"\nsaved -> {os.path.join(a.out, 'results.json')}   ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
