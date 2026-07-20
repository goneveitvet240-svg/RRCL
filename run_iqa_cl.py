"""
RRCL norm-ablation + adaptive-f on IQA datasets (KADID-10k etc.)
Scalar regression: image -> DMOS/MOS quality score.

Same protocol as run_age_cl.py.

Usage (AutoDL):
  python run_iqa_cl.py --config domains_iqa_kadid.json \
    --img-size 518 --backbone dinov2_vitb14 \
    --lam 1e2 --max-per-domain 500 \
    --out runs_real/kadid
"""
import argparse
import json
import os
import time

import numpy as np

from rls_head import ForgettingRidgeRLS
from metrics import forgetting_matrix_stats


def _arr(domains, split, t):
    Xs, ys = [], []
    for f, a, _ in domains.stream(split, t):
        Xs.append(np.asarray(f, dtype=np.float64).reshape(-1))
        ys.append(float(np.ravel(a)[0]))
    return np.asarray(Xs), np.asarray(ys)


def load_all(domains):
    T = domains.n_domains()
    return ({t: _arr(domains, "train", t) for t in range(T)},
            {t: _arr(domains, "test", t) for t in range(T)})


def rel_mae(pred, gt):
    return float(np.mean(np.abs(pred - gt)) / max(np.mean(np.abs(gt)), 1e-6))


def scales(tr, mode):
    return {t: (float(np.mean(tr[t][1])) if mode == "mean" else 1.0) for t in tr}


def aug(X):
    return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)


def train_eval(tr, te, forget, lam, mode, proj_dim=0):
    T = len(tr); d = tr[0][0].shape[1]; s = scales(tr, mode)
    head = ForgettingRidgeRLS(d_in=d, d_out=1, lam=lam, forget=forget, proj_dim=proj_dim)
    M = np.full((T, T), np.nan)
    for t in range(T):
        head.begin_task()
        X, y = tr[t]; head.accumulate(X, (y / s[t]).reshape(-1, 1)); head.solve()
        for i in range(t + 1):
            Xt, yt = te[i]; pred = head.predict(Xt)[:, 0] * s[i]
            M[t, i] = rel_mae(pred, yt)
    return float(np.nanmean(M[T - 1, :T])), M


def sweep(tr, te, forgets, lam, mode):
    rows = []; best = (1.0, 1e18)
    for f in forgets:
        r, _ = train_eval(tr, te, f, lam, mode)
        rows.append((f, r))
        if r < best[1]: best = (f, r)
    return best, rows


def adaptive(tr, te, lam, mode, sel_grid, val_every=5):
    T = len(tr); d = tr[0][0].shape[1]; d_aug = d + 1; s = scales(tr, mode)
    head = ForgettingRidgeRLS(d_in=d, d_out=1, lam=lam)
    Rf = []; Cf = []; Rv = []; Cv = []; Sv = []; nv = []; f_used = []
    M = np.full((T, T), np.nan)
    for t in range(T):
        X, y = tr[t]; yn = (y / s[t]).reshape(-1, 1); Xa = aug(X)
        idx = np.arange(X.shape[0]); vm = (idx % val_every == val_every - 1)
        Xf, yf, Xv, yv = Xa[~vm], yn[~vm], Xa[vm], yn[vm]
        if Xv.shape[0] == 0: Xv, yv = Xf, yf
        Rf.append(Xf.T @ Xf); Cf.append(Xf.T @ yf)
        Rv.append(Xv.T @ Xv); Cv.append(Xv.T @ yv); Sv.append(float(np.sum(yv * yv))); nv.append(Xv.shape[0])
        if t == 0:
            f_t = 1.0
        else:
            bf, bo = 1.0, 1e18
            for f in sel_grid:
                A = lam * np.eye(d_aug); b = np.zeros((d_aug, 1))
                for dd in range(t + 1):
                    w = f ** (t - dd); A += w * Rf[dd]; b += w * Cf[dd]
                W = np.linalg.solve(A, b)
                errs = []
                for dd in range(t + 1):
                    q = float(np.sum(W * (Rv[dd] @ W))); l = float(np.sum(W * Cv[dd]))
                    errs.append((q - 2 * l + Sv[dd]) / max(nv[dd], 1))
                o = float(np.mean(errs))
                if o < bo: bf, bo = float(f), o
            f_t = bf
        f_used.append(f_t)
        head.begin_task(f_t); head.accumulate(X, yn); head.solve()
        for i in range(t + 1):
            Xt, yt = te[i]; pred = head.predict(Xt)[:, 0] * s[i]; M[t, i] = rel_mae(pred, yt)
    return float(np.nanmean(M[T - 1, :T])), f_used


def single_domain(tr, te, lam, mode):
    s = scales(tr, mode); rels = []
    for t in tr:
        d = tr[t][0].shape[1]
        h = ForgettingRidgeRLS(d_in=d, d_out=1, lam=lam); h.begin_task()
        X, y = tr[t]; h.accumulate(X, (y / s[t]).reshape(-1, 1)); h.solve()
        Xt, yt = te[t]; rels.append(rel_mae(h.predict(Xt)[:, 0] * s[t], yt))
    return float(np.mean(rels))


def sgd_seq(tr, te, mode, epochs=15, lr=0.01, l2=1e-4):
    T = len(tr); d = tr[0][0].shape[1]; s = scales(tr, mode)
    mu = tr[0][0].mean(0); sd = tr[0][0].std(0) + 1e-6
    W = np.zeros((d + 1, 1)); b1, b2, eps = 0.9, 0.999, 1e-8
    m = np.zeros_like(W); v = np.zeros_like(W); step = 0
    feat = lambda X: np.concatenate([(X - mu) / sd, np.ones((X.shape[0], 1))], 1)
    M = np.full((T, T), np.nan)
    for t in range(T):
        X, y = tr[t]; Xa = feat(X); yn = (y / s[t]).reshape(-1, 1)
        for _ in range(epochs):
            step += 1
            g = Xa.T @ (Xa @ W - yn) / Xa.shape[0] + l2 * W
            m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
            W -= lr * (m / (1 - b1 ** step)) / (np.sqrt(v / (1 - b2 ** step)) + eps)
        for i in range(t + 1):
            Xt, yt = te[i]; pred = (feat(Xt) @ W)[:, 0] * s[i]; M[t, i] = rel_mae(pred, yt)
    return float(np.nanmean(M[T - 1, :T]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--forgets", default="1.0,0.8,0.6,0.4,0.3,0.25,0.2,0.15,0.1")
    ap.add_argument("--lam", type=float, default=1e2)
    ap.add_argument("--img-size", type=int, default=518)
    ap.add_argument("--backbone", default="dinov2_vitb14")
    ap.add_argument("--max-per-domain", type=int, default=500)
    ap.add_argument("--ranpac-dim", type=int, default=2000)
    ap.add_argument("--out", default="runs_real/kadid")
    a = ap.parse_args()
    forgets = [float(x) for x in a.forgets.split(",")]
    sel_grid = np.round(np.arange(0.05, 1.0001, 0.05), 3)

    from datasets_iqa import IQADomains, build_iqa_config
    with open(a.config) as f:
        cfg = json.load(f)
    domains = IQADomains(build_iqa_config(cfg), backbone=a.backbone, img_size=a.img_size,
                         max_per_domain=a.max_per_domain)
    names = [d["name"] for d in cfg["domains"]]
    t0 = time.time()
    tr, te = load_all(domains)
    print("domains:", names, "| sizes(train/test):",
          [(tr[t][0].shape[0], te[t][0].shape[0]) for t in tr])

    res = {}
    print("\n=== norm-ablation (optimal f by relative MAE) ===")
    for mode in ("none", "mean"):
        (bf, br), rows = sweep(tr, te, forgets, a.lam, mode)
        res[f"opt_f_{mode}"] = bf; res[f"opt_rel_{mode}"] = br
        print(f"  [{mode:4s}] best f = {bf:g}  rel = {br:.4f}   sweep=" +
              " ".join(f"{f:g}:{r:.3f}" for f, r in rows))
    f_mean = res["opt_f_mean"]
    verdict = (f"归一化后最优 f={f_mean:g}{'<1 → 遗忘有效（真实漂移）' if f_mean < 0.9 else '≈1 → 无需遗忘（量级假象或无漂移）'}")
    res["verdict"] = verdict
    print("  VERDICT:", verdict)

    print("\n=== adaptive f / baselines (normalized regime, rel MAE) ===")
    rel_f1, _ = train_eval(tr, te, 1.0, a.lam, "mean")
    rel_ad, f_used = adaptive(tr, te, a.lam, "mean", sel_grid)
    rel_sd = single_domain(tr, te, a.lam, "mean")
    rel_rp, _ = train_eval(tr, te, 1.0, a.lam, "mean", proj_dim=a.ranpac_dim)
    rel_sgd = sgd_seq(tr, te, "mean")
    res.update(dict(f1=rel_f1, adaptive=rel_ad, f_used=f_used, single=rel_sd,
                    ranpac=rel_rp, sgd=rel_sgd))
    print(f"  f1 (=joint)        rel = {rel_f1:.4f}")
    print(f"  adaptive f         rel = {rel_ad:.4f}   f_used={[round(x,2) for x in f_used]}")
    print(f"  single-domain      rel = {rel_sd:.4f}")
    print(f"  RanPAC(proj={a.ranpac_dim},f=1) rel = {rel_rp:.4f}")
    print(f"  SGD-seq(Adam)      rel = {rel_sgd:.4f}")
    print(f"  -> adaptive vs f1: {(rel_f1-rel_ad)/max(rel_f1,1e-9)*100:+.1f}%")

    os.makedirs(a.out, exist_ok=True)
    # tag output by config basename
    tag = os.path.splitext(os.path.basename(a.config))[0]
    out_path = os.path.join(a.out, f"{tag}_result.json")
    with open(out_path, "w") as fh:
        json.dump(dict(config=a.config, names=names, **res), fh, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
