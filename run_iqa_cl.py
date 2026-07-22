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

from adaptive_selector import BalancedSufficientStatsSelector
from rls_head import ForgettingRidgeRLS
from run_provenance import with_provenance
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


def adaptive(
    tr,
    te,
    lam,
    mode,
    sel_grid,
    val_every=5,
    search_mode="continuous",
    abstain_relative_gain=1e-4,
):
    T = len(tr); d = tr[0][0].shape[1]; d_aug = d + 1; s = scales(tr, mode)
    head = ForgettingRidgeRLS(d_in=d, d_out=1, lam=lam)
    selector = BalancedSufficientStatsSelector(
        d_aug, lam, search_mode=search_mode, grid=sel_grid,
        abstain_relative_gain=abstain_relative_gain,
    )
    f_used, drift_scores = [], []
    M = np.full((T, T), np.nan)
    for t in range(T):
        X, y = tr[t]; yn = (y / s[t]).reshape(-1, 1); Xa = aug(X)
        idx = np.arange(X.shape[0]); vm = (idx % val_every == val_every - 1)
        Xf, yf, Xv, yv = Xa[~vm], yn[~vm], Xa[vm], yn[vm]
        if Xv.shape[0] == 0: Xv, yv = Xf, yf
        fit = [Xf.T @ Xf, Xf.T @ yf, float(np.sum(yf * yf)), Xf.shape[0]]
        val = [Xv.T @ Xv, Xv.T @ yv, float(np.sum(yv * yv)), Xv.shape[0]]
        selection = selector.select_and_update(fit, val)
        f_t = selection.factor
        f_used.append(f_t)
        drift_scores.append(selection.drift_score)
        head.begin_task(f_t); head.accumulate(X, yn); head.solve()
        for i in range(t + 1):
            Xt, yt = te[i]; pred = head.predict(Xt)[:, 0] * s[i]; M[t, i] = rel_mae(pred, yt)
    return float(np.nanmean(M[T - 1, :T])), f_used, drift_scores, selector.state_bytes


def vff_iqa(tr, te, lam, mode, gamma=1.5, xi=1e-6, f_min=0.05, val_every=5):
    """Paleologu VFF-RLS adapted to domain-batched IQA (IEEE SPL 2008)."""
    from vff_baselines import paleologu_vff_factor
    T = len(tr); d = tr[0][0].shape[1]; s = scales(tr, mode)
    head = ForgettingRidgeRLS(d_in=d, d_out=1, lam=lam)
    f_used = []; M = np.full((T, T), np.nan)
    for t in range(T):
        X, y = tr[t]; yn = (y / s[t]).reshape(-1, 1)
        idx = np.arange(X.shape[0]); vm = (idx % val_every == val_every - 1)
        Xv, yv = X[vm], yn[vm]
        if Xv.shape[0] == 0: Xv, yv = X, yn
        if t == 0 or head.W is None:
            f_t = 1.0
        else:
            Xva = np.concatenate([Xv, np.ones((Xv.shape[0], 1))], axis=1)
            err_true = yv[:, 0] * s[t]
            error_power = float(np.mean((head.predict(Xv)[:, 0] * s[t] - err_true) ** 2))
            fh = ForgettingRidgeRLS(d_in=d, d_out=1, lam=lam)
            fh.begin_task()
            Xf, yf = X[~vm], yn[~vm]
            if Xf.shape[0] == 0: Xf, yf = X, yn
            fh.accumulate(Xf, yf); fh.solve()
            noise_power = float(np.mean((fh.predict(Xv)[:, 0] * s[t] - err_true) ** 2))
            cov_inv = np.linalg.inv(head.R + lam * np.eye(head.d))
            lev = np.einsum('ij,jk,ik->i', Xva, cov_inv, Xva)
            leverage_power = float(np.mean(lev ** 2))
            f_t = paleologu_vff_factor(error_power, noise_power, leverage_power,
                                       gamma=gamma, xi=xi, f_min=f_min)
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
    ap.add_argument("--selector-search", choices=["continuous", "grid"], default="continuous")
    ap.add_argument("--abstain-relative-gain", type=float, default=1e-4)
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
    rel_ad, f_used, drift_scores, selector_state_bytes = adaptive(
        tr, te, a.lam, "mean", sel_grid,
        search_mode=a.selector_search,
        abstain_relative_gain=a.abstain_relative_gain,
    )
    rel_sd = single_domain(tr, te, a.lam, "mean")
    rel_rp, _ = train_eval(tr, te, 1.0, a.lam, "mean", proj_dim=a.ranpac_dim)
    rel_sgd = sgd_seq(tr, te, "mean")
    rel_vff, vff_f_used = vff_iqa(tr, te, a.lam, "mean")
    res.update(dict(f1=rel_f1, adaptive=rel_ad, f_used=f_used,
                    drift_scores=drift_scores, selector_search=a.selector_search,
                    selector_state_bytes=selector_state_bytes, single=rel_sd,
                    ranpac=rel_rp, vff=rel_vff, vff_f_used=vff_f_used, sgd=rel_sgd))
    print(f"  f1 (=joint)        rel = {rel_f1:.4f}")
    print(f"  adaptive f         rel = {rel_ad:.4f}   f_used={[round(x,2) for x in f_used]}")
    print(f"  single-domain      rel = {rel_sd:.4f}")
    print(f"  RanPAC(proj={a.ranpac_dim},f=1) rel = {rel_rp:.4f}")
    print(f"  VFF-RLS(Paleologu) rel = {rel_vff:.4f}   f_used={[round(x,2) for x in vff_f_used]}")
    print(f"  SGD-seq(Adam)      rel = {rel_sgd:.4f}")
    print(f"  -> adaptive vs f1: {(rel_f1-rel_ad)/max(rel_f1,1e-9)*100:+.1f}%")

    os.makedirs(a.out, exist_ok=True)
    # tag output by config basename
    tag = os.path.splitext(os.path.basename(a.config))[0]
    out_path = os.path.join(a.out, f"{tag}_result.json")
    with open(out_path, "w") as fh:
        payload = with_provenance(dict(config=a.config, names=names, **res), a.config, vars(a))
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
