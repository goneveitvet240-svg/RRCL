"""
Falsifiable test of the predictive theory (RRCL_理论命题草稿):
does the closed-form f*_pred — computed from TRAINING statistics only — match the
empirical oracle f* (test-set sweep), ACROSS tasks (crowd counting + age)?

Unified at the IMAGE level (scalar regression) so both tasks are directly comparable:
  crowd: mean-pooled DINOv2 feature -> total count
  age  : mean-pooled DINOv2 feature -> age

Per domain k, from TRAIN only (normalized targets y/mean):
  w_k  = domain-only ridge solution            (drift carrier)
  mu_k = tr(R_k)/d_aug                          (statistical mass)
  r_k  = mean post-fit residual energy          (noise)
  sig2 = Var(y_norm_k); s_k = mu_k/n_k; rho_k = s_k/sig2_k
Predicted optimum:  f*_pred = argmin_f  J(f),
  J(f) = (1/T)Σ_k rho_k ||Wbar(f)-w_k||^2  +  gain·d_aug·(1/T)Σ_k rho_k · V(f)
  beta_k=f^{tau_k}mu_k/(M_f+λ), Wbar=Σ beta_k w_k, V=Σ f^{2tau}r mu /(M_f+λ)^2, tau_k=T-1-k

Appends one CSV row: config,task,T,f_pred,f_oracle,rel_oracle

Usage:
  python run_theory_predict.py --config domains_jhu_sha_shb.json --task crowd \
    --img-size 518 --backbone vit_base_patch14_dinov2.lvd142m --lam 100 \
    --max-per-domain 400 --csv runs_real/theory_points.csv
"""
import argparse
import csv as csvmod
import json
import os
import numpy as np

from rls_head import ForgettingRidgeRLS


def load_task(cfg, task, backbone, img_size, mpd, sample_seed=42):
    if task == "age":
        from datasets_age import AgeDomains, build_age_config
        dom = AgeDomains(build_age_config(cfg), backbone=backbone, img_size=img_size, max_per_domain=mpd)
        def arr(split, t):
            Xs, ys = [], []
            for f, a, _ in dom.stream(split, t):
                Xs.append(np.asarray(f, dtype=np.float64).reshape(-1)); ys.append(float(np.ravel(a)[0]))
            return np.asarray(Xs), np.asarray(ys)
    else:
        from datasets_real import RealCountingDomains, build_from_config
        dom = RealCountingDomains(
            build_from_config(cfg),
            backbone=backbone,
            img_size=img_size,
            max_per_domain=mpd,
            sample_seed=sample_seed,
        )
        def arr(split, t):
            Xs, ys = [], []
            for X, Y, _ in dom.stream(split, t):
                Xs.append(np.asarray(X, dtype=np.float64).mean(0)); ys.append(float(np.asarray(Y).sum()))
            return np.asarray(Xs), np.asarray(ys)
    T = dom.n_domains()
    tr = {t: arr("train", t) for t in range(T)}
    te = {t: arr("test", t) for t in range(T)}
    return tr, te, [d["name"] for d in cfg["domains"]]


def rel_mae(pred, gt):
    return float(np.mean(np.abs(pred - gt)) / max(np.mean(gt), 1e-6))


def estimate(tr, lam, val_every=5):
    """Per-domain {w_k, mu_k, r_k, rho_k} from TRAIN, normalized targets.
    Noise r_k is estimated on a HELD-OUT split (generalization residual) so a
    high-dim head interpolating the training set doesn't drive r_k -> 0; sigma^2
    is floored to keep rho stable."""
    T = len(tr); pd = []
    for t in range(T):
        X, y = tr[t]; s = max(float(np.mean(y)), 1e-6); yn = (y / s); n = X.shape[0]
        idx = np.arange(n); vm = (idx % val_every == val_every - 1)
        Xf, yf, Xv, yv = X[~vm], yn[~vm], X[vm], yn[vm]
        if Xv.shape[0] == 0:
            Xf, yf, Xv, yv = X, yn, X, yn
        hf = ForgettingRidgeRLS(d_in=X.shape[1], d_out=1, lam=lam); hf.begin_task()
        hf.accumulate(X, yn.reshape(-1, 1)); hf.solve()              # full data -> w_k, mass
        d_aug = hf.R.shape[0]; mu = float(np.trace(hf.R) / d_aug)
        hv = ForgettingRidgeRLS(d_in=X.shape[1], d_out=1, lam=lam); hv.begin_task()
        hv.accumulate(Xf, yf.reshape(-1, 1)); hv.solve()            # fit-only -> held-out noise
        r = float(hv.residual_energy(Xv, yv.reshape(-1, 1)))
        sig2 = max(float(np.var(yn)), 1e-3)
        s_k = mu / max(n, 1)
        pd.append(dict(w=hf.W.copy(), mu=mu, r=r, rho=s_k / sig2, n=n, s=s))
    return pd, d_aug


def f_pred(pd, d_aug, lam, grid, gain=1.0):
    T = len(pd)
    mus = np.array([p["mu"] for p in pd]); rs = np.array([p["r"] for p in pd])
    rhos = np.array([p["rho"] for p in pd]); Ws = [p["w"] for p in pd]
    tau = np.array([T - 1 - k for k in range(T)], dtype=np.float64)
    best = (1.0, 1e30); curve = []
    for f in grid:
        a = (f ** tau) * mus; Mf = a.sum(); beta = a / (Mf + lam)
        Wbar = sum(beta[k] * Ws[k] for k in range(T))
        bias = float(np.mean([rhos[k] * np.sum((Wbar - Ws[k]) ** 2) for k in range(T)]))
        V = float(np.sum((f ** (2 * tau)) * rs * mus) / (Mf + lam) ** 2)
        var = gain * d_aug * float(np.mean(rhos)) * V
        J = bias + var; curve.append((float(f), J))
        if J < best[1]: best = (float(f), J)
    return best[0], curve


def f_oracle(tr, te, forgets, lam):
    T = len(tr); d = tr[0][0].shape[1]
    s = {t: max(float(np.mean(tr[t][1])), 1e-6) for t in tr}
    best = (1.0, 1e30); rows = []
    for f in forgets:
        h = ForgettingRidgeRLS(d_in=d, d_out=1, lam=lam, forget=f)
        for t in range(T):
            h.begin_task(); h.accumulate(tr[t][0], (tr[t][1] / s[t]).reshape(-1, 1)); h.solve()
        rels = [rel_mae(h.predict(te[i][0])[:, 0] * s[i], te[i][1]) for i in range(T)]
        r = float(np.mean(rels)); rows.append((f, r))
        if r < best[1]: best = (float(f), r)
    return best[0], best[1], rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", choices=["crowd", "age"], required=True)
    ap.add_argument("--forgets", default="1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.25,0.2,0.15,0.1,0.05")
    ap.add_argument("--lam", type=float, default=1e2)
    ap.add_argument("--img-size", type=int, default=518)
    ap.add_argument("--backbone", default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--max-per-domain", type=int, default=400)
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--var-gain", type=float, default=1.0)
    ap.add_argument("--csv", default="runs_real/theory_points.csv")
    a = ap.parse_args()
    forgets = [float(x) for x in a.forgets.split(",")]
    grid = np.round(np.arange(0.05, 1.0001, 0.05), 3)

    with open(a.config) as f:
        cfg = json.load(f)
    tr, te, names = load_task(
        cfg,
        a.task,
        a.backbone,
        a.img_size,
        a.max_per_domain,
        sample_seed=a.sample_seed,
    )
    pd, d_aug = estimate(tr, a.lam)
    fp, _ = f_pred(pd, d_aug, a.lam, grid, gain=a.var_gain)
    fo, relo, _ = f_oracle(tr, te, forgets, a.lam)

    print(f"[{os.path.basename(a.config)}] task={a.task} T={len(tr)} "
          f"names={names}")
    print(f"  mu={[round(p['mu'],1) for p in pd]}  r={[round(p['r'],3) for p in pd]}  "
          f"rho={[round(p['rho'],3) for p in pd]}")
    print(f"  ==> f*_pred = {fp:g}   |   f*_oracle = {fo:g}  (rel={relo:.4f})")

    os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
    new = not os.path.exists(a.csv)
    with open(a.csv, "a", newline="") as fh:
        w = csvmod.writer(fh)
        if new: w.writerow(["config", "task", "T", "f_pred", "f_oracle", "rel_oracle"])
        w.writerow([os.path.basename(a.config), a.task, len(tr), fp, fo, round(relo, 4)])
    print(f"  appended -> {a.csv}")


if __name__ == "__main__":
    main()
