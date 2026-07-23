"""
Baselines for the Pareto/CL comparison, SAME setting as adaptive f
(冻结 DINOv2 + image-aux head, per-domain target normalization "mean",
scale-invariant rel_MAE averaged over seen domains, + nBwT).

Methods:
  cf_f1     : closed-form absolute memory f=1  (== matched centralized ridge,
              because sufficient-statistic addition is order invariant)
  single    : per-domain closed-form model, evaluated on its OWN domain only
              (no sharing / no CL reference)
  sgd_seq   : sequential SGD linear probe over domains, NO replay / NO regularization
              (the standard gradient-based CL lower bound; catastrophic forgetting)

Run (per order, on AutoDL):
  python run_baselines.py --config domains_jhu_sha_shb.json \
    --img-size 518 --backbone vit_base_patch14_dinov2.lvd142m \
    --lam 100 --patch-target dct5 --alpha 0.25 --max-per-domain 400 \
    --sgd-epochs 10 --sgd-lr 0.1 --out runs_real/base_jhu_sha_shb
"""
import argparse
import json
import os
import time
import numpy as np

from rls_head import ForgettingRidgeRLS
from metrics import forgetting_matrix_stats
from run_real import _first_dim
from run_real_image_aux import transform_patch_target, _image_feature, _image_target
from run_real_norm_ablation import domain_scale, eval_domain, train_eval
from run_provenance import with_provenance


class LinHead:
    """Linear head with optional feature standardization + bias; predict() matches
    ForgettingRidgeRLS so eval_domain() works unchanged."""
    def __init__(self, d_in, mu=None, sd=None, bias=True):
        self.bias = bias
        self.mu = mu
        self.sd = sd
        self.W = np.zeros((d_in + (1 if bias else 0), 1), dtype=np.float64)

    def _aug(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.mu is not None:
            X = (X - self.mu) / self.sd
        if self.bias:
            X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        return X

    def predict(self, X, non_negative=True):
        Y = self._aug(X) @ self.W
        return np.clip(Y, 0.0, None) if non_negative else Y


class StandardizedClosedFormHead:
    """Ridge head on the exact standardized image features used by SGD."""

    def __init__(self, d_in, lam, mu, sd, proj_dim=0):
        self.mu = mu
        self.sd = sd
        self.head = ForgettingRidgeRLS(
            d_in=d_in, d_out=1, lam=lam, proj_dim=proj_dim
        )

    def begin_task(self, factor=1.0):
        self.head.begin_task(factor)

    def accumulate(self, X, Y):
        self.head.accumulate((np.asarray(X) - self.mu) / self.sd, Y)

    def solve(self):
        self.head.solve()

    def predict(self, X, non_negative=True):
        return self.head.predict(
            (np.asarray(X) - self.mu) / self.sd, non_negative=non_negative
        )


def feature_standardizer(domains, t=0):
    """mu/sd over domain t's patch features (for SGD stability)."""
    n = 0
    s = None
    ss = None
    for X, _, _ in domains.stream("train", t):
        X = np.asarray(X, dtype=np.float64)
        s = X.sum(0) if s is None else s + X.sum(0)
        ss = (X * X).sum(0) if ss is None else ss + (X * X).sum(0)
        n += X.shape[0]
    mu = s / max(n, 1)
    var = ss / max(n, 1) - mu * mu
    sd = np.sqrt(np.maximum(var, 1e-6))
    return mu, sd


def _clip(g, max_norm=1.0):
    nrm = float(np.linalg.norm(g))
    return g if nrm <= max_norm or nrm == 0.0 else g * (max_norm / nrm)


def matrix_metrics(M):
    st = forgetting_matrix_stats(M)
    return st["final_avg_mae"], st["nBwT"]


def run_single_domain(domains, patch_target, alpha, lam):
    """Independent closed-form model per domain, eval on its own test."""
    T = domains.n_domains()
    d_in = _first_dim(domains)
    rels = []
    for t in range(T):
        s_t = domain_scale(domains, t, "mean")
        ph = ForgettingRidgeRLS(d_in=d_in, d_out=1, lam=lam)
        ih = ForgettingRidgeRLS(d_in=d_in, d_out=1, lam=lam)
        ph.begin_task(); ih.begin_task()
        for X, Y, _ in domains.stream("train", t):
            ph.accumulate(X, transform_patch_target(Y, patch_target) / s_t)
            ih.accumulate(_image_feature(X), _image_target(Y) / s_t)
        ph.solve(); ih.solve()
        _, rel = eval_domain(domains, t, ph, ih, alpha, s_t)
        rels.append(rel)
    return float(np.mean(rels)), rels


def run_closed_form_image_seq(domains, lam, proj_dim=0):
    """Matched image-only f=1 baseline; differs from SGD only in optimizer."""
    T = domains.n_domains()
    d_in = _first_dim(domains)
    mu, sd = feature_standardizer(domains, 0)
    image_head = StandardizedClosedFormHead(d_in, lam, mu, sd, proj_dim=proj_dim)
    dummy_patch = LinHead(d_in, mu, sd)
    scales = [domain_scale(domains, t, "mean") for t in range(T)]
    matrix = np.full((T, T), np.nan)
    for t in range(T):
        image_head.begin_task(1.0)
        for X, Y, _ in domains.stream("train", t):
            image_head.accumulate(_image_feature(X), _image_target(Y) / scales[t])
        image_head.solve()
        for previous in range(t + 1):
            _, matrix[t, previous] = eval_domain(
                domains, previous, dummy_patch, image_head, 0.0, scales[previous]
            )
    final, nbwt = matrix_metrics(matrix)
    return final, nbwt, matrix.tolist()


def run_sgd_seq(domains, patch_target, alpha, lam, epochs, lr, l2):
    """Sequential ADAM linear probe at the IMAGE level (predict normalized total).
    The dense patch head is ill-conditioned for gradient training (per-patch errors
    amplify ~Nx when summed over patches → divergence), so the gradient baseline is
    run image-only — the stable, fair configuration (closed-form has no such issue).
    No replay, no anti-forgetting; warm-started across domains → forgets."""
    T = domains.n_domains()
    d_in = _first_dim(domains)
    mu, sd = feature_standardizer(domains, 0)
    image_head = LinHead(d_in, mu, sd)
    dummy_patch = LinHead(d_in, mu, sd)          # W=0 → contributes 0 at alpha=0
    scales = [domain_scale(domains, t, "mean") for t in range(T)]
    b1, b2, eps = 0.9, 0.999, 1e-8
    mi = np.zeros_like(image_head.W); vi = np.zeros_like(image_head.W); step = 0
    M_rel = np.full((T, T), np.nan)
    for t in range(T):
        s_t = scales[t]
        for _ in range(epochs):
            for X, Y, _ in domains.stream("train", t):
                step += 1
                Xi = image_head._aug(_image_feature(X))
                iy = _image_target(Y) / s_t
                gi = Xi.T @ (Xi @ image_head.W - iy) + l2 * image_head.W
                mi = b1 * mi + (1 - b1) * gi; vi = b2 * vi + (1 - b2) * gi * gi
                image_head.W -= lr * (mi / (1 - b1 ** step)) / (np.sqrt(vi / (1 - b2 ** step)) + eps)
        for i in range(t + 1):
            _, M_rel[t, i] = eval_domain(domains, i, dummy_patch, image_head, 0.0, scales[i])
    final, nbwt = matrix_metrics(M_rel)
    return final, nbwt, M_rel.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--patch-target", default="dct5")
    ap.add_argument("--lam", type=float, default=1e2)
    ap.add_argument("--img-size", type=int, default=518)
    ap.add_argument("--backbone", default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--max-per-domain", type=int, default=400)
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--sgd-epochs", type=int, default=10)
    ap.add_argument("--sgd-lr", type=float, default=0.01)   # Adam lr
    ap.add_argument("--sgd-l2", type=float, default=1e-4)
    ap.add_argument("--ranpac-dim", type=int, default=2000)
    ap.add_argument("--out", default="runs_real/base")
    a = ap.parse_args()

    from datasets_real import RealCountingDomains, build_from_config
    with open(a.config) as f:
        cfg = json.load(f)
    domains = RealCountingDomains(build_from_config(cfg), backbone=a.backbone,
                                 img_size=a.img_size, max_per_domain=a.max_per_domain,
                                 sample_seed=a.sample_seed)
    data_manifest = domains.data_manifest()

    t0 = time.time()
    # f1 (== matched centralized ridge, not a task-performance upper bound)
    _, f1_rel, _, f1_M, _ = train_eval(domains, a.patch_target, 1.0, a.alpha, a.lam, "mean")
    f1_final, f1_nbwt = matrix_metrics(np.asarray(f1_M))
    sd_final, sd_rels = run_single_domain(domains, a.patch_target, a.alpha, a.lam)
    cf_image_final, cf_image_nbwt, cf_image_M = run_closed_form_image_seq(
        domains, a.lam, proj_dim=0
    )
    ranpac_image_final, ranpac_image_nbwt, ranpac_image_M = run_closed_form_image_seq(
        domains, a.lam, proj_dim=a.ranpac_dim
    )
    sgd_final, sgd_nbwt, sgd_M = run_sgd_seq(
        domains, a.patch_target, a.alpha, a.lam, a.sgd_epochs, a.sgd_lr, a.sgd_l2)

    rows = [
        ("cf_f1 (=matched joint)", f1_final, f1_nbwt),
        ("cf_f1_image (matched)", cf_image_final, cf_image_nbwt),
        ("ranpac-style image", ranpac_image_final, ranpac_image_nbwt),
        ("single-domain",     sd_final, float("nan")),
        ("sgd_seq(image)",    sgd_final, sgd_nbwt),
    ]
    print("\n===== BASELINES (normalized regime, rel_MAE; lower=better) =====")
    print(f"{'method':22s} {'final_avg_rel':>14s} {'nBwT':>8s}")
    for name, fin, nb in rows:
        print(f"{name:22s} {fin:>14.4f} {('%.3f'%nb) if nb==nb else '   -':>8s}")
    print("\n参考（前序实验，归一化制度 rel_MAE）：")
    print("  oracle best fixed f 与 adaptive f 见 runs_real/adaptf2_*/adaptive_f.json")

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "baselines.json"), "w") as fh:
        payload = dict(config=a.config, alpha=a.alpha, patch_target=a.patch_target,
                       lam=a.lam, sgd=dict(epochs=a.sgd_epochs, lr=a.sgd_lr, l2=a.sgd_l2),
                       data_manifest=data_manifest,
                       cf_f1_full=dict(final_avg_rel=f1_final, nBwT=f1_nbwt,
                                       M_rel=f1_M),
                       cf_f1_image=dict(final_avg_rel=cf_image_final, nBwT=cf_image_nbwt,
                                        M_rel=cf_image_M),
                       ranpac_style_image=dict(
                                         note="Gaussian random projection + ReLU + ridge; "
                                              "not the full RanPAC classifier",
                                         proj_dim=a.ranpac_dim,
                                         final_avg_rel=ranpac_image_final,
                                         nBwT=ranpac_image_nbwt,
                                         M_rel=ranpac_image_M),
                       single_domain=dict(final_avg_rel=sd_final, per_domain=sd_rels),
                       sgd_seq=dict(final_avg_rel=sgd_final, nBwT=sgd_nbwt, M_rel=sgd_M))
        json.dump(with_provenance(payload, a.config, vars(a)), fh,
                  indent=2, ensure_ascii=False)
    print(f"\nsaved -> {os.path.join(a.out, 'baselines.json')}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
