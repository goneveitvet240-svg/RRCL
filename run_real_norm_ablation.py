"""
DECISIVE diagnostic: is the order-dependent optimal forgetting factor f* a
TARGET-MAGNITUDE artifact, or a genuine drift effect?

Synthetic finding (linear-Gaussian): the JHU-first(f<1) / JHU-last(f=1) flip is
reproduced *only* by target-magnitude imbalance, and it collapses once each
domain's targets are normalized on the ACCUMULATION side. A scale-invariant
metric alone does NOT remove it -- the accumulation must be normalized.

This script tests that on REAL data, reusing the existing pipeline unchanged:
  - same DINOv2 features, same patch DCT target, same image-aux head, same RLS.
  - adds per-domain target normalization on accumulation: targets /= s_t,
    where s_t = mean image count of domain t's TRAIN split (current domain only;
    no old samples, no test set -> exemplar-free and CL-legal).
  - reports BOTH raw avg MAE (JHU-dominated, for continuity with old numbers)
    and a scale-invariant RELATIVE MAE = mean_i( |pred-gt| / mean_gt_i ).
  - sweeps f under target-norm in {none, mean} in ONE run, side by side.

Read-off:
  * none mode should reproduce your existing optimum (e.g. JHU-first -> f<1).
  * if under `mean` the optimum moves to f~=1 (flip disappears) -> MAGNITUDE artifact
    (advisor's concern is correct; reframe paper around mass-normalization).
  * if the optimum stays f<1 under `mean` -> genuine drift survives normalization
    (the headline holds; proceed with adaptive f).

Usage (run once per order config on AutoDL):
  python run_real_norm_ablation.py --config domains_jhu_sha_shb.json \
    --img-size 518 --backbone vit_base_patch14_dinov2.lvd142m \
    --lam 100 --patch-target dct5 --alpha 0.25 \
    --forgets 1.0,0.8,0.6,0.4,0.35,0.3,0.25,0.2,0.15,0.1 \
    --max-per-domain 400 --out runs_real/normabl_jhu_sha_shb
"""
import argparse
import json
import os
import time
import numpy as np

from rls_head import ForgettingRidgeRLS
from metrics import mae
from run_real import _first_dim
from run_real_image_aux import transform_patch_target, _image_feature, _image_target
from run_provenance import with_provenance


def domain_scale(domains, t, mode):
    """Per-domain target scale from the TRAIN split (current domain only)."""
    if mode == "none":
        return 1.0
    totals = [float(np.asarray(Y).sum()) for _, Y, _ in domains.stream("train", t)]
    return max(float(np.mean(totals)), 1e-6)


def eval_domain(domains, i, patch_head, image_head, alpha, s_i):
    preds, gts = [], []
    for X, Y, _ in domains.stream("test", i):
        pc = float(patch_head.predict(X)[:, 0].sum())
        ic = float(image_head.predict(_image_feature(X))[0, 0])
        pred = (alpha * pc + (1.0 - alpha) * ic) * s_i      # un-normalize to raw count
        preds.append(max(pred, 0.0))
        gts.append(float(np.asarray(Y).sum()))
    preds = np.asarray(preds); gts = np.asarray(gts)
    abs_mae = mae(preds, gts)
    rel_mae = abs_mae / max(float(np.mean(gts)), 1e-6)
    return abs_mae, rel_mae


def train_eval(domains, patch_target, forget, alpha, lam, target_norm):
    T = domains.n_domains()
    d_in = _first_dim(domains)
    patch_head = ForgettingRidgeRLS(d_in=d_in, d_out=1, lam=lam, forget=forget)
    image_head = ForgettingRidgeRLS(d_in=d_in, d_out=1, lam=lam, forget=forget)
    M_abs = np.full((T, T), np.nan)
    M_rel = np.full((T, T), np.nan)
    scales = []
    for t in range(T):
        s_t = domain_scale(domains, t, target_norm)
        scales.append(s_t)
        patch_head.begin_task(); image_head.begin_task()
        for X, Y, _ in domains.stream("train", t):
            py = transform_patch_target(Y, patch_target) / s_t
            iy = _image_target(Y) / s_t
            patch_head.accumulate(X, py)
            image_head.accumulate(_image_feature(X), iy)
        patch_head.solve(); image_head.solve()
        for i in range(t + 1):
            M_abs[t, i], M_rel[t, i] = eval_domain(
                domains, i, patch_head, image_head, alpha, scales[i]
            )
    final_abs = float(np.nanmean(M_abs[T - 1, :T]))
    final_rel = float(np.nanmean(M_rel[T - 1, :T]))
    return final_abs, final_rel, M_abs.tolist(), M_rel.tolist(), scales


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--forgets", default="1.0,0.8,0.6,0.4,0.35,0.3,0.25,0.2,0.15,0.1")
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--patch-target", default="dct5")
    ap.add_argument("--lam", type=float, default=1e2)
    ap.add_argument("--img-size", type=int, default=518)
    ap.add_argument("--backbone", default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--max-per-domain", type=int, default=400)
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--target-norms", default="none,mean")
    ap.add_argument("--out", default="runs_real/normabl")
    a = ap.parse_args()

    forgets = [float(x) for x in a.forgets.split(",")]
    modes = [x.strip() for x in a.target_norms.split(",") if x.strip()]

    from datasets_real import RealCountingDomains, build_from_config
    with open(a.config) as f:
        cfg = json.load(f)
    domains = RealCountingDomains(build_from_config(cfg), backbone=a.backbone,
                                 img_size=a.img_size, max_per_domain=a.max_per_domain,
                                 sample_seed=a.sample_seed)
    data_manifest = domains.data_manifest()

    t0 = time.time()
    results = {}
    summary = {}
    for mode in modes:
        rows = []
        for f in forgets:
            fa, fr, Ma, Mr, sc = train_eval(domains, a.patch_target, f, a.alpha, a.lam, mode)
            results[f"norm={mode},f={f:g}"] = dict(target_norm=mode, forget=f,
                                                   final_avg_mae=fa, final_avg_rel=fr,
                                                   scales=sc, M_abs=Ma, M_rel=Mr)
            rows.append((f, fa, fr))
            print(f"[norm={mode:4s} f={f:<4g}] raw_avg_MAE={fa:8.3f}  rel_MAE={fr:.4f}")
        best_rel = min(rows, key=lambda r: r[2])
        best_raw = min(rows, key=lambda r: r[1])
        summary[mode] = dict(best_f_by_rel=best_rel[0], best_rel=best_rel[2],
                             best_f_by_raw=best_raw[0], best_raw=best_raw[1])
        print(f"  -> [{mode}] best f by REL = {best_rel[0]:g} (rel={best_rel[2]:.4f}); "
              f"best f by raw = {best_raw[0]:g} (raw={best_raw[1]:.3f})")

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "norm_ablation.json"), "w") as fh:
        payload = dict(config=a.config, alpha=a.alpha, patch_target=a.patch_target,
                       lam=a.lam, forgets=forgets, summary=summary, results=results,
                       data_manifest=data_manifest)
        json.dump(with_provenance(payload, a.config, vars(a)), fh,
                  indent=2, ensure_ascii=False)

    # verdict
    print("\n===== VERDICT (decisive read-off) =====")
    for mode in modes:
        s = summary[mode]
        print(f"  norm={mode:4s}: optimal f (by relative MAE) = {s['best_f_by_rel']:g}")
    if "none" in summary and "mean" in summary:
        f_none = summary["none"]["best_f_by_rel"]
        f_mean = summary["mean"]["best_f_by_rel"]
        if f_none < 0.95 and f_mean >= 0.9:
            print("  => order/forgetting benefit COLLAPSES under normalization "
                  "-> magnitude artifact (advisor likely right; reframe).")
        elif f_none < 0.95 and f_mean < 0.9:
            print("  => forgetting still helps AFTER normalization "
                  "-> genuine drift survives (headline holds).")
        else:
            print("  => f=1 already optimal here even raw; this order is not a forgetting case.")
    print(f"\nsaved -> {os.path.join(a.out, 'norm_ablation.json')}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
