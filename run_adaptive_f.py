"""
Adaptive forgetting factor on the POST-NORMALIZATION drift signal.

Why post-normalization: the magnitude-vs-drift ablation showed raw innovation
conflates target scale with genuine drift (SHA->SHB forgetting was a magnitude
artifact). So we (a) normalize each domain's targets on accumulation, then
(b) measure drift from the innovation of the OLD weights on the NEW domain's
normalized training data -- a scale-invariant drift proxy. This removes the
fragility of the naive innovation rule (which over-forgot on scale-only cases).

Rule (boundary-triggered, exemplar-free, no test set):
  at domain t>=1:
     eps_t  = mean residual energy of current W on domain t's NORMALIZED train
     rho_t  = (eps_t - r_hat)+ / r_hat          # excess error over noise floor
     f_t    = f_star(rho_t)  in [f_min, 1]      # closed-form optimal forgetting
  domain boundaries are known (domain-incremental), so no drift DETECTION needed,
  only drift MAGNITUDE estimation. r_hat is an EMA noise floor from post-fit
  residuals. All of f_star / adaptive_forget_from_innovation / update_noise_floor
  already live in rls_head.py; this script just drives them on normalized targets.

Compares, under normalized accumulation:
   f=1 (absolute memory) | oracle best fixed f (grid; cheats, uses test) | adaptive f
Expected:
   * JHU-first orders : adaptive picks f<1, ~= oracle, beats f=1.
   * SHA->SHB / JHU-last : adaptive STAYS ~1 (no over-forgetting) -> robustness.

Usage (per order, on AutoDL):
  python run_adaptive_f.py --config domains_jhu_sha_shb.json \
    --img-size 518 --backbone vit_base_patch14_dinov2.lvd142m \
    --lam 100 --patch-target dct5 --alpha 0.25 \
    --forgets 1.0,0.8,0.6,0.4,0.35,0.3,0.25,0.2,0.15,0.1 \
    --max-per-domain 400 --out runs_real/adaptf_jhu_sha_shb
"""
import argparse
import json
import os
import time
import numpy as np

from adaptive_selector import BalancedSufficientStatsSelector
from rls_head import ForgettingRidgeRLS, f_star
from run_provenance import with_provenance
from run_real import _first_dim
from run_real_image_aux import transform_patch_target, _image_feature, _image_target
from run_real_norm_ablation import domain_scale, eval_domain, train_eval


def domain_resid_energy(domains, t, patch_head, patch_target, s_t):
    """Mean per-image residual energy of current W on domain t's NORMALIZED train."""
    tot, n = 0.0, 0
    for X, Y, _ in domains.stream("train", t):
        py = transform_patch_target(Y, patch_target) / s_t
        e = patch_head.residual_energy(X, py)
        if e is not None:
            tot += e
            n += 1
    return tot / max(n, 1)


def domain_floor_energy(domains, t, d_in, patch_target, lam, s_t):
    """Best achievable residual on domain t (fit a fresh head on domain t ALONE).
    This is the underfitting-robust baseline: drift = how much WORSE the old W is
    than a domain-only fit, not how big the absolute residual is."""
    h = ForgettingRidgeRLS(d_in=d_in, d_out=1, lam=lam)
    h.begin_task()
    for X, Y, _ in domains.stream("train", t):
        h.accumulate(X, transform_patch_target(Y, patch_target) / s_t)
    h.solve()
    tot, n = 0.0, 0
    for X, Y, _ in domains.stream("train", t):
        e = h.residual_energy(X, transform_patch_target(Y, patch_target) / s_t)
        if e is not None:
            tot += e
            n += 1
    return tot / max(n, 1)


def adaptive_train_eval(domains, patch_target, alpha, lam, f_min, f_max, noise_ema):
    T = domains.n_domains()
    d_in = _first_dim(domains)
    patch_head = ForgettingRidgeRLS(d_in=d_in, d_out=1, lam=lam, adaptive=True,
                                    f_min=f_min, f_max=f_max, noise_ema=noise_ema)
    image_head = ForgettingRidgeRLS(d_in=d_in, d_out=1, lam=lam)   # follows same f_t
    M_rel = np.full((T, T), np.nan)
    scales, f_used, rho_used = [], [], []
    for t in range(T):
        s_t = domain_scale(domains, t, "mean")
        scales.append(s_t)
        if t == 0:
            f_t, rho = 1.0, 0.0
        else:
            eps_old = domain_resid_energy(domains, t, patch_head, patch_target, s_t)
            eps_floor = domain_floor_energy(domains, t, d_in, patch_target, lam, s_t)
            rho = max(eps_old - eps_floor, 0.0) / max(eps_floor, 1e-12)
            f_t = float(np.clip(f_star(rho), f_min, f_max))
        f_used.append(float(f_t))
        rho_used.append(None if rho is None else float(rho))
        patch_head.begin_task(f_t)
        image_head.begin_task(f_t)
        for X, Y, _ in domains.stream("train", t):
            py = transform_patch_target(Y, patch_target) / s_t
            iy = _image_target(Y) / s_t
            patch_head.accumulate(X, py)
            image_head.accumulate(_image_feature(X), iy)
        patch_head.solve()
        image_head.solve()
        for i in range(t + 1):
            _, M_rel[t, i] = eval_domain(domains, i, patch_head, image_head, alpha, scales[i])
    final_rel = float(np.nanmean(M_rel[T - 1, :T]))
    evidence = {
        "domain_names": [spec.name for spec in domains.domains],
        "scales": scales,
        "M_rel": M_rel.tolist(),
    }
    return final_rel, f_used, rho_used, evidence


def _aug_raw(X):
    X = np.asarray(X, dtype=np.float64)
    return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)


def _domain_suffstats_split(domains, t, patch_target, s_t, d_aug, val_every=5):
    """Per-domain sufficient stats, split into FIT and VAL by image index.
    VAL is held-out TRAINING data (never the test set); it lets f-selection see
    generalization, so the selector forgets the RIGHT amount -- training-error
    selection alone under-forgets (more retained data always lowers train error)."""
    def z():
        return [np.zeros((d_aug, d_aug)), np.zeros((d_aug, 1)), 0.0, 0]
    fit, val = z(), z()
    for i, (X, Y, _) in enumerate(domains.stream("train", t)):
        Xa = _aug_raw(X)
        yv = transform_patch_target(Y, patch_target) / s_t
        tgt = val if (i % val_every == val_every - 1) else fit
        tgt[0] += Xa.T @ Xa; tgt[1] += Xa.T @ yv
        tgt[2] += float(np.sum(yv * yv)); tgt[3] += Xa.shape[0]
    return fit, val


def adaptive_train_eval_suff(
    domains,
    patch_target,
    alpha,
    lam,
    f_min,
    f_max,
    sel_grid,
    search_mode="grid",
    abstain_relative_gain=1e-4,
):
    """Select f at each boundary by minimizing HELD-OUT (validation) balanced error
    over seen domains. Historical fit and validation quadratics are accumulated in
    O(d^2) total state; no per-domain matrices or exemplars are retained. The
    deployed head still trains on the FULL domain."""
    T = domains.n_domains()
    d_in = _first_dim(domains)
    d_aug = d_in + 1
    patch_head = ForgettingRidgeRLS(d_in=d_in, d_out=1, lam=lam)   # deployed model
    image_head = ForgettingRidgeRLS(d_in=d_in, d_out=1, lam=lam)
    selector = BalancedSufficientStatsSelector(
        dimension=d_aug,
        lam=lam,
        f_min=f_min,
        f_max=f_max,
        search_mode=search_mode,
        grid=sel_grid,
        abstain_relative_gain=abstain_relative_gain,
    )
    scales, f_used, obj_used, drift_scores, selections = [], [], [], [], []
    M_rel = np.full((T, T), np.nan)
    for t in range(T):
        s_t = domain_scale(domains, t, "mean"); scales.append(s_t)
        fit, val = _domain_suffstats_split(domains, t, patch_target, s_t, d_aug)
        selection = selector.select_and_update(fit, val)
        f_t = selection.factor
        f_used.append(f_t)
        obj_used.append(selection.objective)
        drift_scores.append(selection.drift_score)
        selections.append(
            {
                "domain": domains.domains[t].name,
                "factor": selection.factor,
                "objective": selection.objective,
                "objective_f1": selection.objective_f1,
                "drift_score": selection.drift_score,
                "search_mode": selection.search_mode,
            }
        )
        patch_head.begin_task(f_t); image_head.begin_task(f_t)
        for X, Y, _ in domains.stream("train", t):
            py = transform_patch_target(Y, patch_target) / s_t
            iy = _image_target(Y) / s_t
            patch_head.accumulate(X, py); image_head.accumulate(_image_feature(X), iy)
        patch_head.solve(); image_head.solve()
        for i in range(t + 1):
            _, M_rel[t, i] = eval_domain(domains, i, patch_head, image_head, alpha, scales[i])
    final_rel = float(np.nanmean(M_rel[T - 1, :T]))
    evidence = {
        "domain_names": [spec.name for spec in domains.domains],
        "scales": scales,
        "M_rel": M_rel.tolist(),
        "selections": selections,
    }
    return (
        final_rel,
        f_used,
        obj_used,
        drift_scores,
        selector.state_bytes,
        evidence,
    )


def classify_adaptive_result(
    rel_f1,
    oracle_rel,
    adaptive_rel,
    f_used,
    absolute_tolerance=1e-4,
    min_oracle_recovery=0.8,
):
    """Classify selector behavior without calling a missed oracle gain "GOOD".

    The fixed-factor oracle is diagnostic and test-selected.  Its gain over
    f=1 defines how much recoverable signal exists in this particular sweep.
    When that gain is material, a selector only counts as successful if it
    recovers the requested fraction of it.
    """
    rel_f1 = float(rel_f1)
    oracle_rel = float(oracle_rel)
    adaptive_rel = float(adaptive_rel)
    tolerance = float(absolute_tolerance)
    oracle_gain = max(rel_f1 - oracle_rel, 0.0)
    adaptive_gain = rel_f1 - adaptive_rel
    all_absolute_memory = all(abs(float(factor) - 1.0) <= 1e-9 for factor in f_used)

    if adaptive_gain < -tolerance:
        status = "worse_than_f1"
        success = False
        recovery = (
            adaptive_gain / oracle_gain if oracle_gain > tolerance else None
        )
    elif oracle_gain <= tolerance:
        recovery = None
        if all_absolute_memory:
            status = "correct_abstention_no_fixed_oracle_gain"
            success = True
        else:
            status = "neutral_but_unnecessary_forgetting"
            success = False
    else:
        recovery = adaptive_gain / oracle_gain
        if adaptive_gain <= tolerance:
            status = "missed_fixed_oracle_gain"
            success = False
        elif recovery >= min_oracle_recovery:
            status = "captures_fixed_oracle_gain"
            success = True
        else:
            status = "partial_fixed_oracle_gain"
            success = False

    return {
        "status": status,
        "success": success,
        "oracle_gain_abs": oracle_gain,
        "adaptive_gain_abs": adaptive_gain,
        "oracle_gain_recovery": recovery,
        "all_absolute_memory": all_absolute_memory,
        "absolute_tolerance": tolerance,
        "min_oracle_recovery": float(min_oracle_recovery),
        "oracle_is_test_selected_diagnostic": True,
    }


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
    ap.add_argument("--f-min", type=float, default=0.05)
    ap.add_argument("--f-max", type=float, default=1.0)
    ap.add_argument("--noise-ema", type=float, default=0.5)
    ap.add_argument("--selector", choices=["sufficient", "innovation"], default="sufficient",
                    help="sufficient = suff-stat balanced-error search (recommended); "
                         "innovation = f_star(rho) heuristic")
    ap.add_argument(
        "--selector-search",
        choices=["continuous", "grid"],
        default="grid",
        help="grid is the paper default because the objective is not proven unimodal; "
             "continuous is an ablation with coarse-grid initialization",
    )
    ap.add_argument("--abstain-relative-gain", type=float, default=1e-4)
    ap.add_argument("--verdict-absolute-tolerance", type=float, default=1e-4)
    ap.add_argument("--min-oracle-recovery", type=float, default=0.8)
    ap.add_argument("--out", default="runs_real/adaptf")
    a = ap.parse_args()

    forgets = [float(x) for x in a.forgets.split(",")]

    from datasets_real import RealCountingDomains, build_from_config
    with open(a.config) as f:
        cfg = json.load(f)
    domains = RealCountingDomains(build_from_config(cfg), backbone=a.backbone,
                                 img_size=a.img_size, max_per_domain=a.max_per_domain,
                                 sample_seed=a.sample_seed)
    data_manifest = domains.data_manifest()

    t0 = time.time()
    # all in NORMALIZED accumulation regime (mean), the honest setting
    _, rel_f1, _, f1_matrix, f1_scales = train_eval(
        domains, a.patch_target, 1.0, a.alpha, a.lam, "mean"
    )
    best_f, best_rel = None, float("inf")
    oracle_curve = []
    for f in forgets:
        _, rel, _, _, _ = train_eval(domains, a.patch_target, f, a.alpha, a.lam, "mean")
        oracle_curve.append({"factor": f, "rel_MAE": rel})
        print(f"[oracle scan f={f:<4g}] rel_MAE={rel:.4f}")
        if rel < best_rel:
            best_f, best_rel = f, rel
    if a.selector == "sufficient":
        sel_grid = np.round(np.arange(0.05, 1.0001, 0.05), 3)
        (
            rel_ad,
            f_used,
            selector_objectives,
            drift_scores,
            selector_state_bytes,
            adaptive_evidence,
        ) = (
            adaptive_train_eval_suff(
                domains,
                a.patch_target,
                a.alpha,
                a.lam,
                a.f_min,
                a.f_max,
                sel_grid,
                search_mode=a.selector_search,
                abstain_relative_gain=a.abstain_relative_gain,
            )
        )
        rho_used = None
    else:
        rel_ad, f_used, rho_used, adaptive_evidence = adaptive_train_eval(
            domains, a.patch_target, a.alpha, a.lam, a.f_min, a.f_max, a.noise_ema)
        selector_objectives = None
        drift_scores = rho_used
        selector_state_bytes = None

    gap_to_oracle = rel_ad - best_rel
    improve_over_f1 = (rel_f1 - rel_ad) / max(rel_f1, 1e-12) * 100.0
    verdict = classify_adaptive_result(
        rel_f1,
        best_rel,
        rel_ad,
        f_used,
        absolute_tolerance=a.verdict_absolute_tolerance,
        min_oracle_recovery=a.min_oracle_recovery,
    )

    print("\n===== ADAPTIVE f RESULT (normalized regime) =====")
    print(f"  f=1 (absolute memory)   rel_MAE = {rel_f1:.4f}")
    print(f"  oracle best fixed f={best_f:<4g} rel_MAE = {best_rel:.4f}")
    print(f"  adaptive f              rel_MAE = {rel_ad:.4f}   (f_used={[round(x,2) for x in f_used]})")
    print(f"  -> adaptive vs f=1: {improve_over_f1:+.1f}%   gap-to-oracle: {gap_to_oracle:+.4f}")
    recovery = verdict["oracle_gain_recovery"]
    recovery_text = "n/a" if recovery is None else f"{100.0 * recovery:.1f}%"
    print(
        f"  VERDICT: {verdict['status']} | success={verdict['success']} | "
        f"fixed-oracle gain recovery={recovery_text}"
    )

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "adaptive_f.json"), "w") as fh:
        payload = dict(config=a.config, rel_f1=rel_f1, oracle_f=best_f, oracle_rel=best_rel,
                       adaptive_rel=rel_ad, f_used=f_used, rho_used=rho_used,
                       selector=a.selector, selector_search=a.selector_search,
                       selector_objectives=selector_objectives, drift_scores=drift_scores,
                       selector_state_bytes=selector_state_bytes,
                       adaptive_evidence=adaptive_evidence,
                       f1_evidence=dict(
                           domain_names=[spec.name for spec in domains.domains],
                           scales=f1_scales,
                           M_rel=f1_matrix,
                       ),
                       fixed_oracle_curve=oracle_curve,
                       data_manifest=data_manifest,
                       improve_over_f1_pct=improve_over_f1, gap_to_oracle=gap_to_oracle,
                       verdict=verdict)
        json.dump(with_provenance(payload, a.config, vars(a)), fh, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {os.path.join(a.out, 'adaptive_f.json')}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
