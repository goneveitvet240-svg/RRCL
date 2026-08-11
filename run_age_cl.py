"""
Transfer test: does the f=1-vs-f<1 (magnitude vs drift) phenomenon generalize
from crowd counting to ANOTHER continual visual regression task — age estimation?

Same protocol as the crowd pipeline, scalar regression (image -> age):
  * norm-ablation: optimal forgetting under none vs per-domain (mean) target norm.
      - magnitude-only domains (e.g. UTK young -> UTK old): expect collapse to f=1 under mean.
      - genuine cross-dataset drift (e.g. AgeDB -> UTK): expect f<1 to survive under mean.
  * adaptive f: held-out-validation sufficient-statistic selector (no test set).
  * baselines: f1(=matched joint ridge), single-domain, RanPAC-style random
    features (projection+f1), SGD(Adam, sequential).

Metric: scale-invariant relative MAE = mean_i( MAE_i / mean_age_i ), over seen domains.

Usage:
  python run_age_cl.py --config domains_age_agedb_utk.json \
    --img-size 518 --backbone vit_base_patch14_dinov2.lvd142m \
    --lam 100 --max-per-domain 400 --ranpac-dim 2000 \
    --out runs_real/age_agedb_utk
"""
import argparse
import json
import os
import time
import numpy as np

from adaptive_selector import BalancedSufficientStatsSelector
from evaluation_metrics import balanced_metric_mean, scalar_regression_metrics
from rls_head import ForgettingRidgeRLS
from run_adaptive_f import classify_adaptive_result
from run_provenance import with_provenance
from result_io import dump_result
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
    return float(np.mean(np.abs(pred - gt)) / max(np.mean(gt), 1e-6))


def scales(tr, mode):
    return {t: (float(np.mean(tr[t][1])) if mode == "mean" else 1.0) for t in tr}


def aug(X):
    return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)


def final_age_metrics(head, te, domain_scales, predict=None):
    per_domain = []
    for domain in te:
        X, target = te[domain]
        prediction = (
            predict(X, domain)
            if predict is not None
            else head.predict(X)[:, 0] * domain_scales[domain]
        )
        row = scalar_regression_metrics(prediction, target)
        row["domain_index"] = domain
        per_domain.append(row)
    return {"per_domain": per_domain, "balanced": balanced_metric_mean(per_domain)}


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
    return float(np.nanmean(M[T - 1, :T])), M, final_age_metrics(head, te, s)


def sweep(tr, te, forgets, lam, mode):
    rows = []; best = (1.0, 1e18)
    for f in forgets:
        r, _, _ = train_eval(tr, te, f, lam, mode)
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
    search_mode="grid",
    abstain_relative_gain=1e-4,
    max_component_relative_harm=None,
):
    T = len(tr); d = tr[0][0].shape[1]; d_aug = d + 1; s = scales(tr, mode)
    head = ForgettingRidgeRLS(d_in=d, d_out=1, lam=lam)
    selector = BalancedSufficientStatsSelector(
        d_aug, lam, search_mode=search_mode, grid=sel_grid,
        abstain_relative_gain=abstain_relative_gain,
        max_component_relative_harm=max_component_relative_harm,
    )
    f_used, drift_scores, selections = [], [], []
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
        selections.append(
            {
                "domain_index": t,
                "factor": selection.factor,
                "objective": selection.objective,
                "objective_f1": selection.objective_f1,
                "drift_score": selection.drift_score,
                "historical_relative_harm": selection.historical_relative_harm,
                "current_relative_harm": selection.current_relative_harm,
                "safety_abstained": selection.safety_abstained,
            }
        )
        head.begin_task(f_t); head.accumulate(X, yn); head.solve()
        for i in range(t + 1):
            Xt, yt = te[i]; pred = head.predict(Xt)[:, 0] * s[i]; M[t, i] = rel_mae(pred, yt)
    return (
        float(np.nanmean(M[T - 1, :T])),
        f_used,
        drift_scores,
        selector.state_bytes,
        M.tolist(),
        final_age_metrics(head, te, s),
        selections,
    )


def single_domain(tr, te, lam, mode):
    s = scales(tr, mode); rels = []; task_metrics = []
    for t in tr:
        d = tr[t][0].shape[1]
        h = ForgettingRidgeRLS(d_in=d, d_out=1, lam=lam); h.begin_task()
        X, y = tr[t]; h.accumulate(X, (y / s[t]).reshape(-1, 1)); h.solve()
        Xt, yt = te[t]; prediction = h.predict(Xt)[:, 0] * s[t]
        rels.append(rel_mae(prediction, yt))
        row = scalar_regression_metrics(prediction, yt)
        row["domain_index"] = t
        task_metrics.append(row)
    return float(np.mean(rels)), {
        "per_domain": task_metrics,
        "balanced": balanced_metric_mean(task_metrics),
    }


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
            Xt, yt = te[i]; pred = np.clip((feat(Xt) @ W)[:, 0] * s[i], 0, None); M[t, i] = rel_mae(pred, yt)
    metrics = final_age_metrics(
        None,
        te,
        s,
        predict=lambda X, domain: np.clip(
            (feat(X) @ W)[:, 0] * s[domain], 0, None
        ),
    )
    return float(np.nanmean(M[T - 1, :T])), M.tolist(), metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--forgets", default="1.0,0.8,0.6,0.4,0.3,0.25,0.2,0.15,0.1")
    ap.add_argument("--lam", type=float, default=1e2)
    ap.add_argument("--img-size", type=int, default=518)
    ap.add_argument("--backbone", default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--max-per-domain", type=int, default=400)
    ap.add_argument("--ranpac-dim", type=int, default=2000)
    ap.add_argument(
        "--selector-search",
        choices=["continuous", "grid"],
        default="grid",
        help="grid is the paper default because the selector objective is not "
             "proven unimodal",
    )
    ap.add_argument("--abstain-relative-gain", type=float, default=1e-4)
    ap.add_argument("--max-component-relative-harm", type=float, default=None)
    ap.add_argument("--out", default="runs_real/age")
    a = ap.parse_args()
    forgets = [float(x) for x in a.forgets.split(",")]
    sel_grid = np.round(np.arange(0.05, 1.0001, 0.05), 3)

    from datasets_age import AgeDomains, build_age_config
    with open(a.config) as f:
        cfg = json.load(f)
    domains = AgeDomains(build_age_config(cfg), backbone=a.backbone, img_size=a.img_size,
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
        res[f"sweep_{mode}"] = [
            {"factor": factor, "rel_MAE": score} for factor, score in rows
        ]
        print(f"  [{mode:4s}] best f = {bf:g}  rel = {br:.4f}   sweep=" +
              " ".join(f"{f:g}:{r:.3f}" for f, r in rows))
    f_none, f_mean = res["opt_f_none"], res["opt_f_mean"]
    if f_mean < 0.9:
        normalization_interpretation = (
            f"归一化目标下最优 f={f_mean:g}<1，存在可由历史降权缓解的"
            f"残余负迁移；该结果本身不能证明概念漂移；raw 最优 f={f_none:g}"
        )
    else:
        normalization_interpretation = (
            f"归一化后 f≈1（={f_mean:g}），固定因子扫描未显示遗忘收益；"
            f"raw 最优 f={f_none:g}"
        )
    res["normalization_interpretation"] = normalization_interpretation
    print("  INTERPRETATION:", normalization_interpretation)

    print("\n=== adaptive f / baselines (normalized regime, rel MAE) ===")
    rel_f1, f1_matrix, f1_metrics = train_eval(tr, te, 1.0, a.lam, "mean")
    (
        rel_ad,
        f_used,
        drift_scores,
        selector_state_bytes,
        adaptive_matrix,
        adaptive_metrics,
        selection_records,
    ) = adaptive(
        tr, te, a.lam, "mean", sel_grid,
        search_mode=a.selector_search,
        abstain_relative_gain=a.abstain_relative_gain,
        max_component_relative_harm=a.max_component_relative_harm,
    )
    rel_sd, single_metrics = single_domain(tr, te, a.lam, "mean")
    rel_rp, ranpac_matrix, ranpac_metrics = train_eval(
        tr, te, 1.0, a.lam, "mean", proj_dim=a.ranpac_dim
    )
    rel_sgd, sgd_matrix, sgd_metrics = sgd_seq(tr, te, "mean")
    verdict = classify_adaptive_result(
        rel_f1,
        res["opt_rel_mean"],
        rel_ad,
        f_used,
    )
    res.update(dict(f1=rel_f1, adaptive=rel_ad, f_used=f_used,
                    drift_scores=drift_scores, selector_search=a.selector_search,
                    selector_state_bytes=selector_state_bytes, single=rel_sd,
                    ranpac_style=rel_rp, sgd=rel_sgd, verdict=verdict,
                    evaluation=dict(
                        f1={"M_rel": f1_matrix.tolist(), **f1_metrics},
                        adaptive={"M_rel": adaptive_matrix, **adaptive_metrics},
                        ranpac_style={"M_rel": ranpac_matrix.tolist(), **ranpac_metrics},
                        single=single_metrics,
                        sgd={"M_rel": sgd_matrix, **sgd_metrics},
                    ),
                    selection_records=selection_records,
                    data_manifest=domains.data_manifest()))
    print(f"  f1 (=matched joint ridge) rel = {rel_f1:.4f}")
    print(f"  adaptive f         rel = {rel_ad:.4f}   f_used={[round(x,2) for x in f_used]}")
    print(f"  single-domain      rel = {rel_sd:.4f}")
    print(f"  RanPAC-style(proj={a.ranpac_dim},f=1) rel = {rel_rp:.4f}")
    print(f"  SGD-seq(Adam)      rel = {rel_sgd:.4f}")
    print(f"  -> adaptive vs f1: {(rel_f1-rel_ad)/max(rel_f1,1e-9)*100:+.1f}%")

    os.makedirs(a.out, exist_ok=True)
    payload = with_provenance(
        dict(config=a.config, names=names, **res), a.config, vars(a)
    )
    dump_result(os.path.join(a.out, "age_result.json"), payload)
    print(f"\nsaved -> {os.path.join(a.out, 'age_result.json')}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
