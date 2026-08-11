"""Domain-incremental counting on REAL data (streaming, memory-safe).

Quick check of the whole real-data code path WITHOUT torch/datasets:
    python run_real.py --selftest --forgets 1.0,0.8,0.6

Real run (after filling a config JSON, see README):
    python run_real.py --config domains.json --forgets 0.9 --img-size 518 --max-per-domain 200

It streams each domain image-by-image into the recursive ridge head
(begin_task -> accumulate per image -> solve), evaluates on all seen domains,
and writes the MAE matrix + nBwT to runs_real/results.json.
"""
import argparse
import json
import os
import time
import numpy as np

from rls_head import ForgettingRidgeRLS
from metrics import mae, forgetting_matrix_stats


class _SelfTest:
    """Fake domains with controllable drift — exercises the SAME streaming loop
    as real data, but needs only numpy (no torch/datasets)."""
    def __init__(self, T=4, d=48, P=64, imgs=40, drift=1.5, seed=0):
        self.T, self.d, self.P, self.imgs, self.drift = T, d, P, imgs, drift
        rng = np.random.default_rng(seed)
        self.wb = rng.normal(0, 1, d) / np.sqrt(d)
        self.delta = [rng.normal(0, 1, d) / np.sqrt(d) for _ in range(T)]
        self.mu = [rng.normal(0, 0.6, d) for _ in range(T)]

    def n_domains(self):
        return self.T

    def _w(self, t):
        return self.wb + self.drift * (t / max(1, self.T - 1)) * self.delta[t]

    def stream(self, split, d):
        rng = np.random.default_rng((1 if split == "train" else 2) * 1000 + d)
        n = self.imgs if split == "train" else max(10, self.imgs // 4)
        for _ in range(n):
            X = rng.normal(self.mu[d], 1.0, size=(self.P, self.d))
            y = np.clip(np.round(X @ self._w(d) + 2.0 + rng.normal(0, 0.4, self.P)), 0, None)
            yield X.astype(np.float64), y.reshape(-1, 1), self.P


def _first_dim(domains):
    for X, _, _ in domains.stream("train", 0):
        return X.shape[1]
    raise RuntimeError("domain 0 has no training images")


def _stream_residual_energy(head, stream):
    """Mean squared residual energy over a stream, weighted by patch count."""
    total, count = 0.0, 0
    for X, Y, _ in stream:
        e = head.residual_energy(X, Y)
        if e is None:
            return None
        n = np.asarray(Y).reshape(-1, head.m).shape[0]
        total += e * n
        count += n
    return None if count == 0 else total / count


def _run_setting(domains, T, d_in, lam, label, forget=1.0,
                 adaptive=False, f_min=0.05, f_max=1.0,
                 noise_ema=0.5):
    head = ForgettingRidgeRLS(
        d_in=d_in, d_out=1, lam=lam, forget=forget,
        adaptive=adaptive, f_min=f_min, f_max=f_max, noise_ema=noise_ema,
    )
    M = np.full((T, T), np.nan)
    task_log = []
    for t in range(T):
        if adaptive:
            eps = _stream_residual_energy(head, domains.stream("train", t))
            if eps is None:
                f_used, rho_hat = f_max, None
            else:
                f_used, rho_hat = head.adaptive_forget_from_innovation(eps)
            head.innovation_history.append(None if eps is None else float(eps))
            head.rho_history.append(None if rho_hat is None else float(rho_hat))
            head.begin_task(forget=f_used)
        else:
            f_used, eps, rho_hat = forget, None, None
            head.begin_task()

        n = 0
        for X, Y, _ in domains.stream("train", t):
            head.accumulate(X, Y)
            n += 1
        head.solve()

        post_eps = _stream_residual_energy(head, domains.stream("train", t))
        if adaptive and post_eps is not None:
            head.update_noise_floor(post_eps)

        for i in range(t + 1):
            preds, gts = [], []
            for X, Y, _ in domains.stream("test", i):
                preds.append(float(head.predict(X)[:, 0].sum()))
                gts.append(float(Y.sum()))
            M[t, i] = mae(preds, gts)

        task_log.append({
            "task": t,
            "n_train_images": n,
            "forget": float(f_used),
            "innovation": None if eps is None else float(eps),
            "rho_hat": None if rho_hat is None else float(rho_hat),
            "postfit_residual": None if post_eps is None else float(post_eps),
            "noise_floor": None if head.r_hat is None else float(head.r_hat),
        })
        extra = ""
        if adaptive:
            extra = f"; f_t={f_used:.3f}"
            if eps is not None:
                extra += f", innovation={eps:.4g}"
            if rho_hat is not None:
                extra += f", rho_hat={rho_hat:.3g}"
        print(f"  [{label}] learned domain {t} ({n} imgs){extra}; "
              f"seen-domain MAEs = {np.array2string(M[t, :t+1], precision=2)}")

    summary = forgetting_matrix_stats(M)
    summary.update({
        "matrix": M.tolist(),
        "adaptive": bool(adaptive),
        "fixed_forget": None if adaptive else float(forget),
        "f_history": [float(x) for x in head.f_history],
        "task_log": task_log,
    })
    return summary


def run(domains, forgets, lam, out, adaptive_forget=False,
        f_min=0.05, f_max=1.0, noise_ema=0.5):
    T = domains.n_domains()
    d_in = _first_dim(domains)
    results = {}
    for forget in forgets:
        label = f"forget={forget:g}"
        results[label] = _run_setting(domains, T, d_in, lam, label, forget=forget)
    if adaptive_forget:
        label = "adaptive"
        results[label] = _run_setting(
            domains, T, d_in, lam, label, adaptive=True,
            f_min=f_min, f_max=f_max, noise_ema=noise_ema,
        )
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump({
            "forgets": forgets,
            "lam": lam,
            "d_in": d_in,
            "adaptive_forget": adaptive_forget,
            "adaptive_params": {
                "f_min": f_min,
                "f_max": f_max,
                "noise_ema": noise_ema,
            },
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    return results


def pretty(results):
    print("\n===== Real domain-incremental counting =====")
    for k, r in results.items():
        M = np.array(r["matrix"]); T = M.shape[0]
        print(f"\n--- {k} ---   (rows = after task t, cols = MAE on domain i)")
        print("          " + "".join(f"  D{i:<5}" for i in range(T)))
        for t in range(T):
            cells = "".join(f"  {M[t, i]:5.2f}" if not np.isnan(M[t, i]) else "    . " for i in range(T))
            print(f"  task {t:<2}|{cells}")
        print(f"  final avg MAE={r['final_avg_mae']:.3f}  nBwT={r['nBwT']:.3f}  "
              f"|  oldest D0={r['final_oldest_mae']:.3f}  newest D{T-1}={r['final_newest_mae']:.3f}")
        if r.get("adaptive"):
            hist = ", ".join(f"{x:.3g}" for x in r.get("f_history", []))
            print(f"  adaptive f_t = [{hist}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="domains JSON (see README)")
    ap.add_argument("--selftest", action="store_true", help="run the loop on fake data (no torch)")
    ap.add_argument("--forgets", type=str, default="1.0,0.9")
    ap.add_argument("--lam", type=float, default=1e2)
    ap.add_argument("--img-size", type=int, default=518)
    ap.add_argument("--backbone", type=str, default="dinov2_vitb14")
    ap.add_argument("--max-per-domain", type=int, default=None)
    ap.add_argument("--out", type=str, default="runs_real")
    ap.add_argument("--adaptive-forget", action="store_true",
                    help="also run innovation-driven variable forgetting")
    ap.add_argument("--f-min", type=float, default=0.05)
    ap.add_argument("--f-max", type=float, default=1.0)
    ap.add_argument("--noise-ema", type=float, default=0.5)
    a = ap.parse_args()
    forgets = [float(x) for x in a.forgets.split(",")]

    if a.selftest:
        domains = _SelfTest()
    elif a.config:
        from datasets_real import RealCountingDomains, build_from_config
        with open(a.config) as f:
            cfg = json.load(f)
        domains = RealCountingDomains(build_from_config(cfg), backbone=a.backbone,
                                      img_size=a.img_size, max_per_domain=a.max_per_domain)
    else:
        ap.error("pass --selftest or --config domains.json")

    t0 = time.time()
    results = run(
        domains, forgets, a.lam, a.out,
        adaptive_forget=a.adaptive_forget,
        f_min=a.f_min,
        f_max=a.f_max,
        noise_ema=a.noise_ema,
    )
    pretty(results)
    print(f"\nsaved -> {os.path.join(a.out, 'results.json')}   ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
