"""Unit tests for the precision-shrinkage diagnostic core (pure numpy).

Pre-registered contracts:
  * gamma = 0 reproduces the f=1 joint ridge solution exactly;
  * gamma = 1 reproduces the precision endpoint defined by the diagnostic's
    three-way holdout split (not the production 80/20 raw-precision runner);
  * weights keep mean 1 for every gamma;
  * the final model is invariant to domain order;
  * no test data enters weight estimation or the train-only objective.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_shrinkage import (
    causal_boundary_reference,
    causal_final_weights,
    ensure_clean_worktree,
    gamma_sweep,
    gamma_weights,
    grid_argmin,
    quadratic_mse,
    sign_agreement,
    solve_ridge,
    weighted_system,
)
from result_io import dump_result

LAM = 10.0


def _make_domain(rng, n, d, noise, shift):
    X = rng.normal(shift, 1.0, (n, d))
    Xa = np.concatenate([X, np.ones((n, 1))], axis=1)
    w_true = rng.normal(0, 1, (d + 1, 1))
    y = Xa @ w_true + rng.normal(0, noise, (n, 1))
    return Xa, y


def _stats_from_rows(Xa, y):
    return (Xa.T @ Xa, Xa.T @ y)


def _val_block(Xa, y):
    return {
        "R": Xa.T @ Xa,
        "C": Xa.T @ y,
        "S": float(np.sum(y * y)),
        "n": Xa.shape[0],
    }


def _build_domains(seed=0, T=3, d=6, n=60):
    """Synthetic domains with heterogeneous noise; returns stats + raw rows."""
    rng = np.random.default_rng(seed)
    domain_stats, raw = [], []
    for t in range(T):
        Xa, y = _make_domain(rng, n, d, noise=0.2 + 1.5 * t, shift=float(t))
        n_fit = int(0.8 * n)
        fit_rows, fit_y = Xa[:n_fit], y[:n_fit]
        val_rows, val_y = Xa[n_fit:], y[n_fit:]
        domain_stats.append({
            "full": {
                "patch": _stats_from_rows(Xa, y),
                "image": _stats_from_rows(Xa, y),
            },
            "fit": {
                "patch": _stats_from_rows(fit_rows, fit_y),
                "image": _stats_from_rows(fit_rows, fit_y),
            },
            "val": {
                "patch": _val_block(val_rows, val_y),
                "image": _val_block(val_rows, val_y),
            },
        })
        raw.append((Xa, y))
    return domain_stats, raw


class TestEndpoints(unittest.TestCase):

    def setUp(self):
        self.stats, self.raw = _build_domains()
        self.precisions = [4.0, 1.0, 0.25]
        self.gammas = [0.0, 0.5, 1.0]
        self.records = gamma_sweep(self.stats, self.precisions, self.gammas, LAM)

    def test_gamma0_equals_joint_ridge(self):
        """gamma=0 must equal the f=1 model fit on ALL stacked data."""
        X_all = np.concatenate([Xa for Xa, _ in self.raw], axis=0)
        y_all = np.concatenate([y for _, y in self.raw], axis=0)
        W_joint = np.linalg.solve(
            X_all.T @ X_all + LAM * np.eye(X_all.shape[1]), X_all.T @ y_all
        )
        np.testing.assert_allclose(
            self.records[0]["patch"]["W_deployed"], W_joint, rtol=1e-9, atol=1e-11
        )

    def test_gamma1_equals_raw_causal_precision(self):
        """gamma=1 must equal the final boundary of the causal reference."""
        reference = causal_boundary_reference(
            [d["full"]["patch"] for d in self.stats], self.precisions, LAM
        )
        np.testing.assert_allclose(
            self.records[-1]["patch"]["W_deployed"], reference, rtol=1e-9, atol=1e-11
        )

    def test_weights_mean_one_for_every_gamma(self):
        for record in self.records:
            self.assertAlmostEqual(float(np.mean(record["weights"])), 1.0, places=12)

    def test_gamma_domain_validated(self):
        u = causal_final_weights(self.precisions)
        with self.assertRaises(ValueError):
            gamma_weights(u, -0.01)
        with self.assertRaises(ValueError):
            gamma_weights(u, 1.01)

    def test_precisions_validated(self):
        with self.assertRaises(ValueError):
            causal_final_weights([1.0, 0.0])
        with self.assertRaises(ValueError):
            causal_final_weights([1.0, float("nan")])
        with self.assertRaises(ValueError):
            causal_final_weights([])


class TestOrderInvariance(unittest.TestCase):

    def test_final_model_is_order_invariant(self):
        stats, _ = _build_domains(seed=7)
        precisions = [5.0, 1.0, 0.2]
        permutation = [2, 0, 1]
        stats_perm = [stats[i] for i in permutation]
        precisions_perm = [precisions[i] for i in permutation]
        for gamma in (0.0, 0.4, 1.0):
            a = gamma_sweep(stats, precisions, [gamma], LAM)[0]
            b = gamma_sweep(stats_perm, precisions_perm, [gamma], LAM)[0]
            np.testing.assert_allclose(
                a["patch"]["W_deployed"], b["patch"]["W_deployed"],
                rtol=1e-9, atol=1e-11,
            )
            self.assertEqual(sorted(a["weights"]), sorted(b["weights"]))

    def test_causal_reference_final_is_order_invariant(self):
        stats, _ = _build_domains(seed=9)
        precisions = [3.0, 0.5, 1.5]
        blocks = [d["full"]["patch"] for d in stats]
        W_a = causal_boundary_reference(blocks, precisions, LAM)
        order = [1, 2, 0]
        W_b = causal_boundary_reference(
            [blocks[i] for i in order], [precisions[i] for i in order], LAM
        )
        np.testing.assert_allclose(W_a, W_b, rtol=1e-9, atol=1e-11)


class TestObjective(unittest.TestCase):

    def test_quadratic_mse_matches_rowwise(self):
        rng = np.random.default_rng(3)
        Xa, y = _make_domain(rng, 40, 5, noise=0.5, shift=0.0)
        W = solve_ridge(Xa.T @ Xa, Xa.T @ y, LAM)
        block = _val_block(Xa, y)
        direct = float(np.mean((y - Xa @ W) ** 2))
        via_stats = quadratic_mse(W, block["R"], block["C"], block["S"], block["n"])
        self.assertAlmostEqual(direct, via_stats, places=10)

    def test_objective_uses_only_fit_and_val_partitions(self):
        """The sweep touches nothing beyond the provided train-side stats."""
        stats, _ = _build_domains(seed=11)

        class _Guard(dict):
            def __getitem__(self, key):
                if key == "test":
                    raise AssertionError("test data accessed in gamma_sweep")
                return super().__getitem__(key)

        guarded = []
        for d in stats:
            g = _Guard(d)
            g["test"] = "FORBIDDEN"
            guarded.append(g)
        records = gamma_sweep(guarded, [1.0, 2.0, 0.5], [0.0, 1.0], LAM)
        self.assertEqual(len(records), 2)

    def test_grid_argmin_rejects_nonfinite(self):
        with self.assertRaises(ValueError):
            grid_argmin([1.0, float("nan")])
        self.assertEqual(grid_argmin([3.0, 1.0, 2.0]), 1)

    def test_weighted_system_length_check(self):
        stats, _ = _build_domains(seed=1)
        blocks = [d["full"]["patch"] for d in stats]
        with self.assertRaises(ValueError):
            weighted_system(blocks, [1.0, 2.0])


class TestFusedProxy(unittest.TestCase):

    def test_fused_quadratic_matches_rowwise(self):
        """record['fused'] equals direct MSE of z @ [Wp; Wi] on explicit rows."""
        stats, _ = _build_domains(seed=21)
        rng = np.random.default_rng(22)
        d_aug = stats[0]["full"]["patch"][0].shape[0]
        fused_rows = []
        for d in stats:
            Z = rng.normal(0, 1, (12, 2 * d_aug))
            y = rng.normal(0, 1, (12, 1))
            d["val"]["fused"] = _val_block(Z, y)
            fused_rows.append((Z, y))
        records = gamma_sweep(stats, [2.0, 1.0, 0.5], [0.3], LAM)
        record = records[0]
        theta = np.concatenate(
            [record["patch"]["W_candidate"], record["image"]["W_candidate"]], axis=0
        )
        for j, (Z, y) in enumerate(fused_rows):
            direct = float(np.mean((y - Z @ theta) ** 2))
            self.assertAlmostEqual(
                record["fused"]["val_per_domain"][j], direct, places=10
            )

    def test_fused_absent_keeps_backward_compatibility(self):
        stats, _ = _build_domains(seed=23)
        records = gamma_sweep(stats, [1.0, 1.0, 1.0], [0.0, 1.0], LAM)
        for record in records:
            self.assertNotIn("fused", record)


def _make_mock_crowd_domains(T=3, D=5, P=4, n_train=60, n_test=8, seed=31):
    """Torch-free mock with real image ids for both diagnostic and production."""
    rng = np.random.default_rng(seed)
    train, test = [], []
    for t in range(T):
        w = rng.normal(0, 1, (D, 1))
        noise = 0.2 + 1.0 * t
        train.append([
            (rng.normal(t, 1, (P, D)),
             np.abs(rng.normal(t, 1, (P, D)) @ w) + rng.normal(2.0, noise, (P, 1)) ** 2,
             P, f"img_{t}_{i:03d}.jpg")
            for i in range(n_train)
        ])
        test.append([
            (rng.normal(t, 1, (P, D)),
             np.abs(rng.normal(t, 1, (P, D)) @ w) + rng.normal(2.0, noise, (P, 1)) ** 2,
             P, f"te_{t}_{i:03d}.jpg")
            for i in range(n_test)
        ])

    class _Dom:
        def n_domains(self):
            return T

        def stream(self, split, t, with_ids=False):
            for X, Y, n, image_id in (train if split == "train" else test)[t]:
                yield (X, Y, n, image_id) if with_ids else (X, Y, n)

    return _Dom()


class TestThreeWayEndpointEndToEnd(unittest.TestCase):
    """gamma=1 must equal the causal accumulation of the DIAGNOSTIC's own
    precision_val-role estimates — the declared three-way endpoint.  Runs the
    real collect_domain -> estimate_precision -> gamma_sweep -> deployed_heads
    path; no production precisions are injected."""

    def test_gamma1_equals_three_way_endpoint_end_to_end(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import run_precision_shrinkage_diagnostic as diag
        from run_real_norm_ablation import domain_scale, eval_domain

        T, D, lam = 3, 5, 50.0
        dom = _make_mock_crowd_domains(T=T, D=D)
        scales = [domain_scale(dom, t, "mean") for t in range(T)]
        stats, precisions = [], []
        for t in range(T):
            s, raw, _m = diag.collect_domain(
                dom, t, scales[t], "raw", 0.25, D + 1, 5, 42
            )
            _mse, p_t, _W, _resid = diag.estimate_precision(s, raw, lam, 1e-8)
            stats.append(s)
            precisions.append(p_t)

        record = gamma_sweep(stats, precisions, [1.0], lam)[0]
        for space in ("patch", "image"):
            reference = causal_boundary_reference(
                [d["full"][space] for d in stats], precisions, lam
            )
            np.testing.assert_allclose(
                record[space]["W_deployed"], reference, rtol=1e-9, atol=1e-11
            )
        # And the full deployed evaluation path agrees with reference heads.
        patch_head, image_head = diag.deployed_heads(record, D, lam)
        ref_record = {
            "patch": {"W_deployed": causal_boundary_reference(
                [d["full"]["patch"] for d in stats], precisions, lam)},
            "image": {"W_deployed": causal_boundary_reference(
                [d["full"]["image"] for d in stats], precisions, lam)},
        }
        ref_patch, ref_image = diag.deployed_heads(ref_record, D, lam)
        for i in range(T):
            mine = eval_domain(dom, i, patch_head, image_head, 0.25, scales[i])[1]
            ref = eval_domain(dom, i, ref_patch, ref_image, 0.25, scales[i])[1]
            self.assertAlmostEqual(mine, ref, places=10)


class TestProductionMachineryCheck(unittest.TestCase):
    """Machinery-only check against PRODUCTION run_causal_precision.

    The three-way endpoint and the production 80/20 raw-precision method use
    DIFFERENT precision-estimation protocols and are separate comparators; no
    cross-protocol equality exists.  What must agree is the weighted
    accumulation itself: given IDENTICAL precisions (injected by
    construction), gamma_sweep's gamma=1 model must reproduce production's
    final model exactly."""

    def test_weighted_machinery_matches_production_with_shared_precisions(self):
        try:
            from scripts.run_precision_weighted import (
                _estimate_domain_noise,
                run_causal_precision,
            )
        except Exception as exc:  # torch-free dev boxes
            self.skipTest(f"production precision runner unavailable here: {exc}")

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import run_precision_shrinkage_diagnostic as diag
        from run_real_norm_ablation import domain_scale, eval_domain

        T, D, lam = 3, 5, 50.0
        dom = _make_mock_crowd_domains(T=T, D=D)
        production = run_causal_precision(dom, "raw", 0.25, lam, 1e-8, split_seed=42)
        production_final_row = np.asarray(production[6])[T - 1, :T]

        scales = [domain_scale(dom, t, "mean") for t in range(T)]
        stats, shared_precisions = [], []
        for t in range(T):
            s, _raw, _m = diag.collect_domain(
                dom, t, scales[t], "raw", 0.25, D + 1, 5, 42
            )
            stats.append(s)
            # Injected on purpose: isolates the accumulation machinery.
            _mse, _n, p_t, _man = _estimate_domain_noise(
                dom, t, scales[t], lam, 1e-8, split_seed=42
            )
            shared_precisions.append(p_t)
        record = gamma_sweep(stats, shared_precisions, [1.0], lam)[0]
        patch_head, image_head = diag.deployed_heads(record, D, lam)
        mine = [
            eval_domain(dom, i, patch_head, image_head, 0.25, scales[i])[1]
            for i in range(T)
        ]
        np.testing.assert_allclose(mine, production_final_row, rtol=1e-8, atol=1e-10)


class TestSignAgreement(unittest.TestCase):

    def test_cases(self):
        self.assertEqual(sign_agreement(-0.1, -0.2), "agree")
        self.assertEqual(sign_agreement(0.1, 0.2), "agree")
        self.assertEqual(sign_agreement(-0.1, 0.2), "disagree")
        self.assertEqual(sign_agreement(0.0, 0.2), "val_flat")
        self.assertEqual(sign_agreement(-0.1, 0.0), "test_flat")
        self.assertEqual(sign_agreement(0.0, 0.0), "both_flat")


class TestWorktreeGate(unittest.TestCase):

    def test_dirty_tree_blocks(self):
        with self.assertRaises(SystemExit):
            ensure_clean_worktree(
                allow_dirty=False, git_status_provider=lambda: " M run_adaptive_f.py"
            )

    def test_allow_dirty_bypasses(self):
        state = ensure_clean_worktree(
            allow_dirty=True, git_status_provider=lambda: " M run_adaptive_f.py"
        )
        self.assertTrue(state["git_dirty"])

    def test_clean_tree_passes(self):
        state = ensure_clean_worktree(
            allow_dirty=False, git_status_provider=lambda: ""
        )
        self.assertFalse(state["git_dirty"])


class TestSerialization(unittest.TestCase):

    def test_dump_result_accepts_numpy_and_rejects_inf(self):
        stats, _ = _build_domains(seed=2)
        record = gamma_sweep(stats, [1.0, 2.0, 4.0], [0.5], LAM)[0]
        payload = {
            "gamma": record["gamma"],
            "weights": record["weights"],
            "val_balanced_image": record["image"]["val_balanced"],
            "W_sample": record["patch"]["W_deployed"][:3],
        }
        with tempfile.TemporaryDirectory() as tmp:
            dump_result(os.path.join(tmp, "x.json"), payload)
        with self.assertRaises(ValueError):
            with tempfile.TemporaryDirectory() as tmp:
                dump_result(os.path.join(tmp, "y.json"), {"bad": float("inf")})


if __name__ == "__main__":
    unittest.main()
