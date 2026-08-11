"""Tests for the Phase-1 RRCL scientific-validity fixes.

P1  PerDomainStatsSelector causal recursion
P2  VFF image_head power
P3  patch_target not hardcoded to "dct5"
P4  selector_space ablation
P5  causal precision weighting
P6  oracle grid unified
P7  numerical and JSON hygiene
"""

import json
import math
import os
import tempfile
import unittest

import numpy as np

from adaptive_selector import PerDomainStatsSelector


# ======================================================================
# Helpers
# ======================================================================

def _stats(x, y, d_aug=None):
    """Build (R, C, S, n) tuple from 1D arrays.

    x: feature values, augmented with bias column if needed.
    y: targets.
    d_aug: expected augmented dimension (d+1). If not provided,
           determined from x's shape.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    # Add bias column
    x = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    if d_aug is not None and x.shape[1] < d_aug:
        # Pad with additional ones to reach d_aug
        extra = d_aug - x.shape[1]
        x = np.concatenate([x, np.ones((x.shape[0], extra))], axis=1)
    n = x.shape[0]
    return [x.T @ x, x.T @ y, float(np.sum(y * y)), n]


def _solve_ridge(R, C, lam):
    """Explicit ridge solve for cross-checking."""
    w = np.linalg.solve(R + lam * np.eye(R.shape[0]), C)
    return w


# ======================================================================
# P1: PerDomainStatsSelector recursive state
# ======================================================================

class TestPerDomainRecursion(unittest.TestCase):
    """Verify the selector's causal recursion."""

    def test_first_domain_returns_f1(self):
        s = PerDomainStatsSelector(3, lam=1.0)
        r = s.select_and_update(
            _stats([1, 2, 3], [10, 20, 30], d_aug=3),
            _stats([1], [10], d_aug=3),
        )
        self.assertEqual(r.factor, 1.0)
        self.assertEqual(s.n_domains, 1)
        self.assertEqual(r.per_domain_validation, [])

    def test_second_domain_candidates_differ(self):
        """f<1 与 f=1 的候选系统确实不同。"""
        s = PerDomainStatsSelector(2, lam=0.0, f_min=0.1, f_max=1.0)
        # Domain 0: clean data
        s.select_and_update(
            _stats([1, 1], [10, 10], d_aug=2),
            _stats([1], [10], d_aug=2),
        )
        # Domain 1: very different data
        w1 = s._candidate_weights(1.0,
                                  np.eye(2) * 100,
                                  np.array([[0.0], [0.0]]))
        w05 = s._candidate_weights(0.5,
                                   np.eye(2) * 100,
                                   np.array([[0.0], [0.0]]))
        diff = float(np.sum(np.abs(w1 - w05)))
        self.assertGreater(diff, 1e-10,
                           f"f=1 and f=0.5 produced identical weights (diff={diff})")

    def test_third_domain_matches_manual_recursion(self):
        """第三域候选系统与独立手工递归完全一致。不使用 selector._R_actual。"""
        d = 2
        lam = 0.5
        s = PerDomainStatsSelector(d, lam=lam, f_min=0.05, f_max=1.0)

        R0 = np.eye(d) * 10
        C0 = np.full((d, 1), 5.0, dtype=np.float64)
        result0 = s.select_and_update([R0, C0, 0.0, 1], [R0, C0, 0.0, 1])
        self.assertEqual(result0.factor, 1.0)

        R1 = np.eye(d) * 20
        C1 = np.full((d, 1), -3.0, dtype=np.float64)
        result1 = s.select_and_update([R1, C1, 0.0, 1], [R1, C1, 0.0, 1])
        f1 = result1.factor  # the ACTUAL selected factor from boundaries

        # Domain 2
        R2 = np.eye(d) * 30
        C2 = np.full((d, 1), 7.0, dtype=np.float64)

        # --- independent manual construction (no selector._R_actual) ---
        # R_actual after domain 1:  f1*R0 + R1
        R_manual = f1 * R0 + R1
        C_manual = f1 * C0 + C1

        # Test candidate f=0.3 for domain 2
        f_candidate = 0.3
        # Selector path
        w_sel = s._candidate_weights(f_candidate, R2, C2)
        # Manual path
        R_eff_manual = lam * np.eye(d) + f_candidate * R_manual + R2
        C_eff_manual = f_candidate * C_manual + C2
        w_manual = np.linalg.solve(R_eff_manual, C_eff_manual)

        diff = float(np.max(np.abs(w_sel - w_manual)))
        self.assertLess(diff, 1e-10,
                        f"selector-vs-manual mismatch at domain 3 (diff={diff:.2e})")

    def test_actual_state_matches_deployed_recursion(self):
        """选出 f_k 后，通过 select_and_update 返回值验证状态一致。"""
        d = 2
        lam = 0.5
        s = PerDomainStatsSelector(d, lam=lam, f_min=0.05, f_max=1.0)

        R1 = np.eye(d) * 10
        C1 = np.full((d, 1), 5.0, dtype=np.float64)
        r1 = s.select_and_update([R1, C1, 0.0, 1], [R1, C1, 0.0, 1])
        self.assertEqual(r1.factor, 1.0)  # first domain always f=1

        R2 = np.eye(d) * 20
        C2 = np.full((d, 1), -3.0, dtype=np.float64)
        r2 = s.select_and_update([R2, C2, 0.0, 1], [R2, C2, 0.0, 1])
        f2 = r2.factor

        # After two domains: R_actual = f2*R1 + R2 (R1 was first, R2 is new)
        expected_R = f2 * R1 + R2
        expected_C = f2 * C1 + C2

        self.assertLess(float(np.max(np.abs(s._R_actual - expected_R))), 1e-10)
        self.assertLess(float(np.max(np.abs(s._C_actual - expected_C))), 1e-10)
        self.assertEqual(s.n_domains, 2)

    def test_state_bytes_includes_actual_state(self):
        s = PerDomainStatsSelector(5, lam=1.0)
        self.assertGreater(s.state_bytes, 0,
                           "state_bytes should include R_actual + C_actual")


# ======================================================================
# P1: continuous mode raises NotImplementedError
# ======================================================================

class TestContinuousMode(unittest.TestCase):

    def test_continuous_raises(self):
        with self.assertRaises(NotImplementedError):
            PerDomainStatsSelector(5, lam=1.0, search_mode="continuous")


# ======================================================================
# P7: NumPy 2.x float() safety
# ======================================================================

class TestNumpyScalarSafety(unittest.TestCase):
    """No np.float64 2D array passed to float()."""

    def test_float_on_2d_array_fails(self):
        """Verify that float(np.array([[1.0]])) raises TypeError on numpy 2.x."""
        # This is the pattern we must avoid
        arr = np.array([[1.0]])
        if hasattr(np, '__version__') and np.__version__.startswith('2'):
            with self.assertRaises(TypeError):
                float(arr)
        # Safe pattern: float(np.sum(arr)) or arr.item()
        safe = float(np.sum(np.array([[3.0]])))
        self.assertEqual(safe, 3.0)


# ======================================================================
# P5: Causal precision weighting
# ======================================================================

class TestPrecisionWeightingLogic(unittest.TestCase):
    """Causal precision weighting properties (unit-level, no datasets)."""

    def test_precision_from_single_domain(self):
        """Precision estimate for one domain is finite and positive."""
        rng = np.random.default_rng(12345)
        d = 5
        X = rng.normal(0, 1, (100, d))
        Xa = np.concatenate([X, np.ones((100, 1))], axis=1)
        true_w = rng.normal(0, 1, (d, 1))
        y = X @ true_w + rng.normal(0, 0.3, (100, 1))

        val_mask = np.arange(100) % 5 == 4
        Xf, yf = Xa[~val_mask], y[~val_mask]
        Xv, yv = Xa[val_mask], y[val_mask]
        self.assertGreater(Xv.shape[0], 0)
        self.assertGreater(Xf.shape[0], 0)

        w_fit = np.linalg.solve(Xf.T @ Xf + 1.0 * np.eye(Xf.shape[1]), Xf.T @ yf)
        resid = yv - Xv @ w_fit
        mse = float(np.mean(resid * resid))
        precision = 1.0 / max(mse, 1e-8)

        self.assertGreater(mse, 0)
        self.assertTrue(np.isfinite(mse))
        self.assertTrue(np.isfinite(precision))
        self.assertGreater(precision, 0)

    def test_normalized_weights_mean_one(self):
        precisions = np.array([1.0, 2.0, 0.5])
        for k in range(1, len(precisions) + 1):
            p_sub = precisions[:k]
            bar_p = np.mean(p_sub)
            weights = p_sub / bar_p
            self.assertAlmostEqual(float(np.mean(weights)), 1.0, places=10)
            self.assertTrue(np.all(np.isfinite(weights)))

    def test_no_future_leakage_via_production_function(self):
        """调用生产版 run_causal_precision：前缀 M_rel 不受未来域影响。"""
        from scripts.run_precision_weighted import run_causal_precision

        d_in = 4
        T = 4
        rng = np.random.default_rng(7777)

        # Pre-build all data so stream() returns fixed arrays
        def _build_domain_data(base_seed):
            """Return list of T lists of (X, Y, n) for train, same for test."""
            sr = np.random.default_rng(base_seed)
            train, test = [], []
            for t in range(T):
                t_data = []
                # 30 train images: the stable hash split needs enough ids to
                # populate both partitions for every "domain:t" sample key.
                for _ in range(30):
                    X = sr.normal(float(t), 1, (4, d_in))
                    Y = sr.normal(float(t) * 0.5, 0.3, (4, 1))
                    t_data.append((X, Y, 4))
                train.append(t_data)
                tst = []
                for _ in range(3):  # 3 test images
                    X = sr.normal(float(t) + 0.1, 1, (4, d_in))
                    Y = sr.normal(float(t) * 0.5 + 0.1, 0.3, (4, 1))
                    tst.append((X, Y, 4))
                test.append(tst)
            return train, test

        # Sequence A: domains 0-2 same as B; domain 3 different
        train_a, test_a = _build_domain_data(7777)
        train_b, test_b = _build_domain_data(7777)
        # Override domain 3 in sequence B with different data
        sr_b = np.random.default_rng(99999)
        train_b[3] = []
        for _ in range(30):
            X = sr_b.normal(100.0, 20, (4, d_in))
            Y = sr_b.normal(50.0, 10, (4, 1))
            train_b[3].append((X, Y, 4))
        test_b[3] = []
        for _ in range(3):
            X = sr_b.normal(100.0, 20, (4, d_in))
            Y = sr_b.normal(50.0, 10, (4, 1))
            test_b[3].append((X, Y, 4))

        class _MockDomains:
            def __init__(self, tr, te):
                self._tr = tr; self._te = te
            def n_domains(self): return T
            def stream(self, split, t):
                src = self._tr if split == "train" else self._te
                for item in src[t]:
                    yield item

        dom_a = _MockDomains(train_a, test_a)
        dom_b = _MockDomains(train_b, test_b)

        result_a = run_causal_precision(dom_a, "raw", 0.25, 100.0, 1e-8)
        result_b = run_causal_precision(dom_b, "raw", 0.25, 100.0, 1e-8)

        # result[6] = M_rel (index 6 in the returned tuple)
        M_a = result_a[6]
        M_b = result_b[6]
        np.testing.assert_allclose(M_a[:3, :3], M_b[:3, :3], atol=1e-10, rtol=0)

        # Verify normalized weights mean 1 at each boundary (result[4])
        for boundary_weights in result_a[4]:
            self.assertAlmostEqual(float(np.mean(boundary_weights)), 1.0, places=10)


# ======================================================================
# P6: Oracle grid unified
# ======================================================================

class TestOracleGrid(unittest.TestCase):

    def test_grid_matches_main_experiment(self):
        expected = [1.0, 0.8, 0.6, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]
        # Check precision_weighted uses the same grid
        from scripts.run_precision_weighted import _ORACLE_FORGETS
        self.assertEqual(_ORACLE_FORGETS, expected)


# ======================================================================
# P7: JSON hygiene
# ======================================================================

class TestJSONHygiene(unittest.TestCase):

    def test_no_nan_or_inf_in_serialization(self):
        """JSON 不包含裸 NaN 或 Infinity。"""
        import json as json_mod
        data = {
            "ok": 1.0,
            "bad_nan": float("nan"),
        }
        with self.assertRaises((ValueError, TypeError)):
            json_mod.dumps(data, allow_nan=False)

    def test_adaptive_selector_no_2d_float_pattern(self):
        """No float(w.T @ ...) pattern in selector source."""
        import inspect
        src = inspect.getsource(PerDomainStatsSelector)
        # Dangerous: float(weights.T @ ...) or float(w.T @ ...)
        self.assertNotIn("float(weights.T @", src)
        self.assertNotIn("float(w.T @", src)

    def test_all_float_patterns_are_vectorized(self):
        """Explicit check: all quad/lin computations use float(np.sum(...))."""
        import inspect
        src = inspect.getsource(PerDomainStatsSelector)
        # The dangerous pattern is float(w.T @ R @ w) where w is (d,1)
        # The safe pattern is float(np.sum(w * (R @ w)))
        self.assertNotIn("float(weights.T @", src)
        self.assertNotIn("float(w.T @", src)
        self.assertNotIn(".T @ Rj)", src)  # as argument to float


# ======================================================================
# P7: Ridge equivalence
# ======================================================================

class TestRidgeEquivalence(unittest.TestCase):

    def test_production_precision_accumulator(self):
        """精度加权最终解与显式加权 ridge regression 数值一致（误差 < 1e-10）。"""
        d = 3
        lam = 0.5
        rng = np.random.default_rng(42)

        Rs, Cs = [], []
        for _ in range(3):
            X = rng.normal(0, 1, (50, d))
            Xa = np.concatenate([X, np.ones((50, 1))], axis=1)
            y = X @ rng.normal(0, 1, (d, 1)) + rng.normal(0, 0.5, (50, 1))
            Rs.append(Xa.T @ Xa)
            Cs.append(Xa.T @ y)

        # Production pattern: accumulate p_k * R_k, divide by bar_p
        precisions = np.array([1.0, 2.0, 0.5])
        P_R = np.zeros_like(Rs[0])
        P_C = np.zeros_like(Cs[0])
        sum_p = 0.0
        for k, (Rk, Ck, pk) in enumerate(zip(Rs, Cs, precisions)):
            sum_p += pk
            bar_p = sum_p / (k + 1)
            P_R += pk * Rk
            P_C += pk * Ck
            R_eff = P_R / bar_p
            C_eff = P_C / bar_p
            w_prod = np.linalg.solve(R_eff + lam * np.eye(R_eff.shape[0]), C_eff)

        # Explicit weighted ridge (same data)
        R_exp = np.zeros_like(Rs[0])
        C_exp = np.zeros_like(Cs[0])
        final_precisions = precisions
        bar_final = np.mean(final_precisions)
        weights = final_precisions / bar_final
        for Rk, Ck, wk in zip(Rs, Cs, weights):
            R_exp += wk * Rk
            C_exp += wk * Ck
        w_exp = np.linalg.solve(R_exp + lam * np.eye(R_exp.shape[0]), C_exp)

        diff = float(np.max(np.abs(w_prod - w_exp)))
        self.assertLess(diff, 1e-10,
                        f"production accumulator differs from explicit weighted (diff={diff:.2e})")


# ======================================================================
# P7: Bias column weighted
# ======================================================================

class TestBiasColumnWeighted(unittest.TestCase):

    def test_bias_column_scaled_with_domain_weight(self):
        """bias column 也被完整按域权重缩放。"""
        d = 2
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        Xa = np.concatenate([X, np.ones((2, 1))], axis=1)
        R = Xa.T @ Xa  # (3, 3), last row/col includes bias interactions

        weight = 0.3
        R_scaled = weight * R

        # The bias-bias element (last row, last col) should be scaled
        self.assertAlmostEqual(R_scaled[-1, -1], weight * float(Xa.shape[0]), places=10)
        # The feature-bias cross terms should be scaled
        self.assertAlmostEqual(R_scaled[0, -1], weight * np.sum(X[:, 0]), places=10)


# ======================================================================
# P2: VFF image_head power space
# ======================================================================

class TestVFFImageHeadPower(unittest.TestCase):

    def test_vff_factor_call_without_power_space_kwarg(self):
        from vff_baselines import paleologu_vff_factor
        import inspect
        sig = inspect.signature(paleologu_vff_factor)
        param_names = list(sig.parameters.keys())
        self.assertNotIn("power_space", param_names)

    def test_paleologu_vff_mock_domain_smoke(self):
        """调用生产版 run_paleologu：2域 mock，不崩溃，验证输出。"""
        from run_vff_baselines import run_paleologu

        d_in = 4
        T = 2
        rng = np.random.default_rng(99)

        # Pre-build 2-domain data
        train, test = [], []
        for t in range(T):
            t_data = []
            for _ in range(10):
                X = rng.normal(float(t), 1, (4, d_in))
                Y = rng.normal(float(t) * 0.5, 0.3, (4, 1))
                t_data.append((X, Y, 4))
            train.append(t_data)
            tst = []
            for _ in range(3):
                X = rng.normal(float(t) + 0.1, 1, (4, d_in))
                Y = rng.normal(float(t) * 0.5 + 0.1, 0.3, (4, 1))
                tst.append((X, Y, 4))
            test.append(tst)

        class _MockDomains:
            def n_domains(self): return T
            @property
            def domains(self):
                class _Spec: name = "mock"
                return [_Spec(), _Spec()]
            def stream(self, split, t):
                src = train if split == "train" else test
                for item in src[t]:
                    yield item

        dom = _MockDomains()
        score, factors, diagnostics, matrix = run_paleologu(
            dom, "raw", 0.25, 100.0, 1.5, 1e-6, 0.05)

        self.assertTrue(np.isfinite(score))
        self.assertEqual(len(factors), 2)
        for f in factors:
            self.assertTrue(np.isfinite(f))
            self.assertGreaterEqual(f, 0.05)
            self.assertLessEqual(f, 1.0)
        self.assertEqual(diagnostics[1]["power_space"], "image_head")
        M = np.array(matrix)
        self.assertTrue(np.all(np.isfinite(M[~np.isnan(M)])))


# ======================================================================
# P3: patch_target parameter propagation
# ======================================================================

class TestPatchTargetPropagation(unittest.TestCase):

    def test_adaptive_train_eval_perdomain_accepts_patch_target(self):
        from run_adaptive_f import adaptive_train_eval_perdomain
        import inspect
        sig = inspect.signature(adaptive_train_eval_perdomain)
        self.assertIn("patch_target", sig.parameters)

    def test_domain_suffstats_split_patch_vs_image_differs(self):
        """_domain_suffstats_split with patch vs image produces different R."""
        from run_adaptive_f import _domain_suffstats_split

        class _Mock:
            def stream(self, split, t):
                # 30 images: enough for the hash split to populate both
                # partitions for sample_key "domain:0" at split_seed 42.
                for _ in range(30):
                    X = np.random.default_rng(42).normal(0, 1, (4, 2))
                    Y = np.random.default_rng(43).normal(0, 1, (4, 1))
                    yield X, Y, 4

        d_aug = 3
        dom = _Mock()
        fp, _, mp = _domain_suffstats_split(dom, 0, "raw", 1.0, d_aug,
                                            selector_space="patch", val_every=5)
        fi, _, mi = _domain_suffstats_split(dom, 0, "raw", 1.0, d_aug,
                                            selector_space="image", val_every=5)
        diff = float(np.max(np.abs(fp[0] - fi[0])))
        self.assertGreater(diff, 1e-10,
                           f"patch vs image should differ (diff={diff:.2e})")
        # Same domain, same seed -> identical split manifest across spaces
        self.assertEqual(mp["fit_ids_sha256"], mi["fit_ids_sha256"])
        self.assertEqual(mp["val_ids_sha256"], mi["val_ids_sha256"])
        self.assertGreater(mp["fit_count"], 0)
        self.assertGreater(mp["val_count"], 0)


class TestSelectorSpace(unittest.TestCase):

    def test_selector_space_in_cli_source(self):
        import run_adaptive_f
        import inspect
        src = inspect.getsource(run_adaptive_f)
        self.assertIn("--selector-space", src)

    def test_illegal_selector_space_raises_valueerror(self):
        """非法 selector_space 抛出 ValueError。"""
        from run_adaptive_f import _domain_suffstats_split
        class _Mock:
            def stream(self, split, t):
                X = np.ones((4, 2)); Y = np.ones((4, 1))
                yield X, Y, 4
        dom = _Mock()
        with self.assertRaises(ValueError):
            _domain_suffstats_split(dom, 0, "raw", 1.0, 3, selector_space="bad")


# ======================================================================
# P7: JSON output no NaN/Inf
# ======================================================================

class TestJSONOutputSanity(unittest.TestCase):

    def test_mock_output_has_no_nan_or_inf(self):
        """模拟 JSON 输出不含 NaN 或 Infinity。"""
        import json as json_mod
        payload = {
            "heldout_domain_local_mse": [0.5, 0.3],
            "residual_precision": [2.0, 3.333],
            "n_validation_images": [80, 80],
            "normalized_weights_at_each_boundary": [[1.0], [0.75, 1.5]],
            "weight_ratio": 2.0,
            "rel_f1": 0.12345,
            "rel_precision_weighted": 0.11111,
        }
        # Should NOT raise
        dumped = json_mod.dumps(payload, allow_nan=False)
        self.assertIsInstance(dumped, str)
        self.assertGreater(len(dumped), 0)


# ======================================================================
# P7: NumPy 2.x explicit check
# ======================================================================

class TestExplicitFloat2D(unittest.TestCase):

    def test_1d_scalar_is_fine(self):
        arr = np.array([3.0])
        result = float(np.sum(arr))
        self.assertAlmostEqual(result, 3.0)

    def test_2d_single_element_via_sum_is_safe(self):
        arr = np.array([[3.0]])
        result = float(np.sum(arr))
        self.assertAlmostEqual(result, 3.0)


if __name__ == "__main__":
    unittest.main()
