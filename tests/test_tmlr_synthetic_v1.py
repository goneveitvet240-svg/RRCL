"""Regression tests for the frozen TMLR synthetic-v1 execution chain."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_tmlr_synthetic_v1 import preflight, run
from scripts.validate_tmlr_synthetic_v1 import validate, validate_paper_table
from tmlr_synthetic import (
    METHODS,
    cell_id,
    evaluate_scenario,
    factor_cells,
    load_config,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
FORMAL_CONFIG = ROOT / "configs" / "tmlr_synthetic_v1.json"


def _cell(scale=0, reliability=0, mapping=0, covariance=0):
    return {
        "label_scale": scale,
        "reliability": reliability,
        "mapping_change": mapping,
        "covariance_anisotropy": covariance,
    }


def _tiny_config(tmp: Path):
    config = copy.deepcopy(load_config(FORMAL_CONFIG))
    config["dimensions"]["n_train_per_domain"] = 32
    config["dimensions"]["n_test_per_domain"] = 64
    config["seeds"] = [12345]
    config["factor_grid"] = [0.0, 0.5, 1.0]
    config["gamma_grid"] = [0.0, 0.5, 1.0]
    config["bootstrap"]["replicates"] = 100
    path = tmp / "config.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path, config


def test_config_freezes_complete_factorial_and_orders():
    config = load_config(FORMAL_CONFIG)
    assert len(list(factor_cells())) == 16
    assert tuple(config["method_roster"]) == METHODS
    assert set(config["secondary_methods"]) == {"SIFt", "DOS-ELM"}
    assert config["orders"]["reverse"] == list(reversed(config["orders"]["forward"]))
    assert [cell_id(cell) for cell in factor_cells()][0] == "0000"
    assert [cell_id(cell) for cell in factor_cells()][-1] == "1111"


def test_invalid_grid_is_rejected():
    config = copy.deepcopy(load_config(FORMAL_CONFIG))
    config["factor_grid"] = [0.0, 1.0, 0.5]
    with pytest.raises(ValueError, match="factor_grid"):
        validate_config(config)


def test_scenario_is_deterministic_and_has_full_roster():
    config = load_config(FORMAL_CONFIG)
    cell = _cell(reliability=1, mapping=1, covariance=1)
    first = evaluate_scenario(config, cell, config["seeds"][0], "forward")
    second = evaluate_scenario(config, cell, config["seeds"][0], "forward")
    assert first["data_sha256"] == second["data_sha256"]
    assert first["methods"] == second["methods"]
    assert set(first["methods"]) == set(METHODS)


def test_label_scale_is_exactly_removed_in_normalized_primary_curve():
    config = load_config(FORMAL_CONFIG)
    seed = config["seeds"][0]
    base = evaluate_scenario(config, _cell(), seed, "forward")
    scaled = evaluate_scenario(config, _cell(scale=1), seed, "forward")
    base_curve = base["methods"]["fixed_f_oracle"]["details"]["curve"]
    scaled_curve = scaled["methods"]["fixed_f_oracle"]["details"]["curve"]
    assert base_curve == scaled_curve
    assert (
        base["methods"]["fixed_f_oracle"]["details"]["raw_scale_curve"]
        != scaled["methods"]["fixed_f_oracle"]["details"]["raw_scale_curve"]
    )


def test_reference_and_information_bounded_method_invariants():
    config = load_config(FORMAL_CONFIG)
    result = evaluate_scenario(config, _cell(reliability=1), config["seeds"][0], "reverse")
    methods = result["methods"]
    assert methods["f1"]["relative_utility_vs_f1"] == 0.0
    assert methods["fixed_f_oracle"]["relative_utility_vs_f1"] >= -1e-12
    assert methods["fixed_f_oracle"]["details"]["diagnostic_only"] is True
    weights = np.asarray(methods["precision_weighting"]["details"]["weights"])
    assert np.all(weights > 0)
    assert np.mean(weights) == pytest.approx(1.0)
    assert methods["shrinkage_gamma"]["details"]["gamma"] in config["gamma_grid"]
    assert len(methods["vff_rls"]["details"]["boundary_factors"]) == 4


def test_order_changes_no_generated_data_and_f1_is_order_invariant():
    config = load_config(FORMAL_CONFIG)
    cell = _cell(mapping=1, covariance=1)
    forward = evaluate_scenario(config, cell, config["seeds"][0], "forward")
    reverse = evaluate_scenario(config, cell, config["seeds"][0], "reverse")
    assert forward["data_sha256"] == reverse["data_sha256"]
    assert forward["methods"]["f1"]["population_balanced_mse"] == pytest.approx(
        reverse["methods"]["f1"]["population_balanced_mse"], abs=1e-12
    )


def test_testing_runner_summary_and_validator_round_trip():
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        config_path, config = _tiny_config(temporary)
        output = temporary / "run"
        report = preflight(config_path, output, testing=True)
        assert report["ready"]
        assert report["expected_scenarios"] == 32
        run(config_path, output, testing=True)
        assert len(list((output / "raw").rglob("scenario.json"))) == 32
        summary = json.loads((output / "summary.json").read_text())
        assert summary["raw_scenario_count"] == 32
        assert set(summary["cells"]) == {cell_id(cell) for cell in factor_cells()}
        assert validate(output, config_path, allow_testing=True) == []
        first = json.loads((output / "manifest.json").read_text())["entries"][0]
        path = output / first["path"]
        path.write_text(path.read_text() + " ", encoding="utf-8")
        failures = validate(output, config_path, allow_testing=True)
        assert any("hash mismatch" in failure for failure in failures)


def test_testing_mode_refuses_formal_output_path():
    config = load_config(FORMAL_CONFIG)
    formal_output = ROOT / config["formal_output"]
    report = preflight(FORMAL_CONFIG, formal_output, testing=True)
    assert not report["ready"]
    assert any("testing mode" in issue for issue in report["issues"])


def test_formal_paper_table_matches_frozen_summary_when_available():
    runs = ROOT / "runs_real" / "tmlr_synthetic_v1"
    if not (runs / "summary.json").exists():
        pytest.skip("formal synthetic result package is not present locally")
    assert validate_paper_table(runs) == []
