import copy
import json
from pathlib import Path

import numpy as np

from tmlr_selector_v2 import (
    FAMILIES,
    _sift_risk,
    assemble_result,
    base_record,
    evaluate_meta_test,
    generate_task,
    load_config,
    prepare_selection,
    split_seeds,
    summary_features,
    task_stats,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]


def small_config():
    config = json.loads((ROOT / "configs/tmlr_selector_v2.json").read_text())
    config = copy.deepcopy(config)
    config["dimensions"]["n_train_per_domain"] = 16
    config["factor_grid"] = [0.0, 0.5, 1.0]
    config["sift_domain_factor_grid"] = [0.5, 1.0]
    config["knn_k_grid"] = [1, 3]
    config["meta_splits"] = {
        "train": {"seed_start": 91000, "count": 8},
        "validation": {"seed_start": 92000, "count": 4},
        "test": {"seed_start": 93000, "count": 8},
    }
    config["bootstrap"]["replicates"] = 100
    return validate_config(config)


def test_formal_config_loads_and_has_disjoint_splits():
    config = load_config(ROOT / "configs/tmlr_selector_v2.json")
    train = set(split_seeds(config, "train"))
    validation = set(split_seeds(config, "validation"))
    test = set(split_seeds(config, "test"))
    assert not (train & validation or train & test or validation & test)


def test_task_generation_is_deterministic_and_family_is_seed_fixed():
    config = small_config()
    left = generate_task(config, 93001)
    right = generate_task(config, 93001)
    assert left.family == FAMILIES[93001 % 4]
    for a, b in zip(left.domains, right.domains):
        np.testing.assert_array_equal(a.X, b.X)
        np.testing.assert_array_equal(a.y, b.y)


def test_train_summary_is_finite_fixed_length_and_test_free():
    config = small_config()
    task = generate_task(config, 93002)
    stats = task_stats(config, task)
    features = summary_features(config, task, stats)
    assert len(features) == 51
    assert np.isfinite(features).all()


def test_sift_uses_rank_one_updates_and_domain_equivalent_factor():
    config = small_config()
    task = generate_task(config, 93003)
    record = _sift_risk(config, task, 0.5)
    assert record["all_updates_rank_one"]
    assert record["updates"] == 4 * config["dimensions"]["n_train_per_domain"]
    expected = 0.5 ** (1.0 / config["dimensions"]["n_train_per_domain"])
    assert abs(record["sample_factor"] - expected) < 1e-15


def test_small_end_to_end_experiment_keeps_test_out_of_selection():
    config = small_config()
    train, validation, selection = prepare_selection(config)
    assert selection["meta_train_records_sha256"]
    assert selection["meta_validation_records_sha256"]
    assert all(row["seed"] < 93000 for row in train + validation)
    test = evaluate_meta_test(config, train, selection)
    result = assemble_result(config, train, validation, selection, test)
    assert len(test) == 8
    assert result["information_policy"]["test_generated_after_selection_lock"]
    assert not result["information_policy"]["fdst_reused"]
    assert not result["information_policy"]["model_merging_in_scope"]
