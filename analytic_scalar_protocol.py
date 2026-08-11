"""Shared train-only protocol for analytic scalar-regression baselines."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from analytic_forgetting_baselines import SIFtRLS, dos_elm_factor_update
from evaluation_metrics import balanced_metric_mean, scalar_regression_metrics
from rls_head import ForgettingRidgeRLS


def default_validation_masks(train, val_every=5):
    masks = {}
    for domain, (features, _) in train.items():
        indices = np.arange(features.shape[0])
        mask = indices % int(val_every) == int(val_every) - 1
        if not mask.any() or mask.all():
            raise ValueError(
                f"domain {domain} cannot form a non-empty validation split"
            )
        masks[domain] = mask
    return masks


def grouped_validation_masks(groups, val_every=5):
    masks = {}
    for domain, domain_groups in groups.items():
        domain_groups = np.asarray(domain_groups)
        unique = np.unique(domain_groups)
        if unique.size < 2:
            raise ValueError(
                f"domain {domain} needs at least two groups for validation"
            )
        held_out = unique[int(val_every) - 1 :: int(val_every)]
        if held_out.size == 0:
            held_out = unique[-1:]
        mask = np.isin(domain_groups, held_out)
        if not mask.any() or mask.all():
            raise ValueError(
                f"domain {domain} produced an empty grouped validation split"
            )
        masks[domain] = mask
    return masks


def mean_scales(train):
    return {
        domain: max(float(np.mean(np.abs(target))), 1e-6)
        for domain, (_, target) in train.items()
    }


def _relative_mae(prediction, target):
    return float(
        np.mean(np.abs(np.asarray(prediction) - np.asarray(target)))
        / max(float(np.mean(np.abs(target))), 1e-6)
    )


def _matrix_score(matrix):
    final = np.asarray(matrix, dtype=np.float64)[-1]
    return float(np.nanmean(final))


def _evaluate_head(head, test, scales, matrix, step):
    for domain in range(step + 1):
        features, target = test[domain]
        prediction = head.predict(features, non_negative=True)[:, 0] * scales[domain]
        matrix[step, domain] = _relative_mae(prediction, target)


def _final_metrics(head, test, scales):
    per_domain = []
    for domain in range(len(test)):
        features, target = test[domain]
        prediction = head.predict(features, non_negative=True)[:, 0] * scales[domain]
        row = scalar_regression_metrics(prediction, target)
        row["domain_index"] = domain
        row["relative_mae"] = _relative_mae(prediction, target)
        per_domain.append(row)
    return {
        "per_domain": per_domain,
        "balanced": {
            **balanced_metric_mean(per_domain),
            "relative_mae": float(
                np.mean([row["relative_mae"] for row in per_domain])
            ),
        },
    }


def run_dos_elm_style(
    train,
    test,
    *,
    ridge,
    scales=None,
    initial_factor=1.0,
    factor_min=0.0,
):
    """Run the DOS-ELM dynamic rule on a matched frozen-feature ridge head.

    This is a domain-batched regression adaptation, not a reproduction of the
    original random-hidden-layer ELM.  Following the paper's chronology, the
    score observed after chunk ``t`` changes the factor for chunk ``t+1``.
    """

    scales = mean_scales(train) if scales is None else scales
    domains = len(train)
    dimension = train[0][0].shape[1]
    head = ForgettingRidgeRLS(d_in=dimension, d_out=1, lam=ridge)
    matrix = np.full((domains, domains), np.nan)
    factor = float(initial_factor)
    previous_score = None
    records = []

    for domain in range(domains):
        design_factor = factor
        statistic_factor = design_factor * design_factor
        features, target = train[domain]
        normalized_target = (target / scales[domain]).reshape(-1, 1)
        head.begin_task(statistic_factor)
        head.accumulate(features, normalized_target)
        head.solve()

        normalized_prediction = head.predict(features, non_negative=True)[:, 0]
        normalized_rmse = float(
            np.sqrt(
                np.mean(
                    (normalized_prediction - normalized_target[:, 0]) ** 2
                )
            )
        )
        current_score = -normalized_rmse
        update = None
        if previous_score is not None:
            update = dos_elm_factor_update(
                design_factor,
                previous_score,
                current_score,
                factor_min=factor_min,
                factor_max=1.0,
            )
            factor = update.factor
        records.append(
            {
                "domain_index": domain,
                "design_factor_used": design_factor,
                "statistic_factor_used": statistic_factor,
                "train_score_definition": "negative normalized RMSE; larger is better",
                "train_score": current_score,
                "next_factor_update": asdict(update) if update is not None else None,
            }
        )
        previous_score = current_score
        _evaluate_head(head, test, scales, matrix, domain)

    return {
        "score": _matrix_score(matrix),
        "matrix": matrix.tolist(),
        "records": records,
        "evaluation": _final_metrics(head, test, scales),
        "state_bytes": int(
            head.R.nbytes + head.C.nbytes + (head.W.nbytes if head.W is not None else 0)
        ),
        "adaptation_note": (
            "DOS-ELM-style domain-batched regression adaptation on the matched "
            "frozen-feature ridge head; lambda^2 weights old sufficient statistics"
        ),
    }


def _fit_sift(train, scales, ridge, factor, epsilon, masks=None):
    dimension = train[0][0].shape[1]
    head = SIFtRLS(
        dimension,
        d_out=1,
        ridge=ridge,
        forgetting=factor,
        epsilon=epsilon,
        bias=True,
    )
    diagnostics = []
    for domain in range(len(train)):
        features, target = train[domain]
        if masks is not None:
            features = features[~masks[domain]]
            target = target[~masks[domain]]
        info = head.update(features, (target / scales[domain]).reshape(-1, 1))
        diagnostics.append({"domain_index": domain, **asdict(info)})
    return head, diagnostics


def select_sift_factor(
    train,
    scales,
    validation_masks,
    *,
    ridge,
    factors,
    epsilon,
):
    """Select one fixed SIFt factor using training-only balanced validation."""

    candidates = []
    for factor in sorted({float(value) for value in factors}, reverse=True):
        if not (0.0 < factor <= 1.0):
            raise ValueError("SIFt candidate factors must satisfy 0 < f <= 1")
        head, diagnostics = _fit_sift(
            train,
            scales,
            ridge,
            factor,
            epsilon,
            masks=validation_masks,
        )
        per_domain = []
        for domain in range(len(train)):
            features, target = train[domain]
            mask = validation_masks[domain]
            prediction = (
                head.predict(features[mask], non_negative=True)[:, 0]
                * scales[domain]
            )
            per_domain.append(_relative_mae(prediction, target[mask]))
        candidates.append(
            {
                "factor": factor,
                "balanced_validation_relative_mae": float(np.mean(per_domain)),
                "per_domain_validation_relative_mae": per_domain,
                "fit_diagnostics": diagnostics,
            }
        )
    # Candidates are ordered from high to low f, so exact ties prefer less
    # forgetting.  Test data are never inspected.
    selected = min(
        candidates,
        key=lambda row: row["balanced_validation_relative_mae"],
    )
    return selected["factor"], candidates


def run_sift(
    train,
    test,
    validation_masks,
    *,
    ridge,
    factors,
    epsilon=1e-8,
    scales=None,
):
    """Train-validation-select a fixed-factor SIFt-RLS and evaluate on test."""

    scales = mean_scales(train) if scales is None else scales
    selected_factor, selection_curve = select_sift_factor(
        train,
        scales,
        validation_masks,
        ridge=ridge,
        factors=factors,
        epsilon=epsilon,
    )
    dimension = train[0][0].shape[1]
    head = SIFtRLS(
        dimension,
        d_out=1,
        ridge=ridge,
        forgetting=selected_factor,
        epsilon=epsilon,
        bias=True,
    )
    domains = len(train)
    matrix = np.full((domains, domains), np.nan)
    diagnostics = []
    for domain in range(domains):
        features, target = train[domain]
        info = head.update(features, (target / scales[domain]).reshape(-1, 1))
        diagnostics.append({"domain_index": domain, **asdict(info)})
        _evaluate_head(head, test, scales, matrix, domain)
    return {
        "score": _matrix_score(matrix),
        "matrix": matrix.tolist(),
        "selected_factor": selected_factor,
        "train_only_selection_curve": selection_curve,
        "full_train_diagnostics": diagnostics,
        "evaluation": _final_metrics(head, test, scales),
        "state_bytes": head.state_bytes,
        "adaptation_note": (
            "Original SIFt-RLS information-subspace update on frozen image-level "
            "features; one fixed factor selected without test data"
        ),
    }
