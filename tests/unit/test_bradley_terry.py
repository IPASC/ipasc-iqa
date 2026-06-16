"""Unit tests for the regularized Bradley-Terry estimator."""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from ipasc_iqa.contrib.bradley_terry import fit_bradley_terry


def _simulate(true: dict[str, float], n: int, seed: int = 0) -> list[tuple[str, str]]:
    rng = np.random.default_rng(seed)
    items = list(true)
    out = []
    for _ in range(n):
        i, j = rng.choice(items, 2, replace=False)
        win = rng.random() < expit(true[i] - true[j])
        out.append((i, j) if win else (j, i))
    return out


class TestBradleyTerry:
    def test_recovers_ranking(self) -> None:
        true = {"gt": 4.0, "a": 1.0, "b": 0.0, "c": -1.5}
        res = fit_bradley_terry(list(true), _simulate(true, 1500), anchor="gt")
        ranked = sorted(true, key=lambda k: -res.beta[k])
        assert ranked == ["gt", "a", "b", "c"]

    def test_anchor_is_zero(self) -> None:
        true = {"gt": 3.0, "a": 0.0, "b": -2.0}
        res = fit_bradley_terry(list(true), _simulate(true, 800), anchor="gt")
        assert res.beta["gt"] == 0.0
        assert res.se["gt"] == 0.0
        assert res.beta["a"] < 0 and res.beta["b"] < res.beta["a"]

    def test_separation_stays_finite(self) -> None:
        # 'gt' always wins — MLE would diverge, but the prior keeps it finite.
        comps = [("gt", x) for x in ("a", "b") for _ in range(15)]
        comps += [("a", "b")] * 10
        res = fit_bradley_terry(["gt", "a", "b"], comps, anchor="gt")
        assert np.isfinite(res.beta["a"]) and np.isfinite(res.beta["b"])
        assert res.beta["a"] > res.beta["b"]

    def test_no_data_returns_prior(self) -> None:
        res = fit_bradley_terry(["x", "y"], [], prior_sd=2.5)
        assert res.beta == {"x": 0.0, "y": 0.0}
        assert res.se["x"] == 2.5

    def test_more_data_reduces_se(self) -> None:
        true = {"a": 1.0, "b": -1.0}
        se_small = fit_bradley_terry(list(true), _simulate(true, 30, 1)).se["a"]
        se_large = fit_bradley_terry(list(true), _simulate(true, 1500, 2)).se["a"]
        assert se_large < se_small
