"""Regularized Bradley-Terry model: pairwise preferences → continuous scores.

The Bradley-Terry model gives each item a latent strength ``beta`` such that
``P(i beats j) = sigmoid(beta_i - beta_j)``. We estimate the betas by MAP with a
Gaussian prior ``N(0, prior_sd^2)`` on each strength. The prior is essential
here: an item that wins (or loses) *all* its comparisons — e.g. the ground-truth
reference, which should always be preferred — has no finite maximum-likelihood
estimate, but the prior keeps it finite and shrinks sparse estimates toward
"equal strength", so scores stay well-behaved as the crowd-sourced data grows.

Scores can be re-centred on an ``anchor`` item (its beta set to 0); the reported
standard errors are then the SEs of the differences ``beta_i - beta_anchor``,
computed from the full posterior covariance.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy.special import expit


@dataclass
class BTResult:
    items: list[str]
    beta: dict[str, float]  # strengths (centred on the anchor if one was given)
    se: dict[str, float]  # standard errors (of beta_i - beta_anchor when anchored)


def fit_bradley_terry(
    items: list[str],
    comparisons: Iterable[tuple[str, str]],
    *,
    prior_sd: float = 3.0,
    anchor: str | None = None,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> BTResult:
    """MAP Bradley-Terry fit with a Gaussian prior on the strengths.

    Parameters
    ----------
    items:
        All item ids (the "players").
    comparisons:
        ``(winner_id, loser_id)`` tuples; repeated pairs accumulate as evidence.
    prior_sd:
        Std-dev of the ``N(0, prior_sd^2)`` prior. Smaller ⇒ stronger shrinkage
        toward equal strength (and more robust under separation / little data).
    anchor:
        If given (and present in ``items``), scores are re-centred so this item's
        strength is 0 and SEs become those of ``beta_i - beta_anchor``.
    """
    index = {it: i for i, it in enumerate(items)}
    n = len(items)
    pairs = [(index[w], index[lo]) for w, lo in comparisons]
    tau2 = prior_sd * prior_sd

    if n == 0:
        return BTResult([], {}, {})
    if not pairs:
        # No evidence: prior only ⇒ all strengths 0, SE = prior_sd.
        return BTResult(
            list(items),
            {it: 0.0 for it in items},
            {it: float(prior_sd) for it in items},
        )

    w_idx = np.array([p[0] for p in pairs], dtype=int)
    l_idx = np.array([p[1] for p in pairs], dtype=int)

    beta = np.zeros(n)
    hess = np.eye(n)  # last Hessian, reused for the covariance after convergence
    for _ in range(max_iter):
        diff = np.clip(beta[w_idx] - beta[l_idx], -30.0, 30.0)
        p = expit(diff)

        # Gradient of the negative log-posterior.
        grad = beta / tau2
        np.add.at(grad, w_idx, -(1.0 - p))
        np.add.at(grad, l_idx, (1.0 - p))

        # Hessian (positive-definite thanks to the prior ridge).
        hess = np.eye(n) / tau2
        wgt = p * (1.0 - p)
        np.add.at(hess, (w_idx, w_idx), wgt)
        np.add.at(hess, (l_idx, l_idx), wgt)
        np.add.at(hess, (w_idx, l_idx), -wgt)
        np.add.at(hess, (l_idx, w_idx), -wgt)

        step = np.linalg.solve(hess, grad)
        beta -= step
        if np.max(np.abs(step)) < tol:
            break

    cov = np.linalg.inv(hess)
    if anchor is not None and anchor in index:
        a = index[anchor]
        beta = beta - beta[a]
        var = np.diag(cov) + cov[a, a] - 2.0 * cov[:, a]
        se = np.sqrt(np.clip(var, 0.0, None))
        se[a] = 0.0
    else:
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    return BTResult(
        list(items),
        {it: float(beta[i]) for it, i in index.items()},
        {it: float(se[i]) for it, i in index.items()},
    )
