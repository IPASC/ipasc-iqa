"""Per-dataset standings / leaderboard.

The mockup's "Standings" screen ranks reconstruction algorithms within each
dataset by a score. The default score is the expert quality (z-scored MOS), i.e.
"which reconstruction algorithm produced the best-rated images", but the function
is generic: feed it ``(dataset, group, score)`` triples and it ranks ``group`` per
``dataset`` by mean ``score``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class StandingRow:
    dataset: str
    rank: int
    group: str
    score: float
    n: int

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "rank": self.rank,
            "group": self.group,
            "score": self.score,
            "n": self.n,
        }


def compute_standings(
    records: list[tuple[str, str, float]],
    *,
    higher_is_better: bool = True,
) -> list[StandingRow]:
    """Rank groups within each dataset by mean score.

    Parameters
    ----------
    records:
        ``(dataset, group, score)`` triples — one per image (or per measurement).
    higher_is_better:
        If True, higher mean score ranks first (rank 1).

    Returns
    -------
    Standings sorted by (dataset, rank).
    """
    # dataset -> group -> list of scores
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for dataset, group, score in records:
        if score is None or not np.isfinite(score):
            continue
        buckets[dataset][group].append(float(score))

    rows: list[StandingRow] = []
    for dataset in sorted(buckets):
        group_means = [
            (group, float(np.mean(scores)), len(scores))
            for group, scores in buckets[dataset].items()
            if scores
        ]
        group_means.sort(key=lambda gm: gm[1], reverse=higher_is_better)
        for rank, (group, mean_score, n) in enumerate(group_means, start=1):
            rows.append(
                StandingRow(dataset=dataset, rank=rank, group=group, score=mean_score, n=n)
            )

    return rows
