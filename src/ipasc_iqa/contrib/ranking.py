"""Binary insertion sort over pairwise comparisons (2AFC).

A :class:`RankingSession` ranks a set of items by asking the user to compare two
at a time. Items are placed one by one into a sorted list using binary search:
each placement costs ~log2(k) comparisons, so a full set of n items is ranked in
Θ(n·log n) comparisons (see Breger et al.'s PhotIQA figure).

The session is presentation-randomised in three ways, matching that figure:

1. the initial item order is shuffled (Fisher-Yates, via ``random.shuffle``);
2. the first pair is shown in a random left/right order;
3. every subsequent comparison's left/right order is randomised.

The class is pure (no web/IO); the route layer drives it one comparison at a
time and persists the result when :pyattr:`is_complete` becomes true.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class RankingSession:
    """Stateful binary-insertion-sort ranking of ``items`` (a list of ids).

    The sorted list is maintained worst→best (index 0 = lowest quality), matching
    the "low → high quality" axis in the reference figure.
    """

    items: list[str]
    dataset: str
    reference: str  # stem / reference group id
    task: str | None
    user_uid: str
    seed: int | None = None

    # --- internal state (initialised in __post_init__) ---
    _rng: random.Random = field(init=False)
    _order: list[str] = field(init=False)  # shuffled input order
    sorted_ids: list[str] = field(init=False)  # placed items, worst→best
    next_index: int = field(init=False)  # index into _order of the item being placed
    lo: int = field(init=False)
    hi: int = field(init=False)
    comparisons: list[dict[str, str]] = field(init=False)  # [{"winner","loser"}, ...]

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._order = list(self.items)
        self._rng.shuffle(self._order)  # (1) Fisher-Yates initial order
        # Seed the sorted list with the first element; insert the rest.
        self.sorted_ids = self._order[:1]
        self.next_index = 1
        self.comparisons = []
        self._begin_insertion()

    # -- placement bookkeeping ------------------------------------------------

    def _begin_insertion(self) -> None:
        """Reset the binary-search window for the next item to place."""
        self.lo = 0
        self.hi = len(self.sorted_ids)

    @property
    def is_complete(self) -> bool:
        return self.next_index >= len(self._order)

    @property
    def n_items(self) -> int:
        return len(self._order)

    def _current(self) -> str:
        return self._order[self.next_index]

    def _pivot_index(self) -> int:
        return (self.lo + self.hi) // 2

    def _pivot(self) -> str:
        return self.sorted_ids[self._pivot_index()]

    # -- the API the route layer uses -----------------------------------------

    def current_pair(self) -> dict[str, str] | None:
        """The next pair to show, or ``None`` when complete.

        Returns ``{"left", "right", "current", "pivot"}``. ``left``/``right`` are
        the randomised display positions; ``current``/``pivot`` identify the
        roles so :meth:`submit` can interpret the winner.
        """
        if self.is_complete:
            return None
        current, pivot = self._current(), self._pivot()
        positions = [current, pivot]
        self._rng.shuffle(positions)  # (2,3) randomise left/right each time
        return {
            "left": positions[0],
            "right": positions[1],
            "current": current,
            "pivot": pivot,
        }

    def submit(self, winner: str) -> None:
        """Record that ``winner`` (one of the current pair) was chosen as higher.

        Advances the binary search; when the insertion point is found, the item
        is placed and the next item begins. No-op once complete.
        """
        if self.is_complete:
            return
        current, pivot = self._current(), self._pivot()
        if winner not in (current, pivot):
            raise ValueError(f"winner {winner!r} is not part of the current pair")
        loser = pivot if winner == current else current
        self.comparisons.append({"winner": winner, "loser": loser})

        if winner == current:
            # current is higher quality than the pivot → search the upper half.
            self.lo = self._pivot_index() + 1
        else:
            self.hi = self._pivot_index()

        if self.lo >= self.hi:
            # Insertion point found: place the current item, move to the next.
            self.sorted_ids.insert(self.lo, current)
            self.next_index += 1
            self._begin_insertion()

    def result(self) -> dict:
        """Serialisable summary of a completed session (worst→best order)."""
        return {
            "dataset": self.dataset,
            "reference": self.reference,
            "task": self.task,
            "user_uid": self.user_uid,
            "order": list(self.sorted_ids),
            "comparisons": list(self.comparisons),
            "n_comparisons": len(self.comparisons),
        }
