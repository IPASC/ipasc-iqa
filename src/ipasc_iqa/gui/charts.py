"""Turn scatter data into a render-ready model for an inline SVG plot.

Keeps all pixel-layout maths in one place so the template only iterates over
pre-computed coordinates. Mirrors the look of the PhotIQA paper figures:
points coloured by group, a black rank-mean trend line, and per-group
mean ± std error-bar crosshairs.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from ipasc_iqa.analysis.scatter import ScatterData

# Group colours (cycled). Chosen to read well on the light plot background.
_PALETTE = ["#1d4ed8", "#7c3aed", "#ea580c", "#dc2626", "#0891b2", "#16a34a", "#db2777"]


def _ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    """``n + 1`` evenly spaced tick values across ``[lo, hi]``."""
    if hi <= lo:
        return [lo]
    return [lo + (hi - lo) * i / n for i in range(n + 1)]


def _fmt(v: float) -> str:
    """Compact numeric label."""
    av = abs(v)
    if av != 0 and (av < 1e-2 or av >= 1e4):
        return f"{v:.1e}"
    if av >= 100:
        return f"{v:.0f}"
    return f"{v:.2f}".rstrip("0").rstrip(".")


def scatter_chart(
    data: ScatterData,
    *,
    srcc: float | None,
    krcc: float | None,
    x_label: str,
    y_label: str,
    width: int = 540,
    height: int = 380,
) -> dict[str, Any]:
    """Build an SVG model dict from :class:`ScatterData`."""
    xs = np.asarray(data.xs, dtype=float)
    ys = np.asarray(data.ys, dtype=float)

    # Plot rectangle (leave room for axis labels/ticks).
    ml, mr, mt, mb = 64, 18, 16, 46
    px0, py0 = ml, mt
    pw, ph = width - ml - mr, height - mt - mb

    def _bounds(v: np.ndarray) -> tuple[float, float]:
        lo, hi = float(v.min()), float(v.max())
        if hi <= lo:
            lo, hi = lo - 1.0, hi + 1.0
        pad = (hi - lo) * 0.05
        return lo - pad, hi + pad

    xmin, xmax = _bounds(xs)
    ymin, ymax = _bounds(ys)

    def sx(x: float) -> float:
        return px0 + (x - xmin) / (xmax - xmin) * pw

    def sy(y: float) -> float:
        return py0 + ph - (y - ymin) / (ymax - ymin) * ph

    # Stable colour per group.
    uniq = sorted({g for g in data.groups if g is not None})
    color_of = {g: _PALETTE[i % len(_PALETTE)] for i, g in enumerate(uniq)}
    default_color = "#475569"

    points = [
        {
            "cx": round(sx(x), 1),
            "cy": round(sy(y), 1),
            "color": color_of.get(g, default_color),
        }
        for x, y, g in zip(xs, ys, data.groups, strict=True)
    ]

    trend = " ".join(
        f"{round(sx(tx), 1)},{round(sy(ty), 1)}"
        for tx, ty in zip(data.trend_x, data.trend_y, strict=True)
    )

    # Per-group mean ± std crosshairs (only when groups are present).
    group_bars: list[dict[str, Any]] = []
    if uniq:
        by_group: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for x, y, g in zip(xs, ys, data.groups, strict=True):
            if g is not None:
                by_group[g].append((x, y))
        for g in uniq:
            gx = np.asarray([p[0] for p in by_group[g]], dtype=float)
            gy = np.asarray([p[1] for p in by_group[g]], dtype=float)
            if gx.size == 0:
                continue
            group_bars.append(
                {
                    "label": g,
                    "color": color_of[g],
                    "cx": round(sx(float(gx.mean())), 1),
                    "cy": round(sy(float(gy.mean())), 1),
                    # Half-extents in pixels for the error-bar arms.
                    "exw": round(float(gx.std(ddof=0)) / (xmax - xmin) * pw, 1),
                    "eyh": round(float(gy.std(ddof=0)) / (ymax - ymin) * ph, 1),
                }
            )

    x_ticks = [{"x": round(sx(t), 1), "label": _fmt(t)} for t in _ticks(xmin, xmax)]
    y_ticks = [{"y": round(sy(t), 1), "label": _fmt(t)} for t in _ticks(ymin, ymax)]

    return {
        "width": width,
        "height": height,
        "plot": {"x": px0, "y": py0, "w": pw, "h": ph},
        "points": points,
        "trend": trend,
        "x_ticks": x_ticks,
        "y_ticks": y_ticks,
        "group_bars": group_bars,
        "legend": [{"label": g, "color": color_of[g]} for g in uniq],
        "x_label": x_label,
        "y_label": y_label,
        "measure": data.measure,
        "srcc": srcc,
        "krcc": krcc,
        "n": len(data.xs),
    }
