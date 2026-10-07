"""Statistics for cross-modal questions (used by phase 7 and compare_videos.py).

Everything here is plain NumPy so it can be unit-tested without the models.

Two questions are answered with an explicit null hypothesis instead of a bare
percentage:

1. Are cuts placed on beats more often than chance?
   ``cut_on_beat_test`` compares the observed hit rate with a null in which
   every cut is moved independently by a random offset within ±half the local
   beat period. Each cut stays in the same part of the song (same section,
   same editing rhythm) but its phase relative to the beat is randomized.
   (Shifting all cuts by one common offset does NOT work: beat grids are
   nearly periodic, so any shift close to a multiple of the beat period puts
   on-beat cuts straight back on beats and the test loses its power.)

2. Do visual and audio features co-vary across shots?
   ``correlate`` reports Spearman's rho with a moving-block bootstrap 95% CI.
   Shots next to each other are not independent (a chorus spans many shots),
   so resampling blocks of consecutive shots gives more honest (wider)
   intervals than resampling single shots.
"""
from __future__ import annotations
import numpy as np


def nearest_distance(times, refs) -> np.ndarray:
    """|t - nearest ref| for each t (inf if there are no refs)."""
    times = np.asarray(times, dtype=float)
    refs = np.sort(np.asarray(refs, dtype=float))
    if refs.size == 0:
        return np.full(times.shape, np.inf)
    idx = np.searchsorted(refs, times)
    left = refs[np.clip(idx - 1, 0, refs.size - 1)]
    right = refs[np.clip(idx, 0, refs.size - 1)]
    return np.minimum(np.abs(times - left), np.abs(times - right))


def beat_window_coverage(beats, tol: float, t0: float, t1: float) -> float:
    """Fraction of [t0, t1] lying within ±tol of some beat.

    This is the hit rate expected if cuts were placed uniformly at random.
    """
    if t1 <= t0 or len(beats) == 0:
        return 0.0
    covered, cur_s, cur_e = 0.0, None, None
    for b in sorted(beats):
        s, e = max(t0, b - tol), min(t1, b + tol)
        if e <= s:
            continue
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                covered += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        covered += cur_e - cur_s
    return covered / (t1 - t0)


def cut_on_beat_test(cuts, beats, tol: float, duration: float,
                     n_perm: int = 5000, seed: int = 0) -> dict:
    """Observed on-beat rate vs chance, with a circular-shift permutation p-value."""
    cuts = np.asarray(cuts, dtype=float)
    beats = np.asarray(beats, dtype=float)
    n = int(cuts.size)
    chance = beat_window_coverage(beats, tol, 0.0, duration)
    if n == 0 or beats.size == 0 or duration <= 0:
        return {"tolerance_sec": tol, "n_cuts": n, "hits": 0, "observed_pct": 0.0,
                "chance_pct": round(float(chance) * 100, 1), "p_value": None}
    hits = int((nearest_distance(cuts, beats) <= tol).sum())
    beats = np.sort(beats)
    # Local beat period: the gap between the beats around each cut (cuts before
    # the first / after the last beat use the nearest gap).
    if beats.size > 1:
        idx = np.clip(np.searchsorted(beats, cuts), 1, beats.size - 1)
        period = beats[idx] - beats[idx - 1]
        period = np.where(period > 0, period, float(np.median(np.diff(beats))))
    else:
        period = np.full(n, duration)
    rng = np.random.default_rng(seed)
    offsets = rng.uniform(-0.5, 0.5, (n_perm, n)) * period
    null = cuts + offsets
    null_hits = (nearest_distance(null.ravel(), beats).reshape(n_perm, n) <= tol).sum(axis=1)
    p = (1 + int((null_hits >= hits).sum())) / (1 + n_perm)
    return {
        "tolerance_sec": tol,
        "n_cuts": n,
        "hits": hits,
        "observed_pct": round(100 * hits / n, 1),
        "chance_pct": round(float(chance) * 100, 1),
        "null_mean_pct": round(100 * float(null_hits.mean()) / n, 1),
        "p_value": round(p, 4),
    }


def rankdata(x) -> np.ndarray:
    """Average ranks (ties share the mean rank), like scipy.stats.rankdata."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size)
    sx = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x, y) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size < 3:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def block_bootstrap_ci(x, y, stat=spearman, n_boot: int = 2000, block: int | None = None,
                       seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Moving-block bootstrap CI for stat(x, y) over a time-ordered sequence."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = x.size
    if n < 6:
        return float("nan"), float("nan")
    block = block or max(2, int(round(n ** (1 / 3))))
    starts_max = n - block + 1
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        starts = rng.integers(0, starts_max, n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        v = stat(x[idx], y[idx])
        if not np.isnan(v):
            vals.append(v)
    if len(vals) < n_boot // 2:
        return float("nan"), float("nan")
    lo, hi = np.quantile(vals, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def correlate(name_x: str, x, name_y: str, y, **kw) -> dict:
    """Spearman rho + block-bootstrap CI, skipping shots where either value is missing."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    rho = spearman(x, y)
    lo, hi = block_bootstrap_ci(x, y, **kw)

    def r(v):
        return None if np.isnan(v) else round(v, 3)
    return {"x": name_x, "y": name_y, "n": int(x.size), "rho": r(rho), "ci_low": r(lo), "ci_high": r(hi),
            "significant": bool(not np.isnan(lo) and (lo > 0 or hi < 0))}


def overlap_mean(records: list[dict], key: str, s: float, e: float) -> float:
    """Mean of records[key] over [s, e], weighted by each record's overlap with it.

    records carry start_sec/end_sec; returns nan if nothing overlaps.
    """
    num = den = 0.0
    for r in records:
        rs, re_ = float(r["start_sec"]), float(r["end_sec"])
        ov = min(e, re_) - max(s, rs)
        if ov <= 0:
            continue
        v = r.get(key)
        if v in (None, ""):
            continue
        num += float(v) * ov
        den += ov
    return num / den if den > 0 else float("nan")
