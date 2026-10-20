"""Chance-corrected statistics for cross-modal claims (round 8).

A bare "48 % of cuts land on a beat" is meaningless: with a ±100 ms window
and beats every ~0.46 s (130 BPM) a RANDOM cut lands on a beat ~43 % of the
time. Every cross-modal rate the pipeline reports therefore ships with its
chance level and a p-value.

  - beat_alignment():   hit rate vs an exact per-cut chance model (each
    cut's beat phase uniform), plus the Rayleigh test on beat phase (are
    cuts concentrated at one point of the beat cycle?) and a circular-shift
    sensitivity check.

Pure functions; numpy only.
"""
from __future__ import annotations
import bisect
import math


def hit_rate(events: list[float], beats: list[float], tol: float) -> float:
    """Fraction of events within ±tol of any beat."""
    if not events or not beats:
        return float("nan")
    bs = sorted(beats)
    hits = 0
    for t in events:
        j = bisect.bisect_left(bs, t)
        near = min((abs(t - bs[k]) for k in (j - 1, j) if 0 <= k < len(bs)), default=math.inf)
        hits += near <= tol
    return hits / len(events)


def beat_phases(events: list[float], beats: list[float]) -> list[float]:
    """Phase in [0, 1) of each event within its enclosing beat interval
    (0 = on the beat). Events outside the beat grid are skipped."""
    bs = sorted(beats)
    out = []
    for t in events:
        j = bisect.bisect_right(bs, t) - 1
        if 0 <= j < len(bs) - 1 and bs[j + 1] > bs[j]:
            out.append((t - bs[j]) / (bs[j + 1] - bs[j]))
    return out


def rayleigh(phases: list[float]) -> dict:
    """Rayleigh test of non-uniformity for phases in [0, 1).
    Returns mean resultant length R, mean phase, z and p (Zar's approximation)."""
    n = len(phases)
    if n < 2:
        return {"n": n, "R": None, "mean_phase": None,
                "z": None, "p": None}
    c = sum(math.cos(2 * math.pi * p) for p in phases)
    s = sum(math.sin(2 * math.pi * p) for p in phases)
    Rn = math.hypot(c, s)
    R = Rn / n
    z = Rn * Rn / n
    p = math.exp(math.sqrt(1 + 4 * n + 4 * (n * n - Rn * Rn)) - (1 + 2 * n))
    mean_phase = (math.atan2(s, c) / (2 * math.pi)) % 1.0
    return {"n": n, "R": round(R, 4), "mean_phase": round(mean_phase, 4),
            "z": round(z, 4), "p": min(1.0, p)}


def poisson_binomial_sf(probs: list[float], k: int) -> float:
    """P(X >= k) for X = sum of independent Bernoulli(p_i). Exact DP, O(n^2)."""
    dist = [1.0]
    for p in probs:
        nxt = [0.0] * (len(dist) + 1)
        for j, q in enumerate(dist):
            nxt[j] += q * (1 - p)
            nxt[j + 1] += q * p
        dist = nxt
    return min(1.0, max(0.0, sum(dist[k:])))


def beat_alignment(events: list[float], beats: list[float], duration: float,
                   tol: float = 0.1, n_perm: int = 2000, seed: int = 0) -> dict:
    """Is the cut-on-beat rate higher than chance?

    Primary test (exact): under H0 each cut's phase within its beat interval
    is uniform, so cut i hits with p_i = min(1, 2*tol / local_beat_period).
    Expected hits = sum p_i; p-value = Poisson-binomial P(X >= observed).
    Local periods absorb tempo changes. Cuts outside the beat grid are not
    tested (counted in n_outside_grid).

    Sensitivity check: circular-shift null (whole cut train rotated against
    the beat grid). It keeps the editing rhythm, so it is very conservative
    when cuts are spaced at whole multiples of the beat (then every shift
    moves all cuts to the same phase) — reported, not used for the verdict.
    """
    import numpy as np
    bs = sorted(beats)
    res = {"n_events": len(events), "tolerance_sec": tol,
           "null_model": "per-cut uniform beat phase (exact Poisson-binomial)"}
    probs, hits = [], 0
    for t in events:
        j = bisect.bisect_right(bs, t) - 1
        if not (0 <= j < len(bs) - 1) or bs[j + 1] <= bs[j]:
            continue
        period = bs[j + 1] - bs[j]
        probs.append(min(1.0, 2 * tol / period))
        hits += min(t - bs[j], bs[j + 1] - t) <= tol
    n = len(probs)
    res["n_tested"] = n
    res["n_outside_grid"] = len(events) - n
    if n < 3:
        res.update({"observed_rate": None, "chance_rate": None,
                    "chance_sd": None, "z": None,
                    "p_value": None, "significant": False,
                    "note": "too few cuts inside the beat grid for a test"})
        return res
    mu = sum(probs)
    var = sum(p * (1 - p) for p in probs)
    p_val = poisson_binomial_sf(probs, hits)
    res.update({
        "observed_rate": round(hits / n, 4),
        "chance_rate": round(mu / n, 4),
        "chance_sd": round(math.sqrt(var) / n, 4),
        "z": round((hits - mu) / math.sqrt(var), 3) if var > 0 else None,
        "p_value": round(p_val, 6),
        "significant": bool(p_val < 0.05),
        "phase_rayleigh": rayleigh(beat_phases(events, beats)),
    })
    if duration > 0 and n_perm > 0:
        rng = np.random.default_rng(seed)
        ev = np.asarray(events)
        obs = hit_rate(events, beats, tol)
        null = np.array([hit_rate(np.mod(ev + rng.uniform(0, duration), duration).tolist(),
                                  beats, tol) for _ in range(n_perm)])
        res["circular_shift_check"] = {
            "observed_rate": round(obs, 4),
            "chance_rate": round(float(null.mean()), 4),
            "p_value": round((1 + int((null >= obs - 1e-12).sum())) / (n_perm + 1), 4),
            "n_permutations": n_perm,
        }
    return res


def boundary_alignment(boundaries: list[float], cuts: list[float], duration: float,
                       tol: float = 0.25) -> dict:
    """Do section boundaries coincide with cuts more than chance?

    A boundary "hits" if a cut lies within +-tol. Under H0 boundary positions
    are independent of the edit, so each hits with probability p = share of
    the timeline within tol of some cut (union of [c-tol, c+tol]). Exact
    binomial tail via poisson_binomial_sf.
    """
    res = {"n_boundaries": len(boundaries), "tolerance_sec": tol,
           "null_model": "boundary position independent of cuts (coverage of cut windows)"}
    if len(boundaries) < 2 or not cuts or duration <= 0:
        res.update({"observed_rate": None, "chance_rate": None, "p_value": None,
                    "significant": False, "note": "too few section boundaries or cuts"})
        return res
    spans = sorted((max(0.0, c - tol), min(duration, c + tol)) for c in cuts)
    covered, cur_s, cur_e = 0.0, None, None
    for s, e in spans:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                covered += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    covered += cur_e - cur_s
    p = min(1.0, covered / duration)
    cs = sorted(cuts)
    hits = 0
    for b in boundaries:
        j = bisect.bisect_left(cs, b)
        near = min((abs(b - cs[k]) for k in (j - 1, j) if 0 <= k < len(cs)), default=math.inf)
        hits += near <= tol
    n = len(boundaries)
    pv = poisson_binomial_sf([p] * n, hits)
    res.update({"observed_rate": round(hits / n, 4), "chance_rate": round(p, 4),
                "p_value": round(pv, 6), "significant": bool(pv < 0.05), "hits": hits})
    return res
