"""Simulating Hawkes-type point processes with an arbitrary kernel by thinning.

A *kernel function* is any callable::

    kernel(t_hist, s_hist, t, s) -> np.ndarray

giving the influence of past events at ``(t_hist, s_hist)`` on query points
``(t, s)``. It must follow NumPy broadcasting: times have shape ``[...]`` and
locations ``[..., d]``; for temporal processes ``s_hist`` and ``s`` are ``None``.
It may assume ``t > t_hist``. The intensity is
``lambda(t, s) = mu + sum_{t_i < t} kernel(t_i, s_i, t, s)``.
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional, Sequence

import numpy as np

KernelFn = Callable[[np.ndarray, Optional[np.ndarray], float, Optional[np.ndarray]], np.ndarray]


def _domain(S: Optional[Sequence[Sequence[float]]]) -> tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    if S is None or len(S) == 0:
        return None, None, 1.0
    S_arr = np.asarray(S, dtype=float)
    return S_arr[:, 0], S_arr[:, 1], float(np.prod(S_arr[:, 1] - S_arr[:, 0]))


def simulate(
    kernel: KernelFn,
    mu: float,
    T: Sequence[float],
    S: Optional[Sequence[Sequence[float]]] = None,
    *,
    upper_bound: float,
    n_sequences: int,
    min_events: int = 1,
    rng: Optional[np.random.Generator] = None,
    max_attempts: Optional[int] = None,
) -> list[np.ndarray]:
    """Draw event sequences by sequential thinning (Algorithm 2 of the paper).

    Candidates come from a homogeneous Poisson process with rate
    ``upper_bound`` per unit time and area; each candidate is accepted with
    probability ``lambda(t, s) / upper_bound`` given the events accepted so far.

    A sequence is discarded and re-drawn if the intensity ever exceeds
    ``upper_bound`` or becomes negative (possible with inhibiting kernels), or if
    it has fewer than ``min_events`` events. A warning reports the fraction of
    discarded draws, which should be small.

    Returns
    -------
    List of ``n_sequences`` arrays of shape ``[n_i, 1 + d]`` (``[n_i, 1]`` if temporal).
    """
    rng = np.random.default_rng() if rng is None else rng
    low, high, area = _domain(S)
    d = 0 if low is None else len(low)
    max_attempts = 100 * n_sequences if max_attempts is None else max_attempts
    rate = upper_bound * area
    sequences, reasons = [], {"exceeds_bound": 0, "negative": 0, "too_short": 0}
    attempts = 0
    while len(sequences) < n_sequences:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(f"too many discarded sequences: {reasons}; increase upper_bound?")
        times, locs, failure = [], [], None
        t = float(T[0])
        while True:
            t += rng.exponential(1.0 / rate)
            if t >= T[1]:
                break
            s = rng.uniform(low, high) if d else None
            lam = mu
            if times:
                lam += float(np.sum(kernel(np.asarray(times), np.asarray(locs) if d else None, t, s)))
            if lam > upper_bound:
                failure = "exceeds_bound"
                break
            if lam < 0:
                failure = "negative"
                break
            if rng.uniform() * upper_bound <= lam:
                times.append(t)
                if d:
                    locs.append(s)
        if failure is None and len(times) < min_events:
            failure = "too_short"
        if failure is not None:
            reasons[failure] += 1
            continue
        seq = np.asarray(times)[:, None]
        if d:
            seq = np.concatenate([seq, np.asarray(locs).reshape(-1, d)], axis=1)
        sequences.append(seq)
    discarded = attempts - n_sequences
    if discarded:
        warnings.warn(f"discarded {discarded}/{attempts} simulated sequences: {reasons}", stacklevel=2)
    return sequences


def true_intensity(
    kernel: KernelFn,
    mu: float,
    sequence: np.ndarray,
    t_grid: np.ndarray,
    s_grid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Ground-truth conditional intensity of one sequence on a grid.

    Parameters
    ----------
    sequence:
        ``[n, 1 + d]`` events (or ``[n]`` times).
    t_grid:
        ``[C]`` query times.
    s_grid:
        ``[P, d]`` query locations (spatial processes only).

    Returns
    -------
    ``[C]`` for temporal processes or ``[C, P]`` for spatio-temporal processes.
    """
    sequence = np.asarray(sequence, dtype=float).reshape(len(sequence), -1)
    t_grid = np.asarray(t_grid, dtype=float)
    times = sequence[:, 0]
    if s_grid is None:
        past = times[None, :] < t_grid[:, None]  # [C, n]
        k = kernel(times[None, :], None, t_grid[:, None], None) if len(times) else np.zeros_like(past, float)
        return mu + np.where(past, k, 0.0).sum(-1)
    s_grid = np.asarray(s_grid, dtype=float)
    locs = sequence[:, 1:]
    past = (times[None, None, :] < t_grid[:, None, None])  # [C, 1, n]
    if not len(times):
        return np.full((len(t_grid), len(s_grid)), float(mu))
    k = kernel(times[None, None, :], locs[None, None, :, :], t_grid[:, None, None], s_grid[None, :, None, :])
    return mu + np.where(past, k, 0.0).sum(-1)
