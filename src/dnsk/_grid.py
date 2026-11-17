"""Grid utilities behind the efficient likelihood computation (Sec. 4.2).

* :func:`interp_uniform` -- linear interpolation of functions tabulated on a
  uniform grid. Used for ``phi_l(t_j - t_i)`` so that the temporal displacement
  networks are only evaluated on the grid ``U_t``.
* :func:`cumtrapz_uniform` / :func:`integrate_interp_uniform` -- the integral
  ``F_l(x) = int_0^x phi_l`` of the piecewise-linear interpolant, evaluated
  exactly from the tabulated values.
* :func:`box_sums` -- sums of a tabulated function over axis-aligned boxes in
  O(2^d) per query using a summed-area table. Used for the truncated spatial
  integrals ``int_{B(0, s_max) cap (S - s_i)} v_r``.
"""

from __future__ import annotations

import itertools
import math

import torch
from torch import Tensor


def _bracket(x: Tensor, h: float, n_points: int) -> tuple[Tensor, Tensor]:
    """Left grid index and fractional offset of ``x`` on the grid ``0, h, ..., (n-1)h``."""
    pos = (x / h).clamp(0.0, n_points - 1.0)
    idx = pos.floor().long().clamp(max=n_points - 2)
    return idx, pos - idx


def _gather(values: Tensor, idx: Tensor) -> Tensor:
    """``values[:, idx]`` for an index tensor of arbitrary shape."""
    return values[:, idx.reshape(-1)].reshape(values.shape[0], *idx.shape)


def interp_uniform(values: Tensor, h: float, x: Tensor) -> Tensor:
    """Linearly interpolate tabulated functions.

    Parameters
    ----------
    values:
        ``[K, G]`` values of ``K`` functions on the grid ``0, h, ..., (G-1)h``.
    h:
        Grid spacing.
    x:
        Query points of any shape; clamped to ``[0, (G-1)h]``.

    Returns
    -------
    ``[K, *x.shape]`` interpolated values.
    """
    idx, frac = _bracket(x, h, values.shape[-1])
    v0, v1 = _gather(values, idx), _gather(values, idx + 1)
    return v0 + frac * (v1 - v0)


def cumtrapz_uniform(values: Tensor, h: float) -> Tensor:
    """Cumulative trapezoidal integral ``C[:, g] = int_0^{g h}`` (exact for the interpolant)."""
    steps = 0.5 * h * (values[:, 1:] + values[:, :-1])
    zeros = torch.zeros_like(values[:, :1])
    return torch.cat([zeros, torch.cumsum(steps, dim=1)], dim=1)


def integrate_interp_uniform(values: Tensor, cumulative: Tensor, h: float, x: Tensor) -> Tensor:
    """Exact ``int_0^x`` of the piecewise-linear interpolant of ``values``.

    ``cumulative`` must be ``cumtrapz_uniform(values, h)``. Returns ``[K, *x.shape]``.
    """
    idx, frac = _bracket(x, h, values.shape[-1])
    v0, v1 = _gather(values, idx), _gather(values, idx + 1)
    vx = v0 + frac * (v1 - v0)
    return _gather(cumulative, idx) + 0.5 * frac * h * (v0 + vx)


def midpoint_grid(lower: Tensor, upper: Tensor, n: int) -> tuple[Tensor, float]:
    """Cell centres of a regular ``n^d`` grid on the box ``[lower, upper]``.

    Returns the ``[n^d, d]`` centres (C-order, last dimension fastest) and the cell volume.
    """
    widths = (upper - lower) / n
    axes = [lower[k] + (torch.arange(n, dtype=lower.dtype, device=lower.device) + 0.5) * widths[k]
            for k in range(lower.numel())]
    centres = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, lower.numel())
    return centres, float(torch.prod(widths))


def summed_area_table(values: Tensor) -> Tensor:
    """Zero-padded d-dimensional prefix sums of ``values`` with shape ``[K, n_1, ..., n_d]``."""
    table = values
    for dim in range(1, values.dim()):
        table = torch.cumsum(table, dim=dim)
    pad = [1, 0] * (values.dim() - 1)  # one leading zero on every spatial axis
    return torch.nn.functional.pad(table, pad)


def box_sums(table: Tensor, lo: Tensor, hi: Tensor) -> Tensor:
    """Sum of the tabulated values over the inclusive index boxes ``[lo, hi]``.

    Parameters
    ----------
    table:
        Output of :func:`summed_area_table`, shape ``[K, n_1 + 1, ..., n_d + 1]``.
    lo, hi:
        Integer tensors of shape ``[..., d]`` with ``0 <= lo <= n`` and ``-1 <= hi <= n - 1``.
        Empty boxes (``lo > hi`` in any dimension) yield 0.

    Returns
    -------
    ``[K, ...]`` box sums.
    """
    n_funcs, sizes = table.shape[0], table.shape[1:]
    d = len(sizes)
    strides = [math.prod(sizes[k + 1:]) for k in range(d)]
    flat = table.reshape(n_funcs, -1)
    out = torch.zeros(n_funcs, *lo.shape[:-1], dtype=table.dtype, device=table.device)
    for corner in itertools.product((0, 1), repeat=d):
        index = torch.zeros_like(lo[..., 0])
        sign = 1.0
        for k, use_hi in enumerate(corner):
            if use_hi:
                index = index + (hi[..., k] + 1) * strides[k]
            else:
                index = index + lo[..., k] * strides[k]
                sign = -sign
        out = out + sign * _gather(flat, index)
    nonempty = (lo <= hi).all(dim=-1)
    return out * nonempty
