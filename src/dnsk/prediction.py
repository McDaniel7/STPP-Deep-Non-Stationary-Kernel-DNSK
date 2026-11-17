"""Next-event prediction from a fitted model (Sec. 5.2).

Given the history up to the last event ``t_n``, the density of the next event is::

    f(t, s) = lambda(t, s) * exp(- int_{t_n}^{t} int_S lambda(tau, nu) dnu dtau),  t > t_n.

:func:`predict_next_event` evaluates ``f`` on a grid over ``(t_n, t_end] x S``
and returns ``E[t_{n+1}]`` and ``E[s_{n+1}]`` under ``f`` restricted to that
window and renormalised, i.e. conditional on the next event occurring before
``t_end``.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ._grid import midpoint_grid
from .model import DNSK


@torch.no_grad()
def predict_next_event(
    model: DNSK,
    X: Tensor,
    mask: Tensor,
    t_end: Optional[float] = None,
    n_time: int = 200,
    n_space: int = 50,
    batch_size: int = 16,
) -> tuple[Tensor, Optional[Tensor]]:
    """Expected time (and location) of the next event of each sequence.

    Parameters
    ----------
    model:
        Fitted model.
    X, mask:
        Histories (see :func:`dnsk.data.pad_sequences`).
    t_end:
        End of the prediction window; defaults to the model's ``T1``.
    n_time:
        Number of time points on ``(t_n, t_end]``.
    n_space:
        Grid points per spatial dimension over ``S``.
    batch_size:
        Sequences processed at once (limits memory).

    Returns
    -------
    t_pred:
        ``[B]`` expected next-event times.
    s_pred:
        ``[B, d]`` expected next-event locations, or ``None`` for temporal models.
    """
    device = next(model.parameters()).device
    t_end = model.T1 if t_end is None else float(t_end)
    if model.spatial:
        s_grid, cell = midpoint_grid(model.kernel.S_low, model.kernel.S_high, n_space)
    t_out, s_out = [], []
    for start in range(0, X.shape[0], batch_size):
        xb, mb = X[start:start + batch_size].to(device), mask[start:start + batch_size].to(device)
        has_events = mb.any(-1)
        t_last = torch.where(has_events, (xb[..., 0] * mb).max(-1).values,
                             torch.full_like(xb[:, 0, 0], model.T0))
        dt = (t_end - t_last) / n_time  # [b]
        t_grid = t_last[:, None] + (torch.arange(n_time, device=device) + 0.5) * dt[:, None]  # midpoints
        if model.spatial:
            lam = model.intensity(xb, mb, t_grid, s_grid).clamp(min=0.0)  # [b, C, P]
            rate = lam.sum(-1) * cell  # [b, C] = int_S lambda
        else:
            lam = model.intensity(xb, mb, t_grid).clamp(min=0.0)  # [b, C]
            rate = lam
        cum = torch.cumsum(rate, dim=1) * dt[:, None] - 0.5 * rate * dt[:, None]  # Lambda at midpoints
        density_t = rate * torch.exp(-cum)  # [b, C] = int_S f(t, s) ds
        mass = density_t.sum(-1)
        expected_t = (density_t * t_grid).sum(-1) / mass.clamp(min=1e-12)
        t_out.append(torch.where(mass > 0, expected_t, t_last))  # empty window: no prediction possible
        if model.spatial:
            weights = lam * torch.exp(-cum)[..., None]  # [b, C, P]
            s_weights = weights.sum(1)  # [b, P]
            s_out.append(s_weights @ s_grid / s_weights.sum(-1, keepdim=True).clamp(min=1e-12))
    return torch.cat(t_out).cpu(), (torch.cat(s_out).cpu() if model.spatial else None)
