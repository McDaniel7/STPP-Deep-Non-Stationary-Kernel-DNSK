"""DNSK point process: conditional intensity and efficient log-likelihood (Sec. 4).

Conditional intensity (Eq. 3) with the DNSK kernel of Eq. (4)::

    lambda(t, s) = mu + sum_{(t_i, s_i): t_i < t} k(t_i, t - t_i, s_i, s - s_i)

Log-likelihood (Eq. 2)::

    l = sum_j log lambda(t_j, s_j) - int_T int_S lambda(t, s) ds dt

Efficient computation (Sec. 4.2):

* ``phi_l`` is evaluated only on a uniform grid ``U_t`` over ``[0, tau_max]``;
  ``phi_l(t_j - t_i)`` is obtained by linear interpolation and
  ``F_l(x) = int_0^x phi_l`` exactly from the tabulated values.
* ``v_r`` is evaluated directly, but only on event pairs within the local
  support ``0 < t_j - t_i <= tau_max`` and ``||s_j - s_i|| <= s_max``.
* Because the kernel is a sum of separable terms, the integral decomposes
  (Eq. 8) into ``mu |S| T + sum_i sum_{l,r} alpha_lr psi_l(t_i) u_r(s_i)
  F_l(min(T - t_i, tau_max)) V_r(s_i)`` with
  ``V_r(s_i) = int_{B(0, s_max) cap (S - s_i)} v_r``. ``v_r`` is tabulated once
  on a grid ``U_s`` over the ball and ``V_r(s_i)`` is read off a summed-area
  table, so no numerical integration of ``lambda`` itself is needed.

Data layout: a batch of ``B`` sequences is a zero-padded tensor ``X`` of shape
``[B, N, 1 + d_S]`` (column 0 is time, the rest is location) and a boolean
``mask`` of shape ``[B, N]`` marking real events. Events must be sorted in time.
See :func:`dnsk.data.pad_sequences`.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import torch
from torch import Tensor, nn

from ._grid import (
    box_sums,
    cumtrapz_uniform,
    integrate_interp_uniform,
    interp_uniform,
    midpoint_grid,
    summed_area_table,
)
from .kernel import DNSKKernel


class DNSK(nn.Module):
    """Spatio-temporal (or temporal) point process with a DNSK influence kernel.

    Parameters
    ----------
    T:
        Observation window ``(T0, T1)``.
    S:
        Spatial domain ``[(low_1, high_1), ..., (low_d, high_d)]``; ``None`` for
        temporal point processes.
    mu:
        Background rate ``mu`` (events per unit time per unit area).
    tau_max, s_max:
        Temporal / spatial range of influence (local support of ``phi_l`` / ``v_r``).
    n_basis_time, n_basis_space:
        Temporal rank ``L`` and spatial rank ``R`` of the kernel.
    learn_mu:
        If ``True``, ``mu`` is a learnable (softplus-positive) parameter;
        otherwise it is a fixed hyper-parameter.
    n_grid_time:
        ``|U_t|``: grid size over ``[0, tau_max]`` for ``phi_l``.
    n_grid_space:
        Grid points *per spatial dimension* over ``[-s_max, s_max]^d`` used to
        integrate ``v_r`` (``|U_s| = n_grid_space^d``).
    n_barrier_time, n_barrier_space:
        Size of the log-barrier grid ``U_bar,t x U_bar,s``: points over ``[T0, T1]``
        and points *per spatial dimension* over ``S``.
    log_eps:
        ``log lambda`` is computed as ``log(max(lambda, log_eps))``.
    **kernel_kwargs:
        Forwarded to :class:`dnsk.kernel.DNSKKernel` (network width, depth, initialisation).
    """

    def __init__(
        self,
        T: Sequence[float],
        S: Optional[Sequence[Sequence[float]]] = None,
        *,
        mu: float,
        tau_max: float,
        s_max: Optional[float] = None,
        n_basis_time: int = 1,
        n_basis_space: int = 1,
        learn_mu: bool = False,
        n_grid_time: int = 50,
        n_grid_space: int = 50,
        n_barrier_time: int = 50,
        n_barrier_space: int = 20,
        log_eps: float = 1e-5,
        **kernel_kwargs: Any,
    ) -> None:
        super().__init__()
        if mu <= 0:
            raise ValueError("mu must be positive")
        if n_grid_time < 2:
            raise ValueError("n_grid_time must be >= 2")
        self.config = dict(
            T=[float(T[0]), float(T[1])],
            S=None if S is None else [[float(lo), float(hi)] for lo, hi in S],
            mu=float(mu), tau_max=float(tau_max), s_max=None if s_max is None else float(s_max),
            n_basis_time=n_basis_time, n_basis_space=n_basis_space, learn_mu=learn_mu,
            n_grid_time=n_grid_time, n_grid_space=n_grid_space, n_barrier_time=n_barrier_time,
            n_barrier_space=n_barrier_space, log_eps=log_eps, **kernel_kwargs,
        )
        self.kernel = DNSKKernel(T, S, tau_max=tau_max, s_max=s_max, n_basis_time=n_basis_time,
                                 n_basis_space=n_basis_space, **kernel_kwargs)
        self.T0, self.T1 = self.kernel.T0, self.kernel.T1
        self.spatial = self.kernel.spatial
        self.log_eps = log_eps

        if learn_mu:
            self.mu_raw = nn.Parameter(torch.tensor(math.log(math.expm1(mu))))
        else:
            self.register_buffer("mu_fixed", torch.tensor(float(mu)))

        # Temporal grid U_t over [0, tau_max] for phi_l.
        self.h_time = self.kernel.tau_max / (n_grid_time - 1)
        self.register_buffer("tau_grid", torch.linspace(0.0, self.kernel.tau_max, n_grid_time))
        # Temporal barrier grid U_bar,t.
        self.register_buffer("barrier_t", torch.linspace(self.T0, self.T1, n_barrier_time))

        if self.spatial:
            d, smax = self.kernel.d_space, self.kernel.s_max
            self.area = float(torch.prod(self.kernel.S_high - self.kernel.S_low))
            # Spatial grid U_s over the ball B(0, s_max) (as cells of the box [-s_max, s_max]^d).
            self.n_grid_space = n_grid_space
            self.cell_space = 2.0 * smax / n_grid_space
            ds_grid, cell_vol = midpoint_grid(torch.full((d,), -smax), torch.full((d,), smax), n_grid_space)
            self.cell_vol = cell_vol
            self.register_buffer("ds_grid", ds_grid)
            self.register_buffer("ds_grid_in_ball", torch.linalg.norm(ds_grid, dim=-1) <= smax)
            # Spatial barrier grid U_bar,s over S.
            barrier_s, _ = midpoint_grid(self.kernel.S_low, self.kernel.S_high, n_barrier_space)
            self.register_buffer("barrier_s", barrier_s)
        else:
            self.area = 1.0

    # --------------------------------------------------------------- helpers
    @property
    def mu(self) -> Tensor:
        """Background rate (scalar tensor)."""
        if hasattr(self, "mu_raw"):
            return nn.functional.softplus(self.mu_raw)
        return self.mu_fixed

    def _split(self, X: Tensor) -> tuple[Tensor, Optional[Tensor]]:
        return X[..., 0], (X[..., 1:] if self.spatial else None)

    def _phi_table(self) -> Tensor:
        """``phi_l`` on ``U_t``: ``[L, |U_t|]``."""
        return self.kernel.phi(self.tau_grid)

    # ------------------------------------------------------------ intensity
    def event_intensity(self, X: Tensor, mask: Tensor) -> Tensor:
        """Conditional intensity at the observed events, ``lambda(t_j, s_j)``: ``[B, N]``.

        Only pairs ``(i, j)`` inside the kernel support are evaluated.
        """
        t, s = self._split(X)
        tau = t[:, None, :] - t[:, :, None]  # [B, i, j] = t_j - t_i
        pair = mask[:, :, None] & mask[:, None, :] & (tau > 0) & (tau <= self.kernel.tau_max)
        if self.spatial:
            ds = s[:, None, :, :] - s[:, :, None, :]  # [B, i, j, d] = s_j - s_i
            pair = pair & (torch.linalg.norm(ds, dim=-1) <= self.kernel.s_max)
        b, i, j = pair.nonzero(as_tuple=True)

        lam = self.mu.expand(t.shape).clone()
        if b.numel() == 0:
            return lam
        phi = interp_uniform(self._phi_table(), self.h_time, tau[b, i, j])  # [L, P]
        psi = self.kernel.psi(t)[:, b, i]  # [L, P]
        time = self.kernel.alpha @ (psi * phi)  # [R, P]
        if self.spatial:
            contrib = (time * self.kernel.u(s)[:, b, i] * self.kernel.v(ds[b, i, j])).sum(0)
        else:
            contrib = time[0]
        return lam.index_put((b, j), contrib, accumulate=True)

    def intensity(
        self,
        X: Tensor,
        mask: Tensor,
        t_grid: Tensor,
        s_grid: Optional[Tensor] = None,
    ) -> Tensor:
        """Conditional intensity on a query grid given each sequence's history.

        Parameters
        ----------
        X, mask:
            Batch of sequences (the history).
        t_grid:
            Query times ``[C]`` (shared) or ``[B, C]`` (per sequence). Only events
            strictly before a query time contribute.
        s_grid:
            Query locations ``[P, d]`` (spatial models only).

        Returns
        -------
        ``[B, C]`` for temporal models or ``[B, C, P]`` for spatio-temporal models.
        """
        t, s = self._split(X)
        if t_grid.dim() == 1:
            t_grid = t_grid.expand(t.shape[0], -1)
        tau = t_grid[:, :, None] - t[:, None, :]  # [B, C, N]
        support = mask[:, None, :] & (tau > 0) & (tau <= self.kernel.tau_max)
        phi = interp_uniform(self._phi_table(), self.h_time, tau)  # [L, B, C, N]
        time = torch.einsum("rl,lbcn,lbn->rbcn", self.kernel.alpha, phi, self.kernel.psi(t))
        time = time * support  # [R, B, C, N]
        if not self.spatial:
            return self.mu + time[0].sum(-1)
        if s_grid is None:
            raise ValueError("s_grid is required for spatio-temporal models")
        ds = s_grid[None, :, None, :] - s[:, None, :, :]  # [B, P, N, d]
        near = mask[:, None, :] & (torch.linalg.norm(ds, dim=-1) <= self.kernel.s_max)
        b, p, n = near.nonzero(as_tuple=True)
        space = torch.zeros(self.kernel.n_basis_space, *near.shape, dtype=X.dtype, device=X.device)
        space[:, b, p, n] = self.kernel.u(s)[:, b, n] * self.kernel.v(ds[b, p, n])  # [R, B, P, N]
        return self.mu + torch.einsum("rbcn,rbpn->bcp", time, space)

    def barrier_intensity(self, X: Tensor, mask: Tensor) -> Tensor:
        """Intensity on the log-barrier grid ``U_bar,t x U_bar,s``: ``[B, |U_bar|]``."""
        lam = self.intensity(X, mask, self.barrier_t, self.barrier_s if self.spatial else None)
        return lam.reshape(lam.shape[0], -1)

    # -------------------------------------------------------- likelihood
    def _spatial_integral(self, s: Tensor) -> Tensor:
        """``V_r(s_i) = int_{B(0, s_max) cap (S - s_i)} v_r(ds) d(ds)``: ``[R, B, N]``."""
        n, d, smax = self.n_grid_space, self.kernel.d_space, self.kernel.s_max
        v = self.kernel.v(self.ds_grid) * self.ds_grid_in_ball * self.cell_vol  # [R, n^d]
        table = summed_area_table(v.reshape(-1, *([n] * d)))
        # Displacements ds with s_i + ds in S, i.e. ds in [S_low - s_i, S_high - s_i];
        # convert to (inclusive) ranges of grid cells whose centres fall inside.
        lo = torch.ceil((self.kernel.S_low - s + smax) / self.cell_space - 0.5).long().clamp(0, n)
        hi = torch.floor((self.kernel.S_high - s + smax) / self.cell_space - 0.5).long().clamp(-1, n - 1)
        return box_sums(table, lo, hi)

    def compensator(self, X: Tensor, mask: Tensor) -> Tensor:
        """``int_{T0}^{T1} int_S lambda(t, s) ds dt`` for each sequence: ``[B]`` (Eq. 8)."""
        t, s = self._split(X)
        phi = self._phi_table()
        remaining = (self.T1 - t).clamp(0.0, self.kernel.tau_max)
        F = integrate_interp_uniform(phi, cumtrapz_uniform(phi, self.h_time), self.h_time, remaining)
        time = torch.einsum("rl,lbn->rbn", self.kernel.alpha, self.kernel.psi(t) * F)  # [R, B, N]
        if self.spatial:
            per_event = (time * self.kernel.u(s) * self._spatial_integral(s)).sum(0)
        else:
            per_event = time[0]
        background = self.mu * self.area * (self.T1 - self.T0)
        return background + (per_event * mask).sum(-1)

    def log_likelihood(self, X: Tensor, mask: Tensor) -> Tensor:
        """Log-likelihood (Eq. 2) of each sequence: ``[B]``."""
        lam = self.event_intensity(X, mask)
        log_lam = torch.log(lam.clamp(min=self.log_eps)) * mask
        return log_lam.sum(-1) - self.compensator(X, mask)

    def forward(self, X: Tensor, mask: Tensor) -> Tensor:
        return self.log_likelihood(X, mask)

    # --------------------------------------------------------- persistence
    def save(self, path: str) -> None:
        """Save constructor arguments and weights to ``path``."""
        torch.save({"config": self.config, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, map_location: Any = "cpu") -> DNSK:
        """Load a model written by :meth:`save`."""
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model
