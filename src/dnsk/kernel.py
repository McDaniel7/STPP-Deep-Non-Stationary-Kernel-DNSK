"""The deep non-stationary influence kernel (Sec. 3, Eq. 4).

.. math::

    k(t', t - t', s', s - s') = \\sum_{r=1}^{R} \\sum_{l=1}^{L}
        \\alpha_{lr}\\, \\psi_l(t')\\, \\phi_l(t - t')\\, u_r(s')\\, v_r(s - s')

* ``psi_l`` / ``u_r`` model the (non-stationary) influence of an event that
  happened at time ``t'`` / location ``s'``;
* ``phi_l`` / ``v_r`` model how that influence propagates over the temporal /
  spatial *displacement*. They are supported on the local domain
  ``[0, tau_max] x B(0, s_max)`` and are zero outside of it.
* ``alpha_{lr} = softplus(.) >= 0``. The basis networks have linear outputs, so
  the kernel itself may be negative (inhibition).

For temporal point processes (no space) the kernel reduces to
``k(t', t - t') = sum_l alpha_l psi_l(t') phi_l(t - t')``.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import Tensor, nn

from .basis import BasisBank


class DNSKKernel(nn.Module):
    """Low-rank deep non-stationary kernel with displacement parameterization.

    Parameters
    ----------
    T:
        Observation window ``(T0, T1)``.
    S:
        Spatial domain as a sequence of ``(low, high)`` pairs, one per spatial
        dimension, or ``None`` for purely temporal data.
    tau_max:
        Temporal range of influence; ``phi_l(tau) = 0`` for ``tau > tau_max``.
    s_max:
        Spatial range of influence; ``v_r(ds) = 0`` for ``||ds|| > s_max``.
        Required when ``S`` is given.
    n_basis_time, n_basis_space:
        Temporal rank ``L`` and spatial rank ``R``.
    hidden_dim, n_hidden_layers, softplus_beta, init_gain, init_bias, out_bias:
        Architecture and initialisation of every basis network
        (see :class:`dnsk.basis.BasisBank`).
    init_branching:
        Average initial value of ``int k`` over the kernel support assuming all
        basis functions equal 1, so that the model starts as a weakly
        self-exciting process. Each ``alpha_lr`` is additionally scaled by a
        random factor in ``U(0.1, 2)`` to break the symmetry between components.

    Notes
    -----
    Basis-network inputs are rescaled to O(1) ranges: ``t'`` to ``[0, 1]`` over
    the window, ``tau`` to ``[0, 1]`` over ``[0, tau_max]``, ``s'`` to
    ``[-1, 1]^d`` over ``S`` and ``s - s'`` to the unit ball over ``B(0, s_max)``.
    """

    def __init__(
        self,
        T: Sequence[float],
        S: Optional[Sequence[Sequence[float]]] = None,
        *,
        tau_max: float,
        s_max: Optional[float] = None,
        n_basis_time: int = 1,
        n_basis_space: int = 1,
        hidden_dim: int = 64,
        n_hidden_layers: int = 2,
        softplus_beta: float = 100.0,
        init_gain: float = 0.5,
        init_bias: float = 1e-2,
        out_bias: float = 1.0,
        init_branching: float = 0.1,
    ) -> None:
        super().__init__()
        self.T0, self.T1 = float(T[0]), float(T[1])
        if not self.T1 > self.T0:
            raise ValueError("T must satisfy T0 < T1")
        if tau_max <= 0:
            raise ValueError("tau_max must be positive")
        self.tau_max = float(tau_max)
        self.spatial = S is not None and len(S) > 0
        self.d_space = len(S) if self.spatial else 0
        self.n_basis_time = n_basis_time
        self.n_basis_space = n_basis_space if self.spatial else 1

        net = dict(hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers, softplus_beta=softplus_beta,
                   init_gain=init_gain, init_bias=init_bias, out_bias=out_bias)
        self.psi_net = BasisBank(n_basis_time, 1, **net)  # psi_l(t')
        self.phi_net = BasisBank(n_basis_time, 1, **net)  # phi_l(t - t')
        if self.spatial:
            if s_max is None or s_max <= 0:
                raise ValueError("a positive s_max is required for spatio-temporal kernels")
            self.s_max = float(s_max)
            S_t = torch.tensor([[float(lo), float(hi)] for lo, hi in S])
            if not bool((S_t[:, 1] > S_t[:, 0]).all()):
                raise ValueError("every spatial interval must satisfy low < high")
            self.register_buffer("S_low", S_t[:, 0].clone())
            self.register_buffer("S_high", S_t[:, 1].clone())
            self.u_net = BasisBank(n_basis_space, self.d_space, **net)  # u_r(s')
            self.v_net = BasisBank(n_basis_space, self.d_space, **net)  # v_r(s - s')
        else:
            self.s_max = None
        # Support measure tau_max * |B(0, s_max) cap S| (the ball volume is capped by |S|).
        support = self.tau_max
        if self.spatial:
            d = self.d_space
            ball = math.pi ** (d / 2) / math.gamma(d / 2 + 1) * self.s_max ** d
            support *= min(ball, float(torch.prod(self.S_high - self.S_low)))
        alpha0 = init_branching / (self.n_basis_space * n_basis_time * support)
        alpha = alpha0 * torch.empty(self.n_basis_space, n_basis_time).uniform_(0.1, 2.0)
        self.alpha_raw = nn.Parameter(torch.log(torch.expm1(alpha)))

    # ------------------------------------------------------------------ bases
    @property
    def alpha(self) -> Tensor:
        """Non-negative weights ``alpha`` of shape ``[R, L]``."""
        return nn.functional.softplus(self.alpha_raw)

    def psi(self, t: Tensor) -> Tensor:
        """``psi_l(t')`` for all ``l``: ``[L, *t.shape]``."""
        return self.psi_net(((t - self.T0) / (self.T1 - self.T0)).unsqueeze(-1))

    def phi(self, tau: Tensor) -> Tensor:
        """``phi_l(tau)`` for all ``l`` *without* the truncation mask: ``[L, *tau.shape]``."""
        return self.phi_net((tau / self.tau_max).unsqueeze(-1))

    def u(self, s: Tensor) -> Tensor:
        """``u_r(s')`` for all ``r``: ``[R, *s.shape[:-1]]``."""
        mid, half = (self.S_high + self.S_low) / 2, (self.S_high - self.S_low) / 2
        return self.u_net((s - mid) / half)

    def v(self, ds: Tensor) -> Tensor:
        """``v_r(s - s')`` for all ``r`` *without* the truncation mask: ``[R, *ds.shape[:-1]]``."""
        return self.v_net(ds / self.s_max)

    # ----------------------------------------------------------------- kernel
    def forward(
        self,
        t_hist: Tensor,
        t: Tensor,
        s_hist: Optional[Tensor] = None,
        s: Optional[Tensor] = None,
    ) -> Tensor:
        """Evaluate ``k`` exactly (no grid interpolation) at broadcastable inputs.

        Parameters
        ----------
        t_hist, t:
            Times of the past event ``t'`` and of the query ``t``.
        s_hist, s:
            Locations ``[..., d]`` of the past event and of the query (spatial kernels only).

        Returns
        -------
        Kernel values; zero where ``t - t'`` is outside ``(0, tau_max]`` or
        ``||s - s'|| > s_max``.
        """
        t_hist, t = torch.broadcast_tensors(t_hist, t)
        tau = t - t_hist
        time = self.psi(t_hist) * self.phi(tau)  # [L, ...]
        support = (tau > 0) & (tau <= self.tau_max)
        if not self.spatial:
            return (self.alpha[0][:, None] * time.reshape(self.n_basis_time, -1)).sum(0).reshape(tau.shape) * support
        if s_hist is None or s is None:
            raise ValueError("spatial kernels need s_hist and s")
        s_hist, s = torch.broadcast_tensors(s_hist, s)
        ds = s - s_hist
        space = self.u(s_hist) * self.v(ds)  # [R, ...]
        support = support & (torch.linalg.norm(ds, dim=-1) <= self.s_max)
        value = torch.einsum("rl,l...,r...->...", self.alpha, time, space)
        return value * support
