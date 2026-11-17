"""Neural-network basis functions.

Every basis function in the DNSK kernel (``psi_l``, ``phi_l``, ``u_r``, ``v_r``
in Eq. (4) of the paper) is an independent fully-connected network with
Softplus hidden activations and a *linear* output layer, so that the kernel can
take negative values (inhibiting influence).

:class:`BasisBank` stores ``K`` such networks with identical architecture as
stacked weight tensors and evaluates all of them in a single batched pass. This
is mathematically identical to ``K`` separate ``torch.nn.Sequential`` models
but avoids a Python loop over basis functions.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class BasisBank(nn.Module):
    """A bank of ``n_funcs`` independent MLPs ``R^in_dim -> R``.

    Parameters
    ----------
    n_funcs:
        Number of independent basis functions.
    in_dim:
        Input dimension (1 for temporal bases, ``d_S`` for spatial bases).
    hidden_dim:
        Width of each hidden layer.
    n_hidden_layers:
        Number of hidden layers.
    softplus_beta:
        ``beta`` of the Softplus hidden activation. Large values make the
        activation close to ReLU while remaining smooth.
    init_gain, init_bias:
        Xavier-uniform gain for weights and constant initial value of the hidden-layer biases.
    out_bias:
        Initial bias of the output layer. With the default 1.0 every basis
        function starts close to the constant 1 (plus a small random function),
        which keeps the gradients of the product ``psi * phi * u * v`` O(1) at
        initialisation.
    """

    def __init__(
        self,
        n_funcs: int,
        in_dim: int,
        hidden_dim: int = 64,
        n_hidden_layers: int = 2,
        softplus_beta: float = 100.0,
        init_gain: float = 0.5,
        init_bias: float = 1e-2,
        out_bias: float = 1.0,
    ) -> None:
        super().__init__()
        if n_funcs < 1 or in_dim < 1 or n_hidden_layers < 1:
            raise ValueError("n_funcs, in_dim and n_hidden_layers must be >= 1")
        self.n_funcs = n_funcs
        self.in_dim = in_dim
        dims = [in_dim] + [hidden_dim] * n_hidden_layers + [1]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for k, (fan_in, fan_out) in enumerate(zip(dims[:-1], dims[1:])):
            bound = init_gain * math.sqrt(6.0 / (fan_in + fan_out))
            w = torch.empty(n_funcs, fan_in, fan_out).uniform_(-bound, bound)
            bias = out_bias if k == len(dims) - 2 else init_bias
            b = torch.full((n_funcs, 1, fan_out), float(bias))
            self.weights.append(nn.Parameter(w))
            self.biases.append(nn.Parameter(b))
        self.activation = nn.Softplus(beta=softplus_beta)

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate all basis functions.

        Parameters
        ----------
        x:
            Inputs of shape ``[..., in_dim]``.

        Returns
        -------
        Tensor of shape ``[n_funcs, ...]``.
        """
        batch_shape = x.shape[:-1]
        h = x.reshape(1, -1, self.in_dim).expand(self.n_funcs, -1, -1)  # [K, M, in]
        n_layers = len(self.weights)
        for k, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = torch.baddbmm(b, h, w)  # [K, M, out]
            if k < n_layers - 1:
                h = self.activation(h)
        return h.reshape(self.n_funcs, *batch_shape)
