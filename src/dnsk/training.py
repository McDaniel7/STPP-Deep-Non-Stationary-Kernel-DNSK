"""Maximum-likelihood training with the log-barrier (Sec. 4.1, Algorithm 1).

The constrained problem ``min -l(theta) s.t. lambda >= 0`` is replaced by::

    L(theta) = -l(theta) / B  +  (1 / t) * p(theta, b),
    p(theta, b) = -mean_{(t_c, s_c) in U_bar} log(lambda(t_c, s_c) - b)

where ``U_bar`` is the barrier grid of the model and ``b`` is a lower bound of
``lambda`` on that grid. Following the reference implementation:

* ``b = min(lambda_min - margin, b_max)``, where ``lambda_min`` is the running
  minimum of the barrier-grid intensity within the current epoch and
  ``b_max < 0`` caps ``b`` (defaults: ``margin = mu / 4``, ``b_max = -mu / 2``).
  Within a batch the bound is additionally lowered to that batch's minimum, so
  the logarithm is always well defined.
* the barrier weight ``1 / t`` decays geometrically: ``t <- min(t * t_mul, t_max)``
  after every epoch (``w <- w * a`` in Algorithm 1).
* if the initial model has negative intensity on the barrier grid, a
  feasibility phase first minimises the barrier term alone.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Callable, Optional

import torch
from torch import Tensor

from .data import iterate_batches
from .model import DNSK


@dataclass
class TrainConfig:
    """Hyper-parameters of :func:`fit`."""

    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-2
    optimizer: str = "adam"  # "adam" or "adadelta"
    barrier: bool = True
    t_init: float = 1.0
    t_mul: float = 1.5
    t_max: float = 1e3
    margin: Optional[float] = None  # epsilon in Algorithm 1; default mu / 4
    b_max: Optional[float] = None  # default -mu / 2
    feasibility_epochs: int = 50
    seed: Optional[int] = None
    device: str = "cpu"
    verbose: bool = True


def _make_optimizer(name: str, params, lr: float) -> torch.optim.Optimizer:
    if name == "adadelta":
        return torch.optim.Adadelta(params, lr=lr)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    raise ValueError(f"unknown optimizer {name!r}")


@torch.no_grad()
def evaluate(model: DNSK, X: Tensor, mask: Tensor, batch_size: int = 64) -> dict:
    """Total log-likelihood, number of events and log-likelihood per event."""
    device = next(model.parameters()).device
    total = 0.0
    for xb, mb in iterate_batches(X, mask, batch_size):
        total += float(model.log_likelihood(xb.to(device), mb.to(device)).sum())
    n_events = int(mask.sum())
    return {"loglik": total, "n_events": n_events, "loglik_per_event": total / max(n_events, 1)}


@torch.no_grad()
def _barrier_minimum(model: DNSK, X: Tensor, mask: Tensor, batch_size: int, device: str) -> float:
    return min(float(model.barrier_intensity(xb.to(device), mb.to(device)).min())
               for xb, mb in iterate_batches(X, mask, batch_size))


def fit(
    model: DNSK,
    X: Tensor,
    mask: Tensor,
    config: Optional[TrainConfig] = None,
    X_val: Optional[Tensor] = None,
    mask_val: Optional[Tensor] = None,
    callback: Optional[Callable[[int, dict], None]] = None,
) -> list[dict]:
    """Fit a :class:`~dnsk.model.DNSK` model by penalised maximum likelihood.

    Parameters
    ----------
    model:
        Model to train (modified in place).
    X, mask:
        Training sequences (see :func:`dnsk.data.pad_sequences`).
    config:
        Training hyper-parameters.
    X_val, mask_val:
        Optional held-out sequences; their log-likelihood per event is logged each epoch.
    callback:
        Called as ``callback(epoch, record)`` after every epoch.

    Returns
    -------
    A list with one record (dict) per epoch.
    """
    cfg = config or TrainConfig()
    gen = None
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        gen = torch.Generator().manual_seed(cfg.seed)
    device = cfg.device
    model.to(device)
    mu = float(model.mu.detach())
    margin = mu / 4 if cfg.margin is None else cfg.margin
    b_max = -mu / 2 if cfg.b_max is None else cfg.b_max
    log = print if cfg.verbose else (lambda *a, **k: None)

    # ---- feasibility phase: make the intensity non-negative on the barrier grid.
    b = b_max
    if cfg.barrier:
        lam_min = _barrier_minimum(model, X, mask, cfg.batch_size, device)
        optimizer = _make_optimizer(cfg.optimizer, model.parameters(), cfg.lr)
        for epoch in range(cfg.feasibility_epochs):
            if lam_min >= 0:
                break
            b = min(lam_min - margin, 0.0)
            for xb, mb in iterate_batches(X, mask, cfg.batch_size, shuffle=True, generator=gen):
                optimizer.zero_grad()
                lam = model.barrier_intensity(xb.to(device), mb.to(device))
                (-torch.log(lam - min(b, float(lam.detach().min()) - margin)).mean()).backward()
                optimizer.step()
            lam_min = _barrier_minimum(model, X, mask, cfg.batch_size, device)
            log(f"[feasibility {epoch}] min lambda on barrier grid = {lam_min:.4e}")
        b = min(lam_min - margin, b_max)

    # ---- penalised maximum likelihood.
    optimizer = _make_optimizer(cfg.optimizer, model.parameters(), cfg.lr)
    t = cfg.t_init
    n_events = max(int(mask.sum()), 1)
    history = []
    for epoch in range(cfg.epochs):
        start = time.perf_counter()
        total_ll, total_barrier, n_batches = 0.0, 0.0, 0
        lam_min = math.inf
        model.train()
        for xb, mb in iterate_batches(X, mask, cfg.batch_size, shuffle=True, generator=gen):
            xb, mb = xb.to(device), mb.to(device)
            optimizer.zero_grad()
            ll = model.log_likelihood(xb, mb)
            loss = -ll.mean()
            if cfg.barrier:
                lam = model.barrier_intensity(xb, mb)
                lam_batch_min = float(lam.detach().min())
                b_batch = min(b, lam_batch_min - margin, b_max)
                barrier = -torch.log(lam - b_batch).mean()
                loss = loss + barrier / t
                total_barrier += float(barrier)
                lam_min = min(lam_min, lam_batch_min)
                b = min(lam_min - margin, b_max)
            loss.backward()
            optimizer.step()
            total_ll += float(ll.sum())
            n_batches += 1

        record = {"epoch": epoch, "train_loglik_per_event": total_ll / n_events,
                  "time_sec": time.perf_counter() - start}
        if cfg.barrier:
            record.update(barrier=total_barrier / n_batches, barrier_lambda_min=lam_min, b=b, t=t)
            t = min(t * cfg.t_mul, cfg.t_max)
        if X_val is not None:
            model.eval()
            record["val_loglik_per_event"] = evaluate(model, X_val, mask_val, cfg.batch_size)["loglik_per_event"]
        history.append(record)
        log(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()))
        if callback is not None:
            callback(epoch, record)
    model.eval()
    return history



def fit_restarts(
    build_model: Callable[[], DNSK],
    X: Tensor,
    mask: Tensor,
    X_val: Tensor,
    mask_val: Tensor,
    config: Optional[TrainConfig] = None,
    n_restarts: int = 3,
    callback: Optional[Callable[[int, int, dict, DNSK], None]] = None,
) -> tuple[DNSK, list[list[dict]], int]:
    """Train several randomly initialised models and keep the best on validation data.

    The likelihood has local optima. For example, the temporal profiles of two
    spatial components can be swapped or averaged. Such solutions have a
    noticeably lower held-out log-likelihood, so restarting from a few
    initialisations and selecting by validation likelihood is a cheap remedy.

    Parameters
    ----------
    build_model:
        Zero-argument function returning a freshly initialised model.
    X, mask, X_val, mask_val:
        Training and validation sequences.
    config:
        Training configuration; restart ``r`` uses seed ``config.seed + r`` (or ``r``)
        for both the initialisation and the batch order.
    n_restarts:
        Number of initialisations.
    callback:
        Called as ``callback(restart, epoch, record, model)`` after every epoch.

    Returns
    -------
    best_model, histories (one list of epoch records per restart), index of the best restart.
    """
    cfg = config or TrainConfig()
    base = 0 if cfg.seed is None else cfg.seed
    best_model, best_score, best_index, histories = None, -math.inf, -1, []
    for restart in range(n_restarts):
        seed = base + restart
        torch.manual_seed(seed)
        model = build_model()
        epoch_cb = None
        if callback is not None:
            def epoch_cb(epoch, record, restart=restart, model=model):
                callback(restart, epoch, record, model)
        histories.append(fit(model, X, mask, replace(cfg, seed=seed), X_val, mask_val, epoch_cb))
        score = evaluate(model, X_val, mask_val, cfg.batch_size)["loglik_per_event"]
        if cfg.verbose:
            print(f"[restart {restart}] validation log-likelihood per event = {score:.4f}")
        if score > best_score:
            best_model, best_score, best_index = model, score, restart
    return best_model, histories, best_index
