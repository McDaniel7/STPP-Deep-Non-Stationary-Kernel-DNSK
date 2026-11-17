"""Converting event sequences to padded tensors and iterating over mini-batches."""

from __future__ import annotations

from typing import Iterator, Optional, Sequence

import numpy as np
import torch
from torch import Tensor


def pad_sequences(
    sequences: Sequence[np.ndarray],
    T: Optional[Sequence[float]] = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Stack variable-length event sequences into a zero-padded batch.

    Parameters
    ----------
    sequences:
        Each element is an array of shape ``[n_i, 1 + d_S]`` (time first, then
        location) or ``[n_i]`` for purely temporal data. Empty sequences are allowed.
    T:
        Optional window ``(T0, T1)``; if given, all event times must lie in it.
    dtype:
        Floating point type of the returned tensor.

    Returns
    -------
    X:
        ``[B, N_max, 1 + d_S]`` events, zero-padded.
    mask:
        ``[B, N_max]`` boolean, ``True`` for real events.
    """
    seqs = [np.asarray(seq, dtype=np.float64) for seq in sequences]
    seqs = [seq[:, None] if seq.ndim == 1 else seq for seq in seqs]
    if not seqs:
        raise ValueError("no sequences given")
    dims = {seq.shape[1] for seq in seqs if seq.size}
    if len(dims) > 1:
        raise ValueError(f"inconsistent event dimensions: {sorted(dims)}")
    dim = dims.pop() if dims else 1
    n_max = max(1, max(len(seq) for seq in seqs))
    X = np.zeros((len(seqs), n_max, dim))
    mask = np.zeros((len(seqs), n_max), dtype=bool)
    for k, seq in enumerate(seqs):
        if len(seq) == 0:
            continue
        if np.any(np.diff(seq[:, 0]) < 0):
            raise ValueError(f"sequence {k} is not sorted in time")
        if T is not None and (seq[0, 0] < T[0] or seq[-1, 0] > T[1]):
            raise ValueError(f"sequence {k} has events outside the window {tuple(T)}")
        X[k, : len(seq)] = seq
        mask[k, : len(seq)] = True
    return torch.as_tensor(X, dtype=dtype), torch.as_tensor(mask)


def iterate_batches(
    X: Tensor,
    mask: Tensor,
    batch_size: int,
    shuffle: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Iterator[tuple[Tensor, Tensor]]:
    """Yield ``(X_batch, mask_batch)`` with padding trimmed to the longest sequence in the batch."""
    order = torch.randperm(X.shape[0], generator=generator) if shuffle else torch.arange(X.shape[0])
    for start in range(0, X.shape[0], batch_size):
        idx = order[start:start + batch_size]
        xb, mb = X[idx], mask[idx]
        n = max(1, int(mb.sum(-1).max()))
        yield xb[:, :n], mb[:, :n]


def train_test_split(n: int, test_fraction: float = 0.1, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Random split of sequence indices ``0..n-1`` into train and test indices."""
    perm = np.random.default_rng(seed).permutation(n)
    n_test = max(1, int(round(test_fraction * n)))
    return np.sort(perm[n_test:]), np.sort(perm[:n_test])
