import numpy as np
import pytest
import torch

from dnsk import DNSK, pad_sequences


def random_sequences(n_seq, T, S=None, max_events=25, seed=0):
    rng = np.random.default_rng(seed)
    seqs = []
    for _ in range(n_seq):
        n = int(rng.integers(0, max_events))
        t = np.sort(rng.uniform(T[0], T[1], n))
        if S is None:
            seqs.append(t)
        else:
            s = np.stack([rng.uniform(lo, hi, n) for lo, hi in S], axis=-1)
            seqs.append(np.concatenate([t[:, None], s], axis=1))
    return pad_sequences(seqs)


def zero_kernel_(model):
    """Make phi_l identically zero, i.e. a homogeneous Poisson process with rate mu."""
    with torch.no_grad():
        model.kernel.phi_net.weights[-1].zero_()
        model.kernel.phi_net.biases[-1].zero_()
    return model


@pytest.fixture
def temporal_model():
    torch.manual_seed(0)
    return DNSK(T=(0.0, 20.0), mu=0.5, tau_max=4.0, n_basis_time=2, n_grid_time=2001)


@pytest.fixture
def spatial_model():
    torch.manual_seed(0)
    return DNSK(T=(0.0, 10.0), S=[(-1.0, 1.0), (0.0, 2.0)], mu=0.3, tau_max=3.0, s_max=0.7,
                n_basis_time=2, n_basis_space=2, n_grid_time=2001, n_grid_space=200,
                n_barrier_time=7, n_barrier_space=5, hidden_dim=16)
