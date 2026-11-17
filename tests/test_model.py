"""The efficient likelihood terms must agree with direct (brute-force) evaluation."""

import math

import torch
from conftest import random_sequences, zero_kernel_

from dnsk import DNSK

S2 = [(-1.0, 1.0), (0.0, 2.0)]


def brute_force_event_intensity(model, X, mask):
    t, s = X[..., 0], X[..., 1:]
    lam = torch.full(t.shape, float(model.mu))
    for b in range(X.shape[0]):
        for j in range(X.shape[1]):
            if not mask[b, j]:
                continue
            hist = mask[b] & (t[b] < t[b, j])
            if model.spatial:
                k = model.kernel(t[b, hist], t[b, j], s[b, hist], s[b, j])
            else:
                k = model.kernel(t[b, hist], t[b, j])
            lam[b, j] += k.sum()
    return lam


@torch.no_grad()
def test_event_intensity_temporal(temporal_model):
    X, mask = random_sequences(5, (0.0, 20.0), seed=1)
    fast = temporal_model.event_intensity(X, mask)
    slow = brute_force_event_intensity(temporal_model, X, mask)
    torch.testing.assert_close(fast[mask], slow[mask], rtol=1e-3, atol=1e-4)


@torch.no_grad()
def test_event_intensity_spatial(spatial_model):
    X, mask = random_sequences(4, (0.0, 10.0), S2, seed=2)
    fast = spatial_model.event_intensity(X, mask)
    slow = brute_force_event_intensity(spatial_model, X, mask)
    torch.testing.assert_close(fast[mask], slow[mask], rtol=1e-3, atol=1e-4)


@torch.no_grad()
def test_compensator_temporal(temporal_model):
    X, mask = random_sequences(4, (0.0, 20.0), seed=3)
    n = 50000
    t_grid = (torch.arange(n) + 0.5) * (20.0 / n)
    lam = temporal_model.intensity(X, mask, t_grid)
    brute = lam.sum(-1) * (20.0 / n)
    torch.testing.assert_close(temporal_model.compensator(X, mask), brute, rtol=2e-3, atol=1e-3)


@torch.no_grad()
def test_spatial_integral_matches_direct_sum(spatial_model):
    m = spatial_model
    X, mask = random_sequences(3, (0.0, 10.0), S2, seed=4)
    s = X[..., 1:]
    fast = m._spatial_integral(s)  # [R, B, N]
    v = m.kernel.v(m.ds_grid) * m.ds_grid_in_ball * m.cell_vol  # [R, G]
    target = s[:, :, None, :] + m.ds_grid  # [B, N, G, d]
    inside = ((target >= m.kernel.S_low) & (target <= m.kernel.S_high)).all(-1)
    slow = torch.einsum("rg,bng->rbn", v, inside.to(v.dtype))
    torch.testing.assert_close(fast[:, mask], slow[:, mask], rtol=1e-4, atol=1e-5)


@torch.no_grad()
def test_compensator_spatial(spatial_model):
    m = spatial_model
    X, mask = random_sequences(2, (0.0, 10.0), S2, max_events=15, seed=5)
    n_t, n_s = 1000, 150
    t_grid = (torch.arange(n_t) + 0.5) * (10.0 / n_t)
    xs = -1.0 + (torch.arange(n_s) + 0.5) * (2.0 / n_s)
    ys = 0.0 + (torch.arange(n_s) + 0.5) * (2.0 / n_s)
    s_grid = torch.stack(torch.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)
    brute = torch.zeros(X.shape[0])
    for start in range(0, n_t, 100):
        lam = m.intensity(X, mask, t_grid[start:start + 100], s_grid)
        brute += lam.sum((1, 2)) * (10.0 / n_t) * (4.0 / n_s**2)
    torch.testing.assert_close(m.compensator(X, mask), brute, rtol=2e-2, atol=1e-2)


@torch.no_grad()
def test_barrier_grid_intensity(spatial_model):
    m = spatial_model
    X, mask = random_sequences(2, (0.0, 10.0), S2, seed=6)
    fast = m.barrier_intensity(X, mask).reshape(2, len(m.barrier_t), len(m.barrier_s))
    t, s = X[..., 0], X[..., 1:]
    for b in range(2):
        for c, tc in enumerate(m.barrier_t):
            hist = mask[b] & (t[b] < tc)
            k = m.kernel(t[b, hist][:, None], tc, s[b, hist][:, None, :], m.barrier_s[None])  # [n, P]
            torch.testing.assert_close(fast[b, c], m.mu + k.sum(0), rtol=1e-3, atol=1e-4)


@torch.no_grad()
def test_poisson_limit_is_exact(spatial_model):
    m = zero_kernel_(spatial_model)
    X, mask = random_sequences(3, (0.0, 10.0), S2, seed=7)
    n = mask.sum(-1).double()
    expected = n * math.log(0.3) - 0.3 * 4.0 * 10.0
    torch.testing.assert_close(m.log_likelihood(X, mask).double(), expected, rtol=1e-5, atol=1e-4)


@torch.no_grad()
def test_padding_does_not_change_likelihood(spatial_model):
    X, mask = random_sequences(3, (0.0, 10.0), S2, seed=8)
    ll = spatial_model.log_likelihood(X, mask)
    X_pad = torch.cat([X, torch.zeros(3, 4, 3)], dim=1)
    mask_pad = torch.cat([mask, torch.zeros(3, 4, dtype=torch.bool)], dim=1)
    torch.testing.assert_close(spatial_model.log_likelihood(X_pad, mask_pad), ll)
    torch.testing.assert_close(spatial_model.log_likelihood(X[1:2], mask[1:2]), ll[1:2])


def test_gradients_and_learnable_mu():
    torch.manual_seed(0)
    m = DNSK(T=(0.0, 5.0), mu=0.4, tau_max=1.0, learn_mu=True, hidden_dim=8)
    X, mask = random_sequences(3, (0.0, 5.0), seed=9)
    assert abs(float(m.mu.detach()) - 0.4) < 1e-6
    m.log_likelihood(X, mask).sum().backward()
    assert m.mu_raw.grad is not None
    assert all(p.grad is not None for p in m.parameters())


def test_save_and_load(tmp_path, spatial_model):
    X, mask = random_sequences(2, (0.0, 10.0), S2, seed=10)
    path = tmp_path / "model.pt"
    spatial_model.save(str(path))
    loaded = DNSK.load(str(path))
    with torch.no_grad():
        torch.testing.assert_close(loaded.log_likelihood(X, mask), spatial_model.log_likelihood(X, mask))
