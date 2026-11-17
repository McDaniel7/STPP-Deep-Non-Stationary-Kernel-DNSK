import math

import numpy as np
import torch
from conftest import zero_kernel_

from dnsk import (
    DNSK,
    TrainConfig,
    evaluate,
    fit,
    fit_restarts,
    pad_sequences,
    predict_next_event,
    simulate,
    true_intensity,
)


def exponential_kernel(alpha, beta):
    return lambda t_hist, s_hist, t, s: alpha * np.exp(-beta * (t - t_hist))


def test_poisson_counts():
    rng = np.random.default_rng(0)
    seqs = simulate(lambda *a: 0.0 * a[0], mu=0.5, T=(0, 10), S=[(0, 2), (0, 1)], upper_bound=1.0,
                    n_sequences=400, min_events=0, rng=rng)
    counts = np.array([len(s) for s in seqs])
    assert abs(counts.mean() - 10.0) < 0.5  # E[N] = mu |S| T
    assert all(s.shape[1] == 3 for s in seqs if len(s))
    assert all(np.all((s[:, 1] >= 0) & (s[:, 1] <= 2) & (s[:, 2] >= 0) & (s[:, 2] <= 1)) for s in seqs)


def test_hawkes_counts_and_true_intensity():
    rng = np.random.default_rng(1)
    kernel = exponential_kernel(0.5, 1.0)
    seqs = simulate(kernel, mu=0.5, T=(0, 200), upper_bound=6.0, n_sequences=200, rng=rng)
    counts = np.array([len(s) for s in seqs])
    expected = 0.5 * 200 / (1 - 0.5) - 1.0  # stationary mean minus the start-up transient
    assert abs(counts.mean() - expected) / expected < 0.05
    seq = seqs[0][:, 0]
    grid = np.array([seq[3] + 1e-3, seq[3] + 0.5])
    manual = 0.5 + 0.5 * np.exp(-(grid[:, None] - seq[None, :4])).sum(-1)
    np.testing.assert_allclose(true_intensity(kernel, 0.5, seqs[0], grid), manual)


def test_prediction_matches_poisson_closed_form():
    torch.manual_seed(0)
    model = zero_kernel_(DNSK(T=(0.0, 10.0), S=[(-1.0, 1.0), (-1.0, 1.0)], mu=0.1, tau_max=1.0, s_max=0.5))
    X, mask = pad_sequences([np.array([[2.0, 0.3, -0.2]]), np.array([[1.0, 0.0, 0.0], [6.0, 0.5, 0.5]])])
    t_pred, s_pred = predict_next_event(model, X, mask, n_time=4000, n_space=10)
    rate = 0.1 * 4.0
    for t_last, pred in zip([2.0, 6.0], t_pred):
        H = 10.0 - t_last  # E[t | t <= t_end] for an exponential waiting time
        expected = t_last + 1 / rate - H * math.exp(-rate * H) / (1 - math.exp(-rate * H))
        assert abs(float(pred) - expected) < 1e-2
    torch.testing.assert_close(s_pred, torch.zeros(2, 2), atol=1e-5, rtol=0)


def test_temporal_prediction_matches_poisson_closed_form():
    torch.manual_seed(0)
    model = zero_kernel_(DNSK(T=(0.0, 5.0), mu=0.7, tau_max=1.0))
    X, mask = pad_sequences([np.array([1.0, 3.0])])
    t_pred, s_pred = predict_next_event(model, X, mask, n_time=4000)
    H = 2.0
    expected = 3.0 + 1 / 0.7 - H * math.exp(-0.7 * H) / (1 - math.exp(-0.7 * H))
    assert s_pred is None and abs(float(t_pred[0]) - expected) < 1e-3


def test_fit_improves_likelihood_and_respects_barrier():
    rng = np.random.default_rng(2)
    seqs = simulate(exponential_kernel(0.8, 2.0), mu=0.5, T=(0, 20), upper_bound=8.0, n_sequences=120, rng=rng)
    X, mask = pad_sequences(seqs)
    torch.manual_seed(0)
    model = DNSK(T=(0, 20), mu=0.5, tau_max=3.0, hidden_dim=16)
    before = evaluate(model, X[100:], mask[100:])["loglik_per_event"]
    cfg = TrainConfig(epochs=15, batch_size=20, lr=1e-2, verbose=False, seed=0)
    history = fit(model, X[:100], mask[:100], cfg, X[100:], mask[100:])
    after = history[-1]["val_loglik_per_event"]
    assert after > before
    assert all(math.isfinite(r["train_loglik_per_event"]) for r in history)
    assert all(r["b"] <= -0.5 / 2 for r in history)


def test_fit_restarts_returns_best_validation_model():
    rng = np.random.default_rng(3)
    seqs = simulate(exponential_kernel(0.8, 2.0), mu=0.5, T=(0, 10), upper_bound=8.0, n_sequences=60, rng=rng)
    X, mask = pad_sequences(seqs)
    calls = []
    best, histories, index = fit_restarts(
        lambda: DNSK(T=(0, 10), mu=0.5, tau_max=2.0, hidden_dim=8), X[:40], mask[:40], X[40:], mask[40:],
        TrainConfig(epochs=2, batch_size=20, verbose=False, seed=5), n_restarts=3,
        callback=lambda r, e, rec, m: calls.append((r, e)))
    assert len(histories) == 3 and len(calls) == 6
    scores = [h[-1]["val_loglik_per_event"] for h in histories]
    assert index == int(np.argmax(scores))
    assert abs(evaluate(best, X[40:], mask[40:])["loglik_per_event"] - max(scores)) < 1e-5
