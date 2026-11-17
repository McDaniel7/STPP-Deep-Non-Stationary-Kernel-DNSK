import itertools

import torch

from dnsk._grid import (
    box_sums,
    cumtrapz_uniform,
    integrate_interp_uniform,
    interp_uniform,
    midpoint_grid,
    summed_area_table,
)


def test_interp_is_exact_on_linear_functions():
    h = 0.1
    grid = torch.arange(11) * h
    values = torch.stack([2 * grid + 1, -grid])
    x = torch.rand(7, 3)
    out = interp_uniform(values, h, x)
    torch.testing.assert_close(out[0], 2 * x + 1)
    torch.testing.assert_close(out[1], -x)


def test_integral_of_interpolant_is_exact():
    h = 0.25
    values = torch.randn(3, 9)
    cumulative = cumtrapz_uniform(values, h)
    x = torch.rand(50) * 8 * h
    out = integrate_interp_uniform(values, cumulative, h, x)
    fine = torch.linspace(0, 8 * h, 200001)
    dense = interp_uniform(values, h, fine)
    for k in range(3):
        for q in range(0, 50, 7):
            sel = fine <= x[q]
            ref = torch.trapezoid(dense[k, sel], fine[sel])
            assert abs(out[k, q] - ref) < 1e-3


def test_box_sums_match_brute_force():
    torch.manual_seed(0)
    for d, n in [(1, 13), (2, 9), (3, 5)]:
        values = torch.randn(2, *([n] * d), dtype=torch.float64)
        table = summed_area_table(values)
        lo = torch.randint(0, n + 1, (20, d))
        hi = torch.randint(-1, n, (20, d))
        out = box_sums(table, lo, hi)
        for q in range(20):
            if (lo[q] > hi[q]).any():
                assert torch.all(out[:, q] == 0)
                continue
            region = values[(slice(None),) + tuple(slice(int(a), int(b) + 1) for a, b in zip(lo[q], hi[q]))]
            torch.testing.assert_close(out[:, q], region.reshape(2, -1).sum(-1))


def test_midpoint_grid():
    centres, vol = midpoint_grid(torch.tensor([-1.0, 0.0]), torch.tensor([1.0, 4.0]), 4)
    assert centres.shape == (16, 2)
    assert abs(vol - 0.5) < 1e-12
    expected = list(itertools.product([-0.75, -0.25, 0.25, 0.75], [0.5, 1.5, 2.5, 3.5]))
    torch.testing.assert_close(centres, torch.tensor(expected))
