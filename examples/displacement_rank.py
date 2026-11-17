"""Why parameterize the kernel by displacement? (Figure 1 of the paper.)

The same non-stationary kernel is tabulated as a matrix in two ways:

* ``k(t', t)``      -- rows: history time t', columns: current time t
* ``k(t', t - t')`` -- rows: history time t', columns: displacement tau = t - t'

and the numerical rank of each matrix is computed. The displacement form has a
much lower rank, so far fewer basis functions are needed in the low-rank
decomposition ``k = sum_l alpha_l psi_l(t') phi_l(t - t')``.

Usage::

    python examples/displacement_rank.py [--out rank.png]
"""

from __future__ import annotations

import argparse

import numpy as np


def infinite_rank_kernel(t_prime, tau, n_terms=50):
    """Kernel of the paper's "1D Data set 3": infinitely many terms, strongly non-stationary in t'."""
    j = np.arange(1, n_terms + 1).reshape(-1, *([1] * np.ndim(tau)))
    terms = 2.0**-j * (0.3 + np.cos(2 + (t_prime / 5) ** 0.7 * 1.3 * (j + 1) * np.pi)) * np.exp(-8 * tau**2 * j**2 / 25)
    return 0.3 * terms.sum(0)


def numerical_rank(matrix, rtol=None):
    """Number of singular values above ``rtol * sigma_max``.

    The default ``rtol = max(matrix.shape) * eps`` is NumPy's ``matrix_rank`` tolerance for the matrix's dtype.
    """
    sv = np.linalg.svd(matrix, compute_uv=False)
    if rtol is None:
        rtol = max(matrix.shape) * np.finfo(matrix.dtype).eps
    return int((sv > rtol * sv[0]).sum()), sv, rtol


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=300, help="grid size")
    p.add_argument("--rtol", type=float, default=None,
                   help="singular values below rtol * max are dropped (default: NumPy's matrix_rank tolerance)")
    p.add_argument("--out", default=None, help="optional path of a figure")
    args = p.parse_args()

    T, tau_max = 20.0, 20.0  # same grid as Figure 1 of the paper
    # Uniform n x n grid with spacing T / n, tabulated in single precision as in the paper.
    t_prime = (np.arange(args.n) * T / args.n)[:, None]
    t = (np.arange(args.n) * T / args.n)[None, :]
    tau = (np.arange(args.n) * tau_max / args.n)[None, :]
    K_time = np.where(t >= t_prime, infinite_rank_kernel(t_prime, np.abs(t - t_prime)), 0.0).astype(np.float32)
    K_disp = infinite_rank_kernel(t_prime, tau).astype(np.float32)  # k(t', t - t')

    rank_time, sv_time, rtol = numerical_rank(K_time, args.rtol)
    rank_disp, sv_disp, _ = numerical_rank(K_disp, rtol)
    print(f"numerical rank (rtol={rtol:.2g}):  k(t', t): {rank_time}   k(t', t - t'): {rank_disp}")

    if args.out:
        import _style  # examples/_style.py
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        _style.apply()
        # Panels (a) and (b) reproduce Figure 1 of the paper: "Blues", one shared scale, no colour bar.
        norm = mpl.colors.Normalize(vmin=min(K_time.min(), K_disp.min()), vmax=max(K_time.max(), K_disp.max()))
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True,
                                 gridspec_kw={"width_ratios": [1, 1, 1.35]})
        fig.get_layout_engine().set(w_pad=0.2)
        panels = [(K_time, "t", f"(a) Kernel matrix of k(t', t), rank {rank_time}"),
                  (K_disp, "t − t'", f"(b) Kernel matrix of k(t', t − t'), rank {rank_disp}")]
        for ax, (K, xlabel, title) in zip(axes[:2], panels):
            ax.imshow(K, extent=(0, T, T, 0), cmap="Blues", norm=norm)
            ax.set_xticks([0, T])
            ax.set_yticks([0, T])
            ax.set_xlabel(xlabel, labelpad=-8)
            ax.set_ylabel("t'", labelpad=-10)
            ax.set_title(title, fontsize=11)
            ax.tick_params(length=2)
            for spine in ax.spines.values():
                spine.set(visible=True, color=_style.TEXT_SECONDARY, linewidth=0.8)
        axes[2].semilogy(sv_time / sv_time[0], color=_style.CATEGORICAL[0], label="k(t', t)")
        axes[2].semilogy(sv_disp / sv_disp[0], color=_style.CATEGORICAL[1], label="k(t', t − t')")
        axes[2].axhline(rtol, color=_style.TEXT_SECONDARY, ls="--", lw=1)
        axes[2].text(args.n * 0.5, rtol * 0.3, f"rank tolerance {rtol:.1e}", ha="center", va="top",
                     color=_style.TEXT_SECONDARY, fontsize=9)
        floor = 10 * np.finfo(K_time.dtype).eps ** 2  # below this, single-precision SVD values are rounding noise
        axes[2].set(title="Normalised singular values", xlabel="index i", ylabel="σᵢ / σ₁", ylim=(floor, 3))
        axes[2].grid(alpha=0.25)
        axes[2].legend(loc="center right")
        fig.suptitle("The same non-stationary kernel, tabulated by time vs. by displacement",
                     color=_style.TEXT, fontsize=13)
        fig.savefig(args.out, dpi=150)
        print(f"saved {args.out}")

if __name__ == "__main__":
    main()
