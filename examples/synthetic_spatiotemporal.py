"""End-to-end example: simulate a non-stationary spatio-temporal Hawkes process,
fit DNSK with the log-barrier, and compare the learned kernel and intensity
with the ground truth.

The ground truth is the rank-(2 x 2) non-stationary mixture kernel of the
paper's synthetic "3D Data set 2" (Appendix C)::

    k(t', t - t', s', s - s') = sum_{r, l} alpha_rl u_r(s') v_r(s - s') psi_l(t') phi_l(t - t')

    psi_l(t') = 1 - 0.02 t'                       (influence fades over time)
    u_r(s')   = 1 - 0.3 (s'_2 + 1)                (weaker influence in the north)
    v_1       = N(s - s'; 0, 0.2^2 I)             (local excitation)
    v_2       = N(s - s' - 0.8; 0, 0.2^2 I)       (excitation shifted to the north-east)
    phi_1     = exp(-2 tau)                       (immediate, decaying)
    phi_2     = clip(1.5 tau - 1, 0, 1) 1{tau<=3} (delayed)
    alpha     = [[0.6, 0.15], [0.15, 0.6]]        (local: mostly immediate; north-east: mostly delayed)

The model is trained from several random initialisations and the run with the
best validation log-likelihood is kept (see ``dnsk.fit_restarts``).

Usage::

    python examples/synthetic_spatiotemporal.py --out results/example
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import dnsk

T = (0.0, 20.0)
S = [(-1.0, 1.0), (-1.0, 1.0)]
MU = 0.1
ALPHA = np.array([[0.6, 0.15], [0.15, 0.6]])  # alpha[r, l]
TAU_MAX, S_MAX = 3.0, 2.0
CENTRES = {"local": np.array([0.0, 0.0]), "north-east": np.array([0.8, 0.8])}  # blob centres of k(., ds)
T_PRIME, S_PRIME = 5.0, np.array([0.0, 0.0])  # past event used for kernel visualisation
TAU_PROFILE = np.linspace(0.01, TAU_MAX, 60)


def gaussian(d, sigma):
    return np.exp(-np.sum(d**2, axis=-1) / (2 * sigma**2)) / (2 * np.pi * sigma**2)


def true_kernel(t_hist, s_hist, t, s):
    """Ground-truth kernel; broadcasts over leading dimensions (see dnsk.simulation)."""
    tau, ds = t - t_hist, s - s_hist
    psi = 1 - 0.02 * t_hist
    phi = [np.exp(-2 * tau), np.clip(1.5 * tau - 1, 0, 1) * (tau <= 3)]
    u = 1 - 0.3 * (s_hist[..., 1] + 1)
    v = [gaussian(ds, 0.2), gaussian(ds - 0.8, 0.2)]
    return sum(ALPHA[r, l] * u * v[r] * psi * phi[l] for r in range(2) for l in range(2))


def oracle_loglik_per_event(sequences, n_time=200, n_space=40):
    """Log-likelihood per event of the true model (numerical integral on a midpoint grid)."""
    t_grid = T[0] + (np.arange(n_time) + 0.5) * (T[1] - T[0]) / n_time
    axis = -1 + (np.arange(n_space) + 0.5) * 2 / n_space
    s_grid = np.stack(np.meshgrid(axis, axis, indexing="ij"), -1).reshape(-1, 2)
    cell = (T[1] - T[0]) / n_time * 4.0 / n_space**2
    total, n_events = 0.0, 0
    for seq in sequences:
        t, s = seq[:, 0], seq[:, 1:]
        k = true_kernel(t[None, :], s[None, :, :], t[:, None], s[:, None, :])  # [j, i]
        lam = MU + np.where(t[None, :] < t[:, None], k, 0.0).sum(1)
        total += np.log(lam).sum() - dnsk.true_intensity(true_kernel, MU, seq, t_grid, s_grid).sum() * cell
        n_events += len(seq)
    return total / n_events


def learned_profiles(model):
    """k(t'=5, s'=0 -> t'+tau, s'+c) as a function of tau at both blob centres."""
    taus = torch.tensor(TAU_PROFILE, dtype=torch.float32)
    with torch.no_grad():
        return {name: model.kernel(torch.tensor(T_PRIME), T_PRIME + taus, torch.zeros(2),
                                   torch.tensor(c, dtype=torch.float32).expand(len(taus), 2)).cpu().numpy()
                for name, c in CENTRES.items()}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-sequences", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=Path("results/synthetic_spatiotemporal"))
    p.add_argument("--plot-only", action="store_true", help="redraw the figures of a finished run in --out")
    p.add_argument("--style", choices=["nature", "iclr"], default="nature",
                   help="heatmap colours: 'nature' (default) or 'iclr' (as in the paper; files get an _iclr suffix)")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # 1. Simulate data by thinning and split into train / validation / test sequences.
    sequences = dnsk.simulate(true_kernel, MU, T, S, upper_bound=5.0, n_sequences=args.n_sequences,
                              min_events=5, rng=rng)
    X, mask = dnsk.pad_sequences(sequences, T=T)
    train_idx, test_idx = dnsk.train_test_split(len(sequences), test_fraction=0.1, seed=args.seed)
    train_idx, val_idx = train_idx[: -len(test_idx)], train_idx[-len(test_idx):]
    print(f"{len(sequences)} sequences, {mask.sum(-1).float().mean():.1f} events on average")

    if args.plot_only:
        model = dnsk.DNSK.load(str(args.out / "model.pt"))
        metrics = json.loads((args.out / "metrics.json").read_text())
        saved = json.loads((args.out / "training_profiles.json").read_text())
        profiles = {int(e): {k: np.array(v) for k, v in p.items()} for e, p in saved.items()}
        make_figures(model, sequences[test_idx[0]], profiles, metrics["histories"], metrics["selected_restart"],
                     args.out, args.style)
        return

    # 2. DNSK with temporal rank L = 2 and spatial rank R = 2, trained from several initialisations.
    def build_model():
        return dnsk.DNSK(T, S, mu=MU, tau_max=TAU_MAX, s_max=S_MAX, n_basis_time=2, n_basis_space=2)

    profiles = {}  # restart -> epoch -> learned k(tau) profiles

    def record(restart, epoch, rec, model):
        profiles.setdefault(restart, {})[epoch + 1] = learned_profiles(model)

    config = dnsk.TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                              seed=args.seed, device=args.device)
    model, histories, best = dnsk.fit_restarts(build_model, X[train_idx], mask[train_idx], X[val_idx],
                                               mask[val_idx], config, n_restarts=args.restarts, callback=record)
    model.cpu()
    model.save(str(args.out / "model.pt"))

    # 3. Evaluation: test log-likelihood per event (vs. the true model) and intensity MRE.
    test = dnsk.evaluate(model, X[test_idx], mask[test_idx])
    oracle = oracle_loglik_per_event([sequences[i] for i in test_idx])
    t_grid = np.linspace(*T, 51)[1:]
    s_axis = np.linspace(-1, 1, 21)[:-1] + 0.05
    s_grid = np.stack(np.meshgrid(s_axis, s_axis, indexing="ij"), -1).reshape(-1, 2)
    with torch.no_grad():
        lam_hat = model.intensity(X[test_idx], mask[test_idx], torch.tensor(t_grid, dtype=torch.float32),
                                  torch.tensor(s_grid, dtype=torch.float32)).numpy()
    lam_true = np.stack([dnsk.true_intensity(true_kernel, MU, sequences[i], t_grid, s_grid) for i in test_idx])
    mre = float(np.mean(np.abs(lam_hat - lam_true) / lam_true))
    metrics = {"test_loglik_per_event": test["loglik_per_event"], "true_model_test_loglik_per_event": oracle,
               "test_intensity_mre": mre, "selected_restart": best,
               "val_loglik_per_event": [h[-1]["val_loglik_per_event"] for h in histories],
               "histories": histories}
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.out / "training_profiles.json").write_text(json.dumps(
        {e: {k: v.tolist() for k, v in p.items()} for e, p in profiles[best].items()}))
    print(f"test log-likelihood per event: DNSK {test['loglik_per_event']:.4f} | true model {oracle:.4f}")
    print(f"test intensity MRE: {mre:.4f} (selected restart {best})")

    make_figures(model, sequences[test_idx[0]], profiles[best], histories, best, args.out, args.style)


def make_figures(model, test_sequence, profiles, histories, best, out, style="nature"):
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("matplotlib not installed; skipping figures")
        return
    suffix = "" if style == "nature" else f"_{style}"
    plot_comparison(model, test_sequence, out / f"comparison{suffix}.png", style)
    plot_kernel(model, out / f"kernel_recovery{suffix}.png", style)
    plot_intensity(model, test_sequence, out / f"intensity_recovery{suffix}.png", style)
    plot_training(profiles, histories, best, out / "training_progress.png")


# ----------------------------------------------------------------------------- figures
KERNEL_EXTENT, KERNEL_TICKS = (-1.5, 1.5, -1.5, 1.5), [-1, 0, 1]
SPACE_EXTENT, SPACE_TICKS = (-1, 1, -1, 1), [-0.5, 0, 0.5]


def kernel_fields(model, taus, n=81):
    """True and learned k(t'=5, s'=0 -> t'+tau, s'+ds) on an n x n grid of ds, for each tau."""
    axis = np.linspace(-1.5, 1.5, n)
    ds = np.stack(np.meshgrid(axis, axis, indexing="ij"), -1)
    inside = np.linalg.norm(ds, axis=-1) <= S_MAX
    true = [true_kernel(T_PRIME, S_PRIME, T_PRIME + tau, S_PRIME + ds) * inside for tau in taus]
    with torch.no_grad():
        learned = [model.kernel(torch.tensor(T_PRIME), torch.tensor(T_PRIME + tau),
                                torch.zeros(2), torch.tensor(ds, dtype=torch.float32)).numpy() for tau in taus]
    return true, learned


def intensity_fields(model, sequence, times, n=81):
    """True and learned lambda(t, s) of one sequence on an n x n grid over S, plus recent events."""
    axis = np.linspace(-1, 1, n)
    s_grid = np.stack(np.meshgrid(axis, axis, indexing="ij"), -1).reshape(-1, 2)
    X, mask = dnsk.pad_sequences([sequence])
    with torch.no_grad():
        learned = model.intensity(X, mask, torch.tensor(times, dtype=torch.float32),
                                  torch.tensor(s_grid, dtype=torch.float32))[0].numpy().reshape(len(times), n, n)
    true = dnsk.true_intensity(true_kernel, MU, sequence, times, s_grid).reshape(len(times), n, n)
    recent = [sequence[(sequence[:, 0] < t) & (sequence[:, 0] > t - TAU_MAX), 1:] for t in times]
    return list(true), list(learned), recent


def _panel(ax, img, extent, ticks, cmap, norm, points=None, point_color="#eb6834"):
    import _style  # examples/_style.py

    im = ax.imshow(img.T, origin="lower", extent=extent, cmap=cmap, norm=norm)
    if points is not None and len(points):
        ax.scatter(points[:, 0], points[:, 1], s=24, c=point_color, edgecolors=_style.SURFACE,
                   linewidths=1.2, zorder=3)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set(visible=True, color="#e4e3df", linewidth=0.8)
    return im


def _heatmap_grid(fig, rows, row_labels, col_titles, extent, ticks, kind, style, points=None, xlabel="", ylabel=""):
    """Rows of heatmaps sharing one colour scale and one colour bar."""
    import _style  # examples/_style.py

    cmap, norm = _style.colormap_for([img for row in rows for img in row], kind, style)
    axes = fig.subplots(len(rows), len(col_titles), squeeze=False)
    for r, (row, label) in enumerate(zip(rows, row_labels)):
        for c, img in enumerate(row):
            ax = axes[r, c]
            im = _panel(ax, img, extent, ticks, cmap, norm, None if points is None else points[c],
                        _style.event_color(style))
            if r == 0:
                ax.set_title(col_titles[c])
            if c == 0:
                ax.set_ylabel(f"{label}\n{ylabel}", color=_style.TEXT, fontsize=12)
            else:
                ax.set_yticklabels([])
            if r == len(rows) - 1:
                ax.set_xlabel(xlabel)
            else:
                ax.set_xticklabels([])
    fig.colorbar(im, ax=axes, shrink=0.85, pad=0.03)
    return axes


def _save(fig, path):
    import matplotlib.pyplot as plt

    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def plot_comparison(model, sequence, path, style="nature", taus=(0.1, 2.0), quantiles=(0.2, 0.7)):
    """Two kernel snapshots and two intensity snapshots, true vs. DNSK, in one figure."""
    import _style  # examples/_style.py
    import matplotlib.pyplot as plt

    _style.apply()
    times = np.quantile(sequence[:, 0], quantiles) + 0.05
    k_true, k_learned = kernel_fields(model, taus)
    l_true, l_learned, recent = intensity_fields(model, sequence, times)
    fig = plt.figure(figsize=(14.5, 6.8), constrained_layout=True)
    left, right = fig.subfigures(1, 2, wspace=0.04)
    _heatmap_grid(left, [k_true, k_learned], ["True", "DNSK"], [f"τ = {tau:g}" for tau in taus],
                  KERNEL_EXTENT, KERNEL_TICKS, "kernel", style, xlabel="Δs₁", ylabel="Δs₂")
    left.suptitle("Influence kernel of an event at t' = 5, s' = 0", color=_style.TEXT, fontsize=13)
    _heatmap_grid(right, [l_true, l_learned], ["True", "DNSK"], [f"t = {t:.1f}" for t in times],
                  SPACE_EXTENT, SPACE_TICKS, "intensity", style, points=recent, xlabel="s₁", ylabel="s₂")
    right.suptitle("Conditional intensity of a test sequence", color=_style.TEXT, fontsize=13)
    _save(fig, path)


def plot_kernel(model, path, style="nature", taus=(0.1, 0.6, 1.2, 2.0)):
    import _style  # examples/_style.py
    import matplotlib.pyplot as plt

    _style.apply()
    true, learned = kernel_fields(model, taus)
    fig = plt.figure(figsize=(3.0 * len(taus) + 0.8, 6.6), constrained_layout=True)
    _heatmap_grid(fig, [true, learned], ["True", "DNSK"], [f"τ = {tau:g}" for tau in taus],
                  KERNEL_EXTENT, KERNEL_TICKS, "kernel", style, xlabel="Δs₁", ylabel="Δs₂")
    fig.suptitle("Influence of an event at t' = 5, s' = (0, 0) on the point (t' + τ, s' + Δs)",
                 color=_style.TEXT, fontsize=13)
    _save(fig, path)


def plot_intensity(model, sequence, path, style="nature", quantiles=(0.2, 0.45, 0.7, 0.95)):
    import _style  # examples/_style.py
    import matplotlib.pyplot as plt

    _style.apply()
    times = np.quantile(sequence[:, 0], quantiles) + 0.05
    true, learned, recent = intensity_fields(model, sequence, times)
    fig = plt.figure(figsize=(3.0 * len(times) + 0.8, 6.6), constrained_layout=True)
    _heatmap_grid(fig, [true, learned], ["True", "DNSK"], [f"t = {t:.1f}" for t in times],
                  SPACE_EXTENT, SPACE_TICKS, "intensity", style, points=recent, xlabel="s₁", ylabel="s₂")
    fig.suptitle(f"Conditional intensity λ(t, s) of a test sequence (dots: events in the last {TAU_MAX:g} "
                 "time units)", color=_style.TEXT, fontsize=13)
    _save(fig, path)


def plot_training(profiles, histories, best, path):
    import _style  # examples/_style.py
    import matplotlib.pyplot as plt

    _style.apply()
    epochs = sorted(profiles)
    shown = sorted({e for e in [1, 3, 5, 10, 20, epochs[-1]] if e in profiles})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    for ax, name in zip(axes[:2], CENTRES):
        truth = true_kernel(T_PRIME, S_PRIME, T_PRIME + TAU_PROFILE, S_PRIME + CENTRES[name])
        for k, e in enumerate(shown[:-1]):
            step = int(round(k * (len(_style.ORDINAL_BLUE) - 1) / max(1, len(shown) - 2)))
            ax.plot(TAU_PROFILE, profiles[e][name], color=_style.ORDINAL_BLUE[step], lw=1.5, label=f"epoch {e}")
        ax.plot(TAU_PROFILE, profiles[shown[-1]][name], color=_style.EVENT, lw=2.5, label=f"DNSK, epoch {shown[-1]}")
        ax.plot(TAU_PROFILE, truth, color=_style.TEXT, ls="--", label="true")
        ax.set(title=f"k(τ) at the {name} component", xlabel="time lag τ = t − t'")
        ax.grid(alpha=0.25)
    axes[0].legend(loc="upper right")
    ax = axes[2]
    others = iter([_style.CATEGORICAL[0], _style.CATEGORICAL[2]] * len(histories))
    for r, h in enumerate(histories):
        selected = r == best
        ax.plot([rec["epoch"] + 1 for rec in h], [rec["val_loglik_per_event"] for rec in h],
                color=_style.EVENT if selected else next(others), lw=2.5 if selected else 1.5,
                label=f"restart {r}" + (" (selected)" if selected else ""))
    ax.set(title="Validation log-likelihood per event", xlabel="epoch")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.suptitle("Learning the temporal profiles of the two spatial components", color=_style.TEXT, fontsize=13)
    _save(fig, path)


if __name__ == "__main__":
    main()
