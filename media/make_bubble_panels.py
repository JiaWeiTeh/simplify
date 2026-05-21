"""Generate a two-panel (1x2) vector PDF of the bubble temperature and
density profiles, each simplified to exactly n = 40 points.

The 40 points are the most important ones under the same greedy
worst-error ranking that ``simplify_diagnostic`` uses, so a length-40
prefix is the principled "best 40 points" for a connect-the-dots
reconstruction.  Run from the repository root:

    python media/make_bubble_panels.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simplify import _auto_log_y, _importance_order, simplify_error  # noqa: E402

N_POINTS = 40

PANELS = [
    {
        "file": ROOT / "data" / "bubble_T.dat",
        "ylabel": r"$\log_{10}\,T\;\mathrm{[K]}$",
        "tag": "temperature",
    },
    {
        "file": ROOT / "data" / "bubble_n.dat",
        "ylabel": r"$\log_{10}\,n\;\mathrm{[cm^{-3}]}$",
        "tag": "density",
    },
]


def simplify_n(x, y, n):
    """Keep exactly the ``n`` most important points (sorted by x)."""
    use_log = _auto_log_y(y, "auto")
    y_work = np.log10(y) if use_log else y
    n_keep = max(2, min(n, x.size))
    idx = np.sort(_importance_order(x, y_work, n_keep))
    return x[idx], y[idx], simplify_error(x, y, x[idx], y[idx])


def main():
    plt.style.use(str(ROOT / "simplify.mplstyle"))
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), layout="constrained")

    for ax, panel in zip(axes, PANELS):
        data = np.loadtxt(panel["file"], comments="#")
        r, y = data[:, 0], data[:, 1]
        order = np.argsort(r, kind="stable")
        r, y = r[order], y[order]

        r_s, y_s, m = simplify_n(r, y, N_POINTS)

        ax.plot(r, y, "-", color="0.75", lw=0.9, zorder=1, label="original")
        ax.plot(r_s, y_s, "o-", color="tab:red", ms=3.0, lw=0.8, zorder=2,
                label=f"simplified ($n={m['n_simp']}$)")
        ax.set_xlabel(r"radius $r$ [pc]")
        ax.set_ylabel(panel["ylabel"])
        ax.set_title(
            rf"$n = {m['n_simp']}$,   $R^2 = {m['r_squared']:.5f}$",
            fontsize=10,
        )
        ax.legend(loc="best", fontsize=8)

    out = ROOT / "media" / "bubble_profiles_n40.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
