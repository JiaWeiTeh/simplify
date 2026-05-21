"""Generate a two-column vector PDF of the bubble temperature and density
profiles, each simplified to exactly n = 40 points.

Layout: 2 columns (temperature | density) x 2 rows.
  - top row    : the profile (full curve + the 40 simplified points)
  - bottom row : |y - y_interp|, the absolute residual, on a log y-axis

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
N_ORIG_LABEL = r"$n = 3\times10^4$"

# 1 parsec in cm; the density column is stored as log10(number density per
# pc^3), so subtracting 3*log10(pc/cm) converts it to log10(n / cm^-3).
PC_CM = 3.0856775814913673e18
PC3_TO_CM3_SHIFT = -3.0 * np.log10(PC_CM)

PANELS = [
    {
        "file": ROOT / "data" / "bubble_T.dat",
        "ylabel": r"$\log_{10}\,T\;\mathrm{[K]}$",
        "title": "Temperature profile",
        "log_shift": 0.0,
    },
    {
        "file": ROOT / "data" / "bubble_n.dat",
        "ylabel": r"$\log_{10}\,n\;\mathrm{[cm^{-3}]}$",
        "title": "Density profile",
        "log_shift": PC3_TO_CM3_SHIFT,
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
    fig, axes = plt.subplots(
        2, 2, figsize=(8.0, 5.4), sharex="col",
        gridspec_kw={"height_ratios": [3, 1]}, layout="constrained",
    )

    for col, panel in enumerate(PANELS):
        ax_top, ax_res = axes[0, col], axes[1, col]

        data = np.loadtxt(panel["file"], comments="#")
        r, y = data[:, 0], data[:, 1] + panel["log_shift"]
        order = np.argsort(r, kind="stable")
        r, y = r[order], y[order]

        r_s, y_s, m = simplify_n(r, y, N_POINTS)
        abs_res = np.abs(y - np.interp(r, r_s, y_s))

        # --- top: profile ---
        ax_top.plot(r, y, "-", color="0.75", lw=0.9, zorder=1,
                    label=rf"original ({N_ORIG_LABEL})")
        ax_top.plot(r_s, y_s, "o-", color="tab:red", ms=3.0, lw=0.8, zorder=2,
                    label=rf"simplified ($n={m['n_simp']}$)")
        ax_top.set_ylabel(panel["ylabel"])
        ax_top.set_title(
            rf"{panel['title']}   ($R^2 = {m['r_squared']:.5f}$)",
            fontsize=10,
        )
        ax_top.legend(loc="best", fontsize=8)

        # --- bottom: absolute residual on a log y-axis ---
        ax_res.plot(r, abs_res, "-", color="tab:blue", lw=0.8)
        ax_res.set_yscale("log")
        ax_res.set_xlabel(r"radius $r$ [pc]")
        ax_res.set_ylabel(r"$|y - y_{\mathrm{interp}}|$")
        positive = abs_res[abs_res > 0]
        if positive.size:
            ax_res.set_ylim(positive.min() * 0.5, abs_res.max() * 2.0)

    out = ROOT / "media" / "bubble_profiles_n40.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
