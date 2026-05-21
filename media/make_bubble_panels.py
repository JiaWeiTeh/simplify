"""Generate a single-column vector PDF of the bubble temperature and
density profiles, each simplified to exactly n = 40 points.

Layout: one column, two profile groups stacked vertically (temperature
on top, density below).  Each group is a tall profile panel over a short
residual subrow.  All four panels share the radius x-axis.

Styled to match the project's paper figures: narrow A&A column width,
log-log axes, no grid, a colourblind-friendly (Wong) palette, a dashed
reference line at the bubble edge, compact lower-corner legends, and
physical-unit axis labels on log axes.

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
_LEGEND_FONTSIZE = 9

# Colourblind-friendly (Wong) palette, matching the reference figures.
C_ORIG = "0.6"        # full-resolution curve (faint background)
C_SIMP = "#d62728"    # the n=40 simplification (the hero)
C_RESID = "#0072B2"   # residual subrow
C_VLINE = "0.25"      # bubble-edge reference line

# 1 parsec in cm; the density column is stored as log10(number density per
# pc^3), so subtracting 3*log10(pc/cm) converts it to log10(n / cm^-3).
PC_CM = 3.0856775814913673e18
PC3_TO_CM3_SHIFT = -3.0 * np.log10(PC_CM)

PANELS = [
    {
        "file": ROOT / "data" / "bubble_T.dat",
        "ylabel": r"$T(r)$ [K]",
        "res_ylabel": r"$|\Delta \log T|$",
        "legend_loc": "lower left",
        "r2_xy": (0.97, 0.95),
        "r2_align": ("right", "top"),
        "log_shift": 0.0,
    },
    {
        "file": ROOT / "data" / "bubble_n.dat",
        "ylabel": r"$n(r)$ [cm$^{-3}$]",
        "res_ylabel": r"$|\Delta \log n|$",
        "legend_loc": "lower right",
        "r2_xy": (0.03, 0.95),
        "r2_align": ("left", "top"),
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
    fig = plt.figure(figsize=(4.2, 6.8))

    # Two profile groups, each a tall profile panel over a short residual
    # subrow; a gap between the gridspecs separates the two quantities.
    gs_top = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05,
                              top=0.985, bottom=0.55)
    gs_bot = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05,
                              top=0.47, bottom=0.075)

    ax_T = fig.add_subplot(gs_top[0])
    pairs = [(ax_T, fig.add_subplot(gs_top[1], sharex=ax_T))]
    pairs.append((fig.add_subplot(gs_bot[0], sharex=ax_T),
                  fig.add_subplot(gs_bot[1], sharex=ax_T)))

    # Bubble-edge radius: steepest gradient of log T (the contact
    # discontinuity); shared by both quantities, drawn on every panel.
    t_data = np.loadtxt(PANELS[0]["file"], comments="#")
    t_order = np.argsort(t_data[:, 0], kind="stable")
    r_t, logT = t_data[t_order, 0], t_data[t_order, 1]
    r_edge = float(r_t[np.argmax(np.abs(np.gradient(logT, np.log(r_t))))])

    for (ax_top, ax_res), panel in zip(pairs, PANELS):
        data = np.loadtxt(panel["file"], comments="#")
        r, y = data[:, 0], data[:, 1] + panel["log_shift"]
        order = np.argsort(r, kind="stable")
        r, y = r[order], y[order]

        r_s, y_s, m = simplify_n(r, y, N_POINTS)
        abs_res = np.abs(y - np.interp(r, r_s, y_s))

        # --- top: profile (physical units on a log axis) ---
        ax_top.plot(r, 10.0 ** y, "-", color=C_ORIG, lw=1.0, zorder=1,
                    label=rf"original ({N_ORIG_LABEL})")
        ax_top.plot(r_s, 10.0 ** y_s, "-", color="k", lw=1.2, zorder=3,
                    marker="o", ms=3.5, markerfacecolor=C_SIMP,
                    markeredgecolor="black", mew=0.6,
                    label=rf"simplified ($n={m['n_simp']}$)")
        ax_top.axvline(r_edge, color=C_VLINE, lw=1.2, ls="--", alpha=0.7,
                       zorder=2)
        ax_top.set_xscale("log")
        ax_top.set_yscale("log")
        ax_top.grid(False)
        ax_top.tick_params(labelbottom=False)
        ax_top.set_ylabel(panel["ylabel"])
        ax_top.text(*panel["r2_xy"], rf"$R^2 = {m['r_squared']:.5f}$",
                    transform=ax_top.transAxes,
                    ha=panel["r2_align"][0], va=panel["r2_align"][1],
                    fontsize=_LEGEND_FONTSIZE, color="0.3")
        ax_top.legend(loc=panel["legend_loc"], handlelength=1.6,
                      labelspacing=0.3, fontsize=_LEGEND_FONTSIZE,
                      framealpha=0.9)

        # --- bottom: absolute residual (dex) on a log y-axis ---
        ax_res.plot(r, abs_res, "-", color=C_RESID, lw=1.0)
        ax_res.axvline(r_edge, color=C_VLINE, lw=1.2, ls="--", alpha=0.7,
                       zorder=2)
        ax_res.set_xscale("log")
        ax_res.set_yscale("log")
        ax_res.grid(False)
        ax_res.set_ylabel(panel["res_ylabel"])
        positive = abs_res[abs_res > 0]
        if positive.size:
            ax_res.set_ylim(positive.min() * 0.5, abs_res.max() * 2.0)

    pairs[-1][1].set_xlabel(r"$r$ [pc]")
    ax_T.set_xlim(r.min() * 0.95, r.max() * 1.05)

    out = ROOT / "media" / "bubble_profiles_n40.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
