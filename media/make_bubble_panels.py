"""Generate a single-column vector PDF of the bubble temperature and
density profiles, each simplified to exactly n = 40 points.

The bundled profiles are 100-point files, pre-thinned from a denser
simulation grid, and the legend labels the reference curve with that
true point count.

Layout: one column, two profile groups stacked vertically (temperature
on top, density below).  Each group is a tall profile panel over a short
residual subrow.  All four panels share the radius x-axis.

Styled to match the project's paper figures: narrow A&A column width,
log-log axes, no grid, a colourblind-friendly (Wong) palette, a dashed
reference line at the bubble edge, compact lower-corner legends, and
physical-unit axis labels on log axes.

The 40 points come from :func:`astrosimplify.simplify_n`, the same
greedy worst-error ranking that ``simplify_diagnostic`` uses, so they
are the principled "best 40 points" for a connect-the-dots
reconstruction.  Run from the repository root:

    python media/make_bubble_panels.py
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from astrosimplify import (  # noqa: E402
    _log_ylim_from_positive, simplify_error, simplify_n,
)

STYLE_FILE = ROOT / "media" / "trinity.mplstyle"

N_POINTS = 40

# Colourblind-friendly (Wong) palette, matching the reference figures.
C_ORIG = "0.6"        # full-resolution curve (faint background)
C_SIMP = "#d62728"    # the n=40 simplification (the hero)
C_RESID = "#0072B2"   # residual subrow

PANELS = [
    {
        "file": ROOT / "data" / "bubble_T.dat",
        "ylabel": r"$T(r)$ [K]",
        "res_ylabel": r"$|\Delta \log T|$",
        "legend_loc": "lower left",
    },
    {
        "file": ROOT / "data" / "bubble_n.dat",
        "ylabel": r"$n(r)$ [cm$^{-3}$]",
        "res_ylabel": r"$|\Delta \log n|$",
        "legend_loc": "upper left",
    },
]


def main():
    plt.style.use(str(STYLE_FILE))
    if shutil.which("latex") is None:
        plt.rcParams["text.usetex"] = False   # graceful fallback

    fig = plt.figure(figsize=(4.6, 7.4))

    # Two profile groups, each a tall profile panel over a short residual
    # subrow; a gap between the gridspecs separates the two quantities.
    gs_top = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05,
                              top=0.985, bottom=0.535)
    gs_bot = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05,
                              top=0.495, bottom=0.075)

    ax_T = fig.add_subplot(gs_top[0])
    pairs = [(ax_T, fig.add_subplot(gs_top[1], sharex=ax_T))]
    pairs.append((fig.add_subplot(gs_bot[0], sharex=ax_T),
                  fig.add_subplot(gs_bot[1], sharex=ax_T)))

    for (ax_top, ax_res), panel in zip(pairs, PANELS):
        data = np.loadtxt(panel["file"], comments="#")
        r, y = data[:, 0], data[:, 1]          # y is log10 of the quantity
        order = np.argsort(r, kind="stable")
        r, y = r[order], y[order]

        # The bundled profile is already log10, so rank it linearly: the
        # same ranking simplify_n(r, 10**y) would use in log mode.
        r_s, y_s = simplify_n(r, y, N_POINTS, log_y=False)
        m = simplify_error(r, y, r_s, y_s)
        abs_res = np.abs(y - np.interp(r, r_s, y_s))

        # --- top: profile (physical units on a log axis) ---
        ax_top.plot(r, 10.0 ** y, "-", color=C_ORIG, lw=2.0, zorder=1,
                    label=rf"input ($n = {r.size}$)")
        ax_top.plot(r_s, 10.0 ** y_s, "-", color="k", lw=1.2, zorder=3,
                    marker="o", ms=3.5, markerfacecolor=C_SIMP,
                    markeredgecolor="black", mew=0.6,
                    label=rf"simplified ($n={m['n_simp']}$)")
        ax_top.set_yscale("log")
        ax_top.grid(False)
        ax_top.tick_params(which="both", labelbottom=False)
        ax_top.set_ylabel(panel["ylabel"])
        ax_top.legend(loc=panel["legend_loc"])

        # --- bottom: absolute residual (dex) on a log y-axis ---
        ax_res.plot(r, abs_res, "-", color=C_RESID, lw=1.0)
        ax_res.set_yscale("log")
        ax_res.grid(False)
        ax_res.tick_params(which="both", labelbottom=False)
        ax_res.set_ylabel(panel["res_ylabel"])
        ylim = _log_ylim_from_positive(abs_res, floor_mult=0.5, ceil_mult=2.0)
        if ylim is not None:
            ax_res.set_ylim(*ylim)

    # X-axis numbers only on the very bottom panel.
    ax_bottom = pairs[-1][1]
    ax_bottom.tick_params(which="both", labelbottom=True)
    ax_bottom.set_xlabel(r"$r$ [pc]")
    x_margin = 0.03 * (r.max() - r.min())
    ax_T.set_xlim(r.min() - x_margin, r.max() + x_margin)

    out = ROOT / "media" / "bubble_profiles_n40.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
