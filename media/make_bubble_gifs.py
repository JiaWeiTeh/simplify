"""Generate animated GIFs of the bubble T(r) and n(r) profile
simplification in the same visual style as ``bubble_profiles_n40.pdf``.

For each profile, a single GIF builds the simplified curve up from a
handful of points to the n = 40 set shown in the static PDF, plotted
over a two-panel layout: a tall profile panel (log y-axis) on top of a
short residual subrow (``|Delta log y|`` on a log y-axis).  Original
data is the same faint gray reference curve as the PDF, and the
simplified set uses the PDF's black-line / red-marker styling.  The
final frame is identical to the PDF's n = 40 panels.

Run from the repository root:

    python media/make_bubble_gifs.py
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from astrosimplify import _auto_log_y, _importance_order, simplify_error  # noqa: E402

STYLE_FILE = ROOT / "media" / "trinity.mplstyle"

# Match make_bubble_panels.py exactly so the final GIF frame reproduces
# the static PDF's n = 40 picture.
N_FINAL = 40
N_START = 3
N_ORIG_LABEL = r"$n = 3\times10^4$"

C_ORIG = "0.6"
C_SIMP = "#d62728"
C_RESID = "#0072B2"

PC_CM = 3.0856775814913673e18
PC3_TO_CM3_SHIFT = -3.0 * np.log10(PC_CM)

PANELS = [
    {
        "file": ROOT / "data" / "bubble_T.dat",
        "out": ROOT / "media" / "demo_bubble_T.gif",
        "ylabel": r"$T(r)$ [K]",
        "res_ylabel": r"$|\Delta \log T|$",
        "legend_loc": "lower left",
        "log_shift": 0.0,
    },
    {
        "file": ROOT / "data" / "bubble_n.dat",
        "out": ROOT / "media" / "demo_bubble_n.gif",
        "ylabel": r"$n(r)$ [cm$^{-3}$]",
        "res_ylabel": r"$|\Delta \log n|$",
        "legend_loc": "upper left",
        "log_shift": PC3_TO_CM3_SHIFT,
    },
]


def _build_steps(r, y):
    """Pre-compute (idx, abs_res) for n = 3, 4, ..., N_FINAL."""
    use_log = _auto_log_y(y, "auto")
    y_work = np.log10(y) if use_log else y
    n_cap = min(N_FINAL, r.size)
    order = _importance_order(r, y_work, n_cap)

    steps = []
    for n in range(N_START, n_cap + 1):
        idx = np.sort(order[:n])
        y_interp = np.interp(r, r[idx], y_work[idx])
        abs_res = np.abs(y_work - y_interp)
        steps.append({"n": n, "idx": idx, "abs_res": abs_res})
    return steps


def _make_gif(panel, *, fps=20, duration=5.0, hold_frac=0.18):
    data = np.loadtxt(panel["file"], comments="#")
    r, y_log = data[:, 0], data[:, 1] + panel["log_shift"]
    order_x = np.argsort(r, kind="stable")
    r, y_log = r[order_x], y_log[order_x]
    y_lin = 10.0 ** y_log

    steps = _build_steps(r, y_lin)

    fig = plt.figure(figsize=(4.6, 4.6))
    gs = fig.add_gridspec(
        2, 1, height_ratios=[3, 1], hspace=0.05,
        top=0.975, bottom=0.13, left=0.18, right=0.965,
    )
    ax_top = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1], sharex=ax_top)

    ax_top.plot(r, y_lin, "-", color=C_ORIG, lw=2.0, zorder=1,
                label=rf"original ({N_ORIG_LABEL})")
    line_simp, = ax_top.plot(
        [], [], "-", color="k", lw=1.2, zorder=3,
        marker="o", ms=3.5, markerfacecolor=C_SIMP,
        markeredgecolor="black", mew=0.6,
        label=rf"simplified ($n={steps[-1]['n']}$)",
    )
    ax_top.set_yscale("log")
    ax_top.grid(False)
    ax_top.tick_params(which="both", labelbottom=False)
    ax_top.set_ylabel(panel["ylabel"])
    leg = ax_top.legend(loc=panel["legend_loc"])
    simp_legend_text = leg.get_texts()[1]

    ax_top.set_ylim(y_lin.min() * 0.6, y_lin.max() * 1.8)
    x_margin = 0.03 * (r.max() - r.min())
    ax_top.set_xlim(r.min() - x_margin, r.max() + x_margin)

    res_line, = ax_res.plot([], [], "-", color=C_RESID, lw=1.0)
    ax_res.set_yscale("log")
    ax_res.grid(False)
    ax_res.set_ylabel(panel["res_ylabel"])
    ax_res.set_xlabel(r"$r$ [pc]")

    # Match the static PDF's residual y-axis: derived from the n = 40
    # (final) frame using the same min_pos*0.5 / max*2.0 rule.  Early
    # low-n frames have much larger residuals than the panel can show, so
    # their lines are clipped to the panel's top edge (visible there as a
    # solid bar).  This lets the final frame look exactly like the PDF
    # while the animation still reads as "error shrinks as n grows".
    final = steps[-1]
    final_pos = final["abs_res"][final["abs_res"] > 0]
    res_floor = float(final_pos.min()) * 0.5 if final_pos.size else 1e-6
    res_ceil = float(final["abs_res"].max()) * 2.0 if final["abs_res"].size else 1.0
    ax_res.set_ylim(res_floor, res_ceil)

    n_steps = len(steps)
    n_frames = int(fps * duration)
    sweep_frames = max(1, int(fps * duration * (1.0 - hold_frac)))

    def _update(frame):
        if frame < sweep_frames:
            step_idx = int(frame / max(sweep_frames - 1, 1) * (n_steps - 1))
            step_idx = min(step_idx, n_steps - 1)
        else:
            step_idx = n_steps - 1
        s = steps[step_idx]
        idx = s["idx"]
        line_simp.set_data(r[idx], y_lin[idx])
        simp_legend_text.set_text(rf"simplified ($n={s['n']}$)")
        # Only clip the top so low-n frames don't blow up the panel.
        # Zero residuals (at kept points) survive as log gaps, matching
        # the static PDF's natural breaks between retained samples.
        clipped = np.minimum(s["abs_res"], res_ceil)
        res_line.set_data(r, clipped)
        return line_simp, simp_legend_text, res_line

    anim = FuncAnimation(fig, _update, frames=n_frames, blit=False)
    anim.save(str(panel["out"]), writer="pillow", fps=fps, dpi=120)
    plt.close(fig)
    print(f"Saved {panel['out']}")


def main():
    plt.style.use(str(STYLE_FILE))
    if shutil.which("latex") is None:
        plt.rcParams["text.usetex"] = False
    for panel in PANELS:
        _make_gif(panel)


if __name__ == "__main__":
    main()
