"""Demo GIFs for ``simplify_path``: a star's orbit in a galactic disk.

The orbit uses the epicycle approximation (Binney & Tremaine, Galactic
Dynamics) for a flat rotation curve, where the epicyclic frequency is
kappa = sqrt(2) * Omega, plus a vertical oscillation at nu = 2.7 * Omega,
roughly the solar-neighbourhood ratio.  Amplitudes are exaggerated so
the motion is visible.  Units: guiding radius 1, orbital period 2 pi.

The 2-D GIF shows the orbit in the disk plane, the 3-D GIF the full
orbit.  Both keep every sample within 1e-3 (in guiding radii) of the
simplified orbit at the same time.  Run from the repository root:

    python media/make_path_gifs.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from astrosimplify import simplify_path_animate  # noqa: E402

OMEGA = 1.0
KAPPA = np.sqrt(2.0) * OMEGA
NU = 2.7 * OMEGA
R_GUIDE, X_EPI, Z_AMP = 1.0, 0.15, 0.1
MAX_ERR = 1e-3

t = np.linspace(0.0, 4 * 2 * np.pi / OMEGA, 20000)
R = R_GUIDE + X_EPI * np.cos(KAPPA * t)
phi = OMEGA * t - (2 * OMEGA / KAPPA) * (X_EPI / R_GUIDE) * np.sin(KAPPA * t)
x, y, z = R * np.cos(phi), R * np.sin(phi), Z_AMP * np.cos(NU * t)

simplify_path_animate(
    np.column_stack([x, y]), str(ROOT / "media" / "demo_path_2d.gif"),
    t=t, max_err=MAX_ERR, title="Disk orbit, in-plane",
)
simplify_path_animate(
    np.column_stack([x, y, z]), str(ROOT / "media" / "demo_path_3d.gif"),
    t=t, max_err=MAX_ERR, title="Disk orbit, 3-D",
)
