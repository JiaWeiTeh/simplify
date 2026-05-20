# simplify

Heuristic downsampling of 1-D curves while preserving sharp bends, local
extrema, and overall shape. Single file, no dependencies beyond NumPy.

![Simplification demo](demo_nonoise.gif)

The animation progressively adds points to a 10 000-point random test
curve (seed 42, no noise).  Three panels: the simplified overlay (top),
the signed residual showing where the approximation over-/undershoots
(middle), and RMSE vs point count on a log-log scale (bottom).  R² ≥ 0.9
is first reached at **n = 57** (green dashed line).  Generated with:

```bash
python simplify.py --random --seed 42 --no-noise --animate demo_nonoise.gif --animate-duration 5
```

### Real data — Starburst99 5 Myr SED

| `R² ≥ 0.999` | `R² ≥ 0.999` + `max_err = 0.05` |
|:---:|:---:|
| ![Starburst99 R²=0.999](demo_sb99_loose.gif) | ![Starburst99 R²=0.999 + max_err](demo_sb99_tight.gif) |
| 1221 pts → **38** pts (32× compression) | 1221 pts → **105** pts (12× compression) |

The curve is the Starburst99 `LOG (TOTAL)` SED column at 5 Myr
(instantaneous burst, Z = Z☉) — a real astrophysical spectrum spanning
91 Å to 1.6 × 10⁶ Å in wavelength and ~6 dex in luminosity.  `x` is
`log10(λ/Å)` and `y` is `log10(L_λ / (erg s⁻¹ Å⁻¹))`.  The left GIF
shows the default simplification — the arc-length sampler distributes
points symmetrically across the curve's normalised `[0, 1]²` length.
The right GIF adds `max_err = 0.05`, which inserts points at the
worst-error locations until no point deviates by more than 0.05 dex
(≈ 12 %), yielding 105 points.  Generated with:

```bash
python simplify.py --randomSB99 --animate demo_sb99_loose.gif --animate-duration 6 --r2-target 0.999
python simplify.py --randomSB99 --animate demo_sb99_tight.gif --animate-duration 6 --r2-target 0.999 --max-err 0.05
```

## Quick start

```bash
python simplify.py --random --no-noise --animate simplify.gif
```

Generates a synthetic test curve and an animated GIF of the
simplification process. Other quick demos:

```bash
python simplify.py --random --metrics                          # error table
python simplify.py --random --plot                             # comparison plot
python simplify.py --random --animate demo.gif                 # animated GIF
python simplify.py --randomSB99 --animate sb99.gif             # real SED data
python simplify.py --randomSB99 --max-err 0.05 --animate sb99_tight.gif  # bounded error
```

## Command line

```bash
python simplify.py data.csv -o reduced.csv                    # basic
python simplify.py data.csv --metrics --plot                   # inspect quality
python simplify.py data.csv --nmin 200                         # denser output
python simplify.py data.csv --max-err 0.1                      # bound worst-case error
python simplify.py data.csv --animate output.gif               # animation
python simplify.py data.csv --grad-inc 0.5                     # lower curvature threshold
```

Run `python simplify.py --help` for all options.

## Python API

```python
import numpy as np
from simplify import _simplify, _simplify_error, _simplify_plot

x = np.linspace(0, 10, 10000)
y = np.sin(x) + 0.5 * np.sin(5 * x)

# Simplify (default warn_below_r2 = 0.9)
x_s, y_s = _simplify(x, y)

# Higher nmin for denser output
x_s, y_s = _simplify(x, y, nmin=200)

# Bound the worst-case pointwise error
x_s, y_s = _simplify(x, y, max_err=0.1)

# Error metrics
metrics = _simplify_error(x, y, x_s, y_s)
print(f"R² = {metrics['r_squared']:.4f}, compression = {metrics['compression']:.1f}x")

# Plot
_simplify_plot(x, y, x_s, y_s, save_path="comparison.png")
```

## How it works (the short version)

You have a curve with thousands of points and you want to keep only the
few that matter. Which ones matter? `simplify` answers that with four
simple ideas:

- **Keep the bends.** Where the curve turns sharply, you need a point to
  capture the corner. Long straight stretches barely need any.
- **Keep the peaks and valleys.** The high and low points are what your
  eye remembers, so they always make the cut.
- **Spread the rest out fairly.** Don't crowd every point into the busy
  region — make sure every part of the curve gets some representation,
  and rank features by how *important* they are so the big ones never
  get dropped.
- **Check the error and clean up.** Measure how far the simplified curve
  strays from the original. Drop any leftover point that sits on a
  straight line between its neighbours (it adds nothing), and optionally
  keep adding points back until the worst gap is under a limit you set.

That's it. The section below is the same four ideas spelled out
precisely, with the math and the edge cases.

## Algorithm

The input is sorted by x and near-duplicate consecutive samples are
collapsed (ODE-solver stagnation).  Three independent feature detectors
populate a candidate pool, a topological-persistence filter marks the
visually important extrema as mandatory, x-uniform coverage promotes
one point per chunk, an optional greedy loop bounds the worst-case
pointwise error, and a final collinearity pass removes points that lie
on the chord between their neighbours:

1. **Scale-invariant bend detection** — Menger curvature κ (the
   reciprocal of the circumradius of each consecutive triplet) is
   computed in *normalised* `(x, y)` coordinates (both rescaled to
   `[0, 1]`). A point is flagged when κ exceeds `grad_inc`.  Running
   in normalised coordinates makes `grad_inc` dimensionless — the same
   value works at any axis scale.

2. **Refined sign-change detection** — every point where the first
   derivative changes sign is a candidate extremum.  Because the
   sign flip occurs between consecutive samples, the refinement picks
   whichever of the two neighbours is more extreme (higher for a max,
   lower for a min).

3. **Arc-length sampling** — the differential arc length
   `ds = sqrt((Δx/range_x)² + (Δy/range_y)²)` in normalised
   `[0, 1]²` coordinates treats both axes symmetrically.  The
   cumulative arc length is divided into `nmin` equal bins and one
   point is selected at each bin boundary — dense where the curve
   changes rapidly, sparse where it is flat.  Unlike the older
   y-only metric `sum(|Δy|)`, a long gentle slope receives points
   proportional to its x-span, which prevents high-amplitude regions
   from starving gradual ones.

4. **Topological persistence (mandatory set)** — for each extremum,
   `_peak_prominences` computes the peak prominence (minimum descent
   from a local max, or ascent from a local min, needed to reach a
   strictly more extreme point) in a single O(n log n) monotonic-stack
   pass. Extrema with prominence ≥ 5 % of the y-range are marked
   **mandatory** and always retained, so big features never flicker
   in and out with changing `nmin`.

4b. **X-uniform mandatory coverage** — the x-domain is split into 20
   equal-width chunks and the feature-pool point nearest each chunk
   centre is promoted into the mandatory set alongside the prominent
   extrema.  This thin x-uniform skeleton guarantees every part of
   the x-axis gets at least one retained point, complementing the
   arc-length sampler for chunks with very little curve length per
   x-span.

5. **Greedy max-error reduction** — when `max_err` is set, a greedy
   loop finds the original data point with the largest interpolation
   error and inserts it into the retained set, repeating until the
   worst-case absolute error drops below `max_err`.  Points inserted
   by this loop are protected from the subsequent collinearity prune.

6. **Collinearity prune** — a vectorised sweep drops any point whose
   y value is within 0.1 % of the y-range of the chord between its
   two surviving neighbours. Endpoints, mandatory extrema, and
   `max_err` insertions are protected. This removes redundant samples
   on linear and horizontal segments.

After all stages, the output's R² is checked against `warn_below_r2`
and a `UserWarning` is emitted if it falls short — purely
informational, the output is not changed.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nmin` | 100 | Target minimum output samples for arc-length sampling |
| `grad_inc` | 1.0 | Bend sensitivity (dimensionless, scale-invariant); fires when Menger curvature in normalised coords > `grad_inc` |
| `warn_below_r2` | 0.9 | Soft R² quality threshold; a `UserWarning` is emitted when the output falls below this value.  Pass `None` to disable. |
| `max_err` | `None` | Maximum allowed absolute interpolation error.  A greedy loop inserts points until the worst-case error drops below this value.  Operates in the same y-space as the pipeline (log10 when `log_y` is active, so `0.1` means ≤ 0.1 dex ≈ 26 % multiplicative). |
| `log_y` | `"auto"` | Work in log-y space for every internal feature detector.  `"auto"` activates when every `y > 0` and `max(y)/min(y) > 100`; pass `True` / `False` to force. |
| `dedup_tol` | `1e-6` | Stagnation tolerance for collapsing near-duplicate consecutive samples (ODE-solver artefact). Set to `0` to disable. |

## Multi-decade data (density, temperature, flux profiles)

Astrophysical profiles that span several orders of magnitude — e.g.
density or temperature inside an interstellar bubble, where ISM
(~1 cm⁻³) to shocked shell (~10⁶ cm⁻³) is routine — need log-y
handling, because the linear arc-length sampler would be dominated by
the peak amplitude.  `log_y="auto"` (the default) switches every
internal detector onto `log10(y)` when the input is strictly positive
and spans ≥ 2 decades; the output arrays still contain the caller's
raw `y` values.

The `_simplify_error` helper and the CLI's `--metrics` flag report
`log_r_squared`, `log_rms_err`, and `log_max_dex_err` whenever `y` is
strictly positive, so quality assessments on such data are in the
right scale by default.  When rendering, use `plt.semilogy` (or equivalent)
on the simplified points — that interpolates between them in log-y
space, which is what the algorithm's internal arc-length sampler assumed.

## Dependencies

- **numpy** (required)
- **matplotlib** (optional — plotting and animation)
