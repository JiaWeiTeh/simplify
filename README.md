# simplify

Heuristic downsampling of 1-D curves while preserving sharp bends, local
extrema, and overall shape. Single file, no dependencies beyond NumPy.

![Simplification demo](demo.gif)

The animation above was produced with:

```bash
python simplify.py --random --seed 42 --animate demo.gif --animate-duration 5
```

It progressively adds points to a 10 000-point noisy test curve and
stops once R² ≥ 0.9 (green dashed line, here at n = 40). The bottom
panel shows RMSE on a log-log scale with dashed reference lines at
R² = 0.9, 0.99, 0.999 so the remaining distance to a perfect fit is
visible at a glance.

## Quick start

```bash
python simplify.py --random --animate simplify.gif
```

Generates a synthetic test curve and an animated GIF of the
simplification process. Other quick demos:

```bash
python simplify.py --random --metrics                          # error table
python simplify.py --random --plot                             # comparison plot
python simplify.py --random --animate demo.gif --r2-target 0.95  # tighter fit
```

## Command line

```bash
python simplify.py data.csv -o reduced.csv                    # basic
python simplify.py data.csv --metrics --plot                   # inspect quality
python simplify.py data.csv --r2-target 0.95                   # tighter R² target
python simplify.py data.csv --r2-target 0.8                    # aggressive compression
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

# Simplify (default R² target = 0.9)
x_s, y_s = _simplify(x, y)

# Tighter quality target
x_s, y_s = _simplify(x, y, r2_target=0.99)

# Keep all feature-detected points (no R² thinning)
x_s, y_s = _simplify(x, y, r2_target=None)

# Error metrics
metrics = _simplify_error(x, y, x_s, y_s)
print(f"R² = {metrics['r_squared']:.4f}, compression = {metrics['compression']:.1f}x")

# Plot
_simplify_plot(x, y, x_s, y_s, save_path="comparison.png")
```

## Algorithm

Three independent feature detectors populate a candidate pool, a
topological-persistence filter promotes the visually important extrema
to a mandatory set, an R²-driven binary search picks the smallest
subset that meets the quality target, and a final collinearity pass
removes points that lie on the line between their neighbours:

1. **Scale-invariant bend detection** — both the discrete Menger
   curvature and the turning angle between adjacent segments are
   computed in *normalised* `(x, y)` coordinates (both rescaled to
   `[0, 1]`). A point is flagged when either curvature exceeds
   `grad_inc` or the turning angle exceeds `0.1 * grad_inc` radians.
   The two detectors are complementary: curvature catches tight
   corners (shocks, vertical drops), the angle detector catches wide,
   gentle bends whose triplet triangle is too large for curvature
   alone to fire. Running in normalised coordinates makes `grad_inc`
   dimensionless — the same value works at any axis scale.

2. **Sign-change detection** — keeps every point where the first
   derivative changes sign. Sub-noise extrema (prominence below
   0.5 % of the y-range) are dropped so noise-driven sign flips on
   near-flat regions do not flood the candidate pool.

3. **Cumulative-distance sampling** — divides the total variation of y
   (`sum(|diff(y)|)`) into `nmin` equal bins and selects one point at
   each bin boundary. Dense where y changes rapidly, sparse where flat.

4. **Topological persistence (mandatory set)** — for each extremum,
   `_peak_prominences` computes the peak prominence (minimum descent
   from a local max, or ascent from a local min, needed to reach a
   strictly more extreme point) in a single O(n log n) monotonic-stack
   pass. Extrema with prominence ≥ 5 % of the y-range are marked
   **mandatory** and included in every trial subset, so big features
   never flicker in and out with changing `n`. Extrema below 0.5 % are
   dropped as noise (step 2).

5. **R²-based thinning** — the remaining candidates are traversed in
   hierarchical bisection order (endpoints → midpoint → quartiles → …)
   so trial subsets are strictly nested. A binary search plus a
   3-in-a-row stability check picks the smallest `k` for which
   `mandatory ∪ bisection[:k]` reaches `r2_target`.

6. **Collinearity prune** — a vectorised sweep drops any point whose
   y value is within 0.1 % of the y-range of the chord between its
   two surviving neighbours. Endpoints and mandatory extrema are
   protected. This removes the redundant samples the bisector leaves
   on linear and horizontal segments.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nmin` | 100 | Target minimum output samples for distance sampling |
| `grad_inc` | 1.0 | Bend sensitivity (dimensionless, scale-invariant); fires when Menger curvature in normalised coords > `grad_inc`, or turning angle > `0.1 * grad_inc` rad |
| `r2_target` | 0.9 | Target R²; set to `None` to keep all detected points |

## Dependencies

- **numpy** (required)
- **matplotlib** (optional — plotting and animation)
