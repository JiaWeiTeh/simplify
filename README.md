# astrosimplify

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/JiaWeiTeh/simplify)](https://github.com/JiaWeiTeh/simplify/releases)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20332764.svg)](https://doi.org/10.5281/zenodo.20332764)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Dependencies](https://img.shields.io/badge/requires-numpy-blue.svg)

Heuristic downsampling of 1-D astrophysical profiles — radial density and
temperature profiles, stellar evolution tracks, spectral energy
distributions, and any other 1-D scalar curve — while preserving sharp
transitions, local extrema, and overall shape.  Requires only NumPy;
matplotlib is optional, for the plotting and animation helpers.

<img src="https://raw.githubusercontent.com/JiaWeiTeh/simplify/main/media/demo_nonoise.gif" alt="Simplification demo" width="560">

The animation progressively adds points to a 10 000-point random test
curve (seed 42, no noise).  Top: the simplified overlay.  Middle: the
absolute residual `|y − y_interp|` on a log axis.  Bottom: RMSE versus
point count.  R² ≥ 0.9 is first reached at **n = 34** (green dashed line).

```bash
astrosimplify --random --seed 42 --no-noise --animate demo.gif --animate-duration 5
```

Typical inputs are curves where the *solver* needed 10⁴–10⁶ points for
stability but the *saved output* does not: radial `ρ(r)`, `T(r)`, `v(r)`
from 1-D hydrodynamics, stellar evolution tracks, SEDs and spectral
templates, tabulated cooling functions and opacities, and dense output
from adaptive integrators.  `astrosimplify` thins on *shape* rather than
on stride, so shocks, contact discontinuities, and extrema survive while
smooth stretches are reduced to a handful of points.

## Installation

Python 3.9 or newer.  The only hard dependency is NumPy.

```bash
pip install "astrosimplify @ git+https://github.com/JiaWeiTeh/simplify.git"
```

Add the `plot` extra to pull in matplotlib for `--plot`, `--animate`, and
the diagnostic grid:

```bash
pip install "astrosimplify[plot] @ git+https://github.com/JiaWeiTeh/simplify.git"
```

To work on the source or rerun the figure scripts in `media/`:

```bash
git clone https://github.com/JiaWeiTeh/simplify.git
cd simplify
pip install -e ".[plot]"
```

Check the install with `astrosimplify --help`.

## Quick start

```python
import numpy as np
import astrosimplify as asimp

x = np.linspace(0, 10, 10000)
y = np.sin(x) + 0.5 * np.sin(5 * x)

x_s, y_s = asimp.simplify(x, y)                   # 10000 → ~200 points
metrics = asimp.simplify_error(x, y, x_s, y_s)
print(f"R² = {metrics['r_squared']:.4f}, "
      f"compression = {metrics['compression']:.1f}x")
```

Common variations:

```python
x_s, y_s = asimp.simplify(x, y, nmin=200)                       # denser output
x_s, y_s = asimp.simplify(x, y, max_err=0.1, log_y=False)       # bound the worst-case error
asimp.simplify_plot(x, y, x_s, y_s, save_path="comparison.png")  # before/after figure
```

The same operations from the command line, on a two-column text file:

```bash
astrosimplify data.csv -o reduced.csv                 # basic
astrosimplify data.csv --metrics --plot               # inspect quality
astrosimplify data.csv --nmin 200                     # denser output
astrosimplify data.csv --max-err 0.1 --log-y off      # bound worst-case error
astrosimplify data.csv --animate output.gif           # animation
```

Built-in demo curves need no input file:

```bash
astrosimplify --random --metrics                      # synthetic curve, error table
astrosimplify --random --diagnostic                   # size/error sweep
astrosimplify --randomSB99 --animate sb99.gif         # bundled Starburst99 SED
```

`--max-err` always needs an explicit `--log-y on|off` (or `log_y=True/False`
in Python) because its units follow the y-space: dex in log mode, raw
y-units otherwise.

## Real-data examples

### Starburst99 5 Myr SED

<img src="https://raw.githubusercontent.com/JiaWeiTeh/simplify/main/media/demo_sb99_tight.gif" alt="Starburst99 SED with max_err" width="560">

1221 → **360** points (3.4× compression).  The curve is the Starburst99
`LOG (TOTAL)` column at 5 Myr (instantaneous burst, Z = Z☉), spanning
91 Å to 1.6 × 10⁶ Å and ~6 dex in luminosity, with `x = log10(λ/Å)` and
`y = log10(L_λ)`.  With `max_err = 0.05` points are inserted at the
worst-error locations until no sample deviates by more than 0.05 dex.
The data ships with the package as `load_sb99_5myr()`.

```bash
astrosimplify --randomSB99 --animate sb99.gif --animate-duration 6 --r2-target 0.999 --max-err 0.05 --log-y off
```

### Stellar-wind bubble profiles

Temperature `T(r)` and density `n(r)` inside a stellar-wind bubble: a
smooth interior with one sharp transition at the contact discontinuity.

| temperature `log₁₀ T(r)` | density `log₁₀ n(r)` |
|:---:|:---:|
| ![Bubble temperature profile](https://raw.githubusercontent.com/JiaWeiTeh/simplify/main/media/demo_bubble_T.gif) | ![Bubble density profile](https://raw.githubusercontent.com/JiaWeiTeh/simplify/main/media/demo_bubble_n.gif) |

The original profiles sit on a **30 000-point** radial grid; fewer than
**60 points reproduce either curve to R² ≥ 0.999**, a ~500× reduction.
The bend detector spends its budget on the edge and barely touches the
smooth interior.  The bundled `data/bubble_T.dat` and `data/bubble_n.dat`
are a 100-point pre-simplification of the full grid so the demo
regenerates quickly; see `media/make_bubble_gifs.py`.

## Choosing `nmin` and `max_err`

`--diagnostic` sweeps several relative output sizes
(`nrel = n_out / n_orig`), keeping the most important points first, and
prints the point count, worst-case error, and R² for each.  No output
file is written.

```bash
astrosimplify --random --seed 67 --diagnostic
```

```text
   nrel    n_out   compression      max_err        R^2
  --------------------------------------------------------
   0.80     8000          1.2x    3.056e-03   1.000000
   0.30     3000          3.3x    1.573e-02   0.999976
   0.10     1000         10.0x    2.921e-02   0.999922
   0.01      100        100.0x    8.997e-02   0.999665
```

Read it as a budget-versus-accuracy table.  The `max_err` column is in
the same y-space the pipeline optimised (dex when `log_y` is active), so
it drops straight into `--max-err`.  Customise the targets with `--nrels`
and add `--plot` or `--plot-save PATH` for a grid of before/after panels:

```bash
astrosimplify --random --seed 67 --diagnostic --nrels 0.9,0.5,0.25 --plot
```

<img src="https://raw.githubusercontent.com/JiaWeiTeh/simplify/main/media/demo_random_diagnostic.png" alt="Diagnostic grid" width="560">

From Python, `simplify_diagnostic` returns one dict per `nrel` with the
metrics and the simplified arrays:

```python
from astrosimplify import random_test_curve, simplify_diagnostic

x, y = random_test_curve(seed=67)
rows = simplify_diagnostic(x, y, nrels=[0.6, 0.3], plot=False)
print(rows[0]["n_out"], rows[0]["max_err"], rows[0]["r_squared"])
```

## Input and output format

The CLI reads a plain two-column text file of `x y` pairs, one sample
per line, in any order (it is sorted by `x` internally):

- `.csv` files are read comma-delimited; any other extension is read
  whitespace-delimited.
- Lines beginning with `#` are comments, so a `# x,y` header is fine.
- Exactly two columns: the first is `x`, the second is `y`.

Output goes to `--output/-o` (default `simplified_output.csv`) as a
comma-delimited file with a `# x,y` header, so simplified files can be
fed straight back in.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nmin` | 100 | Target minimum number of output samples for arc-length sampling. |
| `grad_inc` | 1.0 | Bend sensitivity (dimensionless). A point is kept when the Menger curvature in normalised coordinates exceeds this value. |
| `warn_below_r2` | 0.9 | Emit a `UserWarning` if the output R² falls below this. `None` disables the check. |
| `max_err` | `None` | Maximum allowed vertical (fixed-x) interpolation error. A greedy loop inserts points until no sample deviates by more than this. Units follow `log_y` (dex when on, raw y-units when off), so `log_y` must be set explicitly when `max_err` is used. |
| `log_y` | `"auto"` | Run the feature detectors in `log10(y)`. `"auto"` activates when every `y > 0` and `max(y)/min(y) > 100`; pass `True`/`False` to force. The returned arrays always contain the raw `y`. |
| `dedup_tol` | `1e-6` | Tolerance for collapsing near-duplicate consecutive samples (ODE-solver stagnation). `0` disables. |

## How it works

The input is sorted by `x` and near-duplicate consecutive samples are
collapsed.  Then:

1. **Bend detection.** Menger curvature of each consecutive triplet,
   computed in `(x, y)` coordinates normalised to `[0, 1]`, flags corners
   where it exceeds `grad_inc`.  Normalising makes `grad_inc`
   scale-invariant.
2. **Extrema.** Every sign change of the first derivative is a candidate
   extremum; the more extreme of the two neighbours is taken.
3. **Arc-length sampling.** The cumulative arc length in normalised
   coordinates is split into `nmin` equal bins and one point is taken per
   bin, so dense regions get more points without starving gentle slopes.
4. **Mandatory set.** Extrema with topological prominence ≥ 5 % of the
   y-range are always kept, and the x-axis is split into 20 chunks with
   one feature point promoted per chunk, so every stretch of `x` is
   represented.
5. **Greedy error bound** (when `max_err` is set).  Repeatedly insert the
   original point with the largest vertical interpolation error until the
   worst case is below `max_err`.
6. **Collinearity prune.** Drop any remaining point lying within 0.1 % of
   the y-range of the chord between its neighbours.  Endpoints, mandatory
   extrema, and `max_err` insertions are protected.

Finally the output R² is compared against `warn_below_r2` and a
`UserWarning` is emitted if it falls short; the output is not changed.

For multi-decade data (density, temperature, flux) `log_y="auto"`
switches every detector onto `log10(y)` when the input is strictly
positive and spans at least two decades.  `simplify_error` and
`--metrics` then also report `log_r_squared`, `log_rms_err`, and
`log_max_dex_err`.  Plot such output with `plt.semilogy` so the
interpolation between kept points happens in log space, as the sampler
assumed.

## Limitations

- **Single-valued curves only.** Closed loops, self-intersecting paths,
  and multi-valued data are not supported; points are sorted by `x`.
- **Heuristic, not optimal.** It does not minimise point count for a
  given error bound the way Douglas–Peucker variants do.
- **Noise.** Heavily noisy data can trip the bend and sign-change
  detectors into keeping noise spikes; pre-smooth or raise `grad_inc`.
- **`log_y` needs positive data.** Curves crossing zero stay in linear
  space.
- **In-memory only.** The whole curve is processed at once; there is no
  streaming mode.

## Citation

If you use `astrosimplify` in your research, please cite the Zenodo
record:

> Teh, J. W. (2026). *astrosimplify: heuristic downsampling of
> one-dimensional astrophysical profiles* (v2.0.0). Zenodo.
> https://doi.org/10.5281/zenodo.20332764

A machine-readable [`CITATION.cff`](CITATION.cff) is included, so the
"Cite this repository" button on GitHub produces BibTeX and APA entries.

## License

Released under the [MIT License](LICENSE).
