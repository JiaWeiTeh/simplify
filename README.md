# simplify

Intelligent curve downsampling that preserves sharp bends, local extrema, and overall shape.

## Quick start

Try it immediately with a random test curve:

```bash
python simplify.py --random --animate simplify.gif
```

This generates a synthetic curve (spikes, plateaus, steps, noise) and produces an animated GIF showing the simplification process.

Other quick demos:

```bash
python simplify.py --random --metrics             # print error metrics
python simplify.py --random --plot                 # interactive comparison plot
python simplify.py --random --animate simplify.gif --r2-target 0.95  # tighter fit
```

## Usage

### Command line

```bash
# Simplify a data file
python simplify.py data.csv -o reduced.csv

# Control quality with R² target (default: 0.9)
python simplify.py data.csv --r2-target 0.95       # keep more points, better fit
python simplify.py data.csv --r2-target 0.8        # aggressive compression

# Show metrics and plot
python simplify.py data.csv --metrics --plot

# Generate animation
python simplify.py data.csv --animate output.gif
```

### Python API

```python
from simplify import _simplify, _simplify_error, _simplify_plot

# Basic usage
x_s, y_s = _simplify(x, y)

# Custom R² target
x_s, y_s = _simplify(x, y, r2_target=0.95)

# Check quality
metrics = _simplify_error(x, y, x_s, y_s)
print(f"R² = {metrics['r_squared']:.4f}, compression = {metrics['compression']:.1f}x")

# Visualise
_simplify_plot(x, y, x_s, y_s)
```

## How it works

Three strategies independently select important points, which are merged together:

1. **Gradient-change detection** -- keeps sharp bends and discontinuities
2. **Sign-change detection** -- keeps local minima and maxima
3. **Cumulative-distance sampling** -- uniform arc-length coverage

An R² target (default 0.9) then thins the result to the minimum points needed to achieve that quality level.

## Dependencies

- `numpy`
- `matplotlib` (optional, for plotting and animation)
