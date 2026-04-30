#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone curve-simplification module.

Heuristic downsampling of 1-D curves while preserving physically and
visually important features (sharp bends, local extrema, arc-length
uniformity).  No dependencies beyond numpy / matplotlib / stdlib.

Functions
---------
_simplify          Core downsampling algorithm.
_simplify_error    Error metrics (RMSE, MAE, R², compression, …).
_simplify_plot     Static before/after comparison plot.
_simplify_animate  Animated GIF/MP4 of the simplification process.
_peak_prominences  1-D topological persistence (O(n log n)).
_random_test_curve Generate a random curve that exercises all strategies.
_load_sb99_5myr    Bundled Starburst99 SED at 5 Myr (log λ vs log L_λ).
_simplify_cli      Command-line interface (reads two-column text files).
"""

from pathlib import Path
from typing import Tuple, Union, Sequence

import numpy as np

# Path to the bundled matplotlib style file.
_STYLE_FILE = Path(__file__).parent / "simplify.mplstyle"

# Bundled Starburst99 SED (instantaneous burst, 5 Myr slice).
_SB99_5MYR_FILE = Path(__file__).parent / "data" / "sb99_5myr.dat"

# Number of equal-x-width chunks for x-uniform mandatory coverage.
_COVERAGE_CHUNKS = 20


def _prev_next_strict(y: np.ndarray, greater: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    One-pass monotonic-stack computation of previous/next strictly-greater
    (or strictly-less) indices for every position in ``y``.

    Each index is pushed and popped at most once, so the total cost is
    amortised O(n) despite the nested ``while`` loop.  Returns two int64
    arrays ``prev_s`` and ``next_s`` such that for every ``i``:

    * ``prev_s[i]`` is the largest ``j < i`` with ``y[j] > y[i]`` (greater
      case) or ``y[j] < y[i]`` (less case); ``-1`` if no such ``j`` exists.
    * ``next_s[i]`` is the smallest ``j > i`` with the same condition;
      ``n`` if no such ``j`` exists.

    The inner loop is the hottest path in the whole module.  To shave
    per-iteration overhead we:

    * work off ``y.tolist()`` so every comparison is between native
      Python floats — numpy-scalar fetches cost ~100 ns each, Python
      floats cost ~10 ns;
    * write the output directly into a preallocated Python list
      (``next_s_list``) and realise it as a numpy array at the end;
    * keep ``stk`` and the two local ``prev_s`` / ``next_s_list``
      bindings in tight lexical scope so CPython's LOAD_FAST path
      is used on every access.

    These are micro-optimisations but they roughly halve the runtime
    of this routine on million-point inputs, which lifts the whole
    ``_peak_prominences`` call out of the profile's top slot.
    """
    n = y.size
    prev_s_list = [-1] * n
    next_s_list = [n] * n
    y_list = y.tolist()
    stk: list = []
    stk_append = stk.append
    stk_pop = stk.pop
    if greater:
        for i in range(n):
            yi = y_list[i]
            # Pop anything not strictly greater than y[i]; those are
            # elements for which i is the next strictly-greater position.
            while stk and y_list[stk[-1]] <= yi:
                next_s_list[stk_pop()] = i
            if stk:
                prev_s_list[i] = stk[-1]
            stk_append(i)
    else:
        for i in range(n):
            yi = y_list[i]
            while stk and y_list[stk[-1]] >= yi:
                next_s_list[stk_pop()] = i
            if stk:
                prev_s_list[i] = stk[-1]
            stk_append(i)
    return (
        np.asarray(prev_s_list, dtype=np.int64),
        np.asarray(next_s_list, dtype=np.int64),
    )


def _sparse_table(y: np.ndarray, reducer) -> np.ndarray:
    """
    Build a sparse table for O(1) range-min or range-max queries on ``y``.

    ``reducer`` is ``np.minimum`` or ``np.maximum``.  Preprocessing is
    O(n log n) — fully vectorised numpy — and storage is (log2(n)+1, n).
    A later query over ``[lo, hi]`` uses two overlapping blocks of
    length 2**k, cf. Bender–Farach-Colton.
    """
    n = y.size
    k_max = max(1, int(np.floor(np.log2(max(n, 1)))) + 1)
    st = np.empty((k_max, n), dtype=y.dtype)
    st[0] = y
    for k in range(1, k_max):
        step = 1 << (k - 1)
        span = 1 << k
        limit = n - span + 1
        if limit <= 0:
            st[k] = st[k - 1]
        else:
            st[k, :limit] = reducer(
                st[k - 1, :limit], st[k - 1, step:step + limit]
            )
            st[k, limit:] = st[k - 1, limit:]
    return st


def _rmq(st: np.ndarray, lo: np.ndarray, hi: np.ndarray, reducer) -> np.ndarray:
    """
    Vectorised range-min/range-max query over inclusive intervals
    ``[lo[i], hi[i]]`` using a precomputed sparse table ``st``.
    """
    length = hi - lo + 1
    k = np.floor(np.log2(length)).astype(np.int64)
    return reducer(st[k, lo], st[k, hi - (1 << k) + 1])


def _peak_prominences(y: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """
    Compute topological persistence (peak prominence) for local extrema.

    For each index ``p`` in ``idx`` (a local maximum or minimum of ``y``),
    returns how much the curve must descend from a max (or ascend from a
    min) before reaching a point more extreme than ``y[p]``, or the
    boundary.  This is equivalent to the persistence of the extremum in
    the sublevel-set filtration of ``y``.

    A very tall, narrow spike has large prominence.  A small wiggle has
    small prominence.  The measure is *not* affected by a feature's
    width — only its amplitude relative to surrounding terrain.

    Complexity
    ----------
    O(n log n) total, fully deterministic: two monotonic-stack passes
    of length ``n`` produce the prev/next strictly-greater and
    strictly-less indices, two sparse tables give O(1) range-min /
    range-max queries, and all per-candidate work is vectorised.
    The monotonic-stack step runs in Python and is the dominant cost
    on very large inputs; callers should skip this routine entirely
    when its output is not needed (e.g. via ``r2_target is None``).

    Parameters
    ----------
    y : np.ndarray
        The 1-D signal.
    idx : np.ndarray
        Indices of local extrema (peaks and troughs).

    Returns
    -------
    prominences : np.ndarray
        Non-negative prominence value for each index in ``idx``.
    """
    n = y.size
    proms = np.zeros(idx.size, dtype=float)
    if idx.size == 0 or n == 0:
        return proms

    p = np.asarray(idx, dtype=np.int64)
    y_p = y[p]

    # Classify each candidate as (weak) max / min from its immediate
    # neighbours in the full signal.
    lo_nb = np.maximum(p - 1, 0)
    hi_nb = np.minimum(p + 1, n - 1)
    is_max = (y_p >= y[lo_nb]) & (y_p >= y[hi_nb])
    is_min = (y_p <= y[lo_nb]) & (y_p <= y[hi_nb]) & ~is_max

    # MAX candidates: walk outward until a point with y > y[p]; track
    # the minimum of y on each side using a min-RMQ.
    if np.any(is_max):
        PG, NG = _prev_next_strict(y, greater=True)
        st_min = _sparse_table(y, np.minimum)
        pm = p[is_max]
        pg = PG[pm]
        ng = NG[pm]
        # Inclusive walk ranges: left = [pg+1, pm-1], right = [pm+1, ng-1].
        # Boundaries: pg = -1 gives lo=0; ng = n gives hi=n-1.
        left_lo = pg + 1
        left_hi = pm - 1
        right_lo = pm + 1
        right_hi = ng - 1
        left_valid = left_lo <= left_hi
        right_valid = right_lo <= right_hi
        left_min = np.full(pm.size, np.inf)
        right_min = np.full(pm.size, np.inf)
        if np.any(left_valid):
            left_min[left_valid] = _rmq(
                st_min, left_lo[left_valid], left_hi[left_valid], np.minimum
            )
        if np.any(right_valid):
            right_min[right_valid] = _rmq(
                st_min, right_lo[right_valid], right_hi[right_valid], np.minimum
            )
        # Walks always include at least one neighbour for a true extremum,
        # so at least one side is valid.  Use the valid sides; if a side
        # is empty (shouldn't happen for real extrema) treat its shoulder
        # as +inf so the other side dominates.
        shoulder = np.maximum(left_min, right_min)
        proms[is_max] = y[pm] - shoulder

    # MIN candidates: mirror image using max-RMQ.
    if np.any(is_min):
        PL, NL = _prev_next_strict(y, greater=False)
        st_max = _sparse_table(y, np.maximum)
        pn = p[is_min]
        pl = PL[pn]
        nl = NL[pn]
        left_lo = pl + 1
        left_hi = pn - 1
        right_lo = pn + 1
        right_hi = nl - 1
        left_valid = left_lo <= left_hi
        right_valid = right_lo <= right_hi
        left_max = np.full(pn.size, -np.inf)
        right_max = np.full(pn.size, -np.inf)
        if np.any(left_valid):
            left_max[left_valid] = _rmq(
                st_max, left_lo[left_valid], left_hi[left_valid], np.maximum
            )
        if np.any(right_valid):
            right_max[right_valid] = _rmq(
                st_max, right_lo[right_valid], right_hi[right_valid], np.maximum
            )
        shoulder = np.minimum(left_max, right_max)
        proms[is_min] = shoulder - y[pn]

    # Clamp tiny negative values from floating-point rounding (prominence
    # is non-negative by definition).
    np.clip(proms, 0.0, None, out=proms)
    return proms


def _prune_collinear(
    idx: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    tol: float,
    protected: np.ndarray,
) -> np.ndarray:
    """
    Drop interior entries of ``idx`` whose y value lies within ``tol`` of
    the chord between their two surviving neighbours.

    Rationale
    ---------
    After R² thinning, the hierarchical-bisection sampler can still
    leave points that interpolate to themselves from their neighbours —
    most often on long horizontal or linear stretches.  Such points
    carry no reconstructive information and are safe to remove.  This
    pass is the primary reason the output no longer over-populates
    straight segments.

    Implementation
    --------------
    Each sweep *vectorises* the residual computation over every
    interior triplet of surviving indices and picks a **non-adjacent**
    subset of the candidates to drop.  Non-adjacency matters: removing
    both of two adjacent candidates in one pass can violate ``tol``
    because the chord between their outer neighbours is longer than
    the chord against which each was individually tested.  The
    non-adjacent pick is a single linear scan over the (already small)
    candidate list.  The outer ``while`` loop iterates until a sweep
    produces no removals — typically 2–3 sweeps because each pass
    drops a large fraction of its candidates.

    Protection
    ----------
    The first and last entries of ``idx`` are always kept, as are any
    indices appearing in ``protected`` (the "mandatory" prominent
    extrema chosen by the caller).

    Complexity
    ----------
    O(m) per sweep, fully vectorised, with ``m = idx.size`` (the
    post-thinning candidate count, typically tens to a few hundred).
    Total work is O(m · p) with p = number of sweeps (≤ log₂ m in
    practice because each sweep strictly decreases the active set).
    """
    if idx.size <= 2 or tol <= 0:
        return idx

    # ``protected_mask[i] = True`` means ``idx[i]`` must never be
    # removed.  Endpoints are always protected; explicit mandatory
    # indices are folded in via np.isin (O(m + |protected|)).
    protected_mask = np.zeros(idx.size, dtype=bool)
    protected_mask[0] = True
    protected_mask[-1] = True
    if protected.size:
        protected_mask |= np.isin(idx, protected)

    # Pre-gather coordinates at the candidate positions — cheaper than
    # fancy-indexing into the full x/y arrays on every sweep.
    xi = x[idx]
    yi = y[idx]
    alive = np.ones(idx.size, dtype=bool)

    while True:
        active = np.flatnonzero(alive)
        if active.size <= 2:
            break

        # Interior triplets of currently-alive indices.  prev/mid/next
        # are parallel arrays of length ``active.size - 2``.
        prev_pos = active[:-2]
        mid_pos = active[1:-1]
        next_pos = active[2:]

        xp = xi[prev_pos]; xm = xi[mid_pos]; xn = xi[next_pos]
        yp = yi[prev_pos]; ym = yi[mid_pos]; yn = yi[next_pos]

        # Chord residual at each mid point.  The ``where`` dodges
        # division by zero for degenerate (xp == xn) triplets, which
        # are excluded from the candidate set below.
        dx_pn = xn - xp
        safe_dx = np.where(dx_pn != 0.0, dx_pn, 1.0)
        y_chord = yp + (yn - yp) * (xm - xp) / safe_dx
        residual = np.abs(ym - y_chord)

        candidate = (
            (dx_pn != 0.0)
            & (residual <= tol)
            & ~protected_mask[mid_pos]
        )
        if not candidate.any():
            break

        # Greedy non-adjacent pick over the candidate array.  Indexing
        # by position-in-`active` (k = 1 .. active.size - 2) lets a
        # simple ``k - last >= 2`` test catch adjacency in the current
        # surviving curve.  The loop runs over candidate count, which
        # is bounded by ``active.size`` and shrinks every sweep.
        cand_k = np.flatnonzero(candidate)
        picks = np.empty(cand_k.size, dtype=np.int64)
        n_pick = 0
        last = -2
        for k in cand_k.tolist():
            if k - last >= 2:
                picks[n_pick] = mid_pos[k]
                n_pick += 1
                last = k
        if n_pick == 0:
            break
        alive[picks[:n_pick]] = False

    return idx[alive]


def _auto_log_y(y: np.ndarray, log_y) -> bool:
    """
    Resolve the ``log_y`` parameter into a plain boolean.

    ``log_y`` may be:

    * ``True``  – always transform.  Raises ``ValueError`` if ``y`` has
      any non-positive values, because ``log10`` would be undefined.
    * ``False`` – never transform.
    * ``"auto"`` (default) – use a log transform when every finite value
      of ``y`` is strictly positive **and** the dynamic range spans at
      least two decades (``max(y) / min(y) > 100``).  This rule keeps
      ordinary [−A, A]-style traces on the linear path and only kicks in
      for physics profiles — density, temperature, flux — where log
      handling is the natural choice.

    The log path is what makes the algorithm good for multi-decade data:
    every feature detector (bend, extrema, cumulative distance,
    persistence) then works on ``log10(y)`` so samples are balanced
    across decades instead of being dominated by the peak amplitude.
    """
    if log_y is True:
        finite = y[np.isfinite(y)]
        if finite.size and np.any(finite <= 0):
            raise ValueError(
                "log_y=True but y contains non-positive values; "
                "pass log_y=False or log_y='auto' instead."
            )
        return True
    if log_y is False:
        return False
    if log_y != "auto":
        raise ValueError(
            f"log_y must be True, False, or 'auto'; got {log_y!r}"
        )
    finite = y[np.isfinite(y)]
    if finite.size == 0 or np.any(finite <= 0):
        return False
    y_min = float(finite.min())
    if y_min <= 0:
        return False
    return float(finite.max()) / y_min > 100.0


def _x_uniform_coverage_idx(
    x: np.ndarray,
    pool_idx: np.ndarray,
    n_chunks: int,
) -> np.ndarray:
    """
    Pool indices nearest the centres of ``n_chunks`` equal-x-width chunks.

    Used by the R²-thinning step to guarantee a thin x-uniform skeleton
    in the mandatory set: global R² is amplitude-weighted, so regions
    with little y-variance contribute little to ``SS_tot`` and would
    otherwise be dropped even when the local fit is poor.  Promoting
    ~``n_chunks`` evenly-spaced pool points to mandatory fixes that.

    Parameters
    ----------
    x : np.ndarray
        Monotonically increasing x-values of the raw curve.
    pool_idx : np.ndarray
        Sorted indices into ``x`` naming the feature-pool candidates.
    n_chunks : int
        Number of equal-x-width chunks.

    Returns
    -------
    np.ndarray
        Sorted, unique indices into ``x`` (subset of ``pool_idx``).
    """
    x_pool = x[pool_idx]
    centres = np.linspace(x[0], x[-1], n_chunks + 1)
    centres = 0.5 * (centres[:-1] + centres[1:])
    ins = np.searchsorted(x_pool, centres)
    lo = np.clip(ins - 1, 0, x_pool.size - 1)
    hi = np.minimum(ins, x_pool.size - 1)
    pick_hi = np.abs(x_pool[hi] - centres) <= np.abs(x_pool[lo] - centres)
    return np.unique(pool_idx[np.where(pick_hi, hi, lo)])


def _simplify(
    x_arr: Union[np.ndarray, Sequence[float]],
    y_arr: Union[np.ndarray, Sequence[float]],
    nmin: int = 100,
    grad_inc: float = 1.0,
    warn_below_r2: Union[float, None] = 0.9,
    max_err: Union[float, None] = None,
    log_y: Union[bool, str] = "auto",
    dedup_tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Heuristic downsampling of a curve y(x) to approximately ``nmin`` points,
    preserving the most physically and visually important features.

    This is useful when a simulation or measurement produces thousands of
    data points but only a compact, faithful representation is needed for
    output, plotting, or storage.

    Algorithm overview
    ------------------
    The input is sorted by x and stagnant runs are collapsed (so
    ODE-solver output with quasi-duplicate samples doesn't dominate the
    feature pool).  Three independent strategies then select "important"
    indices — bend detection, refined sign-change extrema, and
    arc-length sampling — which are merged together with the two
    endpoints into a feature pool.  A prominence filter marks the
    visually important extrema as mandatory and x-uniform coverage
    promotes one feature-pool point per chunk so low-amplitude regions
    are never starved.  An optional greedy max-error loop bounds the
    worst-case pointwise error and a collinearity pass removes points
    that lie on the chord between their neighbours.  Finally a soft R²
    check warns when the result falls below ``warn_below_r2``.

    When ``log_y`` is active (and it auto-activates on strictly positive
    inputs spanning ≥ 2 decades) every stage below operates on
    ``log10(y)`` instead of raw ``y``.  The only effect on the caller
    is that each decade contributes equally to the arc-length sampler
    and to the error metrics, which is the right behaviour for density,
    temperature, and flux profiles.  Output arrays still carry the
    caller's original ``y`` values.

    1. **Scale-invariant bend detection**
       Menger curvature κ (the reciprocal of the circumradius of each
       consecutive triplet) is computed in *normalised* ``(x, y)``
       coordinates — both axes rescaled to ``[0, 1]`` — and a point is
       flagged when κ > ``grad_inc``.  Running in normalised coordinates
       makes ``grad_inc`` dimensionless, so the same value works at any
       axis scale.

    2. **Refined sign-change detection** (local extrema)
       Every point where the first derivative changes sign is a
       candidate local minimum / maximum.  Because ``np.sign(grad)``
       flips between consecutive samples, the true extremum lies at one
       of the two — the refinement picks whichever is more extreme
       (higher for a max, lower for a min).

    3. **Arc-length sampling** (symmetric, axis-equal)
       The differential arc length ``ds = sqrt((Δx/range_x)² +
       (Δy/range_y)²)`` is computed in normalised ``[0, 1]²`` coordinates
       so both axes contribute equally — a long gentle slope receives
       points proportional to its x-span instead of being starved by
       the y-only metric ``sum(|Δy|)``.  The cumulative arc length is
       divided into ``nmin`` equal bins and one point is selected at
       each bin boundary, giving dense sampling where the curve runs
       quickly through ``[0, 1]²`` and sparse sampling where it is
       nearly flat.

    4. **Topological-persistence filter** (mandatory extrema)
       ``_peak_prominences`` computes the prominence of every local
       extremum (the minimum descent required to reach a strictly
       higher point) in a single O(n log n) pass.  Extrema whose
       prominence exceeds 5 % of the y-range are flagged *mandatory*
       and always retained so deep dips and tall spikes never flicker
       in and out as ``nmin`` changes.

    4b. **X-uniform mandatory coverage**
       The x-domain is split into ``_COVERAGE_CHUNKS`` equal-width
       chunks and the feature-pool point nearest each chunk centre is
       promoted into the mandatory set alongside the prominent extrema.
       This thin x-uniform skeleton complements the arc-length sampler:
       even after the symmetric metric distributes points evenly, very
       short stretches of curve length per chunk could still produce
       gaps, and the coverage promotion guarantees every part of the
       x-axis keeps at least one retained point.

    5. **Greedy max-error reduction** (optional, ``max_err``)
       The arc-length pool gives a balanced output, but the worst-case
       pointwise error is not bounded.  When ``max_err`` is set, a
       greedy loop finds the original data point with the largest
       interpolation error and inserts it into the retained set,
       repeating until every original sample agrees within ``max_err``.
       Points inserted by this loop are protected from the subsequent
       collinearity prune.

    6. **Collinearity prune**
       A vectorised sweep drops any point whose y value is within
       0.1 % of the y-range of the chord between its two surviving
       neighbours; endpoints, mandatory extrema, and ``max_err``
       insertions are protected.  This removes redundant samples on
       linear and horizontal segments that survive the feature pool.

    The final output's R² (in the working y-space, log when ``log_y``
    is active) is checked against ``warn_below_r2`` and a
    ``UserWarning`` is emitted if it falls short — this is informational
    only and does not change which points are retained.

    For a perfectly flat curve (zero total arc-length), the algorithm
    falls back to uniformly spaced indices.

    Parameters
    ----------
    x_arr : array-like
        Independent variable (e.g., position, time, wavelength).
        Must be the same length as ``y_arr``.
    y_arr : array-like
        Dependent variable (e.g., temperature, density, flux).
        Must be the same length as ``x_arr``.
    nmin : int, optional
        Target *minimum* number of output samples.  Acts as the number
        of bins for the arc-length sampler; the final count may differ
        after curvature / sign-change / coverage merging, the optional
        ``max_err`` insertions, and the collinearity prune.  Clamped to
        >= 100.  Default is 100.
    grad_inc : float, optional
        Bend-sensitivity threshold for Menger curvature κ in NORMALISED
        ``(x, y)`` coordinates so it is scale-invariant — the same value
        works equally well for near-vertical drops and gentle slopes
        regardless of the axes' units.  Lower values keep more points
        (more sensitive to bends); higher values keep fewer.  Default
        is 1.0.
    warn_below_r2 : float or None, optional
        Soft quality target.  After the simplification finishes, the
        residual R² of the output (linearly interpolated back onto the
        original grid) is compared to this value and a ``UserWarning``
        is emitted if it is lower — purely informational, the output
        is not changed.  Pass ``None`` to disable the warning.  Default
        is 0.9.  When ``log_y`` is active the comparison is in log-y
        space.
    max_err : float or None, optional
        Maximum allowed absolute interpolation error.  When set, a
        greedy loop after the feature pool is built finds the original
        data point with the largest interpolation error and inserts it
        into the retained set, repeating until the worst-case error
        drops below this value.  Operates on the same y-space as the
        rest of the pipeline (log10 y when ``log_y`` is active).
        ``None`` (default) disables the constraint.
    log_y : bool or "auto", optional
        Work in log-y space for every internal feature detector and for
        the residual / warning computations.  This is the recommended
        mode for physics profiles that span multiple decades (density,
        temperature, flux) because each decade contributes equally to
        the arc-length sampler and to the error metrics.  Pass ``True``
        to force it (errors if any ``y <= 0``), ``False`` to keep the
        classic linear pipeline, or ``"auto"`` (default) to switch to
        log-y whenever every finite ``y`` is positive and the
        peak-to-trough ratio exceeds two decades.  The output arrays
        always carry the caller's original ``y`` values at the selected
        indices — only the *internal* computations are log-transformed.
    dedup_tol : float, optional
        Stagnation tolerance for collapsing near-duplicate consecutive
        samples (a common ODE-solver artefact).  Two consecutive points
        are treated as redundant when both ``|Δx| < dedup_tol * (x.max
        - x.min)`` and ``|Δy| < dedup_tol * (y.max - y.min)``; the
        second of each such pair is dropped before any feature
        detection runs.  Default ``1e-6`` — far below any meaningful
        feature scale.  Set to ``0`` to disable.

    Returns
    -------
    x_out : np.ndarray
        Downsampled independent variable.
    y_out : np.ndarray
        Downsampled dependent variable (same length as ``x_out``).

    Raises
    ------
    ValueError
        If ``x_arr`` and ``y_arr`` have different lengths.

    Examples
    --------
    **1. Basic usage with numpy arrays:**

    >>> import numpy as np
    >>> from simplify import _simplify
    >>> x = np.linspace(0, 10, 5000)
    >>> y = np.sin(x) + 0.5 * np.sin(5 * x)
    >>> x_s, y_s = _simplify(x, y, nmin=100)
    >>> print(f"Reduced {x.size} points -> {x_s.size} points")

    **2. Works with plain Python lists (no numpy needed at call site):**

    >>> x = [0.0, 0.1, 0.2, 0.3, ..., 10.0]
    >>> y = [1.2, 1.5, 1.3, 1.8, ..., 0.9]
    >>> x_s, y_s = _simplify(x, y)

    **3. Tuning sensitivity with ``grad_inc``:**

    Lower ``grad_inc`` = keep more points (more sensitive to bends).
    Higher ``grad_inc`` = keep fewer points (only the sharpest features).

    >>> # Sensitive: keep points where Menger curvature > 0.5
    >>> x_s, y_s = _simplify(x, y, nmin=100, grad_inc=0.5)
    >>> # Aggressive: only keep the sharpest bends (curvature > 2.0)
    >>> x_s, y_s = _simplify(x, y, nmin=100, grad_inc=2.0)

    **4. Reading from a file and simplifying:**

    >>> data = np.loadtxt("profile.csv", delimiter=",")
    >>> x_s, y_s = _simplify(data[:, 0], data[:, 1], nmin=200)
    >>> np.savetxt("profile_simplified.csv", np.column_stack([x_s, y_s]),
    ...            delimiter=",", header="x,y")

    **5. Using in a plotting script (e.g. with matplotlib):**

    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(0, 4 * np.pi, 10000)
    >>> y = np.exp(-0.1 * x) * np.sin(x)
    >>> x_s, y_s = _simplify(x, y, nmin=150)
    >>> plt.plot(x, y, 'k-', alpha=0.3, label=f"original ({x.size} pts)")
    >>> plt.plot(x_s, y_s, 'ro-', ms=2, label=f"simplified ({x_s.size} pts)")
    >>> plt.legend()
    >>> plt.show()

    **6. Checking simplification quality:**

    >>> x_s, y_s = _simplify(x, y, nmin=100)
    >>> metrics = _simplify_error(x, y, x_s, y_s)
    >>> print(f"RMSE = {metrics['rms_err']:.2e}")
    >>> print(f"R²   = {metrics['r_squared']:.6f}")
    >>> print(f"Compression = {metrics['compression']:.1f}x")

    **7. Visualising before/after with error panel:**

    >>> _simplify_plot(x, y, x_s, y_s, title="My data", save_path="comparison.png")

    **8. CLI usage (no scripting needed):**

    .. code-block:: bash

        python simplify.py profile.csv -o reduced.csv --nmin 200
        python simplify.py profile.csv --metrics          # print error table
        python simplify.py profile.csv --plot             # interactive plot
        python simplify.py profile.csv --animate out.gif  # animated GIF
    """
    import warnings

    # --- Input validation ---
    x = np.asarray(x_arr, dtype=float)
    y_raw = np.asarray(y_arr, dtype=float)

    # Nothing to simplify for empty arrays.
    if x.size == 0 or y_raw.size == 0:
        return x, y_raw
    if x.size != y_raw.size:
        raise ValueError(
            f"_simplify(): x and y must have the same length. "
            f"Got {x.size} and {y_raw.size}"
        )

    # --- Sort by x if not already monotone ---
    # The pipeline (np.interp, the merged-index ordering, and the
    # collinearity prune all assume x is non-decreasing.  Stable sort
    # so equal x-values preserve their input order.
    if x.size > 1 and not np.all(np.diff(x) >= 0):
        order = np.argsort(x, kind="stable")
        x = x[order]
        y_raw = y_raw[order]

    # Resolve log-y mode.  ``y`` below is the working signal that every
    # feature detector and the R² check operate on — either the
    # caller's raw array or its decimal logarithm.  The output arrays
    # always carry ``y_raw`` at the selected indices, so callers never
    # see transformed values.
    use_log = _auto_log_y(y_raw, log_y)
    y = np.log10(y_raw) if use_log else y_raw

    # --- ODE-solver stagnation collapse ---
    # ODE outputs frequently emit runs of nearly-identical (x, y)
    # samples around stiff regions; left in place they swamp every
    # downstream detector with redundant points.  Drop the trailing
    # member of every consecutive pair where both Δx and Δy are
    # below ``dedup_tol`` × the global range.  Endpoints stay pinned
    # so the output still bounds the input.
    if dedup_tol > 0 and x.size > 2:
        x_rng_full = float(np.ptp(x)) or 1.0
        y_rng_full = float(np.ptp(y)) or 1.0
        stagnant = (
            (np.abs(np.diff(x)) < dedup_tol * x_rng_full) &
            (np.abs(np.diff(y)) < dedup_tol * y_rng_full)
        )
        if np.any(stagnant):
            keep = np.ones(x.size, dtype=bool)
            keep[1:][stagnant] = False
            keep[0] = True
            keep[-1] = True
            x = x[keep]
            y = y[keep]
            y_raw = y_raw[keep]

    # If the array is already short enough, return as-is.
    if nmin >= x.size:
        return x, y_raw
    # Enforce a floor of 100 samples so the output is always useful.
    nmin = max(int(nmin), 100)

    # =====================================================================
    # Strategy 1: Scale-invariant bend detection (Menger curvature)
    # =====================================================================
    # Curvature runs in NORMALISED coordinates (x, y rescaled to [0, 1])
    # so the threshold is dimensionless — the same ``grad_inc`` value
    # fires equally on a near-vertical drop and on a gentle ramp,
    # regardless of the units on the axes.
    x_min = float(np.nanmin(x))
    y_min = float(np.nanmin(y))
    x_range = float(np.nanmax(x)) - x_min
    y_range = float(np.nanmax(y)) - y_min
    x_scale = x_range if x_range > 0 else 1.0
    y_scale = y_range if y_range > 0 else 1.0
    x_n = (x - x_min) / x_scale
    y_n = (y - y_min) / y_scale

    # Consecutive segment deltas in normalised coords.
    dx1 = np.diff(x_n[:-1])                        # x[i]   - x[i-1]
    dy1 = np.diff(y_n[:-1])                        # y[i]   - y[i-1]
    dx2 = np.diff(x_n[1:])                         # x[i+1] - x[i]
    dy2 = np.diff(y_n[1:])                         # y[i+1] - y[i]

    # κ = 2·(signed area) / (|a|·|b|·|c|) — the reciprocal of the
    # circumradius of the triangle, so tight corners give large κ.
    cross = dx1 * (dy1 + dy2) - dy1 * (dx1 + dx2)   # 2× signed area
    a = np.sqrt(dx1 * dx1 + dy1 * dy1)              # |P_{i-1}P_i|
    b = np.sqrt(dx2 * dx2 + dy2 * dy2)              # |P_iP_{i+1}|
    c_len = np.sqrt((dx1 + dx2) ** 2 + (dy1 + dy2) ** 2)  # |P_{i-1}P_{i+1}|
    denom = a * b * c_len
    np.maximum(denom, 1e-30, out=denom)             # guard degenerate triplets
    kappa = 2.0 * np.abs(cross) / denom

    important_curv = np.where(kappa > grad_inc)[0] + 1   # +1: triplet → mid

    # =====================================================================
    # Strategy 2: Refined sign-change extrema
    # =====================================================================
    # ``np.sign(grad)`` flips between consecutive samples ``sc`` and
    # ``sc+1``; the actual local extremum is at whichever has the more
    # extreme y-value (higher for a max, lower for a min).
    grad = np.gradient(y)
    sc = np.where(np.diff(np.sign(grad)) != 0)[0]
    if sc.size > 0:
        sc1 = np.minimum(sc + 1, x.size - 1)
        is_max_sc = grad[sc] > 0     # gradient was + before flip → local max
        pick_sc1 = np.where(is_max_sc, y[sc1] > y[sc], y[sc1] < y[sc])
        important_sign = np.unique(np.where(pick_sc1, sc1, sc))
    else:
        important_sign = sc

    # ---------------------------------------------------------------
    # Topological persistence: extrema whose prominence exceeds 5 %
    # of the y-range are MANDATORY (always retained so big features
    # never flicker in or out as ``nmin`` changes).  The prominence
    # pass is O(n log n) but only runs when there are extrema to
    # filter and the y-range is non-degenerate.
    # ---------------------------------------------------------------
    prom_thresh = 0.05 * y_range
    prominent_idx = np.array([], dtype=int)
    if important_sign.size > 0 and y_range > 0:
        proms_all = _peak_prominences(y, important_sign)
        prominent_idx = important_sign[proms_all >= prom_thresh]

    # =====================================================================
    # Strategy 3: Arc-length sampling (symmetric in [0, 1]² space)
    # =====================================================================
    # ``ds = sqrt((Δx/range_x)² + (Δy/range_y)²)`` treats both axes
    # equally, so a long gentle slope receives points proportional to
    # its x-span — the y-only ``sum(|Δy|)`` predecessor would have
    # starved that region and let a high-amplitude bump elsewhere
    # consume the entire budget.
    inv_x = 1.0 / x_scale
    inv_y = 1.0 / y_scale
    dx_raw = np.diff(x)
    dy_raw = np.diff(y)
    ds = np.sqrt((dx_raw * inv_x) ** 2 + (dy_raw * inv_y) ** 2)
    arc_cum = np.cumsum(ds)
    total_arc = float(arc_cum[-1]) if arc_cum.size > 0 else 0.0

    if not np.isfinite(total_arc) or total_arc == 0:
        # Perfectly flat curve (or all NaN): no arc length means
        # no feature can be distinguished, so fall back to uniform
        # spacing at the requested budget.  Return raw y so the
        # caller's original values are preserved even when log_y is on.
        idx = np.unique(np.linspace(0, x.size - 1, nmin).astype(int))
        return x[idx], y_raw[idx]

    # Bin-boundary crossings of the cumulative arc length give one
    # sample per ``total_arc / nmin`` of length traversed in [0, 1]².
    maxdist = total_arc / nmin
    bins = (arc_cum / maxdist).astype(int)
    idx_dist = np.where(bins[:-1] != bins[1:])[0]

    # =====================================================================
    # Merge feature pool: endpoints ∪ bend ∪ extrema ∪ arc samples
    # =====================================================================
    mask = np.zeros(x.size, dtype=bool)
    mask[0] = True
    mask[-1] = True
    mask[important_curv] = True
    mask[important_sign] = True
    mask[idx_dist] = True
    merged = np.where(mask)[0]

    # X-uniform mandatory coverage: even after the symmetric arc-length
    # metric, a chunk that contains very little curve length per x-span
    # could still be left empty by the bin-boundary scheme.  Promote one
    # feature-pool point per chunk into the mandatory set so every
    # part of the x-axis is covered by at least one retained point.
    coverage_idx = _x_uniform_coverage_idx(x, merged, _COVERAGE_CHUNKS)
    mandatory = np.unique(np.concatenate([prominent_idx, coverage_idx]))

    # =====================================================================
    # Strategy 5: greedy max-error reduction (optional)
    # =====================================================================
    maxerr_inserted = np.array([], dtype=int)
    if max_err is not None and merged.size > 1:
        pre_merged = merged.copy()
        in_merged = np.zeros(x.size, dtype=bool)
        in_merged[merged] = True
        while True:
            y_interp = np.interp(x, x[merged], y[merged])
            abs_res = np.abs(y - y_interp)
            worst_idx = int(np.argmax(abs_res))
            if abs_res[worst_idx] <= max_err:
                break
            if in_merged[worst_idx]:
                break
            in_merged[worst_idx] = True
            merged = np.sort(np.append(merged, worst_idx))
        maxerr_inserted = np.setdiff1d(merged, pre_merged)

    # =====================================================================
    # Strategy 6: collinearity prune
    # =====================================================================
    if merged.size > 2 and y_range > 0:
        protected = (
            np.unique(np.concatenate([mandatory, maxerr_inserted]))
            if maxerr_inserted.size > 0 else mandatory
        )
        merged = _prune_collinear(
            merged, x, y, tol=1e-3 * y_range, protected=protected,
        )

    # --- Soft R² check ---
    # Informational only: warn when the simplification fell short of
    # the requested quality, so the caller can raise ``nmin`` or lower
    # ``grad_inc``.  Computed in the working y-space (log when
    # ``log_y`` is active) to match what the algorithm optimised.
    if warn_below_r2 is not None and merged.size > 1:
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        if ss_tot > 0:
            y_interp = np.interp(x, x[merged], y[merged])
            r2 = 1.0 - float(np.sum((y - y_interp) ** 2)) / ss_tot
            if r2 < warn_below_r2:
                warnings.warn(
                    f"_simplify: R² = {r2:.4f} below warn_below_r2 = "
                    f"{warn_below_r2:.4f} (n_out={merged.size}).  "
                    f"Consider raising nmin or lowering grad_inc.",
                    UserWarning,
                    stacklevel=2,
                )

    # ``y`` above was the working (optionally log-transformed) signal
    # used for every internal decision; the output sends the caller's
    # original values back so simplification is transparent on the
    # input's native scale.
    return x[merged], y_raw[merged]


def _simplify_error(
    x_orig: Union[np.ndarray, Sequence[float]],
    y_orig: Union[np.ndarray, Sequence[float]],
    x_simp: Union[np.ndarray, Sequence[float]],
    y_simp: Union[np.ndarray, Sequence[float]],
) -> dict:
    """
    Compute error metrics comparing a simplified curve to the original.

    The simplified curve is linearly interpolated back onto the original
    x-grid, and the pointwise residuals are used to compute several
    standard error measures.

    Parameters
    ----------
    x_orig, y_orig : array-like
        Original (full-resolution) curve.
    x_simp, y_simp : array-like
        Simplified (downsampled) curve, as returned by ``_simplify()``.

    Returns
    -------
    metrics : dict
        Dictionary with the following keys:

        - ``"max_abs_err"`` : float
            Maximum absolute error (L-infinity norm).  The worst-case
            pointwise deviation between simplified and original.
        - ``"mean_abs_err"`` : float
            Mean absolute error (MAE).  Average pointwise deviation.
        - ``"rms_err"`` : float
            Root-mean-square error (RMSE).  Penalises large deviations
            more than MAE.
        - ``"max_rel_err"`` : float
            Maximum relative error, ``max(|residual| / |y_orig|)``,
            skipping points where ``|y_orig| < 1e-30``.  Useful when
            the signal spans many orders of magnitude.
        - ``"r_squared"`` : float
            Coefficient of determination (R^2).  1.0 = perfect
            reconstruction; values close to 1.0 indicate the simplified
            curve captures nearly all variance of the original.
        - ``"compression"`` : float
            Compression ratio, ``len(x_orig) / len(x_simp)``.  Higher
            means more aggressive downsampling.
        - ``"n_orig"`` : int
            Number of points in the original curve.
        - ``"n_simp"`` : int
            Number of points in the simplified curve.
        - ``"log_r_squared"`` : float
            R² computed in decimal-log y-space against a log-linear
            reconstruction (straight line between samples in
            ``(x, log10 y)``).  Decade-balanced — the right fidelity
            measure for density / temperature / flux profiles that
            span multiple orders of magnitude.  ``nan`` if any
            ``y <= 0``.
        - ``"log_rms_err"`` : float
            RMSE of the log10 residual, in dex.  ``nan`` if any
            ``y <= 0``.
        - ``"log_max_dex_err"`` : float
            Worst-case log-space deviation in dex (L∞).  ``0.01`` ≈ 2 %
            multiplicative; ``0.1`` ≈ 26 %; ``1.0`` = one whole decade
            off.  ``nan`` if any ``y <= 0``.
        - ``"log_mean_dex_err"`` : float
            Mean absolute log-space deviation in dex.  ``nan`` if any
            ``y <= 0``.

    Examples
    --------
    >>> x = np.linspace(0, 10, 5000)
    >>> y = np.sin(x)
    >>> x_s, y_s = _simplify(x, y)
    >>> metrics = _simplify_error(x, y, x_s, y_s)
    >>> print(f"RMSE = {metrics['rms_err']:.2e}, R² = {metrics['r_squared']:.6f}")
    """
    x_o = np.asarray(x_orig, dtype=float)
    y_o = np.asarray(y_orig, dtype=float)
    x_s = np.asarray(x_simp, dtype=float)
    y_s = np.asarray(y_simp, dtype=float)

    # Linear reconstruction: the naive "connect the dots" interpretation
    # of the simplified points.  All linear-space metrics below are
    # taken against this.
    y_interp = np.interp(x_o, x_s, y_s)
    residual = y_o - y_interp

    # --- Linear-space error metrics ---
    max_abs = float(np.max(np.abs(residual)))
    mean_abs = float(np.mean(np.abs(residual)))
    rms = float(np.sqrt(np.mean(residual ** 2)))

    # Relative error (skip near-zero original values to avoid division blow-up).
    eps = 1e-30
    mask = np.abs(y_o) > eps
    if np.any(mask):
        max_rel = float(np.max(np.abs(residual[mask]) / np.abs(y_o[mask])))
    else:
        max_rel = 0.0

    # R² (coefficient of determination) in linear space.
    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum((y_o - np.mean(y_o)) ** 2)
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0

    metrics = {
        "max_abs_err": max_abs,
        "mean_abs_err": mean_abs,
        "rms_err": rms,
        "max_rel_err": max_rel,
        "r_squared": r_squared,
        "compression": float(x_o.size) / float(x_s.size) if x_s.size > 0 else float("inf"),
        "n_orig": int(x_o.size),
        "n_simp": int(x_s.size),
    }

    # --- Log-space error metrics -------------------------------------
    # When every y is strictly positive we can also report metrics in
    # decimal-log space, which is what callers with density /
    # temperature / flux profiles actually care about (each decade
    # contributes equally).  The reconstruction used for the log
    # metrics is *log-linear* — i.e. a straight line between samples
    # in ``(x, log10 y)`` — because that is how matplotlib's log-y
    # plots render segments and is also how the log-y path of
    # _simplify judges quality internally.  All four fields are set to
    # NaN when the log transform is undefined (any y <= 0).
    log_keys = ("log_r_squared", "log_rms_err", "log_max_dex_err",
                "log_mean_dex_err")
    if np.all(y_o > 0) and np.all(y_s > 0) and y_o.size > 0 and x_s.size > 0:
        log_y_o = np.log10(y_o)
        log_interp = np.interp(x_o, x_s, np.log10(y_s))
        log_res = log_y_o - log_interp
        ss_res_log = float(np.sum(log_res ** 2))
        ss_tot_log = float(np.sum((log_y_o - np.mean(log_y_o)) ** 2))
        metrics["log_r_squared"] = (
            1.0 - ss_res_log / ss_tot_log if ss_tot_log > 0 else 1.0
        )
        metrics["log_rms_err"] = float(np.sqrt(np.mean(log_res ** 2)))
        metrics["log_max_dex_err"] = float(np.max(np.abs(log_res)))
        metrics["log_mean_dex_err"] = float(np.mean(np.abs(log_res)))
    else:
        for k in log_keys:
            metrics[k] = float("nan")

    return metrics


def _simplify_plot(
    x_orig: Union[np.ndarray, Sequence[float]],
    y_orig: Union[np.ndarray, Sequence[float]],
    x_simp: Union[np.ndarray, Sequence[float]],
    y_simp: Union[np.ndarray, Sequence[float]],
    title: str = "Curve simplification",
    save_path: Union[str, None] = None,
    show: bool = True,
) -> None:
    """
    Visualise original vs. simplified curves with an error panel.

    Produces a two-panel figure:

    - **Top panel**: original curve (grey line) overlaid with simplified
      points (red dots + line).
    - **Bottom panel**: pointwise residual (original minus linearly
      interpolated simplified curve), with max/mean/RMS error annotated.

    Parameters
    ----------
    x_orig, y_orig : array-like
        Original (full-resolution) curve.
    x_simp, y_simp : array-like
        Simplified (downsampled) curve.
    title : str, optional
        Figure title.  Default: ``"Curve simplification"``.
    save_path : str or None, optional
        If given, save the figure to this file path (e.g. ``"plot.png"``).
        The format is inferred from the extension.  Default: ``None`` (no save).
    show : bool, optional
        Whether to call ``plt.show()``.  Set to ``False`` in non-interactive
        environments or when saving only.  Default: ``True``.

    Examples
    --------
    >>> x = np.linspace(0, 10, 5000)
    >>> y = np.sin(x) + 0.5 * np.sin(5 * x)
    >>> x_s, y_s = _simplify(x, y, nmin=100)
    >>> _simplify_plot(x, y, x_s, y_s, save_path="comparison.png")
    """
    import matplotlib.pyplot as plt

    with plt.style.context(str(_STYLE_FILE)):
        x_o = np.asarray(x_orig, dtype=float)
        y_o = np.asarray(y_orig, dtype=float)
        x_s = np.asarray(x_simp, dtype=float)
        y_s = np.asarray(y_simp, dtype=float)

        # Compute metrics and residual.
        metrics = _simplify_error(x_o, y_o, x_s, y_s)
        residual = y_o - np.interp(x_o, x_s, y_s)

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(7, 5.5), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
            layout="constrained",
        )

        # --- Top panel: curves ---
        ax1.plot(x_o, y_o, "-", color="0.6", lw=0.8,
                 label=f"original ({metrics['n_orig']} pts)")
        ax1.plot(x_s, y_s, "o-", color="tab:red", ms=2.5, lw=0.8,
                 label=f"simplified ({metrics['n_simp']} pts)")
        ax1.set_ylabel(r"$y$")
        ax1.set_title(title)
        ax1.legend(loc="best")

        # --- Bottom panel: residual ---
        ax2.fill_between(x_o, residual, 0, color="tab:blue", alpha=0.3)
        ax2.plot(x_o, residual, "-", color="tab:blue", lw=0.5)
        ax2.axhline(0, color="k", lw=0.5, ls="--")
        ax2.set_ylabel(r"residual")
        ax2.set_xlabel(r"$x$")

        # Annotate with key metrics inside the residual panel.
        info = (
            rf"RMSE $= {metrics['rms_err']:.2e}$    "
            rf"MAE $= {metrics['mean_abs_err']:.2e}$    "
            rf"$R^2 = {metrics['r_squared']:.6f}$    "
            rf"compression $= {metrics['compression']:.1f}\times$"
        )
        ax2.text(
            0.5, 0.02, info,
            transform=ax2.transAxes, ha="center", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="0.8",
                      alpha=0.9),
        )

        if save_path is not None:
            fig.savefig(save_path)
            print(f"Figure saved to '{save_path}'.")

        if show:
            plt.show()
        else:
            plt.close(fig)


def _simplify_animate(
    x_orig: Union[np.ndarray, Sequence[float]],
    y_orig: Union[np.ndarray, Sequence[float]],
    save_path: str = "simplify.gif",
    fps: int = 30,
    duration: float = 6.0,
    title: str = "Curve simplification",
    n_steps: int = 30,
    r2_target: float = 0.9,
    max_err: Union[float, None] = None,
    n_chunks: int = 20,
    xlabel: str = r"$x$",
    ylabel: str = r"$y$",
) -> None:
    """
    Create an animated GIF showing progressive curve simplification.

    The animation builds the simplified curve up from 5 points to the
    full feature-detected set, using the same mandatory / hierarchical-
    bisection ordering as :func:`_simplify` so that frames are strictly
    nested (each frame is a superset of the previous one).  Three panels
    are shown:

    - **Top panel**: the underlying curve as a thin grey line, with the
      current simplified points overlaid as red dots + line.
    - **Middle panel**: signed residual ``y - y_interp`` vs x, showing
      *where* and in which direction the approximation deviates.  When
      ``max_err`` is set, horizontal dashed lines mark the ± threshold.
    - **Bottom panel**: log-log RMSE vs. number of retained points,
      with dashed reference hlines at the RMSE corresponding to
      R² = 0.9, 0.99, 0.999 (equally spaced half-decade rungs), and a
      persistent vertical line at the ``n`` where ``r2_target`` is
      first reached.

    Parameters
    ----------
    x_orig, y_orig : array-like
        Original (full-resolution) curve.
    save_path : str, optional
        Output file path.  Default: ``"simplify.gif"``.
        Also supports ``.mp4`` if ``ffmpeg`` is installed.
    fps : int, optional
        Frames per second.  Default: 30.
    duration : float, optional
        Total animation duration in seconds.  Default: 6.0.
    title : str, optional
        Figure title.  Default: ``"Curve simplification"``.
    n_steps : int, optional
        Number of distinct simplification levels to sweep through,
        log-spaced from 5 up to the total number of feature points.
        Default: 30.
    r2_target : float, optional
        Quality target used to mark the vertical "R² reached" line in
        the bottom panel.  Does not affect which frames are rendered.
        Default: 0.9.

    Examples
    --------
    >>> x = np.linspace(0, 10, 5000)
    >>> y = np.sin(x) + 0.5 * np.sin(5 * x)
    >>> _simplify_animate(x, y, "demo.gif", duration=6.0)
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    plt.style.use(str(_STYLE_FILE))

    x_o = np.asarray(x_orig, dtype=float)
    y_o = np.asarray(y_orig, dtype=float)
    n_orig = x_o.size

    # --- Precompute simplification at increasing point counts ---
    # Start from 5 points and build up to the full feature-detected set.
    # First get all feature-detected indices (no R² thinning).
    x_full, y_full = _simplify(x_o, y_o, warn_below_r2=None)
    n_full = x_full.size

    # Generate log-spaced point counts from 5 up to full.
    pt_counts = np.unique(
        np.logspace(np.log10(5), np.log10(max(n_full, 6)), n_steps)
        .astype(int)
    )
    pt_counts = np.unique(np.concatenate([[5], pt_counts, [n_full]]))
    pt_counts = np.sort(pt_counts)  # ascending: few → many

    # Build each simplification level using the same mandatory/optional
    # split _simplify uses internally: prominent extrema are mandatory
    # (always present, even at the smallest frame), and the remaining
    # feature-detected indices are added in hierarchical-bisection order.
    full_idx = np.sort(np.searchsorted(x_o, x_full))

    # Identify topologically persistent extrema (mandatory set) using
    # the same prominence-threshold rule as _simplify.
    grad_o = np.gradient(y_o)
    sign_idx = np.where(np.diff(np.sign(grad_o)) != 0)[0]
    y_rng = float(np.nanmax(y_o) - np.nanmin(y_o))
    prom_thresh = 0.05 * y_rng
    if sign_idx.size > 0 and y_rng > 0:
        proms = _peak_prominences(y_o, sign_idx)
        prominent_idx = sign_idx[proms >= prom_thresh]
    else:
        prominent_idx = np.array([], dtype=int)

    # Hierarchical bisection of full_idx positions.
    n_f = len(full_idx)
    order = np.empty(n_f, dtype=int)
    order[0] = 0
    order[1] = n_f - 1
    count = 2
    queue = [(0, n_f - 1)]
    while queue:
        nq = []
        for lo_q, hi_q in queue:
            if hi_q - lo_q <= 1:
                continue
            mid_q = (lo_q + hi_q) // 2
            order[count] = mid_q
            count += 1
            nq.append((lo_q, mid_q))
            nq.append((mid_q, hi_q))
        queue = nq
    bisection_pool = full_idx[order[:count]]

    # Mandatory set mirrors _simplify: prominent extrema ∪ x-uniform
    # coverage skeleton, so gradual low-amplitude regions keep a
    # representative point in every animation frame.
    coverage_idx = _x_uniform_coverage_idx(x_o, full_idx, _COVERAGE_CHUNKS)
    mandatory = np.unique(np.concatenate([prominent_idx, coverage_idx]))
    optional = bisection_pool[~np.isin(bisection_pool, mandatory)]

    steps = []
    for npts in pt_counts:
        # Every frame is mandatory ∪ optional[:k] — the mandatory set
        # is present from frame 1, so big features never flicker in
        # and out across the animation.
        k = max(0, int(npts) - int(mandatory.size))
        k = min(k, len(optional))
        trial = np.sort(np.concatenate([mandatory, optional[:k]]))
        x_s, y_s = x_o[trial], y_o[trial]
        y_interp = np.interp(x_o, x_s, y_s)
        residual = y_o - y_interp
        m = _simplify_error(x_o, y_o, x_s, y_s)
        steps.append({
            "npts": m["n_simp"],
            "x": x_s,
            "y": y_s,
            "rms": m["rms_err"],
            "r2": m["r_squared"],
            "residual": residual,
            "max_err": float(np.abs(residual).max()),
        })

    # Deduplicate steps with same npts (can happen when _simplify returns
    # more points than nmin).
    seen = set()
    unique_steps = []
    for s in steps:
        if s["npts"] not in seen:
            seen.add(s["npts"])
            unique_steps.append(s)
    steps = unique_steps

    # Collect arrays for the error subplot.
    all_npts = np.array([s["npts"] for s in steps])
    all_rms = np.array([s["rms"] for s in steps])

    # --- Animation timing ---
    # Each step gets equal time, plus a short hold on the last frame.
    n_anim_steps = len(steps)
    hold_frac = 0.15  # fraction of duration to hold the final frame
    sweep_duration = duration * (1 - hold_frac)
    hold_duration = duration * hold_frac
    n_frames = int(fps * duration)
    sweep_frames = int(fps * sweep_duration)

    # --- Set up figure ---
    fig, (ax, ax_res, ax_err) = plt.subplots(
        3, 1, figsize=(7, 7.5),
        gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.35},
    )
    margin = 0.05 * (np.nanmax(y_o) - np.nanmin(y_o) + 1e-30)
    ax.set_xlim(x_o[0], x_o[-1])
    ax.set_ylim(np.nanmin(y_o) - margin, np.nanmax(y_o) + margin)
    ax.set_ylabel(ylabel)
    ax.tick_params(labelbottom=False)
    ax.set_title(title)

    # Thin underlying curve (always visible).
    ax.plot(x_o, y_o, "-", color="0.75", lw=0.8, zorder=1)

    # Simplified dots + connecting line (updated each frame).
    line_simp, = ax.plot([], [], "-", color="tab:red", lw=0.6, zorder=2)
    scatter_simp = ax.scatter([], [], s=6, color="tab:red", zorder=3,
                              edgecolors="none")
    info_text = ax.text(
        0.98, 0.96, "", transform=ax.transAxes, ha="right", va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9),
    )

    # --- Residual subplot: (y - y_interp) vs x ---
    all_max_errs = np.array([s["max_err"] for s in steps])
    res_ceil = float(np.max(all_max_errs)) * 1.2
    ax_res.set_xlim(x_o[0], x_o[-1])
    ax_res.set_ylim(-res_ceil, res_ceil)
    ax_res.set_xlabel(xlabel)
    ax_res.set_ylabel(r"$y - y_{\mathrm{interp}}$")
    ax_res.axhline(0, color="0.5", ls="-", lw=0.4)
    if max_err is not None:
        ax_res.axhline(max_err, color="tab:green", ls="--", lw=1.0,
                       alpha=0.8, label=rf"$\pm\,${max_err:g}")
        ax_res.axhline(-max_err, color="tab:green", ls="--", lw=1.0,
                       alpha=0.8)
        ax_res.legend(loc="upper right", fontsize=8)
    res_fill = ax_res.fill_between(x_o, 0, np.zeros_like(x_o),
                                   color="tab:orange", alpha=0.4)
    res_line, = ax_res.plot([], [], "-", color="tab:orange", lw=0.6)

    # --- Error subplot: log-RMSE vs n_points (log-log) ---
    sigma_y = float(np.sqrt(np.mean((y_o - np.mean(y_o)) ** 2)))
    r2_levels = (0.9, 0.99, 0.999)
    rms_levels = [sigma_y * np.sqrt(1.0 - r) for r in r2_levels]

    rms_positive = all_rms[all_rms > 0]
    rms_floor = (min(rms_positive.min(), min(rms_levels)) * 0.5
                 if rms_positive.size > 0 else min(rms_levels) * 0.5)
    rms_ceil = (max(all_rms.max(), max(rms_levels)) * 2.0
                if rms_positive.size > 0 else max(rms_levels) * 2.0)
    ax_err.set_xlim(max(all_npts.min() * 0.7, 1), all_npts.max() * 1.3)
    ax_err.set_ylim(rms_floor, rms_ceil)
    ax_err.set_xscale("log")
    ax_err.set_yscale("log")
    ax_err.set_xlabel(r"Number of points $n$")
    ax_err.set_ylabel(r"RMSE")

    # Reference hlines at R^2 = 0.9 / 0.99 / 0.999.
    for r2_lvl, rms_lvl in zip(r2_levels, rms_levels):
        ax_err.axhline(rms_lvl, color="0.6", ls=":", lw=0.8, zorder=0)
        ax_err.text(
            all_npts.max() * 1.25, rms_lvl,
            rf"$R^2={r2_lvl:g}$",
            va="center", ha="right", fontsize=8, color="0.4",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                      alpha=0.8),
        )

    # --- Find the step where R² target is first reached ---
    all_r2 = np.array([s["r2"] for s in steps])
    r2_hit_idx = None
    r2_hit_npts = None
    if r2_target is not None:
        for i, s in enumerate(steps):
            if s["r2"] >= r2_target:
                r2_hit_idx = i
                r2_hit_npts = s["npts"]
                break

    # Draw persistent vertical line at R² target in bottom panel with label.
    if r2_hit_npts is not None:
        ax_err.axvline(r2_hit_npts, color="tab:green", ls="--", lw=1.2,
                       alpha=0.8, zorder=1,
                       label=f"$R^2 \\geq {r2_target}$ at $n={r2_hit_npts}$")
        ax_err.legend(loc="upper right")

    err_line, = ax_err.plot([], [], "o-", color="tab:blue", ms=3, lw=1.0)
    err_marker = ax_err.scatter([], [], s=40, color="tab:red", zorder=5,
                                edgecolors="black", linewidths=0.5)

    def _update(frame):
        nonlocal res_fill
        if frame < sweep_frames:
            # Map frame to step index.
            step_idx = int(frame / max(sweep_frames - 1, 1) * (n_anim_steps - 1))
            step_idx = min(step_idx, n_anim_steps - 1)
        else:
            # Hold phase — stay on last step.
            step_idx = n_anim_steps - 1

        s = steps[step_idx]

        # --- Top panel: update simplified points ---
        line_simp.set_data(s["x"], s["y"])
        scatter_simp.set_offsets(np.column_stack([s["x"], s["y"]]))
        info_text.set_text(
            f"$n = {s['npts']}$    "
            f"$R^2 = {s['r2']:.6f}$    "
            f"max $|\\Delta y| = {s['max_err']:.4f}$"
        )

        # --- Middle panel: update residual ---
        res_fill.remove()
        res_fill = ax_res.fill_between(x_o, 0, s["residual"],
                                       color="tab:orange", alpha=0.4)
        res_line.set_data(x_o, s["residual"])

        # --- Bottom panel: build up error curve ---
        # Show all steps up to current.
        vis_npts = all_npts[:step_idx + 1]
        vis_rms = all_rms[:step_idx + 1]
        err_line.set_data(vis_npts, vis_rms)
        # Highlight current point.
        err_marker.set_offsets([[s["npts"], s["rms"]]])

        return line_simp, scatter_simp, info_text, res_fill, res_line, err_line, err_marker


    anim = FuncAnimation(fig, _update, frames=n_frames, blit=False)

    # Save — use pillow for GIF, ffmpeg for mp4.
    if save_path.lower().endswith(".mp4"):
        writer = "ffmpeg"
    else:
        writer = "pillow"
    anim.save(save_path, writer=writer, fps=fps, dpi=120)
    plt.close(fig)

    # Restore default rcParams.
    plt.rcParams.update(plt.rcParamsDefault)

    print(f"Animation saved to '{save_path}' ({n_frames} frames, {duration:.1f}s).")


# =============================================================================
# Random test-curve generator
# =============================================================================
def _random_test_curve(
    npts: int = 10_000,
    seed: Union[int, None] = None,
    noise: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a random 1-D curve that exercises every simplification strategy.

    The curve is built from several additive components, each designed to
    trigger a different branch of ``_simplify``:

    - **Smooth base** – sum of 3–6 sinusoids with random frequencies,
      amplitudes and phases.  Produces gentle curvature and many local
      extrema (tests sign-change detection).
    - **Flat plateaus** – 1–3 constant-value segments spliced into the
      curve.  These are "redundant" regions where cumulative-distance
      sampling should thin aggressively.
    - **Sharp spikes** – 2–5 narrow Gaussian pulses of random height.
      Their steep flanks trigger the bend detector.
    - **Step discontinuities** – 1–3 Heaviside-like jumps (smoothed over
      a handful of points) that create abrupt level shifts.
    - **Gaussian noise** – low-amplitude noise added everywhere when
      ``noise=True``, so the algorithm must distinguish real features
      from jitter.  Set ``noise=False`` to produce the deterministic
      feature skeleton with no added jitter; useful for timing runs
      where noise-driven sign flips would otherwise flood the feature
      pool and dominate the profile.

    Parameters
    ----------
    npts : int, optional
        Number of points in the output curve.  Default: 10 000.
    seed : int or None, optional
        Random seed for reproducibility.  ``None`` gives a different curve
        each time.
    noise : bool, optional
        If ``True`` (default) add Gaussian noise at 0.1 % of the curve's
        peak-to-peak range.  Set ``False`` for a noise-free reference.

    Returns
    -------
    x : np.ndarray, shape (npts,)
        Independent variable on [0, 1].
    y : np.ndarray, shape (npts,)
        Dependent variable (the composite random curve).

    Examples
    --------
    >>> x, y = _random_test_curve(npts=8000, seed=42)
    >>> x_s, y_s = _simplify(x, y, nmin=120)
    >>> _simplify_plot(x, y, x_s, y_s, title="random test curve")
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, npts)
    y = np.zeros_like(x)

    # ------------------------------------------------------------------
    # 1. Smooth sinusoidal base (sign-change + gentle curvature)
    # ------------------------------------------------------------------
    n_sines = rng.integers(2, 5)                       # 2–4 terms
    for _ in range(n_sines):
        freq  = rng.uniform(1.0, 12.0)                # cycles across [0,1]
        amp   = rng.uniform(0.3, 2.0)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        y += amp * np.sin(2.0 * np.pi * freq * x + phase)

    # ------------------------------------------------------------------
    # 2. Flat plateaus (redundant regions)
    #    Use a smooth blend so the entry/exit don't create artificial
    #    discontinuities that swamp the curvature detector.
    # ------------------------------------------------------------------
    n_plateaus = rng.integers(1, 4)                    # 1–3 plateaus
    for _ in range(n_plateaus):
        width = rng.uniform(0.05, 0.15)               # 5–15 % of domain
        centre = rng.uniform(width, 1.0 - width)
        level = rng.uniform(np.min(y), np.max(y))
        # Smooth top-hat: product of two logistics (rise then fall).
        blend_k = npts / 15                            # transition ~15 pts
        rise = 1.0 / (1.0 + np.exp(np.clip(-blend_k * (x - (centre - width / 2)), -500, 500)))
        fall = 1.0 / (1.0 + np.exp(np.clip( blend_k * (x - (centre + width / 2)), -500, 500)))
        window = rise * fall
        y = y * (1.0 - window) + level * window

    # ------------------------------------------------------------------
    # 3. Sharp spikes (bend detector)
    # ------------------------------------------------------------------
    n_spikes = rng.integers(2, 6)                      # 2–5 spikes
    for _ in range(n_spikes):
        loc   = rng.uniform(0.05, 0.95)
        sigma = rng.uniform(0.002, 0.008)              # narrow but resolved
        amp   = rng.uniform(2.0, 6.0) * rng.choice([-1, 1])
        y += amp * np.exp(-0.5 * ((x - loc) / sigma) ** 2)

    # ------------------------------------------------------------------
    # 4. Step discontinuities (sharp level shifts)
    #    Steepness is set so the transition spans ~20 grid points —
    #    sharp enough to trigger the curvature detector but not so steep
    #    that hundreds of transition points are all flagged.
    # ------------------------------------------------------------------
    n_steps = rng.integers(1, 4)                       # 1–3 steps
    for _ in range(n_steps):
        loc    = rng.uniform(0.10, 0.90)
        height = rng.uniform(1.0, 4.0) * rng.choice([-1, 1])
        transition_pts = 20                            # grid points across step
        steepness = npts / max(transition_pts, 1)
        arg = np.clip(-steepness * (x - loc), -500.0, 500.0)
        y += height / (1.0 + np.exp(arg))

    # ------------------------------------------------------------------
    # 5. Optional Gaussian noise — a realistic jitter floor so the
    #    noise-level filters (persistence-threshold on extrema,
    #    collinearity prune) get exercised.  Off ⇒ a clean reference
    #    curve whose sign-change detector fires at most a few hundred
    #    times, making the fast path easy to time.
    # ------------------------------------------------------------------
    if noise:
        noise_amp = 0.001 * (np.max(y) - np.min(y) + 1e-30)
        y += rng.normal(0.0, noise_amp, size=npts)

    return x, y


def _load_sb99_5myr() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the bundled Starburst99 SED at 5 Myr.

    Returns the luminosity curve as ``(x, y) = (log10 wavelength [Å],
    log10 L_λ [erg / s / Å])``.  Working in log–log coordinates lets the
    linear-axis plotters and the core simplifier treat the curve
    uniformly across its ~4 decades of wavelength and ~6 dex dynamic
    range, exactly as spectral-energy-distribution plots are
    conventionally displayed.

    The 5 Myr column has no missing-data sentinels so no masking is
    needed.  The other age columns in the Starburst99 table are
    intentionally discarded at extract time and are not available here.

    Returns
    -------
    x : np.ndarray, shape (1221,)
        ``log10(wavelength [Å])``, monotonically increasing from ~1.96
        (91 Å) to 6.20 (1.6 × 10⁶ Å).
    y : np.ndarray, shape (1221,)
        ``log10(L_λ [erg / s / Å])`` at 5 Myr after an instantaneous
        burst (Z = Z_☉, Kroupa IMF, default Starburst99 tracks).

    Examples
    --------
    >>> x, y = _load_sb99_5myr()
    >>> x_s, y_s = _simplify(x, y, r2_target=0.99)
    """
    data = np.loadtxt(_SB99_5MYR_FILE, comments="#")
    wavelength = data[:, 0]
    log_lum    = data[:, 1]
    return np.log10(wavelength), log_lum


# =============================================================================
# CLI entry point
# =============================================================================
def _simplify_cli():
    """
    Command-line interface for the _simplify curve downsampling function.

    Reads x and y data from a two-column text file (whitespace- or
    comma-separated), runs the simplification algorithm, and writes the
    reduced data to an output file.

    Usage
    -----
    python simplify.py input.csv -o output.csv --nmin 150 --grad-inc 0.5
    python simplify.py input.csv --metrics --plot
    python simplify.py input.csv --animate simplify.gif --animate-duration 5

    Positional arguments
    --------------------
    infile : str
        Path to input data file.  Must contain two columns (x, y).
        Lines starting with '#' are treated as comments and skipped.
        Both whitespace- and comma-delimited formats are accepted.

    Optional arguments
    ------------------
    -o, --output : str
        Path to the output file.  Default: ``simplified_output.csv``.
    --nmin : int
        Minimum number of output samples (default: 100).
    --grad-inc : float
        Bend sensitivity (default: 1.0, dimensionless in normalised coords).
    --metrics : flag
        Print error metrics (RMSE, MAE, R², etc.) after simplification.
    --plot : flag
        Show an interactive before/after comparison plot with residuals.
    --plot-save : str
        Save the comparison plot to a file (e.g. ``comparison.png``).
    --animate : str
        Save an animated GIF/MP4 showing the original curve morphing
        into the simplified points (e.g. ``simplify.gif``).
    --animate-duration : float
        Animation duration in seconds (default: 3.0).
    --animate-fps : int
        Animation frames per second (default: 30).
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="_simplify",
        description=(
            "Downsample a two-column (x, y) data file while preserving "
            "sharp features, local extrema, and overall curve shape."
        ),
        epilog=(
            "Example:\n"
            "  python simplify.py data.csv -o reduced.csv --nmin 200\n\n"
            "The algorithm combines eight stages:\n"
            "  1.  Scale-invariant bend detection (Menger curvature +\n"
            "      turning angle in normalised coords, gated by --grad-inc)\n"
            "  2.  Sign-change detection (local extrema, noise-filtered)\n"
            "  3.  Cumulative-distance sampling (uniform arc-length in y)\n"
            "  4.  Topological-persistence filter (mandatory prominent extrema)\n"
            "  4b. X-uniform mandatory coverage (--n-chunks even-x skeleton)\n"
            "  5.  R²-based thinning (galloping + binary search to --r2-target)\n"
            "  5b. Greedy max-error reduction (--max-err worst-case bound)\n"
            "  6.  Collinearity prune (drops points on the chord between\n"
            "      their neighbours)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "infile",
        nargs="?",
        default=None,
        help="Path to input data file (two columns: x y).  "
             "Not required when --random is used.",
    )
    parser.add_argument(
        "-o", "--output",
        default="simplified_output.csv",
        help="Path to output file (default: simplified_output.csv).",
    )
    parser.add_argument(
        "--nmin",
        type=int,
        default=100,
        help="Minimum number of output samples (default: 100).",
    )
    parser.add_argument(
        "--grad-inc",
        type=float,
        default=1.0,
        help="Bend sensitivity (dimensionless, scale-invariant).  "
             "Fires when normalised Menger curvature > grad_inc or turning "
             "angle > 0.1 * grad_inc rad.  Default: 1.0.",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Print error metrics (RMSE, MAE, R², etc.) after simplification.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show a before/after comparison plot with residuals.",
    )
    parser.add_argument(
        "--plot-save",
        default=None,
        metavar="PATH",
        help="Save the comparison plot to a file (e.g. comparison.png).",
    )
    parser.add_argument(
        "--animate",
        default=None,
        metavar="PATH",
        help=(
            "Save an animated GIF/MP4 showing the original curve morphing "
            "into the simplified points (e.g. simplify.gif)."
        ),
    )
    parser.add_argument(
        "--animate-duration",
        type=float,
        default=3.0,
        help="Animation duration in seconds (default: 3.0).",
    )
    parser.add_argument(
        "--animate-fps",
        type=int,
        default=30,
        help="Animation frames per second (default: 30).",
    )
    parser.add_argument(
        "--r2-target",
        type=float,
        default=0.9,
        help="Target R² quality (default: 0.9).  Points are added until "
             "R² reaches this value.  Use None to keep all detected points.",
    )
    parser.add_argument(
        "--max-err",
        type=float,
        default=None,
        help="Maximum allowed absolute interpolation error.  After R²-based "
             "thinning, a greedy loop inserts points until the worst-case "
             "error drops below this value.  Default: None (no constraint).",
    )
    parser.add_argument(
        "--log-y",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "Work in log-y space for every internal feature detector "
            "and the R² thinning step.  'auto' (default) switches to "
            "log when every y > 0 and max(y)/min(y) > 100 — the right "
            "behaviour for density / temperature / flux profiles.  "
            "Use 'on' to force, 'off' to keep the classic linear path."
        ),
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help=(
            "Generate a random test curve instead of reading a file.  "
            "The curve contains sinusoidal bases, flat plateaus, sharp "
            "spikes, step discontinuities, and noise — designed to "
            "exercise every simplification strategy."
        ),
    )
    parser.add_argument(
        "--randomSB99",
        action="store_true",
        help=(
            "Use the bundled Starburst99 SED at 5 Myr as the input "
            "curve (log10 wavelength vs log10 luminosity) instead of "
            "reading a file.  A real-data example: broad UV bump, "
            "Balmer/Paschen jumps, and a smooth Rayleigh–Jeans tail "
            "across ~4 decades of wavelength."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for --random (default: None = non-reproducible).",
    )
    parser.add_argument(
        "--random-npts",
        type=int,
        default=10_000,
        help="Number of points in the random curve (default: 10000).",
    )
    # --noise / --no-noise: whether --random should add Gaussian jitter.
    # BooleanOptionalAction (Python 3.9+) gives both spellings from one
    # argument and carries the default forward for --help output.
    parser.add_argument(
        "--noise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add Gaussian noise (0.1 %% of y-range) to the --random "
             "curve.  Pass --no-noise for a deterministic, jitter-free "
             "reference that is much faster to simplify because the "
             "sign-change detector fires at most a handful of times.",
    )

    args = parser.parse_args()

    # --- Obtain input data ---
    if args.random and args.randomSB99:
        parser.error("--random and --randomSB99 are mutually exclusive.")
    if args.random:
        x, y = _random_test_curve(
            npts=args.random_npts, seed=args.seed, noise=args.noise,
        )
        seed_str = f", seed={args.seed}" if args.seed is not None else ""
        noise_str = "" if args.noise else ", no noise"
        source_label = (
            f"random curve ({args.random_npts} pts{seed_str}{noise_str})"
        )
        print(f"Generated {source_label}.")
    elif args.randomSB99:
        x, y = _load_sb99_5myr()
        source_label = f"Starburst99 SED at 5 Myr ({x.size} pts)"
        print(f"Loaded {source_label}.")
    elif args.infile is not None:
        # Try comma-delimited first, fall back to whitespace.
        try:
            data = np.loadtxt(args.infile, delimiter=",", comments="#")
        except ValueError:
            data = np.loadtxt(args.infile, comments="#")

        if data.ndim != 2 or data.shape[1] < 2:
            raise SystemExit(
                f"Error: expected at least 2 columns in '{args.infile}', "
                f"got shape {data.shape}."
            )
        x, y = data[:, 0], data[:, 1]
        source_label = f"'{args.infile}'"
    else:
        parser.error(
            "either provide an input file, use --random, or use --randomSB99."
        )

    # --- Simplify ---
    log_y_arg = {"auto": "auto", "on": True, "off": False}[args.log_y]
    x_out, y_out = _simplify(
        x, y, nmin=args.nmin, grad_inc=args.grad_inc,
        warn_below_r2=args.r2_target, max_err=args.max_err, log_y=log_y_arg,
    )

    # --- Write output ---
    np.savetxt(
        args.output,
        np.column_stack([x_out, y_out]),
        delimiter=",",
        header="x,y",
        comments="# ",
    )
    print(
        f"Simplified {x.size} points -> {x_out.size} points.  "
        f"Written to '{args.output}'."
    )

    # --- Optional: print error metrics ---
    if args.metrics or args.plot or args.plot_save:
        metrics = _simplify_error(x, y, x_out, y_out)

    if args.metrics:
        print()
        print("  Error metrics")
        print("  " + "-" * 40)
        print(f"  Max absolute error : {metrics['max_abs_err']:.4e}")
        print(f"  Mean absolute error: {metrics['mean_abs_err']:.4e}")
        print(f"  RMS error          : {metrics['rms_err']:.4e}")
        print(f"  Max relative error : {metrics['max_rel_err']:.4e}")
        print(f"  R-squared          : {metrics['r_squared']:.6f}")
        # Log-space metrics are NaN when y has any non-positive values;
        # skip their printout in that case to avoid a confusing "nan" row.
        if not np.isnan(metrics["log_r_squared"]):
            print(f"  R-squared (log10y) : {metrics['log_r_squared']:.6f}")
            print(f"  RMS error  (log10y): {metrics['log_rms_err']:.4e} dex")
            print(f"  Max dex error      : {metrics['log_max_dex_err']:.4e} dex")
        print(f"  Compression ratio  : {metrics['compression']:.1f}x")

    # --- Optional: plot ---
    if args.plot or args.plot_save:
        _simplify_plot(
            x, y, x_out, y_out,
            title=f"Simplification of {source_label}",
            save_path=args.plot_save,
            show=args.plot,
        )

    # --- Optional: animation ---
    if args.animate:
        anim_kwargs = {}
        if args.randomSB99:
            anim_kwargs["xlabel"] = r"$\log_{10}(\lambda\ [\mathrm{\AA}])$"
            anim_kwargs["ylabel"] = r"$\log_{10}(L_{\lambda}\ [\mathrm{erg/s/\AA}])$"
        _simplify_animate(
            x, y,
            save_path=args.animate,
            fps=args.animate_fps,
            duration=args.animate_duration,
            title=f"Simplification of {source_label}",
            r2_target=args.r2_target,
            max_err=args.max_err,
            **anim_kwargs,
        )


if __name__ == "__main__":
    import sys

    # If command-line arguments are given, run the file-based CLI.
    # Otherwise, run a quick interactive demo so users can see how it works.
    if len(sys.argv) > 1:
        _simplify_cli()
    else:
        # ---- Quick demo ----
        print("=" * 60)
        print("  _simplify() — interactive demo")
        print("=" * 60)
        print()

        # Generate a test signal: decaying sine with a sharp spike.
        x = np.linspace(0, 4 * np.pi, 5000)
        y = np.exp(-0.1 * x) * np.sin(x)
        # Add a sharp spike at the midpoint to test feature detection.
        y[2500] += 3.0

        x_s, y_s = _simplify(x, y, nmin=100)

        # Show error metrics.
        metrics = _simplify_error(x, y, x_s, y_s)

        print(f"  Original  : {metrics['n_orig']} points")
        print(f"  Simplified: {metrics['n_simp']} points")
        print(f"  Compression: {metrics['compression']:.1f}x")
        print()
        print("  Error metrics:")
        print(f"    RMSE             = {metrics['rms_err']:.4e}")
        print(f"    MAE              = {metrics['mean_abs_err']:.4e}")
        print(f"    Max |error|      = {metrics['max_abs_err']:.4e}")
        print(f"    Max relative err = {metrics['max_rel_err']:.4e}")
        print(f"    R-squared        = {metrics['r_squared']:.6f}")
        print()
        print("  Endpoints preserved:")
        print(f"    first = ({x_s[0]:.4f}, {y_s[0]:.4f})")
        print(f"    last  = ({x_s[-1]:.4f}, {y_s[-1]:.4f})")
        print()
        print("  To use in your own script:")
        print()
        print("    from simplify import _simplify")
        print("    from simplify import _simplify_error")
        print("    from simplify import _simplify_plot")
        print("    from simplify import _simplify_animate")
        print()
        print("    x_s, y_s = _simplify(x, y, nmin=100)")
        print("    metrics  = _simplify_error(x, y, x_s, y_s)")
        print("    _simplify_plot(x, y, x_s, y_s)")
        print("    _simplify_animate(x, y, 'simplify.gif')")
        print()
        print("  Or from the command line:")
        print()
        print("    python simplify.py data.csv -o out.csv --metrics")
        print("    python simplify.py data.csv --plot")
        print("    python simplify.py data.csv --animate simplify.gif")
        print()
        print("  Run with --help for all options.")
        print("=" * 60)