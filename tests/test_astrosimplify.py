"""Regression tests for astrosimplify.

Each test pins a behaviour that was broken at some point, plus the
numbers the README quotes, so a change that moves them fails loudly.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import astrosimplify as a

ROOT = Path(__file__).resolve().parent.parent


def worst_error(x, y, xs, ys):
    return float(np.max(np.abs(y - np.interp(x, xs, ys))))


def quick_start_curve(n=10000):
    x = np.linspace(0, 10, n)
    return x, np.sin(x) + 0.5 * np.sin(5 * x)


# --------------------------------------------------------------------------
# max_err is a guarantee
# --------------------------------------------------------------------------
@pytest.mark.parametrize("max_err", [1e-2, 1e-3, 1e-4])
def test_max_err_bound_holds_linear(max_err):
    x, y = quick_start_curve()
    xs, ys = a.simplify(x, y, max_err=max_err, log_y=False)
    assert worst_error(x, y, xs, ys) <= max_err


@pytest.mark.parametrize("max_err", [0.05, 0.01, 0.005])
def test_max_err_bound_holds_on_sb99(max_err):
    x, y = a.load_sb99_5myr()
    xs, ys = a.simplify(x, y, max_err=max_err, log_y=False)
    assert worst_error(x, y, xs, ys) <= max_err


def test_max_err_bound_holds_on_noisy_curve():
    x, y = a.random_test_curve(seed=3)
    xs, ys = a.simplify(x, y, max_err=0.01, log_y=False)
    assert worst_error(x, y, xs, ys) <= 0.01


def test_max_err_bound_holds_in_log_space():
    x = np.linspace(1, 100, 20000)
    y = 10 ** (3 * np.sin(x / 7)) * (1 + 0.2 * np.sin(3 * x))
    xs, ys = a.simplify(x, y, max_err=1e-3, log_y=True)
    assert worst_error(x, np.log10(y), xs, np.log10(ys)) <= 1e-3


def test_repeated_x_jump_warns_and_does_not_stall_the_bound():
    x, y = quick_start_curve(10000)
    x = np.insert(x, 5000, x[5000])
    y = np.insert(y, 5000, y[5000] + 1.0)
    with pytest.warns(UserWarning, match="could not be met"):
        xs, ys = a.simplify(x, y, max_err=1e-4, log_y=False)
    away = np.abs(x - x[5000]) > 0.5
    residual = np.abs(y - np.interp(x, xs, ys))
    assert residual[away].max() <= 1e-4


# --------------------------------------------------------------------------
# dedup never deletes real structure
# --------------------------------------------------------------------------
def test_dedup_compares_against_last_kept_point():
    # Steps of 3e-4 rad are each below dedup_tol * range, but runs of them
    # add up to large changes.  Every dropped sample must stay within the
    # tolerance of a kept one, so the peak survives.
    x = np.linspace(0, 2 * np.pi, 20001)
    y = np.sin(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xs, ys = a.simplify(x, y, dedup_tol=1e-3, log_y=False)
    assert ys.max() > 0.999
    assert worst_error(x, y, xs, ys) < 0.01


def test_dedup_mask_keeps_every_drop_within_tolerance():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.uniform(0, 2e-3, 5000))
    y = np.cumsum(rng.normal(0, 1e-3, 5000))
    tx, ty = 5e-3, 5e-3
    keep = a._dedup_keep_mask(x, y, tx, ty)
    assert keep[0] and keep[-1]
    kept = np.flatnonzero(keep)
    for i in np.flatnonzero(~keep):
        prev_kept = kept[kept < i][-1]
        assert abs(x[i] - x[prev_kept]) < tx and abs(y[i] - y[prev_kept]) < ty


# --------------------------------------------------------------------------
# prominence
# --------------------------------------------------------------------------
def naive_prominence(y, p):
    """scipy-style prominence by brute force."""
    n = y.size
    is_max = (y[p] >= y[max(p - 1, 0)]) and (y[p] >= y[min(p + 1, n - 1)])
    s = y if is_max else -y
    left = p
    while left > 0 and s[left - 1] <= s[p]:
        left -= 1
    right = p
    while right < n - 1 and s[right + 1] <= s[p]:
        right += 1
    base = max(s[left:p + 1].min(), s[p:right + 1].min())
    return s[p] - base


def test_flat_topped_extrema_keep_their_prominence():
    np.testing.assert_allclose(
        a._peak_prominences(np.array([0.0, 3, 3, 0, 1, 0]), np.array([1, 2])), [3, 3]
    )
    np.testing.assert_allclose(
        a._peak_prominences(np.array([5.0, 0, 0, 5, 4, 5]), np.array([1, 2])), [5, 5]
    )
    assert a._prev_next_strict(np.array([1.0, 2, 2, 1]), True)[1].tolist() == [1, 4, 4, 4]


def test_prominence_matches_brute_force_on_quantised_data():
    rng = np.random.default_rng(1)
    for _ in range(300):
        y = rng.integers(0, 5, rng.integers(3, 25)).astype(float)
        dy = np.sign(np.diff(y))
        p = np.array([i for i in range(1, y.size - 1) if dy[i - 1] * dy[i] <= 0 and (dy[i - 1] or dy[i])])
        if p.size == 0:
            continue
        got = a._peak_prominences(y, p)
        want = [naive_prominence(y, int(i)) for i in p]
        np.testing.assert_allclose(got, want)


# --------------------------------------------------------------------------
# arc-length sampler and nmin
# --------------------------------------------------------------------------
def test_step_is_reproduced_without_curvature_detection():
    x = np.linspace(0, 1, 10000)
    y = np.where(np.arange(x.size) < 5000, 0.0, 1.0)
    xs, ys = a.simplify(x, y, grad_inc=1e4, log_y=False)
    assert worst_error(x, y, xs, ys) == 0.0


def test_nmin_below_100_is_honoured():
    x, y = quick_start_curve()
    assert a.simplify(x, y, nmin=10)[0].size < a.simplify(x, y, nmin=100)[0].size


@pytest.mark.parametrize("bad", [0, 1, 2.5])
def test_invalid_nmin_raises(bad):
    x, y = quick_start_curve(100)
    with pytest.raises(ValueError, match="nmin"):
        a.simplify(x, y, nmin=bad)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_tiny_inputs_are_returned_unchanged(n):
    x = np.arange(n, dtype=float)
    xs, ys = a.simplify(x, x ** 2, nmin=2)
    assert xs.size == n


# --------------------------------------------------------------------------
# ranking, simplify_n, diagnostic
# --------------------------------------------------------------------------
def naive_importance_order(x, y, n_max):
    n = x.size
    in_set = np.zeros(n, bool)
    order = []
    for s in (0, n - 1):
        if not in_set[s] and len(order) < n_max:
            in_set[s] = True
            order.append(s)
    while len(order) < n_max:
        sel = np.flatnonzero(in_set)
        err = np.abs(y - np.interp(x, x[sel], y[sel]))
        err[in_set] = -1.0
        w = int(np.argmax(err))
        if err[w] <= 0:
            break
        in_set[w] = True
        order.append(w)
    order.extend(np.flatnonzero(~in_set)[: n_max - len(order)].tolist())
    return np.asarray(order)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_heap_ranking_matches_naive_greedy(seed):
    x, y = a.random_test_curve(npts=1500, seed=seed)
    np.testing.assert_array_equal(
        a._importance_order(x, y, 1200), naive_importance_order(x, y, 1200)
    )


def test_simplify_n_returns_exact_count():
    x, y = quick_start_curve()
    xs, ys = a.simplify_n(x, y, 40)
    assert xs.size == 40 and np.all(np.diff(xs) > 0)
    assert xs[0] == x[0] and xs[-1] == x[-1]


def test_simplify_n_matches_diagnostic_prefix():
    x, y = a.random_test_curve(seed=67)
    row = a.simplify_diagnostic(x, y, nrels=[0.004], plot=False)[0]
    xs, _ = a.simplify_n(x, y, row["n_out"])
    np.testing.assert_array_equal(xs, row["x_simp"])


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def test_max_rel_err_does_not_depend_on_units():
    x = np.arange(100.0)
    xs = x[::10]
    rel = [
        a.simplify_error(x, k * (2 + np.sin(x / 5)), xs, k * (2 + np.sin(xs / 5)))["max_rel_err"]
        for k in (1.0, 1e-31, 1e30)
    ]
    np.testing.assert_allclose(rel, rel[0], rtol=1e-9)


def test_r_squared_for_constant_original():
    x = np.arange(10.0)
    assert np.isnan(a.simplify_error(x, np.full(10, 5.0), x, np.zeros(10))["r_squared"])
    assert a.simplify_error(x, np.full(10, 5.0), x, np.full(10, 5.0))["r_squared"] == 1.0


def test_simplify_error_sorts_simplified_points():
    x, y = quick_start_curve(2000)
    xs, ys = a.simplify(x, y)
    o = np.random.default_rng(0).permutation(xs.size)
    shuffled, ordered = a.simplify_error(x, y, xs[o], ys[o]), a.simplify_error(x, y, xs, ys)
    assert shuffled.keys() == ordered.keys()
    np.testing.assert_array_equal(list(shuffled.values()), list(ordered.values()))


def test_log_y_auto_threshold_is_strictly_more_than_two_decades():
    assert a._auto_log_y(np.array([1.0, 100.0]), "auto") is False
    assert a._auto_log_y(np.array([1.0, 100.01]), "auto") is True


# --------------------------------------------------------------------------
# plotting and animation
# --------------------------------------------------------------------------
def test_animate_accepts_unsorted_input_and_preserves_rcparams(tmp_path):
    import matplotlib.pyplot as plt

    x = np.linspace(0, 10, 600)
    o = np.random.default_rng(0).permutation(x.size)
    plt.rcParams["lines.linewidth"] = 7.5
    try:
        a.simplify_animate(x[o], np.sin(x[o]), str(tmp_path / "a.gif"), duration=0.4, fps=5)
        assert plt.rcParams["lines.linewidth"] == 7.5
    finally:
        plt.rcdefaults()
    assert (tmp_path / "a.gif").stat().st_size > 0


def test_animate_forwards_simplify_settings(tmp_path, monkeypatch):
    seen = {}
    real = a.simplify

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(a, "simplify", spy)
    x = np.linspace(0, 10, 600)
    a.simplify_animate(x, np.sin(x), str(tmp_path / "b.gif"), duration=0.4, fps=5,
                       nmin=40, grad_inc=3.0, dedup_tol=0.0)
    assert (seen["nmin"], seen["grad_inc"], seen["dedup_tol"]) == (40, 3.0, 0.0)


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------
def run_cli(monkeypatch, tmp_path, *argv):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["astrosimplify", *argv])
    a.main()


def test_cli_rejects_extra_columns(monkeypatch, tmp_path):
    np.savetxt(tmp_path / "three.txt", np.column_stack([np.arange(50), np.arange(50), np.arange(50)]))
    with pytest.raises(SystemExit, match="exactly 2 columns"):
        run_cli(monkeypatch, tmp_path, "three.txt", "-o", "out.csv")


def test_cli_accepts_one_row_file(monkeypatch, tmp_path):
    (tmp_path / "one.csv").write_text("1.0,2.0\n")
    run_cli(monkeypatch, tmp_path, "one.csv", "-o", "out.csv")
    np.testing.assert_allclose(np.loadtxt(tmp_path / "out.csv", delimiter=","), [1.0, 2.0])


def test_cli_demo_does_not_overwrite_default_output(monkeypatch, tmp_path):
    (tmp_path / "simplified_output.csv").write_text("precious\n")
    run_cli(monkeypatch, tmp_path, "--random", "--random-npts", "500")
    assert (tmp_path / "simplified_output.csv").read_text() == "precious\n"


def test_cli_file_mode_writes_default_output(monkeypatch, tmp_path):
    x = np.linspace(0, 10, 500)
    np.savetxt(tmp_path / "in.csv", np.column_stack([x, np.sin(x)]), delimiter=",")
    run_cli(monkeypatch, tmp_path, "in.csv")
    assert (tmp_path / "simplified_output.csv").stat().st_size > 0


def test_cli_handles_unsorted_file_with_plot(monkeypatch, tmp_path):
    x = np.linspace(0, 10, 500)
    o = np.random.default_rng(0).permutation(x.size)
    np.savetxt(tmp_path / "u.csv", np.column_stack([x[o], np.sin(x[o])]), delimiter=",")
    run_cli(monkeypatch, tmp_path, "u.csv", "-o", "out.csv", "--plot-save", "p.png")
    assert (tmp_path / "p.png").stat().st_size > 0


def test_cli_rejects_nmin_below_two(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, tmp_path, "--random", "--nmin", "1")


# --------------------------------------------------------------------------
# numbers quoted in the README
# --------------------------------------------------------------------------
def test_readme_sb99_point_count():
    x, y = a.load_sb99_5myr()
    assert a.simplify(x, y, max_err=0.05, log_y=False)[0].size == 359


def test_readme_diagnostic_table():
    x, y = a.random_test_curve(seed=67)
    rows = a.simplify_diagnostic(x, y, plot=False)
    assert [r["n_out"] for r in rows] == [8000, 3000, 1000, 100]
    np.testing.assert_allclose([r["max_err"] for r in rows],
                               [3.056e-03, 1.573e-02, 2.921e-02, 8.997e-02], rtol=1e-3)


@pytest.mark.parametrize("name", ["bubble_T.dat", "bubble_n.dat"])
def test_readme_bubble_claim(name):
    r, y = np.loadtxt(ROOT / "data" / name).T
    first = next(
        n for n in range(2, r.size + 1)
        if a.simplify_error(r, y, *a.simplify_n(r, y, n, log_y=False))["r_squared"] >= 0.999
    )
    assert first == 12


def test_bubble_density_is_in_cm3_and_isobaric():
    _, log_t = np.loadtxt(ROOT / "data" / "bubble_T.dat").T
    _, log_n = np.loadtxt(ROOT / "data" / "bubble_n.dat").T
    assert -1 < log_n.min() and log_n.max() < 4          # cm^-3, not pc^-3
    assert np.ptp(log_n + log_t) < 1e-6                   # n T = const
