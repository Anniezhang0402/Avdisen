from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np

from .estimation import (
    RecurrentEventData,
    fit_spline,
    fit_spline_fixed,
    _spline_bcif_fn,
)

def _one_bootstrap(
    seed,
    data: RecurrentEventData,
    order,
    n_interior,
    fix_knots,
    interior_grid,
    t_grid,
):
    rng = np.random.default_rng(seed)
    w = rng.exponential(
        1.0,
        size=data.n_units,
    )

    if fix_knots:
        fit = fit_spline_fixed(
            n_interior,
            data,
            order=order,
            weights=w,
        )
    else:
        fit = fit_spline(
            data,
            order=order,
            weights=w,
            interior_grid=interior_grid,
        )
    bcif = _spline_bcif_fn(
        fit["coefs"],
        fit["knots"],
        fit["order"],
    )
    return bcif(t_grid)

def bootstrap_bcif(
    data: RecurrentEventData,
    t_grid,
    B=1000,
    order=3,
    fix_knots=True,
    interior_grid=(1,2,3,4,5,6,8,10),
    n_jobs=1,
    seed=0,
):
    t_grid = np.asarray(
        t_grid,
        float,
    )

    point_fit = fit_spline(
        data,
        order=order,
        interior_grid=interior_grid,
    )
    n_interior = point_fit["n_interior"]
    seeds = np.random.SeedSequence(seed).spawn(B)
    seeds = [
        int(s.generate_state(1)[0])
        for s in seeds
    ]
    worker = partial(
        _one_bootstrap,
        data=data,
        order=order,
        n_interior=n_interior,
        fix_knots=fix_knots,
        interior_grid=interior_grid,
        t_grid=t_grid,
    )

    if n_jobs and n_jobs > 1:

        with ProcessPoolExecutor(
            max_workers=n_jobs
        ) as ex:

            curves = list(
                ex.map(worker, seeds)
            )
    else:

        curves = [
            worker(s)
            for s in seeds
        ]
    return np.vstack(curves), point_fit

def pointwise_ci(curves, alpha=0.05):
    lo = np.quantile(
        curves,
        alpha/2,
        axis=0,
    )
    hi = np.quantile(
        curves,
        1-alpha/2,
        axis=0,
    )
    return lo, hi

def _coverage_at_alpha_p(curves, alpha_p):
    lo = np.quantile(
        curves,
        alpha_p/2,
        axis=0,
    )
    hi = np.quantile(
        curves,
        1-alpha_p/2,
        axis=0,
    )
    inside = np.all(
        (curves >= lo)
        &
        (curves <= hi),
        axis=1,
    )
    return inside.mean()

def simultaneous_band(
    curves,
    alpha=0.05,
    tol=1e-3,
    max_iter=60,
):
    target = 1 - alpha
    lo_a = 1e-4
    hi_a = 0.999
    alpha_c = alpha

    for _ in range(max_iter):

        mid = 0.5 * (lo_a + hi_a)

        cp = _coverage_at_alpha_p(
            curves,
            mid,
        )

        if abs(cp-target) < tol:
            alpha_c = mid
            break

        if cp > target:
            lo_a = mid
        else:
            hi_a = mid

        alpha_c = mid

    lo = np.quantile(
        curves,
        alpha_c/2,
        axis=0,
    )

    hi = np.quantile(
        curves,
        1-alpha_c/2,
        axis=0,
    )

    return lo, hi, alpha_c

def parametric_within_scb(
    param_bcif_curve,
    scb_lo,
    scb_hi,
):

    return bool(
        np.all(
            (param_bcif_curve >= scb_lo)
            &
            (param_bcif_curve <= scb_hi)
        )
    )




