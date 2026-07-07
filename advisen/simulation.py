from __future__ import annotations 
import numpy as np 
from .data import (
    make_true_spline_bcif,
    simulate_nhpp_data,
    default_month_end,
    synthetic_mileage_pool,
)
from .estimation import (
    fit_spline_fixed,
    fit_parametric,
    _spline_bcif_fn,
)
from .inference import (
    bootstrap_bcif,
    simultaneous_band,
    parametric_within_scb,
)

SCENARIOS = {
    1: [6, 16, 23, 11, 4],
    2: [8, 12, 28, 0, 12],
    3: [5, 25, 0, 30, 0],
}

def run_relrmse_study(
    scenario=1,
    sample_sizes=(50,100,200),
    n_reps=50,
    t_grid=None,
    seed=0,
):
    coefs = SCENARIOS[scenario]
    tau = 730.0
    me = default_month_end()
    pool = synthetic_mileage_pool()
    true_bcif, true_bif, _ = make_true_spline_bcif(
        coefs,
        tau=tau,
    )

    if t_grid is None:
        t_grid = np.linspace(
            1,
            tau,
            100,
        )

    true_curve = true_bcif(t_grid)

    rng = np.random.default_rng(seed)

    relrmse_by_n = {}

    for n in sample_sizes:
        sq_err = np.zeros_like(t_grid)
        for r in range(n_reps):
            s = int(
                rng.integers(1_000_000)
            )
            data = simulate_nhpp_data(
                n,
                true_bif,
                pool,
                me,
                seed=s,
            )
            fit = fit_spline_fixed(
                3,
                data,
            )
            est = _spline_bcif_fn(
                fit["coefs"],
                fit["knots"],
                fit["order"],
            )(t_grid)
            sq_err += (
                est - true_curve
            ) ** 2
        rmse = np.sqrt(
            sq_err / n_reps
        )

        with np.errstate(
            divide="ignore",
            invalid="ignore",
        ):

            relrmse = np.where(
                true_curve > 0,
                rmse / true_curve,
                0.0,
            )

        relrmse_by_n[n] = relrmse

    return {
        "scenario": scenario,
        "t_grid": t_grid,
        "relrmse_by_n": relrmse_by_n,
        "true_curve": true_curve,
    }

def run_coverage_study(
    scenario=1,
    sample_sizes=(50,100,200),
    n_reps=30,
    B=200,
    seed=0,
):
    coefs = SCENARIOS[scenario]
    tau = 730.0
    me = default_month_end()
    pool = synthetic_mileage_pool()
    true_bcif, true_bif, _ = make_true_spline_bcif(
        coefs,
        tau=tau,
    )
    t_grid = np.linspace(
        1,
        tau,
        60,
    )

    true_curve = true_bcif(t_grid)
    rng = np.random.default_rng(seed)
    cp_by_n = {}
    acc_by_n = {}

    for n in sample_sizes:
        cov = 0
        acc = 0
        for r in range(n_reps):
            s = int(
                rng.integers(1_000_000)
            )
            data = simulate_nhpp_data(
                n,
                true_bif,
                pool,
                me,
                seed=s,
            )

            curves, _ = bootstrap_bcif(
                data,
                t_grid,
                B=B,
                seed=s,
            )

            lo, hi, _ = simultaneous_band(
                curves,
                alpha=0.05,
            )

            if np.all(
                (true_curve >= lo)
                &
                (true_curve <= hi)
            ):
                cov += 1 
            
            gfit = fit_parametric(
                "Gompertz",
                data,
            )

            gcurve = gfit["model"].bcif(
                t_grid,
                gfit["theta_hat"],
            )
            if parametric_within_scb(
                gcurve,
                lo,
                hi,
            ):
                acc += 1
        cp_by_n[n] = cov / n_reps 
        acc_by_n[n] = acc / n_reps 
    
    return {
        "scenario": scenario,
        "cp_by_n": cp_by_n,
        "acc_by_n": acc_by_n,
    }

