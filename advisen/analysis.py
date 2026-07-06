from __future__ import annotations
import numpy as np 

from .estimation import (
    RecurrentEventData,
    fit_parametric,
    fit_spline,
    _spline_bcif_fn,
)
from .models import PARAMETRIC_MODELS

from .inference import (
    bootstrap_bcif,
    pointwise_ci,
    simultaneous_band,
    parametric_within_scb,
)
from .frailty import heterogeneity_lr_test

def select_best_parametric(data: RecurrentEventData, seed=0):
    """
    Select the best parametric model using AIC.
    """

    results = {}

    for name in PARAMETRIC_MODELS:
        results[name] = fit_parametric(
            name,
            data,
            n_restarts=5,
            seed=seed,
        )

    best_name = min(results, key=lambda k: results[k]["aic"])
    return best_name, results

def analyze_manufacturer(
    data: RecurrentEventData,
    name="manufacturer",
    B=500,
    t_grid=None,
    n_jobs=1,
    seed=0,
    run_bootstrap=True,
    run_frailty=True,
):
    """
    Complete reliaility analysis for a single manufacturer.

    Workflow
    --------
    1. Fit all parametric models.
    2. Select the best model by AIC.
    3. Fit the spline model.
    4. (Optional) Perform bootstrap inference and construct confidence bands.
    5. (Optional) Perform heterogeneity testing using the best parametric model.
    """
    if t_grid is None:
        t_grid = np.linspace(1, data.tau, 200)

    best_param, param_results = select_best_parametric(
        data,
        seed=seed,
    )

    spline_fit = fit_spline(data)

    aic_table = {
        k: v["aic"]
        for k, v in param_results.items()
    }
    aic_table["Spline"] = spline_fit["aic"]

    out = {
        "name": name,
        "parametric": param_results,
        "best_param": best_param,
        "spline": spline_fit,
        "aic_table": aic_table,
    }

    if run_bootstrap:
        curves, point_fit = bootstrap_bcif(
            data,
            t_grid,
            B=B,
            n_jobs=n_jobs,
            seed=seed,
        )

        pci_lo, pci_hi = pointwise_ci(
            curves,
            alpha=0.05,
        )

        scb_lo, scb_hi, alpha_c = simultaneous_band(
            curves,
            alpha=0.05,
        )

        out["bootstrap_curves"] = curves
        out["pci"] = (t_grid, pci_lo, pci_hi)
        out["scb"] = (
            t_grid,
            scb_lo,
            scb_hi,
            alpha_c,
        )

        param_in_scb = {}

        for k, v in param_results.item():
            model = v["model"]
            pcurve = model.bcif(
                t_grid,
                v["theta_hat"],
            )
            param_in_scb[k] = parametric_within_scb(
                pcurve,
                scb_lo,
                scb_hi,
            )
        out["param_in_scb"] = param_in_scb

    if run_frailty:
        out["frailty"] = heterogeneity_lr_test(
            best_param,
            data,
            seed=seed,
        )
    return out 