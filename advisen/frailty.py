from __future__ import annotations
import numpy as np 
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import chi2

from .estimation import (
    RecurrentEventData,
    loglik_parametric,
    fit_parametric,
    _delta_bcif,
)
from .models import PARAMETRIC_MODELS
_LOG_FLOOR = 1e-300

def marginal_loglik_frailty(theta, phi, model, data: RecurrentEventData):
    n = data.n_units
    def bcif(t):
        return model.bcif(t, theta)
    def bif(t):
        return model.bif(t, theta)
    dLambda = _delta_bcif(
        bcif,
        data.month_start,
        data.month_end,
    )
    c = np.sum(
        data.mileage * dLambda[None, :],
        axis=1,
    )
    inv_phi = 1.0 / phi
    total = 0.0

    for i in range(n):
        ev = data.events[i]
        ni = ev.size
        if ni > 0:
            xi = data.mileage_at(i, ev)
            lam = bif(ev)

            log_prod = np.sum(
                np.log(np.maximum(xi, _LOG_FLOOR))
                +
                np.log(np.maximum(lam, _LOG_FLOOR))
            )
        else:
            log_prod = 0.0 

        marg = (
            inv_phi * np.log(inv_phi)
            +
            gammaln(ni + inv_phi)
            -
            gammaln(inv_phi)
            -
            (ni + inv_phi) * np.log(c[i] + inv_phi)
        )
        total += log_prod + marg
    return total 

def fit_frailty(
    model_name,
    data: RecurrentEventData,
    theta_init=None,
    phi_init=0.1,
    seed=0,
):
    model = PARAMETRIC_MODELS[model_name]()
    if theta_init is None:
        base = fit_parametric(
            model_name,
            data,
            n_restarts=3,
            seed=seed,
        )
        theta_init = base["theta_hat"]
    u_theta0 = model.to_unconstrained(
        np.asarray(theta_init, float)
    )
    u0 = np.concatenate([
        u_theta0,
        [np.log(phi_init)],
    ])

    def negll(u):
        theta = model.to_constrained(u[:-1])
        phi = np.exp(u[-1])
        val = marginal_loglik_frailty(
            theta,
            phi,
            model,
            data,
        )
        return -val if np.isfinite(val) else 1e12

    res = minimize(
        negll,
        u0,
        method="Nelder-Mead",
        options={
            "maxiter": 8000,
            "xatol": 1e-6,
            "fatol": 1e-6,
        },
    )

    theta_hat = model.to_constrained(res.x[:-1])
    phi_hat = float(np.exp(res.x[-1]))
    ll = -res.fun
    return {
        "theta_hat": theta_hat,
        "phi_hat": phi_hat,
        "loglik": ll,
        "model": model,
    }

def heterogeneity_lr_test(
    model_name,
    data: RecurrentEventData,
    seed=0,
):
    base = fit_parametric(
        model_name,
        data,
        n_restarts=3,
        seed=seed,
    )
    ll_nhpp = base["loglik"]

    fr = fit_frailty(
        model_name,
        data,
        theta_init=base["theta_hat"],
        seed=seed,
    )
    ll_frailty = fr["loglik"]

    stat = -2.0 * (
        ll_nhpp
        -
        ll_frailty
    )
    stat = max(stat, 0.0)

    p_value = chi2.sf(
        stat,
        df=1,
    )

    return {
        "stat": stat,
        "p_value": p_value,
        "phi_hat": fr["phi_hat"],
        "ll_nhpp": ll_nhpp,
        "ll_frailty": ll_frailty,
        "best_model": model_name,
    }
    