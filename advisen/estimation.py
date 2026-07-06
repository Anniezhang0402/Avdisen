"""
NHPP recurrent-event likelihood and maximum likelihood estimation.
"""
from __future__ import annotations
from dataclasses import dataclass, field 
import numpy as np 
from scipy.optimize import minimize

from .models import ParametricModel, PARAMETRIC_MODELS
from .splines import (
    build_knots,
    ispline_basis,
    ispline_derivative,
)

_LOG_FLOOR = 1e-300

@dataclass
class RecurrentEventData:
    events: list
    mileage: np.ndarray
    month_end: np.ndarray 
    tau: float = 730.0
    month_start: np.ndarray = field(init=False)

    def __post_init__(self):
        self.mileage = np.asarray(
            self.mileage,
            dtype=float,
        )

        self.month_end = np.asarray(
            self.month_end,
            dtype=float,
        )

        self.month_start = np.concatenate(
            [[0.0], self.month_end[:-1]]
        )
    
    @property
    def n_units(self):
        return len(self.events)
    
    def mileage_at(self, i, t):
        t = np.atleast_1d(t)
        idx = np.searchsorted(
            self.month_end,
            t,
            side="left",
        )
        idx = np.clip(
            idx,
            0,
            self.mileage.shape[1]-1,
        )
        return self.mileage[i, idx]

def _delta_bcif(bcif_fn, month_start, month_end):
    return (
        bcif_fn(month_end)
        -
        bcif_fn(month_start)
    )

def loglik_parametric(
    theta,
    model: ParametricModel,
    data: RecurrentEventData,
    weights=None,
):
    n = data.n_units 
    if weights is None:
        w = np.ones(n)
    else: 
        w = np.asarray(weights, float)

    def bcif(t):
        return model.bcif(
            t,
            theta,
        )

    def bif(t):
        return model.bif(
            t,
            theta,
        )

    term_event = 0.0

    for i in range(n):
        ev = data.events[i]
        if ev.size == 0:
            continue
        xi = data.mileage_at(i, ev)
        lam = bif(ev)
        contrib = (
            np.log(
                np.maximum(
                    xi,
                    _LOG_FLOOR,
                )
            )
            +
            np.log(
                np.maximum(
                    lam,
                    _LOG_FLOOR,
                )
            )
        )

        term_event += (
            w[i]
            *
            np.sum(contrib)
        )

    dLambda = _delta_bcif(
        bcif,
        data.month_start,
        data.month_end,
    )

    term_cum = np.sum(
        w[:, None]
        *
        data.mileage
        *
        dLambda[None, :]
    )

    return (
        term_event
        -
        term_cum
    )

def fit_parametric(
    model_name,
    data: RecurrentEventData,
    weights=None,
    n_restarts=5,
    seed=0,
):
    model = PARAMETRIC_MODELS[model_name]()
    rng = np.random.default_rng(seed)

    def negll_u(u):
        theta = model.to_constrained(u)
        val = loglik_parametric(theta, model, data, weights)

        if not np.isfinite(val):
            return 1e12
        return -val 
    
    u0 = model.to_unconstrained(model.init_params())
    best = None

    for r in range(n_restarts):
        start = u0 if r == 0 else u0 + rng.normal(scale=0.5, size=u0.size)
        res = minimize(
            negll_u,
            start,
            method="Nelder-Mead",
            options={
                "maxiter": 5000,
                "xatol": 1e-6,
                "fatol": 1e-6,
            },
        )

        if best is None or res.fun < best.fun:
            best = res 
    
    theta_hat = model.to_constrained(best.x)
    ll = -best.fun
    aic = -2 * ll + 2 * model.n_params
    return {
        "theta_hat": theta_hat,
        "loglik": ll,
        "aic": aic,
        "model": model,
        "n_params": model.n_params,
    }


class _SplineCache:
    def __init__(self, data, knots, order):
        self.knots = knots
        self.order = order 
        self.n_bases = len(knots) - order 
        self.event_deriv = []
        self.event_mileage = []
        all_ev = []
        counts = []

        for i in range(data.n_units):
            ev = data.events[i]
            all_ev.append(ev)
            counts.append(ev.size)
        
        if sum(counts) > 0:
            flat = np.concatenate([e for e in all_ev if e.size > 0])
            flat_deriv = ispline_derivative(flat, knots, order)
        pos = 0

        for i in range(data.n_units):
            ni = counts[i]
            if ni == 0:
                self.event_deriv.append(np.zeros((0, self.n_bases)))
                self.event_mileage.append(np.zeros(0))
            
            else:
                self.event_deriv.append(flat_deriv[pos:pos + ni])
                self.event_mileage.append(data.mileage_at(i, all_ev[i]))
                pos += ni
            
        I_end = ispline_basis(data.month_end, knots, order)
        I_start = ispline_basis(data.month_start, knots, order)
        self.delta_I = I_end - I_start
        self.mileage = data.mileage

def loglik_spline_cached(coefs, cache: _SplineCache, weights):
    n = len(cache.event_deriv)
    w = weights 
    term_event = 0.0 

    for i in range(n):
        D = cache.event_deriv[i]
        if D.shape[0] == 0:
            continue
        lam = D @ coefs
        xi = cache.event_mileage[i]
        contrib = (
            np.log(np.maximum(xi, _LOG_FLOOR))
            +
            np.log(np.maximum(lam, _LOG_FLOOR))
        )
        term_event += w[i] * np.sum(contrib)

    dLambda = cache.delta_I @ coefs
    term_cum = np.sum(
        w[:, None]
        *
        cache.mileage
        *
        dLambda[None, :]
    )
    return term_event - term_cum

def _spline_bcif_fn(coefs, knots, order):
    def bcif(t):
        I = ispline_basis(np.atleast_1d(t), knots, order)
        return I @ coefs

    return bcif

def _spline_bif_fn(coefs, knots, order):
    def bif(t):
        D = ispline_derivative(np.atleast_1d(t), knots, order)
        return D @ coefs
    return bif 

def loglik_spline(coefs, knots, order, data: RecurrentEventData, weights=None):
    n = data.n_units
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    bif = _spline_bif_fn(coefs, knots, order)
    bcif = _spline_bcif_fn(coefs, knots, order)

    term_event = 0.0

    for i in range(n):
        ev = data.events[i]

        if ev.size == 0:
            continue 
        xi = data.mileage_at(i, ev)
        lam = bif(ev)
        contrib = (
            np.log(np.maximum(xi, _LOG_FLOOR))
            +
            np.log(np.maximum(lam, _LOG_FLOOR))
        )

        term_event += w[i] * np.sum(contrib)
    dLambda = _delta_bcif(
        bcif,
        data.month_start,
        data.month_end,
    )

    term_cum = np.sum(
        w[:, None]
        *
        data.mileage
        *
        dLambda[None, :]
    )

    return term_event - term_cum

def _interior_knots_from_events(data: RecurrentEventData, n_interior):
    all_ev = np.concatenate([
        e for e in data.events
        if e.size > 0
    ])
    if n_interior == 0 or all_ev.size == 0:
        return np.array([])
    qs = np.linspace(0, 1, n_interior + 2)[1:-1]
    knots = np.quantile(all_ev, qs)
    knots = np.unique(knots)
    return knots 

def fit_spline_fixed(
    n_interior,
    data: RecurrentEventData,
    order=3,
    weights=None,
):
    interior = _interior_knots_from_events(
        data,
        n_interior,
    )
    knots = build_knots(
        interior,
        boundary=(0.0, data.tau),
        order=order,
    )
    n_bases = len(knots) - order
    w = (
        np.ones(data.n_units)
        if weights is None
        else np.asarray(weights, float)
    )
    cache = _SplineCache(
        data,
        knots,
        order,
    )
    def negll(coefs):
        val = loglik_spline_cached(
            coefs,
            cache,
            w,
        )
        return -val if np.isfinite(val) else 1e12
    x0 = np.full(n_bases, 1.0)
    bounds = [(0.0, None)] * n_bases

    res = minimize(
        negll,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": 5000,
            "ftol": 1e-9,
        },
    )
    coefs = np.maximum(res.x, 0.0)
    ll = -res.fun
    df = int(np.sum(coefs > 1e-8))
    aic = -2 * ll + 2 * df
    return {
        "coefs": coefs,
        "knots": knots,
        "order": order,
        "loglik": ll,
        "aic": aic,
        "df": df,
        "n_interior": n_interior,
    }

def fit_spline(
    data: RecurrentEventData,
    order=3,
    weights=None,
    interior_grid=(1, 2, 3, 4, 5, 6, 8, 10),
):
    best = None
    for nk in interior_grid:
        try:
            fit = fit_spline_fixed(
                nk,
                data,
                order=order,
                weights=weights,
            )
        except ValueError:
            continue
        if best is None or fit["aic"] < best["aic"]:
            best = fit
    return best 








