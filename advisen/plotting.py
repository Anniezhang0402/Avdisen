from __future__ import annotations 
import numpy as np 
from .estimation import (
    _spline_bcif_fn,
    _spline_bif_fn,
)

def plot_bcif_with_scb(result, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(6, 4.5)
        )
    else:
        fig = ax.figure
    spline = result["spline"]
    t_grid, scb_lo, scb_hi, _ = result["scb"]
    sp_bcif = _spline_bcif_fn(
        spline["coefs"],
        spline["knots"],
        spline["order"],
    )
    ax.fill_between(
        t_grid,
        scb_lo,
        scb_hi,
        color="0.8",
        alpha=0.7,
        label="95% SCB",
    )
    ax.plot(
        t_grid,
        sp_bcif(t_grid),
        "k-",
        lw=2,
        label="Spline Model",
    )

    styles = {
        "Gompertz": "r--",
        "Musa-Okumoto": "g--",
        "Weibull": "b-.",
    }
    for name, res in result["parametric"].items():
        model = res["model"]
        ax.plot(
            t_grid,
            model.bcif(
                t_grid,
                res["theta_hat"],
            ),
            styles.get(name, "m:"),
            lw=1.3,
            label=name,
        )
    ax.set_xlabel("Time in Days")
    ax.set_ylabel("Cumulative Intensity Function")
    ax.set_title(
        result.get("name", "")
    )
    ax.legend(fontsize=8)
    return fig, ax

def plot_bif(result, ax=None, logy=True):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(6, 4.5)
        )
    else:
        fig = ax.figure

    if "scb" in result:
        t_grid = result["scb"][0]
    else:
        t_grid = np.linspace(
            1,
            730,
            200,
        )
    spline = result["spline"]

    sp_bif = _spline_bif_fn(
        spline["coefs"],
        spline["knots"],
        spline["order"],
    )

    ax.plot(
        t_grid,
        sp_bif(t_grid),
        "k-",
        lw=2,
        label="Spline Model",
    )
    styles = {
        "Gompertz": "r--",
        "Musa-Okumoto": "g--",
        "Weibull": "b-.",
    }

    for name, res in result["parametric"].items():

        model = res["model"]

        ax.plot(
            t_grid,
            model.bif(
                t_grid,
                res["theta_hat"],
            ),
            styles.get(name, "m:"),
            lw=1.3,
            label=name,
        )

    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Time in Days")
    ax.set_ylabel(
        "Intensity Function (events per k-mile)"
    )
    ax.set_title(
        result.get("name", "")
    )
    ax.legend(fontsize=8)
    return fig, ax 

def plot_simulation_relrmse(sim_result, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(6, 4.5)
        )
    else:
        fig = ax.figrue 
    t_grid = sim_result["t_grid"]

    for n, rr in sim_result["relrmse_by_n"].items():
        ax.plot(
            t_grid,
            rr,
            label=f"n={n}",
        )
    ax.set_xlabel("Time in Days")
    ax.set_ylabel("Relative RMSE")
    ax.set_ylim(0,1)
    ax.legend(fontsize=8)
    return fig, ax 




