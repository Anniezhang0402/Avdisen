from __future__ import annotations
import numpy as np 
from .estimation import RecurrentEventData
from .splines import build_knots, ispline_basis, ispline_derivative

def make_true_spline_bcif(coefs, tau=730.0, order=3, n_interior=None):
    coefs = np.asarray(coefs, float)
    if n_interior is None:
        n_interior = len(coefs) - order 

        if n_interior < 0:
            raise ValueError(
                f"Coefficient length {len(coefs)} is smaller than order {order}; "
                "cannot construct spline basis."
            )
    interior = np.linspace(0, tau, n_interior + 2)[1:-1]
    
    knots = build_knots(interior, boundary=(0.0, tau), order=order)

    def bcif(t):
        return ispline_basis(np.atleast_1d(t), knots, order) @ coefs
    
    def bif(t):
        return ispline_derivative(np.atleast_1d(t), knots, order) @ coefs 
    return bcif, bif, knots 

def _sample_events_one_unit(
    bif,
    mileage_row,
    month_end,
    month_start,
    active_mask,
    rng,
    max_rate=None,
):
    events = []
    n_tau = len(month_end)

    for l in range(n_tau):
        if not active_mask[l]:
            continue
        x_il = mileage_row[l]

        if x_il <= 0:
            continue
        a, b = month_start[l], month_end[l]

        grid = np.linspace(a, b, 20)

        lam_up = np.max(bif(grid)) * x_il 

        if lam_up <= 0:
            continue
        t = a 

        while True:
            t += rng.exponential(1.0 / lam_up)

            if t >= b:
                break
            lam_true = float(bif(np.array([t]))[0]) * x_il 

            if rng.uniform() <= lam_true / lam_up:
                events.append(t)
    return np.array(sorted(events))

def simulate_nhpp_data(
    n_units,
    true_bif,
    mileage_pool,
    month_end,
    avg_active_months=12,
    seed=0,
):
    """
    Generate a window-observed recurrent-event dataset.
    """
    rng = np.random.default_rng(seed)
    
    n_tau = len(month_end)

    month_start = np.concatenate([[0.0], month_end[:-1]])

    mileage = np.zeros((n_units, n_tau))

    events = []

    for i in range(n_units):
        row = mileage_pool[rng.integers(len(mileage_pool))].copy()
        n_active = min(
            n_tau,
            max(1, int(rng.poisson(avg_active_months))),
        )

        start_month = (
            rng.integers(0, n_tau - n_active + 1)
            if n_tau > n_active 
            else 0
        )

        active_mask = np.zeros(n_tau, dtype=bool)
        active_mask[start_month:start_month + n_active] = True 

        row[~active_mask] = 0.0 

        mileage[i] = row 

        ev = _sample_events_one_unit(
            true_bif,
            row,
            month_end,
            month_start,
            active_mask,
            rng,
        )
        events.append(ev)

    return RecurrentEventData(
        events=events,
        mileage=mileage,
        month_end=month_end,
        tau=float(month_end[-1]),
    )

def default_month_end(n_tau=24, days_per_month=30.4):
    """
    Construct a month-end day sequence with 24 months and total duration
    approximately 730 days.
    """

    ends = np.cumsum(np.full(n_tau, days_per_month))
    ends[-1] = 730.0

    return ends 

def synthetic_mileage_pool(n_pool=123, n_tau=24, seed=1):
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.02, 0.12, size=(n_pool, 1))
    wobble = 0.03 * rng.standard_normal((n_pool, n_tau))
    pool = np.clip(base + wobble, 0.0, 0.14)
    return pool 

def load_dmv_disengagement(events_csv, mileage_csv, tau=730.0):
    """
    Load California DMV disengagement event data from cleaned csv files.
    """
    import pandas as pd 
    ev_df = pd.read_csv(events_csv)
    mi_df = pd.read_csv(mileage_csv)

    unit_ids = sorted(mi_df["unit_id"].unique())
    
    id_to_idx = {
        u: k 
        for k, u in enumerate(unit_ids)
    }
    n = len(unit_ids)
    n_tau = int(mi_df["month_index"].max()) + 1
    mileage = np.zeros((n, n_tau))

    for _, r in mi_df.iterrows():
        mileage[
            id_to_idx[r["unit_id"]],
            int(r["month_index"]),
        ] = r["daily_kmiles"]
        
    events = [
        np.array([], dtype=float)
        for _ in range(n)
    ]

    for u, grp in ev_df.groupby("unit_id"):
        if u in id_to_idx:
            events[id_to_idx[u]] = np.sort(
                grp["event_day"].values.astype(float)
            )
    month_end = default_month_end(n_tau)
    return RecurrentEventData(
        events=events,
        mileage=mileage,
        month_end=month_end,
        tau=tau,
    )

