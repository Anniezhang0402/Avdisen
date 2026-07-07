from __future__ import annotations
import numpy as np 
def build_knots(
    interior_knots,
    boundary=(0.0,730.0),
    order=3,
):
    lo, hi = float(boundary[0]), float(boundary[1])
    interior = np.asarray(
        interior_knots,
        dtype=float,
    )

    if interior.size:
        if np.any(interior <= lo) or np.any(interior >= hi):
            raise ValueError(
                "Interior knots must lie strictly inside the boundary."
            )
        if np.any(np.diff(interior) <= 0):
            raise ValueError(
                "Interior knots must be strictly increasing."
            )

    left = np.repeat(
        lo,
        order,
    )
    right = np.repeat(
        hi,
        order,
    )
    return np.concatenate([
        left,
        interior,
        right,
    ])

def _mspline_order1(
    t,
    knots,
    n_bases,
):
    t = np.atleast_1d(t).astype(float)
    out = np.zeros(
        (t.size,n_bases)
    )

    for q in range(n_bases):
        left = knots[q]
        right = knots[q+1]
        width = right-left 

        if width<=0:
            continue
        mask = (
            (t>=left)
            &
            (t<right)
        )
        out[mask,q] = 1.0/width 
    return out

def mspline_basis(
    t,
    knots,
    order,
):
    t = np.atleast_1d(t).astype(float)
    n_knots = len(knots)
    m = _mspline_order1(
        t,
        knots,
        n_knots-1,
    )

    for h in range(2,order+1):
        n_bases_h = n_knots-h 
        m_new = np.zeros(
            (t.size,n_bases_h)
        )
        for q in range(n_bases_h):
            denom = knots[q+h]-knots[q]

            if denom<=0:
                continue
            term_left = (
                t-knots[q]
            )*m[:,q]
            term_right = (
                knots[q+h]-t
            )*m[:,q+1]
            m_new[:,q] = (
                h
                *
                (term_left+term_right)
                /
                ((h-1)*denom)
            )
        m = m_new 
    return m 

_INTEGRATION_GRID = 2001

def ispline_basis(
    t,
    knots,
    order,
):
    t = np.atleast_1d(t).astype(float)
    n_bases = len(knots)-order 
    lo = knots[0]
    hi = knots[-1]

    grid = np.linspace(
        lo,
        hi,
        _INTEGRATION_GRID,
    )

    m_grid = mspline_basis(
        grid,
        knots,
        order,
    )

    dz = np.diff(grid)

    trap = (
        0.5
        *
        (m_grid[1:]+m_grid[:-1])
        *
        dz[:,None]
    )

    cum = np.zeros_like(m_grid)

    cum[1:] = np.cumsum(
        trap,
        axis=0,
    )
    totals = cum[-1].copy()
    totals[totals<=0]=1.0
    cum_norm = cum/totals
    out = np.empty(
        (t.size,n_bases)
    )

    for q in range(n_bases):

        out[:,q] = np.interp(
            t,
            grid,
            cum_norm[:,q],
            left=0.0,
            right=1.0,
        )

    return out

def _ispline_normalizers(
    knots,
    order,
):
    lo = knots[0]

    hi = knots[-1]

    grid = np.linspace(
        lo,
        hi,
        _INTEGRATION_GRID,
    )

    m_grid = mspline_basis(
        grid,
        knots,
        order,
    )

    dz = np.diff(grid)

    trap = (
        0.5
        *
        (m_grid[1:]+m_grid[:-1])
        *
        dz[:,None]
    )

    totals = trap.sum(axis=0)

    totals[totals<=0]=1.0

    return totals

def ispline_derivative(
    t,
    knots,
    order,
):
    m = mspline_basis(
        t,
        knots,
        order,
    )
    z = _ispline_normalizers(
        knots,
        order,
    )
    return m / z 
