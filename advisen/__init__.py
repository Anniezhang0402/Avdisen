from .estimation import (
    RecurrentEventData,
    fit_parametric,
    fit_spline,
    fit_spline_fixed,
    loglik_parametric,
    loglik_spline,
)

from .models import (
    MusaOkumoto,
    Gompertz,
    Weibull,
    PARAMETRIC_MODELS,
)

from .splines import (
    build_knots,
    mspline_basis,
    ispline_basis,
    ispline_derivative,
)

from .inference import (
    bootstrap_bcif,
    pointwise_ci,
    simultaneous_band,
    parametric_within_scb,
)

from .frailty import (
    fit_frailty,
    heterogeneity_lr_test,
    marginal_loglik_frailty,
)

from .data import (
    make_true_spline_bcif,
    simulate_nhpp_data,
    default_month_end,
    synthetic_mileage_pool,
    load_dmv_disengagement,
)

from .analysis import analyze_manufacturer, select_best_parametric
from .simulation import run_relrmse_study, run_coverage_study, SCENARIOS

__version__ = "0.1.0"

__all__ = [
    "RecurrentEventData",
    "fit_parametric", "fit_spline", "fit_spline_fixed",
    "loglik_parametric", "loglik_spline",
    "MusaOkumoto", "Gompertz", "Weibull", "PARAMETRIC_MODELS",
    "build_knots", "mspline_basis", "ispline_basis", "ispline_derivative",
    "bootstrap_bcif", "pointwise_ci", "simultaneous_band", "parametric_within_scb",
    "fit_frailty", "heterogeneity_lr_test", "marginal_loglik_frailty",
    "make_true_spline_bcif", "simulate_nhpp_data", "default_month_end",
    "synthetic_mileage_pool", "load_dmv_disengagement",
    "analyze_manufacturer", "select_best_parametric",
    "run_relrmse_study", "run_coverage_study", "SCENARIOS",
    "__version__",
]