from __future__ import annotations
import numpy as np
_EPS = 1e-300

class ParametricModel:
    name = "base"
    n_params = 0

    def bif(self, t, theta):
        raise NotImplementedError

    def bcif(self, t, theta):
        raise NotImplementedError

    def init_params(self):
        raise NotImplementedError

    def to_unconstrained(self, theta):
        raise NotImplementedError

    def to_constrained(self, u):
        raise NotImplementedError

class MusaOkumoto(ParametricModel):
    name = "Musa-Okumoto"
    n_params = 2

    def bcif(self, t, theta):
        th1, th2 = theta 
        t = np.asarray(t, dtype=float)
        if th1 <= 0:
            return th2 * t
        return np.log1p(
            th2 * th1 * t
        ) / th1 
    
    def bif(self, t, theta):
        th1, th2 = theta 
        t = np.asarray(t, dtype=float)
        return (
            th2 
            /
            (
                1.0
                +
                th2 * th1 * t
            )
        )

    def init_params(self):
        return np.array([
            0.01,
            0.1,
        ])

    def to_unconstrained(self, theta):
        return np.log(
            np.maximum(theta, _EPS)
        )

    def to_constrained(self, u):
        return np.exp(u)
        
class Gompertz(ParametricModel):
    name = "Gompertz"
    n_params = 3

    def bcif(self, t, theta):
        th1, th2, th3 = theta 
        t = np.asarray(t, dtype=float)

        log2 = np.log(
            np.clip(
                th2,
                _EPS,
                1 - 1e-12,
            )
        )

        log3 = np.log(
            np.clip(
                th3,
                _EPS,
                1 - 1e-12,
            )
        )

        a = np.exp(t * log2)
        return (
            th1
            *
            np.exp(a * log3)
            -
            th1 * th3
        )
    
    def bif(self, t, theta):

        th1, th2, th3 = theta

        t = np.asarray(t, dtype=float)

        log2 = np.log(
            np.clip(
                th2,
                _EPS,
                1 - 1e-12,
            )
        )

        log3 = np.log(
            np.clip(
                th3,
                _EPS,
                1 - 1e-12,
            )
        )

        a = np.exp(t * log2)
        return (
            th1
            *
            a
            *
            np.exp(a * log3)
            *
            log2
            *
            log3
        )

    def init_params(self):

        return np.array([
            50.0,
            0.99,
            0.5,
        ])

    def to_unconstrained(self, theta):
        th1, th2, th3 = theta 
        u1 = np.log(max(th1, _EPS))
        u2 = np.log(
            th2 / (1-th2)
        )
        u3 = np.log(
            th3 / (1-th3)
        )
        return np.array([
            u1,
            u2,
            u3,
        ])

    def to_constrained(self, u):

        u1, u2, u3 = u

        th1 = np.exp(u1)

        th2 = 1.0 / (
            1.0 + np.exp(-u2)
        )
        th3 = 1.0 / (
            1.0 + np.exp(-u3)
        )

        return np.array([
            th1,
            th2,
            th3,
        ])

class Weibull(ParametricModel):
    name = "Weibull"
    n_params = 3

    def bcif(self, t, theta):
        th1, th2, th3 = theta 
        t = np.asarray(t, dtype=float)

        tp = np.power(
            np.maximum(t, 0.0),
            th3,
        )

        return (
            th1
            *
            (
                1.0
                -
                np.exp(-th2 * tp)
            )
        )

    def bif(self, t, theta):
        th1, th2, th3 = theta 
        t = np.asarray(t, dtype=float)
        tt = np.maximum(
            t,
            _EPS,
        )
        tp = np.power(
            tt,
            th3,
        )

        return (
            th1
            *
            th2
            *
            th3
            *
            np.power(
                tt,
                th3=1.0,
            )
            *
            np.exp(-th2 * tp)
        )

    def init_params(self):

        return np.array([
            50.0,
            0.01,
            1.0,
        ])

    def to_unconstrained(self, theta):
        return np.log(
            np.maximum(
                theta,
                _EPS,
            )
        )

    def to_constrained(self, u):
        return np.exp(u)

PARAMETRIC_MODELS = {
    "Musa-Okumoto": MusaOkumoto,
    "Gompertz": Gompertz,
    "Weibull": Weibull,
}