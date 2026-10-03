"""Modelo de goles: Poisson bivariado tipo Dixon-Coles con tasas en función del Elo.

    x      = (rating_A - rating_B) / 400
    log λA = μ + β·x + γ·local_A + ω·mundial
    log λB = μ − β·x − δ·local_A + ω·mundial

ρ (Dixon-Coles) corrige la probabilidad de los marcadores bajos (0-0, 1-0, 0-1, 1-1),
que un Poisson independiente subestima (sobre todo los empates).
Todos los parámetros se estiman por máxima verosimilitud con datos históricos,
reemplazando la fórmula heurística anterior `λ = 1.35 + diff/200`.
"""
from dataclasses import dataclass, asdict, field

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln

from .elo import expected_score

MAX_GOALS = 10


def dc_tau(x, y, la, lb, rho):
    tau = np.ones(np.broadcast(x, y, la).shape)
    tau = np.where((x == 0) & (y == 0), 1 - la * lb * rho, tau)
    tau = np.where((x == 0) & (y == 1), 1 + la * rho, tau)
    tau = np.where((x == 1) & (y == 0), 1 + lb * rho, tau)
    tau = np.where((x == 1) & (y == 1), 1 - rho, tau)
    return tau


def score_matrix(la, lb, rho=0.0, max_goals=MAX_GOALS):
    """Matriz P(goles_A=i, goles_B=j). Acepta escalares o arrays (devuelve (n, G, G))."""
    la = np.atleast_1d(np.asarray(la, float))
    lb = np.atleast_1d(np.asarray(lb, float))
    g = np.arange(max_goals + 1)
    log_fact = gammaln(g + 1)
    pa = np.exp(g[None, :] * np.log(la[:, None]) - la[:, None] - log_fact)
    pb = np.exp(g[None, :] * np.log(lb[:, None]) - lb[:, None] - log_fact)
    m = pa[:, :, None] * pb[:, None, :]
    if rho:
        m[:, 0, 0] *= 1 - la * lb * rho
        m[:, 0, 1] *= 1 + la * rho
        m[:, 1, 0] *= 1 + lb * rho
        m[:, 1, 1] *= 1 - rho
    m /= m.sum(axis=(1, 2), keepdims=True)
    return m


def outcome_probs(m):
    """(P(gana A), P(empate), P(gana B)) a partir de matrices de marcador (n, G, G)."""
    p_draw = np.trace(m, axis1=1, axis2=2)
    p_a = np.tril(m, -1).sum(axis=(1, 2))
    p_b = np.triu(m, 1).sum(axis=(1, 2))
    return p_a, p_draw, p_b


@dataclass
class GoalModel:
    mu: float = 0.25
    beta: float = 1.0
    gamma: float = 0.15
    delta: float = 0.15
    omega: float = 0.0
    rho: float = -0.05
    et_factor: float = 1.0 / 3.0   # fracción de goles esperados en 30' de prórroga
    fitted_on: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    def lambdas(self, ra, rb, home_a=0.0, wc=0.0):
        x = (np.asarray(ra, float) - np.asarray(rb, float)) / 400.0
        la = np.exp(self.mu + self.beta * x + self.gamma * home_a + self.omega * wc)
        lb = np.exp(self.mu - self.beta * x - self.delta * home_a + self.omega * wc)
        return la, lb

    def matrix(self, ra, rb, home_a=0.0, wc=0.0):
        la, lb = self.lambdas(ra, rb, home_a, wc)
        return score_matrix(la, lb, self.rho)

    def probs(self, ra, rb, home_a=0.0, wc=0.0):
        return outcome_probs(self.matrix(ra, rb, home_a, wc))

    # --- Ajuste por máxima verosimilitud ---
    @staticmethod
    def _neg_loglik(theta, x, home, wc, ga, gb, w):
        mu, beta, gamma, delta, omega, rho = theta
        la = np.exp(mu + beta * x + gamma * home + omega * wc)
        lb = np.exp(mu - beta * x - delta * home + omega * wc)
        tau = dc_tau(ga, gb, la, lb, rho)
        if np.any(tau <= 0):
            return 1e12
        ll = (np.log(tau) + ga * np.log(la) - la - gammaln(ga + 1)
              + gb * np.log(lb) - lb - gammaln(gb + 1))
        return -np.sum(w * ll)

    @classmethod
    def fit(cls, ra, rb, home, wc, ga, gb, weights=None, fit_omega=True):
        ra, rb = np.asarray(ra, float), np.asarray(rb, float)
        x = (ra - rb) / 400.0
        home = np.asarray(home, float)
        wc = np.asarray(wc, float) if fit_omega else np.zeros_like(x)
        ga, gb = np.asarray(ga, float), np.asarray(gb, float)
        w = np.ones_like(x) if weights is None else np.asarray(weights, float)
        theta0 = np.array([0.2, 1.0, 0.15, 0.15, 0.0, -0.05])
        bounds = [(-2, 2), (0, 5), (-1, 1), (-1, 1), (-1, 1), (-0.3, 0.3)]
        res = minimize(cls._neg_loglik, theta0, args=(x, home, wc, ga, gb, w),
                       method='L-BFGS-B', bounds=bounds)
        mu, beta, gamma, delta, omega, rho = res.x
        return cls(mu=mu, beta=beta, gamma=gamma, delta=delta, omega=omega, rho=rho,
                   fitted_on={'n_matches': int(len(x)), 'neg_loglik': float(res.fun)})


@dataclass
class ShootoutModel:
    """P(ganar tanda) = expected_score(scale · diferencia_Elo). scale≈0 ⇒ moneda al aire."""
    scale: float = 0.15
    p_kick: float = 0.75

    def to_dict(self):
        return asdict(self)

    def win_prob(self, ra, rb):
        return expected_score(self.scale * (ra - rb))

    @classmethod
    def fit(cls, rating_diffs, a_won):
        d = np.asarray(rating_diffs, float)
        y = np.asarray(a_won, float)

        def nll(s):
            p = np.clip(expected_score(s * d), 1e-9, 1 - 1e-9)
            return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

        res = minimize_scalar(nll, bounds=(0.0, 1.0), method='bounded')
        return cls(scale=float(res.x))
