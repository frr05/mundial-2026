"""Modelo completo: Elo + corrección por confederación + Dixon-Coles + penales.

`build_model(cutoff)` entrena todo usando solo partidos ANTERIORES a `cutoff`, lo que
permite evaluar honestamente (sin mirar el futuro) y reentrenar con datos nuevos.
"""
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from .config import GOAL_MODEL_FROM
from .data import (CONFEDERATION, home_team_advanced, load_results, load_shootouts,
                   world_cup_knockout_mask)
from .elo import EloParams, EloSystem
from .goals import GoalModel, ShootoutModel, dc_tau, outcome_probs, score_matrix

CONFEDERATIONS = ['UEFA', 'CONMEBOL', 'CONCACAF', 'CAF', 'AFC', 'OFC']
# Desviación estándar del prior (puntos Elo) para las correcciones por confederación.
CONF_PRIOR_SD = 50.0
CONF_FIT_FROM = '1998-01-01'


@dataclass
class Model:
    ratings: dict
    goal_model: GoalModel
    shootout: ShootoutModel
    elo_params: EloParams
    conf_offsets: dict = field(default_factory=dict)
    ko_sharpness: float = 1.0   # multiplica la diferencia de fuerza en eliminación directa
    cutoff: str = ''

    def rating(self, team):
        return self.ratings.get(team, self.elo_params.initial)

    def strength(self, team):
        """Rating efectivo para predecir: Elo + corrección de su confederación."""
        return self.rating(team) + self.conf_offsets.get(CONFEDERATION.get(team), 0.0)

    def probs(self, team_a, team_b, home_a=0.0, wc=1.0):
        pa, pd_, pb = self.goal_model.probs(self.strength(team_a), self.strength(team_b), home_a, wc)
        return float(pa[0]), float(pd_[0]), float(pb[0])

    def advance_prob(self, team_a, team_b, home_a=0.0):
        """P(team_a avanza) en eliminación directa: 90' + prórroga + penales."""
        diff = self.strength(team_a) - self.strength(team_b)
        return float(advance_probs(self.goal_model, self.shootout, diff, home_a,
                                   self.ko_sharpness)[0])

    def ratings_table(self, teams=None):
        teams = list(self.ratings) if teams is None else teams
        df = pd.DataFrame({
            'team': teams,
            'elo': [self.rating(t) for t in teams],
            'confederation': [CONFEDERATION.get(t, '') for t in teams],
        })
        df['strength'] = [self.strength(t) for t in teams]
        return df.sort_values('strength', ascending=False).reset_index(drop=True)

    def to_dict(self):
        return {
            'cutoff': self.cutoff,
            'elo_params': self.elo_params.to_dict(),
            'goal_model': self.goal_model.to_dict(),
            'shootout': self.shootout.to_dict(),
            'conf_offsets': self.conf_offsets,
            'ko_sharpness': self.ko_sharpness,
        }

    def save(self, config_path, ratings_path):
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=float)
        self.ratings_table().to_csv(ratings_path, index=False)


def advance_probs(goal_model, shootout, diff, home=0.0, sharpness=1.0):
    """Vectorizado: P(avanza A) dada la diferencia de fuerza A−B (90' + prórroga + penales)."""
    gm = goal_model
    diff = np.atleast_1d(np.asarray(diff, float))
    la, lb = gm.lambdas(diff * sharpness, 0.0, home, 1.0)
    la, lb = np.broadcast_arrays(la, lb)
    p_win, p_draw, _ = outcome_probs(score_matrix(la, lb, gm.rho))
    e_win, e_draw, _ = outcome_probs(score_matrix(la * gm.et_factor, lb * gm.et_factor))
    p_pen = shootout.win_prob(diff, 0.0)
    return p_win + p_draw * (e_win + e_draw * p_pen)


# --- Ajuste ----------------------------------------------------------------

def fit_goal_model(results, pre_h, pre_a, since=GOAL_MODEL_FROM):
    m = (results['date'] >= since).to_numpy()
    d = results[m]
    return GoalModel.fit(
        pre_h[m], pre_a[m], (~d['neutral']).to_numpy(float),
        (d['tournament'] == 'FIFA World Cup').to_numpy(float),
        d['home_score'].to_numpy(), d['away_score'].to_numpy())


def fit_conf_offsets(results, pre_h, pre_a, goal_model, since=CONF_FIT_FROM,
                     prior_sd=CONF_PRIOR_SD, exclude_years=()):
    """Estima cuántos puntos Elo sobre/sub-valora el sistema a cada confederación en Mundiales.

    Se usa solo partidos de Mundial entre confederaciones distintas, con un prior normal
    (ridge) centrado en 0 para no sobreajustar. UEFA es la referencia (offset 0).
    """
    m = ((results['tournament'] == 'FIFA World Cup') & (results['date'] >= since)).to_numpy()
    if exclude_years:
        m &= ~results['date'].dt.year.isin(exclude_years).to_numpy()
    d = results[m]
    ca = d['home_team'].map(CONFEDERATION).to_numpy()
    cb = d['away_team'].map(CONFEDERATION).to_numpy()
    keep = pd.notna(ca) & pd.notna(cb) & (ca != cb)
    if keep.sum() == 0:
        return {}
    ra, rb = pre_h[m][keep], pre_a[m][keep]
    ca, cb = ca[keep], cb[keep]
    home = (~d['neutral']).to_numpy(float)[keep]
    ga, gb = d['home_score'].to_numpy(float)[keep], d['away_score'].to_numpy(float)[keep]
    free = CONFEDERATIONS[1:]
    ia = np.array([free.index(c) if c in free else -1 for c in ca])
    ib = np.array([free.index(c) if c in free else -1 for c in cb])
    gm = goal_model

    def nll(off):
        o = np.append(off, 0.0)  # índice -1 → UEFA = 0
        x = (ra + o[ia] - rb - o[ib]) / 400.0
        la = np.exp(gm.mu + gm.beta * x + gm.gamma * home + gm.omega)
        lb = np.exp(gm.mu - gm.beta * x - gm.delta * home + gm.omega)
        tau = np.clip(dc_tau(ga, gb, la, lb, gm.rho), 1e-12, None)
        ll = np.log(tau) + ga * np.log(la) - la + gb * np.log(lb) - lb
        return -ll.sum() + 0.5 * np.sum((off / prior_sd) ** 2)

    res = minimize(nll, np.zeros(len(free)), method='L-BFGS-B')
    offsets = {'UEFA': 0.0}
    offsets.update({c: float(v) for c, v in zip(free, res.x)})
    return offsets


def fit_shootout_model(results, pre_h, pre_a, shootouts):
    keys = ['date', 'home_team', 'away_team']
    matches = results[keys].assign(diff=pre_h - pre_a).drop_duplicates(keys)
    d = shootouts[keys + ['winner']].merge(matches, on=keys, how='inner')
    return ShootoutModel.fit(d['diff'], d['winner'] == d['home_team'])


def fit_ko_sharpness(results, shootouts, pre_h, pre_a, goal_model, shootout, offsets,
                     exclude_years=()):
    """Ajusta la "nitidez" de las eliminatorias de Mundial (desde 1986) por máxima verosimilitud.

    Históricamente el favorito avanza más seguido de lo que predice el modelo de 90'
    (73% real vs 65% predicho en 192 partidos); este factor corrige ese sesgo.
    """
    m = world_cup_knockout_mask(results)
    if exclude_years:
        m &= ~results['date'].dt.year.isin(exclude_years).to_numpy()
    d = results[m]
    if len(d) == 0:
        return 1.0
    home_adv = home_team_advanced(d, shootouts)
    ok = ~np.isnan(home_adv)
    off = lambda teams: teams.map(CONFEDERATION).map(offsets).fillna(0.0).to_numpy()
    diff = (pre_h[m] + off(d['home_team']) - pre_a[m] - off(d['away_team']))[ok]
    home = (~d['neutral']).to_numpy(float)[ok]
    y = home_adv[ok]

    def nll(kappa):
        p = np.clip(advance_probs(goal_model, shootout, diff, home, kappa), 1e-9, 1 - 1e-9)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    return float(minimize_scalar(nll, bounds=(0.5, 3.0), method='bounded').x)


def build_model(cutoff=None, elo_params=None, results=None, shootouts=None,
                use_conf_offsets=True, use_ko_sharpness=True, exclude_years=()):
    """Entrena el modelo con todos los partidos jugados antes de `cutoff` (YYYY-MM-DD)."""
    results = load_results() if results is None else results
    shootouts = load_shootouts() if shootouts is None else shootouts
    if cutoff is not None:
        results = results[results['date'] < cutoff].reset_index(drop=True)
        shootouts = shootouts[shootouts['date'] < cutoff]
    elo = EloSystem(elo_params or EloParams())
    pre_h, pre_a = elo.run(results)
    gm = fit_goal_model(results, pre_h, pre_a)
    offsets = (fit_conf_offsets(results, pre_h, pre_a, gm, exclude_years=exclude_years)
               if use_conf_offsets else {})
    so = fit_shootout_model(results, pre_h, pre_a, shootouts)
    kappa = (fit_ko_sharpness(results, shootouts, pre_h, pre_a, gm, so, offsets, exclude_years)
             if use_ko_sharpness else 1.0)
    return Model(ratings=dict(elo.ratings), goal_model=gm, shootout=so,
                 elo_params=elo.params, conf_offsets=offsets, ko_sharpness=kappa,
                 cutoff=cutoff or str(results['date'].max().date()))
