"""Sistema Elo para selecciones (estilo World Football Elo Ratings / eloratings.net).

Diferencias con la versión anterior del proyecto:
- K depende de la importancia del torneo (Mundial 60, continentales 50, eliminatorias 40,
  otros torneos 30, amistosos 20) en vez de un K fijo o amistosos con K≈0.
- K se multiplica según la diferencia de goles (golear pesa más que ganar 1-0).
- La ventaja de local solo se aplica cuando el partido NO es en cancha neutral.
- Se calcula sobre toda la historia (1872 → hoy), así no hay ratings "recién nacidos" en 1500.
"""
from dataclasses import dataclass, asdict

import numpy as np

CONTINENTAL_FINALS = {
    'UEFA Euro', 'Copa América', 'African Cup of Nations', 'AFC Asian Cup', 'Gold Cup',
    'CONCACAF Championship', 'Oceania Nations Cup', 'Confederations Cup',
}
MAJOR_OTHER = {'UEFA Nations League', 'CONCACAF Nations League'}


def tournament_k(tournament):
    if tournament == 'FIFA World Cup':
        return 60.0
    if tournament in CONTINENTAL_FINALS:
        return 50.0
    if 'qualification' in tournament or tournament in MAJOR_OTHER:
        return 40.0
    if tournament == 'Friendly':
        return 20.0
    return 30.0


def gd_multiplier(goal_diff):
    n = abs(goal_diff)
    if n <= 1:
        return 1.0
    if n == 2:
        return 1.5
    return (11.0 + n) / 8.0


def expected_score(rating_diff):
    """Probabilidad esperada (victoria=1, empate=0.5) para una diferencia de rating."""
    return 1.0 / (1.0 + 10.0 ** (-rating_diff / 400.0))


@dataclass
class EloParams:
    k_scale: float = 1.0       # multiplica todos los K por torneo
    home_adv: float = 100.0    # puntos Elo de ventaja para el local (no neutral)
    initial: float = 1500.0
    use_gd: bool = True

    def to_dict(self):
        return asdict(self)


class EloSystem:
    def __init__(self, params=None):
        self.params = params or EloParams()
        self.ratings = {}

    def rating(self, team):
        return self.ratings.get(team, self.params.initial)

    def update(self, home, away, home_goals, away_goals, k_base, neutral=True):
        """Actualiza ratings tras un partido. Devuelve (rating_local_antes, rating_visita_antes).

        Un partido definido por penales se trata como empate (marcador igualado).
        """
        p = self.params
        rh, ra = self.rating(home), self.rating(away)
        dr = rh - ra + (0.0 if neutral else p.home_adv)
        exp_h = expected_score(dr)
        gd = home_goals - away_goals
        score_h = 1.0 if gd > 0 else (0.0 if gd < 0 else 0.5)
        k = k_base * p.k_scale * (gd_multiplier(gd) if p.use_gd else 1.0)
        delta = k * (score_h - exp_h)
        self.ratings[home] = rh + delta
        self.ratings[away] = ra - delta
        return rh, ra

    def run(self, results, history=False):
        """Procesa todos los partidos en orden. Devuelve arrays de rating previo (local, visita).

        Si history=True devuelve además un DataFrame-friendly list con el historial.
        """
        n = len(results)
        pre_h = np.empty(n)
        pre_a = np.empty(n)
        hist = [] if history else None
        cols = zip(results['home_team'], results['away_team'], results['home_score'],
                   results['away_score'], results['tournament'], results['neutral'], results['date'])
        k_cache = {}
        for i, (h, a, hs, as_, tour, neutral, date) in enumerate(cols):
            k = k_cache.get(tour)
            if k is None:
                k = k_cache[tour] = tournament_k(tour)
            pre_h[i], pre_a[i] = self.update(h, a, hs, as_, k, neutral)
            if history:
                hist.append((date, h, a, hs, as_, tour, pre_h[i], pre_a[i],
                             self.ratings[h], self.ratings[a]))
        return (pre_h, pre_a, hist) if history else (pre_h, pre_a)

    def table(self, teams=None):
        import pandas as pd
        items = self.ratings.items() if teams is None else ((t, self.rating(t)) for t in teams)
        df = pd.DataFrame(items, columns=['team', 'rating'])
        return df.sort_values('rating', ascending=False).reset_index(drop=True)
