"""Simulación Monte Carlo del Mundial 2026 (fase de grupos + Partidos 73-104).

Correcciones respecto a la versión del notebook:
- Las probabilidades por fase se calculan como "alcanzó al menos la fase X", por lo que
  son monótonas y nunca superan 100% (antes se contaban eliminados dos veces).
- Desempates de grupo según reglamento FIFA 2026: puntos, luego enfrentamiento directo
  (puntos, DG, GF), luego DG y GF generales y, finalmente, el rating (proxy del ranking FIFA).
- Ventaja de local solo para anfitriones jugando en su país.
- Prórroga y penales con el modelo calibrado; en eliminación directa la diferencia de
  fuerza se multiplica por `ko_sharpness` (calibrado con Mundiales 1986-2026).
- Elo dinámico opcional durante el torneo.
- Se pueden fijar resultados ya jugados (`known`) para simular solo lo que falta.
"""
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

from .data import (HOST_COUNTRY, KNOCKOUT_FIXTURE, OFFICIAL_THIRDS_ASSIGNMENT, R32_FIXTURE,
                   STAGES, THIRD_PLACE_ALLOWED, THIRD_PLACE_MATCH, WC2026_GROUPS)
from .elo import expected_score, gd_multiplier
from .goals import score_matrix

STAGE_INDEX = {s: i for i, s in enumerate(STAGES)}
THIRD_SLOT_WINNERS = ['A', 'B', 'D', 'E', 'G', 'I', 'K', 'L']


def default_group_fixtures(groups=WC2026_GROUPS):
    """Round-robin por grupo (local nominal = primero listado). Sede: neutral salvo anfitriones."""
    fixtures = []
    for g, teams in groups.items():
        for a, b in combinations(teams, 2):
            host_country = HOST_COUNTRY.get(a) or HOST_COUNTRY.get(b)
            fixtures.append((g, a, b, host_country))
    return fixtures


def group_fixtures_from_results(results_all, groups=WC2026_GROUPS):
    """Fixture real (con país sede) a partir de results.csv, incluso si aún no se juega."""
    wc = results_all[(results_all['tournament'] == 'FIFA World Cup')
                     & (results_all['date'] >= '2026-06-11') & (results_all['date'] <= '2026-06-27')]
    team_group = {t: g for g, ts in groups.items() for t in ts}
    fixtures = [(team_group[r.home_team], r.home_team, r.away_team, r.country)
                for r in wc.itertuples() if r.home_team in team_group]
    return fixtures if len(fixtures) == 72 else default_group_fixtures(groups)


def assign_thirds(qualified_groups):
    """Empareja los 8 mejores terceros con los primeros (Anexo C si se conoce; si no, backtracking)."""
    key = frozenset(qualified_groups)
    if key in OFFICIAL_THIRDS_ASSIGNMENT:
        return OFFICIAL_THIRDS_ASSIGNMENT[key]

    def backtrack(i, available, assignment):
        if i == len(THIRD_SLOT_WINNERS):
            return assignment
        w = THIRD_SLOT_WINNERS[i]
        for g in THIRD_PLACE_ALLOWED[w]:
            if g in available:
                res = backtrack(i + 1, available - {g}, {**assignment, w: g})
                if res is not None:
                    return res
        return None

    res = backtrack(0, set(qualified_groups), {})
    if res is None:  # no debería ocurrir: el reglamento garantiza solución
        raise ValueError(f'Sin emparejamiento válido para terceros {sorted(qualified_groups)}')
    return res


def rank_group(teams, matches, tiebreak_value):
    """Ordena un grupo. matches: lista de (a, b, ga, gb). tiebreak_value: rating (mayor = mejor)."""
    def table(subset, ms):
        t = {x: [0, 0, 0] for x in subset}  # pts, gd, gf
        for a, b, ga, gb in ms:
            if a in t and b in t:
                t[a][1] += ga - gb
                t[a][2] += ga
                t[b][1] += gb - ga
                t[b][2] += gb
                if ga > gb:
                    t[a][0] += 3
                elif gb > ga:
                    t[b][0] += 3
                else:
                    t[a][0] += 1
                    t[b][0] += 1
        return t

    overall = table(teams, matches)

    def order(subset, depth=0):
        if len(subset) <= 1:
            return list(subset)
        if depth == 0:
            key = {x: tuple(overall[x][:1]) for x in subset}
        else:
            h2h = table(subset, matches)
            key = {x: tuple(h2h[x]) for x in subset}
        buckets = defaultdict(list)
        for x in subset:
            buckets[key[x]].append(x)
        out = []
        for k in sorted(buckets, reverse=True):
            tied = buckets[k]
            if len(tied) == 1:
                out += tied
            elif depth == 0 or len(tied) < len(subset):
                out += order(tied, depth + 1)  # aplicar (o re-aplicar) enfrentamiento directo
            else:
                out += sorted(tied, key=lambda x: (overall[x][1], overall[x][2], tiebreak_value(x)),
                              reverse=True)
        return out

    ranked = order(list(teams))
    return ranked, overall


class TournamentSimulator:
    def __init__(self, model, groups=WC2026_GROUPS, group_fixtures=None, known=None,
                 dynamic_k=60.0, seed=None):
        """
        model: modelo.model.Model
        known: {(etapa, frozenset({a, b})): {a: goles, b: goles, 'winner': equipo}}
        dynamic_k: K del Elo que se actualiza dentro de cada simulación (0 = ratings fijos).
                   Representa la incertidumbre sobre la fuerza real de cada equipo.
        """
        self.model = model
        self.groups = groups
        self.fixtures = group_fixtures or default_group_fixtures(groups)
        self.known = known or {}
        self.dynamic_k = dynamic_k
        self.rng = np.random.default_rng(seed)
        self.gm = model.goal_model
        self.home_adv = model.elo_params.home_adv
        self._cache = {}

    # --- partidos ---
    def _cum_matrix(self, diff, home):
        key = (int(round(diff)), home)
        cm = self._cache.get(key)
        if cm is None:
            la, lb = self.gm.lambdas(diff, 0.0, home, 1.0)
            m = score_matrix(la, lb, self.gm.rho)[0]
            cm = (np.cumsum(m.ravel()), m.shape[1], float(np.squeeze(la)), float(np.squeeze(lb)))
            self._cache[key] = cm
        return cm

    def _play_90(self, ra, rb, home, sharpness=1.0):
        cum, n, la, lb = self._cum_matrix((ra - rb) * sharpness, home)
        idx = int(np.searchsorted(cum, self.rng.random() * cum[-1]))
        return idx // n, idx % n, la, lb

    def _home_flag(self, team_a, country):
        return 1.0 if country is not None and HOST_COUNTRY.get(team_a) == country else 0.0

    def _update(self, r, a, b, ga, gb, home):
        if not self.dynamic_k:
            return
        dr = r[a] - r[b] + home * self.home_adv
        gd = ga - gb
        s = 1.0 if gd > 0 else (0.0 if gd < 0 else 0.5)
        delta = self.dynamic_k * gd_multiplier(gd) * (s - expected_score(dr))
        r[a] += delta
        r[b] -= delta

    def play_group_match(self, r, a, b, country):
        k = self.known.get(('GROUP', frozenset((a, b))))
        home = self._home_flag(a, country)
        if home == 0.0 and self._home_flag(b, country):
            a, b, home = b, a, 1.0
        if k is not None:
            ga, gb = k[a], k[b]
        else:
            ga, gb, _, _ = self._play_90(r[a], r[b], home)
        self._update(r, a, b, ga, gb, home)
        return a, b, ga, gb

    def play_knockout(self, r, a, b, country, stage):
        """Devuelve (ganador, perdedor)."""
        home = self._home_flag(a, country)
        if home == 0.0 and self._home_flag(b, country):
            a, b, home = b, a, 1.0
        k = self.known.get((stage, frozenset((a, b))))
        if k is not None:
            ga, gb, winner = k[a], k[b], k['winner']
        else:
            ga, gb, la, lb = self._play_90(r[a], r[b], home, self.model.ko_sharpness)
            if ga == gb:  # prórroga
                f = self.gm.et_factor
                ga += self.rng.poisson(la * f)
                gb += self.rng.poisson(lb * f)
            if ga != gb:
                winner = a if ga > gb else b
            else:  # penales
                p_a = self.model.shootout.win_prob(r[a], r[b])
                winner = a if self.rng.random() < p_a else b
        self._update(r, a, b, ga, gb, home)
        return winner, (b if winner == a else a)

    # --- torneo ---
    def simulate_once(self):
        r = {t: self.model.strength(t) for ts in self.groups.values() for t in ts}
        base = dict(r)
        tiebreak = base.get  # rating inicial como proxy del ranking FIFA
        group_matches = defaultdict(list)
        for g, a, b, country in self.fixtures:
            group_matches[g].append(self.play_group_match(r, a, b, country))

        slots, positions, thirds, points = {}, {}, [], {}
        for g, teams in self.groups.items():
            ranked, tab = rank_group(teams, group_matches[g], tiebreak)
            for pos, t in enumerate(ranked, 1):
                positions[t] = pos
                points[t] = tab[t][0]
                slots[f'{pos}{g}'] = t
            t3 = ranked[2]
            thirds.append((tab[t3][0], tab[t3][1], tab[t3][2], tiebreak(t3), g))
        thirds.sort(reverse=True)
        qualified_third_groups = [x[4] for x in thirds[:8]]
        for w, g in assign_thirds(qualified_third_groups).items():
            slots[f'T{w}'] = slots[f'3{g}']

        reached = {t: 0 for t in base}
        winners = {}
        for mid, (sa, sb, country) in R32_FIXTURE.items():
            a, b = slots[sa], slots[sb]
            reached[a] = reached[b] = STAGE_INDEX['R32']
            winners[mid], _ = self.play_knockout(r, a, b, country, 'R32')

        losers = {}
        for mid, (ma, mb, country, stage) in KNOCKOUT_FIXTURE.items():
            a, b = winners[ma], winners[mb]
            reached[a] = reached[b] = STAGE_INDEX[stage]
            winners[mid], losers[mid] = self.play_knockout(r, a, b, country, stage)
        champion = winners[104]
        reached[champion] = STAGE_INDEX['CHAMPION']

        _, country3 = THIRD_PLACE_MATCH
        third, _ = self.play_knockout(r, losers[101], losers[102], country3, '3RD')
        return {'reached': reached, 'positions': positions, 'points': points,
                'champion': champion, 'runner_up': losers[104], 'third': third,
                'final_ratings': r}

    def run(self, n_sims=10000, progress=False):
        teams = [t for ts in self.groups.values() for t in ts]
        reach_counts = {t: np.zeros(len(STAGES)) for t in teams}
        pos_counts = {t: np.zeros(4) for t in teams}
        pts_sum = defaultdict(float)
        third_counts = defaultdict(int)
        for i in range(n_sims):
            if progress and (i + 1) % max(1, n_sims // 10) == 0:
                print(f'  simulación {i + 1}/{n_sims}')
            s = self.simulate_once()
            for t, st in s['reached'].items():
                reach_counts[t][:st + 1] += 1
            for t, p in s['positions'].items():
                pos_counts[t][p - 1] += 1
                pts_sum[t] += s['points'][t]
            third_counts[s['third']] += 1

        team_group = {t: g for g, ts in self.groups.items() for t in ts}
        rows = []
        for t in teams:
            rc = reach_counts[t] / n_sims * 100
            pc = pos_counts[t] / n_sims * 100
            rows.append({
                'team': t, 'group': team_group[t],
                'elo': round(self.model.rating(t), 1),
                'strength': round(self.model.strength(t), 1),
                'pts_promedio': round(pts_sum[t] / n_sims, 2),
                '1ro_grupo_%': pc[0], '2do_grupo_%': pc[1], '3ro_grupo_%': pc[2], '4to_grupo_%': pc[3],
                'Dieciseisavos (R32) %': rc[STAGE_INDEX['R32']],
                'Octavos (R16) %': rc[STAGE_INDEX['R16']],
                'Cuartos (QF) %': rc[STAGE_INDEX['QF']],
                'Semifinal (SF) %': rc[STAGE_INDEX['SF']],
                'Final (F) %': rc[STAGE_INDEX['FINAL']],
                'Campeón %': rc[STAGE_INDEX['CHAMPION']],
                'Tercer puesto %': third_counts[t] / n_sims * 100,
            })
        df = pd.DataFrame(rows).sort_values(['Campeón %', 'Final (F) %', 'strength'], ascending=False)
        return df.reset_index(drop=True)
