"""Backtesting del modelo contra los resultados reales del Mundial 2026 y Mundiales anteriores."""
from copy import copy

import numpy as np
import pandas as pd

from . import metrics as M
from .config import FIFA_RATINGS_CSV, WC2026_START
from .data import home_team_advanced, known_results_from, world_cup_knockout_mask
from .elo import EloSystem
from .goals import outcome_probs, score_matrix
from .model import build_model
from .tournament import TournamentSimulator

MODEL_NAMES = {
    'original': 'Original (ranking FIFA + λ heurístico)',
    'nuevo_pre': 'Nuevo, congelado pre-Mundial',
    'nuevo_live': 'Nuevo, Elo actualizado partido a partido',
}


# --- Modelo original del proyecto (simulacion_eliminacion_directa.ipynb) ---
def legacy_lambdas(r_home, r_away):
    diff = np.asarray(r_home, float) - np.asarray(r_away, float)
    return np.maximum(0.2, 1.35 + diff / 200), np.maximum(0.2, 1.35 - diff / 300)


def legacy_probs(r_home, r_away):
    la, lb = legacy_lambdas(r_home, r_away)
    return np.column_stack(outcome_probs(score_matrix(la, lb, 0.0, max_goals=15)))


def legacy_advance(r_home, r_away):
    la, lb = legacy_lambdas(r_home, r_away)
    w, d, _ = outcome_probs(score_matrix(la, lb, 0.0, max_goals=15))
    ew, ed, _ = outcome_probs(score_matrix(la / 3, lb / 3, 0.0, max_goals=15))
    return w + d * (ew + ed * 0.5)


def load_fifa_ratings(path=FIFA_RATINGS_CSV):
    df = pd.read_csv(path)
    return dict(zip(df['Equipo'], df['Rating']))


# --- Backtest partido a partido ---
def match_backtest(results, shootouts, wc, pre_model=None):
    """Probabilidades de cada modelo para los 104 partidos del Mundial 2026.

    Fase de grupos: 1X2. Eliminación directa: P(equipo "home" avanza).
    """
    pre_model = pre_model or build_model(WC2026_START, results=results, shootouts=shootouts)
    fifa = load_fifa_ratings()

    # Elo "en vivo": mismos parámetros que el modelo pre-Mundial, pero los ratings
    # se actualizan con cada resultado del torneo (sin mirar partidos futuros).
    live = _live_ratings(results, pre_model, wc)
    live_model = copy(pre_model)

    rows = []
    for r, (ra, rb) in zip(wc.itertuples(), live):
        a, b = r.home_team, r.away_team
        home = 0.0 if r.neutral else 1.0
        live_model.ratings = {a: ra, b: rb}
        row = {'date': r.date.date(), 'stage': r.stage, 'home_team': a, 'away_team': b,
               'home_score': r.home_score, 'away_score': r.away_score, 'winner': r.winner,
               'penalties': r.penalties}
        if r.stage == 'GROUP':
            for name, m in (('nuevo_pre', pre_model), ('nuevo_live', live_model)):
                p = m.probs(a, b, home)
                row.update({f'{name}_p1': p[0], f'{name}_pX': p[1], f'{name}_p2': p[2]})
            p = legacy_probs([fifa[a]], [fifa[b]])[0]
            row.update({'original_p1': p[0], 'original_pX': p[1], 'original_p2': p[2]})
        else:
            row['nuevo_pre_adv'] = pre_model.advance_prob(a, b, home)
            row['nuevo_live_adv'] = live_model.advance_prob(a, b, home)
            row['original_adv'] = float(legacy_advance([fifa[a]], [fifa[b]])[0])
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_backtest(bt):
    grp = bt[bt['stage'] == 'GROUP']
    ko = bt[bt['stage'] != 'GROUP']
    outcomes = M.outcome_index(grp['home_score'], grp['away_score'])
    y = (ko['winner'] == ko['home_team']).astype(int).to_numpy()
    rows = []
    for name, label in MODEL_NAMES.items():
        p = grp[[f'{name}_p1', f'{name}_pX', f'{name}_p2']].to_numpy()
        g = M.summary(p, outcomes)
        k = M.binary_scores(ko[f'{name}_adv'].to_numpy(), y)
        rows.append({
            'modelo': label,
            'grupos_log_loss': g['log_loss'], 'grupos_brier': g['brier'], 'grupos_rps': g['rps'],
            'grupos_acierto': g['accuracy'], 'grupos_empates_predichos': p[:, 1].mean(),
            'ko_log_loss': k['log_loss'], 'ko_brier': k['brier'], 'ko_acierto': k['accuracy'],
        })
    out = pd.DataFrame(rows)
    out.attrs['empates_reales'] = float((outcomes == 1).mean())
    return out


def biggest_surprises(bt, n=10):
    """Partidos con menor probabilidad asignada (modelo pre-Mundial) al resultado real."""
    grp = bt[bt['stage'] == 'GROUP'].copy()
    o = M.outcome_index(grp['home_score'], grp['away_score'])
    p = grp[['nuevo_pre_p1', 'nuevo_pre_pX', 'nuevo_pre_p2']].to_numpy()
    grp['p_resultado_real'] = p[np.arange(len(o)), o]
    ko = bt[bt['stage'] != 'GROUP'].copy()
    ko['p_resultado_real'] = np.where(ko['winner'] == ko['home_team'],
                                      ko['nuevo_pre_adv'], 1 - ko['nuevo_pre_adv'])
    cols = ['date', 'stage', 'home_team', 'home_score', 'away_score', 'away_team', 'winner',
            'p_resultado_real']
    return pd.concat([grp, ko])[cols].sort_values('p_resultado_real').head(n).reset_index(drop=True)


# --- Evolución de probabilidades a lo largo del torneo ---
CHECKPOINTS = [
    ('Pre-Mundial', []),
    ('Tras fase de grupos', ['GROUP']),
    ('Tras dieciseisavos', ['GROUP', 'R32']),
    ('Tras octavos', ['GROUP', 'R32', 'R16']),
    ('Tras cuartos', ['GROUP', 'R32', 'R16', 'QF']),
    ('Tras semifinales', ['GROUP', 'R32', 'R16', 'QF', 'SF']),
]


def tournament_evolution(pre_model, wc, group_fixtures, n_sims=5000, seed=7, dynamic_k=60.0):
    """Probabilidad de campeón de cada equipo en cada punto del torneo (resultados reales fijados)."""
    frames = []
    for label, stages in CHECKPOINTS:
        known = known_results_from(wc[wc['stage'].isin(stages)])
        sim = TournamentSimulator(pre_model, group_fixtures=group_fixtures, known=known,
                                  dynamic_k=dynamic_k, seed=seed)
        df = sim.run(n_sims)[['team', 'Campeón %']].rename(columns={'Campeón %': label})
        frames.append(df.set_index('team'))
    out = pd.concat(frames, axis=1).fillna(0.0)
    return out.sort_values('Pre-Mundial', ascending=False)


# --- Validación en Mundiales anteriores (¿las mejoras generalizan?) ---
HISTORY_VARIANTS = [
    ('base (Elo + Dixon-Coles)', dict(use_conf_offsets=False, use_ko_sharpness=False)),
    ('+ corrección confederación', dict(use_conf_offsets=True, use_ko_sharpness=False)),
    ('+ nitidez eliminatorias (final)', dict(use_conf_offsets=True, use_ko_sharpness=True)),
]


def world_cup_history_backtest(results, shootouts, years=(2006, 2010, 2014, 2018, 2022, 2026)):
    """Evalúa cada Mundial entrenando SOLO con datos previos a ese Mundial (sin mirar el futuro).

    Grupos: 1X2 con ratings congelados al inicio. Eliminatorias: P(avanzar) con el Elo
    actualizado hasta el día anterior (lo que se sabía antes de cada partido).
    """
    wcm = results[results['tournament'] == 'FIFA World Cup']
    ko_mask = world_cup_knockout_mask(results)
    rows = []
    for y in years:
        w = wcm[wcm['date'].dt.year == y]
        start = str(w['date'].min().date())
        g = w.iloc[:72 if y == 2026 else 48]
        o = M.outcome_index(g['home_score'], g['away_score'])
        ko = results[ko_mask & (results['date'].dt.year == y).to_numpy()]
        y_ko = home_team_advanced(ko, shootouts)
        ok = ~np.isnan(y_ko)
        for label, kwargs in HISTORY_VARIANTS:
            mdl = build_model(start, results=results, shootouts=shootouts, **kwargs)
            p = np.array([mdl.probs(a, b, 0.0 if n else 1.0)
                          for a, b, n in zip(g['home_team'], g['away_team'], g['neutral'])])
            s = M.summary(p, o)
            live = _live_ratings(results, mdl, ko)
            p_ko = []
            for (a, b, n), (ra, rb) in zip(zip(ko['home_team'], ko['away_team'], ko['neutral']), live):
                m = copy(mdl)
                m.ratings = {a: ra, b: rb}
                p_ko.append(m.advance_prob(a, b, 0.0 if n else 1.0))
            k = M.binary_scores(np.array(p_ko)[ok], y_ko[ok])
            rows.append({'mundial': y, 'variante': label,
                         'grupos_log_loss': s['log_loss'], 'grupos_rps': s['rps'],
                         'grupos_acierto': s['accuracy'],
                         'ko_log_loss': k['log_loss'], 'ko_brier': k['brier'],
                         'ko_acierto': k['accuracy'], 'ko_sharpness': mdl.ko_sharpness})
    return pd.DataFrame(rows)


def _live_ratings(results, model, matches):
    """Rating Elo de cada equipo justo antes de cada partido de `matches`."""
    upto = results[results['date'] <= matches['date'].max()].reset_index(drop=True) \
        if len(matches) else results.iloc[:0]
    elo = EloSystem(model.elo_params)
    pre_h, pre_a = elo.run(upto)
    key = upto['date'].astype(str) + '|' + upto['home_team'] + '|' + upto['away_team']
    idx = pd.Series(np.arange(len(upto)), index=key)
    idx = idx[~idx.index.duplicated()]
    out = []
    for r in matches.itertuples():
        i = idx[f'{r.date.date()}|{r.home_team}|{r.away_team}']
        out.append((pre_h[i], pre_a[i]))
    return out
