"""CLI del modelo predictivo del Mundial 2026.

Uso:
    python simulator.py todo                 # corre todo y escribe Output/ + reporte
    python simulator.py pre   --sims 20000   # probabilidades pre-Mundial (modelo congelado al 10-jun)
    python simulator.py live  --hasta R16    # simula fijando los resultados reales hasta esa fase
    python simulator.py backtest             # evalúa modelos contra los 104 partidos reales
    python simulator.py ratings              # modelo final re-entrenado con el Mundial 2026 incluido
"""
import argparse
import time

import pandas as pd

from modelo.config import MODEL_CONFIG_JSON, OUTPUT_DIR, WC2026_START
from modelo.data import (WC2026_GROUPS, known_results_from, load_results, load_shootouts,
                         wc2026_matches)
from modelo.elo import EloSystem
from modelo.evaluate import (biggest_surprises, match_backtest, summarize_backtest,
                             tournament_evolution, world_cup_history_backtest)
from modelo.model import build_model
from modelo.tournament import TournamentSimulator, group_fixtures_from_results

STAGE_ORDER = ['GROUP', 'R32', 'R16', 'QF', 'SF', '3RD', 'FINAL']
WC_TEAMS = [t for ts in WC2026_GROUPS.values() for t in ts]


class Context:
    """Carga los datos una sola vez y entrena los modelos bajo demanda."""

    def __init__(self):
        self.results = load_results()
        self.results_all = load_results(played_only=False)
        self.shootouts = load_shootouts()
        self.wc = wc2026_matches(self.results, self.shootouts)
        self.fixtures = group_fixtures_from_results(self.results_all)
        self._pre = self._post = None

    @property
    def pre_model(self):
        if self._pre is None:
            self._pre = build_model(WC2026_START, results=self.results, shootouts=self.shootouts)
        return self._pre

    @property
    def post_model(self):
        if self._post is None:
            self._post = build_model(None, results=self.results, shootouts=self.shootouts)
        return self._post


def cmd_pre(ctx, sims, seed):
    print(f'Simulando el Mundial desde cero con el modelo pre-Mundial ({sims} simulaciones)...')
    df = TournamentSimulator(ctx.pre_model, group_fixtures=ctx.fixtures, seed=seed).run(sims)
    df.to_csv(OUTPUT_DIR / 'tournament_probabilities.csv', index=False)
    print(df.head(15)[['team', 'strength', 'Octavos (R16) %', 'Semifinal (SF) %', 'Campeón %']]
          .round(1).to_string(index=False))
    return df


def cmd_live(ctx, until, sims, seed):
    stages = STAGE_ORDER[:STAGE_ORDER.index(until) + 1] if until else []
    known = known_results_from(ctx.wc[ctx.wc['stage'].isin(stages)])
    print(f'Simulando con {len(known)} resultados reales fijados (hasta {until})...')
    df = TournamentSimulator(ctx.pre_model, group_fixtures=ctx.fixtures, known=known,
                             seed=seed).run(sims)
    df.to_csv(OUTPUT_DIR / f'tournament_probabilities_hasta_{until}.csv', index=False)
    print(df.head(10)[['team', 'Semifinal (SF) %', 'Final (F) %', 'Campeón %']]
          .round(1).to_string(index=False))
    return df


def cmd_backtest(ctx, sims, seed):
    print('Backtest partido a partido (104 partidos)...')
    bt = match_backtest(ctx.results, ctx.shootouts, ctx.wc, ctx.pre_model)
    bt.to_csv(OUTPUT_DIR / 'backtest_mundial2026_partidos.csv', index=False)
    summary = summarize_backtest(bt)
    summary.to_csv(OUTPUT_DIR / 'backtest_mundial2026_resumen.csv', index=False)
    print(summary.round(3).to_string(index=False))
    print(f"Empates reales en grupos: {summary.attrs['empates_reales']:.1%}")

    surprises = biggest_surprises(bt)
    surprises.to_csv(OUTPUT_DIR / 'backtest_mundial2026_sorpresas.csv', index=False)

    print('Validación en Mundiales 2006-2026 (entrenando solo con datos previos)...')
    hist = world_cup_history_backtest(ctx.results, ctx.shootouts)
    hist.to_csv(OUTPUT_DIR / 'backtest_mundiales_historicos.csv', index=False)

    print(f'Evolución de probabilidades de campeón ({sims} simulaciones por punto)...')
    evo = tournament_evolution(ctx.pre_model, ctx.wc, ctx.fixtures, n_sims=sims, seed=seed)
    evo.to_csv(OUTPUT_DIR / 'evolucion_probabilidad_campeon.csv')
    print(evo.head(10).round(1).to_string())
    return {'backtest': bt, 'summary': summary, 'surprises': surprises, 'history': hist,
            'evolution': evo}


def cmd_ratings(ctx):
    post = ctx.post_model
    post.save(MODEL_CONFIG_JSON, OUTPUT_DIR / 'ratings_actuales.csv')
    ctx.pre_model.ratings_table().to_csv(OUTPUT_DIR / 'ratings_pre_mundial.csv', index=False)

    # Historial de Elo (formato largo: date, team, rating) desde 2018 para journalist.py
    elo = EloSystem(post.elo_params)
    _, _, hist = elo.run(ctx.results, history=True)
    h = pd.DataFrame(hist, columns=['date', 'home_team', 'away_team', 'home_score', 'away_score',
                                    'tournament', 'home_before', 'away_before', 'home_after',
                                    'away_after'])
    h = h[h['date'] >= '2018-01-01']
    long = pd.concat([
        h[['date', 'home_team', 'home_after']].set_axis(['date', 'team', 'rating'], axis=1),
        h[['date', 'away_team', 'away_after']].set_axis(['date', 'team', 'rating'], axis=1),
    ]).sort_values('date', kind='stable')
    long['rating'] = long['rating'].round(1)
    long.to_csv(OUTPUT_DIR / 'elo_history.csv', index=False)

    pre_t = ctx.pre_model.ratings_table(WC_TEAMS).set_index('team')
    post_t = post.ratings_table(WC_TEAMS).set_index('team')
    change = pd.DataFrame({
        'elo_pre': pre_t['elo'], 'elo_post': post_t['elo'],
        'strength_post': post_t['strength'],
    })
    change['cambio'] = change['elo_post'] - change['elo_pre']
    change = change.sort_values('strength_post', ascending=False)
    change.round(1).to_csv(OUTPUT_DIR / 'cambio_ratings_mundial2026.csv')
    print('Ranking tras el Mundial (top 15):')
    print(change.head(15).round(1).to_string())
    return change


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('comando', choices=['pre', 'live', 'backtest', 'ratings', 'todo'])
    parser.add_argument('--sims', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--hasta', choices=STAGE_ORDER, default='GROUP',
                        help='(live) última fase con resultados reales fijados')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    t0 = time.time()
    ctx = Context()
    if args.comando == 'pre':
        cmd_pre(ctx, args.sims, args.seed)
    elif args.comando == 'live':
        cmd_live(ctx, args.hasta, args.sims, args.seed)
    elif args.comando == 'backtest':
        cmd_backtest(ctx, min(args.sims, 5000), args.seed)
    elif args.comando == 'ratings':
        cmd_ratings(ctx)
    else:
        from modelo.report import write_report
        pre = cmd_pre(ctx, args.sims, args.seed)
        bt = cmd_backtest(ctx, min(args.sims, 5000), args.seed)
        change = cmd_ratings(ctx)
        path = write_report(ctx, pre, bt, change)
        print(f'\nReporte: {path}')
    print(f'Listo en {time.time() - t0:.0f}s. Archivos en {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
