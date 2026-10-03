"""Genera Output/REPORTE_MUNDIAL_2026.md con los resultados del modelo y el backtest."""
from datetime import date

from .config import OUTPUT_DIR
from .evaluate import HISTORY_VARIANTS


def md_table(df, floatfmt='{:.3f}', index=False):
    if index:
        df = df.reset_index()
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(str(c) for c in cols) + ' |',
             '|' + '|'.join('---' for _ in cols) + '|']
    for _, row in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in row]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def write_report(ctx, pre_probs, bt, change, path=None):
    path = path or OUTPUT_DIR / 'REPORTE_MUNDIAL_2026.md'
    pre, post = ctx.pre_model, ctx.post_model
    summary = bt['summary']
    hist = bt['history']
    wc = ctx.wc
    final = wc[wc['stage'] == 'FINAL'].iloc[0]
    champion = final['winner']
    champ_row = pre_probs[pre_probs['team'] == champion].iloc[0]
    rank_champ = int(pre_probs.index[pre_probs['team'] == champion][0]) + 1
    grp = wc[wc['stage'] == 'GROUP']

    out = [
        '# Mundial 2026 — Reporte del modelo predictivo',
        f'_Generado el {date.today()} con `python simulator.py todo`._',
        '',
        '## 1. Resultado real vs. lo que predijo el modelo',
        f'- Campeón real: **{champion}** (final: {final["home_team"]} {final["home_score"]}-'
        f'{final["away_score"]} {final["away_team"]}).',
        f'- Antes del torneo el modelo le daba a {champion} **{champ_row["Campeón %"]:.1f}%** de ser '
        f'campeón (favorito #{rank_champ}) y {champ_row["Final (F) %"]:.1f}% de llegar a la final.',
        f'- Fase de grupos real: {(grp["home_score"] + grp["away_score"]).mean():.2f} goles/partido y '
        f'{summary.attrs["empates_reales"]:.1%} de empates.',
        '',
        '### Probabilidades pre-Mundial (top 12; modelo con todas las mejoras, entrenado solo con datos hasta el 10-jun-2026)',
        md_table(pre_probs.head(12)[['team', 'strength', 'Octavos (R16) %', 'Cuartos (QF) %',
                                     'Semifinal (SF) %', 'Final (F) %', 'Campeón %']],
                 '{:.1f}'),
        '',
        '## 2. Backtest sobre los 104 partidos reales',
        'Grupos: pronóstico 1X2 (log-loss y RPS: menor es mejor). Eliminatorias: probabilidad de '
        'avanzar (incluye prórroga y penales).',
        '',
        md_table(summary),
        '',
        '## 3. Qué se aprendió del Mundial 2026 y qué se cambió en el modelo',
        'Cada cambio candidato se validó en los Mundiales 2006-2026 entrenando **solo con datos '
        'previos** a cada torneo; solo se adoptaron los que mejoran fuera de muestra.',
        '',
        '| Hallazgo en 2026 | ¿Generaliza? | Decisión |',
        '|---|---|---|',
        '| Equipos AFC y CONCACAF rindieron por debajo de su Elo y UEFA por encima '
        '(igual que en Mundiales anteriores) | Sí: mejora log-loss de grupos en 4 de 6 Mundiales | '
        '**Adoptado**: corrección Elo por confederación (prior ridge, UEFA = 0) |',
        '| En eliminatorias el favorito avanzó 84% de las veces vs ~68% predicho; '
        'en 1986-2026 fue 73% vs 65% (192 partidos) | Sí: mejora 2010, 2014 y 2026; empeora 2018 | '
        '**Adoptado**: `ko_sharpness` (multiplica la diferencia de fuerza en KO) |',
        '| Más goles (≈3 por partido) y más empates (28%) que lo esperado | No: una capa '
        '"específica de Mundial" (goles, empates, pendiente) empeora en los 8 Mundiales probados | '
        'Descartado (sería sobreajustar 72 partidos) |',
        '| Actualizar el Elo partido a partido dentro del torneo | Casi igual que congelado en 2026 | '
        'Se mantiene en la simulación (K=60) para reflejar incertidumbre |',
        '',
        '### Validación por Mundial (promedios)',
        md_table(hist.groupby('variante', sort=False)[['grupos_log_loss', 'grupos_rps', 'ko_log_loss',
                                                       'ko_brier']].mean()
                 .reindex([v for v, _ in HISTORY_VARIANTS]), '{:.4f}', index=True),
        '',
        'Detalle por Mundial: `Output/backtest_mundiales_historicos.csv`.',
        '',
        '## 4. Cómo cambió la probabilidad de campeón durante el torneo',
        'Simulando el resto del torneo con los resultados reales fijados hasta cada fase:',
        '',
        md_table(bt['evolution'].head(10), '{:.1f}', index=True),
        '',
        '## 5. Mayores sorpresas (probabilidad que el modelo pre-Mundial daba al resultado real)',
        md_table(bt['surprises'].fillna({'winner': 'empate'})),
        '',
        f'## 6. Ranking después del Mundial (modelo final, partidos hasta {post.cutoff})',
        '`strength` = Elo + corrección por confederación (lo que usa el modelo para predecir).',
        '',
        md_table(change.head(20), '{:.1f}', index=True),
        '',
        'Mayores subidas: ' + ', '.join(f'{t} ({v:+.0f})' for t, v in
                                         change['cambio'].sort_values(ascending=False).head(5).items()),
        '',
        'Mayores caídas: ' + ', '.join(f'{t} ({v:+.0f})' for t, v in
                                        change['cambio'].sort_values().head(5).items()),
        '',
        '## 7. Parámetros',
        '| Parámetro | Pre-Mundial | Final (con 2026) |',
        '|---|---|---|',
    ]
    gpre, gpost = pre.goal_model, post.goal_model
    params = [
        ('μ (log goles base)', gpre.mu, gpost.mu),
        ('β (efecto diferencia Elo)', gpre.beta, gpost.beta),
        ('γ (ataque local)', gpre.gamma, gpost.gamma),
        ('δ (defensa local)', gpre.delta, gpost.delta),
        ('ω (ajuste goles en Mundial)', gpre.omega, gpost.omega),
        ('ρ (Dixon-Coles)', gpre.rho, gpost.rho),
        ('Escala penales', pre.shootout.scale, post.shootout.scale),
        ('Nitidez eliminatorias', pre.ko_sharpness, post.ko_sharpness),
    ] + [(f'Corrección {c}', pre.conf_offsets.get(c, 0.0), post.conf_offsets.get(c, 0.0))
         for c in post.conf_offsets]
    out += [f'| {n} | {a:.3f} | {b:.3f} |' for n, a, b in params]
    out += ['', 'Configuración completa: `Output/model_config.json`.', '']
    path.write_text('\n'.join(out), encoding='utf-8')
    return path
