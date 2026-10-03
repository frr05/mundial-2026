# Mundial 2026 — Reporte del modelo predictivo
_Generado el 2026-10-03 con `python simulator.py todo`._

## 1. Resultado real vs. lo que predijo el modelo
- Campeón real: **Spain** (final: Spain 1-0 Argentina).
- Antes del torneo el modelo le daba a Spain **28.2%** de ser campeón (favorito #1) y 44.0% de llegar a la final.
- Fase de grupos real: 2.99 goles/partido y 27.8% de empates.

### Probabilidades pre-Mundial (top 12; modelo con todas las mejoras, entrenado solo con datos hasta el 10-jun-2026)
| team | strength | Octavos (R16) % | Cuartos (QF) % | Semifinal (SF) % | Final (F) % | Campeón % |
|---|---|---|---|---|---|---|
| Spain | 2219.1 | 82.6 | 70.2 | 60.2 | 44.0 | 28.2 |
| Argentina | 2196.1 | 75.3 | 67.4 | 52.0 | 37.3 | 22.6 |
| France | 2125.2 | 82.1 | 56.5 | 40.4 | 20.4 | 11.0 |
| England | 2090.9 | 73.7 | 48.4 | 29.2 | 13.8 | 6.6 |
| Brazil | 2075.1 | 69.7 | 48.4 | 27.7 | 12.6 | 5.9 |
| Colombia | 2070.2 | 69.7 | 41.4 | 19.6 | 10.5 | 4.6 |
| Portugal | 2046.5 | 65.0 | 35.4 | 16.4 | 8.7 | 3.8 |
| Ecuador | 2034.1 | 69.0 | 32.2 | 18.9 | 7.4 | 3.1 |
| Germany | 2004.9 | 61.9 | 26.2 | 13.7 | 5.0 | 1.9 |
| Netherlands | 2011.0 | 49.4 | 34.9 | 14.7 | 5.4 | 1.9 |
| Morocco | 1991.1 | 52.1 | 30.7 | 14.0 | 4.8 | 1.6 |
| Norway | 1971.9 | 50.1 | 23.9 | 11.1 | 3.9 | 1.4 |

## 2. Backtest sobre los 104 partidos reales
Grupos: pronóstico 1X2 (log-loss y RPS: menor es mejor). Eliminatorias: probabilidad de avanzar (incluye prórroga y penales).

| modelo | grupos_log_loss | grupos_brier | grupos_rps | grupos_acierto | grupos_empates_predichos | ko_log_loss | ko_brier | ko_acierto |
|---|---|---|---|---|---|---|---|---|
| Original (ranking FIFA + λ heurístico) | 1.012 | 0.570 | 0.176 | 0.611 | 0.156 | 0.418 | 0.143 | 0.750 |
| Nuevo, congelado pre-Mundial | 0.887 | 0.525 | 0.158 | 0.625 | 0.223 | 0.401 | 0.129 | 0.781 |
| Nuevo, Elo actualizado partido a partido | 0.888 | 0.526 | 0.160 | 0.597 | 0.223 | 0.406 | 0.130 | 0.844 |

## 3. Qué se aprendió del Mundial 2026 y qué se cambió en el modelo
Cada cambio candidato se validó en los Mundiales 2006-2026 entrenando **solo con datos previos** a cada torneo; solo se adoptaron los que mejoran fuera de muestra.

| Hallazgo en 2026 | ¿Generaliza? | Decisión |
|---|---|---|
| Equipos AFC y CONCACAF rindieron por debajo de su Elo y UEFA por encima (igual que en Mundiales anteriores) | Sí: mejora log-loss de grupos en 4 de 6 Mundiales | **Adoptado**: corrección Elo por confederación (prior ridge, UEFA = 0) |
| En eliminatorias el favorito avanzó 84% de las veces vs ~68% predicho; en 1986-2026 fue 73% vs 65% (192 partidos) | Sí: mejora 2010, 2014 y 2026; empeora 2018 | **Adoptado**: `ko_sharpness` (multiplica la diferencia de fuerza en KO) |
| Más goles (≈3 por partido) y más empates (28%) que lo esperado | No: una capa "específica de Mundial" (goles, empates, pendiente) empeora en los 8 Mundiales probados | Descartado (sería sobreajustar 72 partidos) |
| Actualizar el Elo partido a partido dentro del torneo | Casi igual que congelado en 2026 | Se mantiene en la simulación (K=60) para reflejar incertidumbre |

### Validación por Mundial (promedios)
| variante | grupos_log_loss | grupos_rps | ko_log_loss | ko_brier |
|---|---|---|---|---|
| base (Elo + Dixon-Coles) | 0.9621 | 0.1945 | 0.5459 | 0.1832 |
| + corrección confederación | 0.9468 | 0.1888 | 0.5357 | 0.1796 |
| + nitidez eliminatorias (final) | 0.9468 | 0.1888 | 0.5176 | 0.1730 |

Detalle por Mundial: `Output/backtest_mundiales_historicos.csv`.

## 4. Cómo cambió la probabilidad de campeón durante el torneo
Simulando el resto del torneo con los resultados reales fijados hasta cada fase:

| team | Pre-Mundial | Tras fase de grupos | Tras dieciseisavos | Tras octavos | Tras cuartos | Tras semifinales |
|---|---|---|---|---|---|---|
| Spain | 27.8 | 24.2 | 25.8 | 31.0 | 30.5 | 52.7 |
| Argentina | 23.0 | 31.9 | 30.4 | 33.0 | 35.8 | 47.3 |
| France | 10.7 | 17.6 | 20.1 | 18.7 | 22.3 | 0.0 |
| England | 6.6 | 5.6 | 6.1 | 10.9 | 11.4 | 0.0 |
| Brazil | 5.9 | 5.6 | 6.3 | 0.0 | 0.0 | 0.0 |
| Colombia | 4.7 | 4.7 | 4.1 | 0.0 | 0.0 | 0.0 |
| Portugal | 3.8 | 2.2 | 2.4 | 0.0 | 0.0 | 0.0 |
| Ecuador | 3.3 | 0.6 | 0.0 | 0.0 | 0.0 | 0.0 |
| Germany | 1.9 | 0.8 | 0.0 | 0.0 | 0.0 | 0.0 |
| Netherlands | 1.9 | 2.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## 5. Mayores sorpresas (probabilidad que el modelo pre-Mundial daba al resultado real)
| date | stage | home_team | home_score | away_score | away_team | winner | p_resultado_real |
|---|---|---|---|---|---|---|---|
| 2026-06-15 | GROUP | Spain | 0 | 0 | Cape Verde | empate | 0.067 |
| 2026-06-20 | GROUP | Ecuador | 0 | 0 | Curaçao | empate | 0.095 |
| 2026-06-13 | GROUP | Qatar | 1 | 1 | Switzerland | empate | 0.104 |
| 2026-06-23 | GROUP | England | 0 | 0 | Ghana | empate | 0.111 |
| 2026-06-15 | GROUP | Saudi Arabia | 1 | 1 | Uruguay | empate | 0.156 |
| 2026-06-14 | GROUP | Ivory Coast | 1 | 0 | Ecuador | Ivory Coast | 0.160 |
| 2026-06-21 | GROUP | Uruguay | 2 | 2 | Cape Verde | empate | 0.182 |
| 2026-06-17 | GROUP | Ghana | 1 | 0 | Panama | Ghana | 0.204 |
| 2026-06-17 | GROUP | Portugal | 1 | 1 | DR Congo | empate | 0.205 |
| 2026-06-13 | GROUP | Australia | 2 | 0 | Turkey | Australia | 0.209 |

## 6. Ranking después del Mundial (modelo final, partidos hasta 2026-08-26)
`strength` = Elo + corrección por confederación (lo que usa el modelo para predecir).

| team | elo_pre | elo_post | strength_post | cambio |
|---|---|---|---|---|
| Spain | 2219.1 | 2327.1 | 2327.1 | 108.0 |
| Argentina | 2190.1 | 2249.8 | 2244.0 | 59.8 |
| England | 2090.9 | 2198.9 | 2198.9 | 108.0 |
| France | 2125.2 | 2141.0 | 2141.0 | 15.8 |
| Colombia | 2064.2 | 2086.8 | 2081.0 | 22.6 |
| Brazil | 2069.0 | 2077.1 | 2071.3 | 8.1 |
| Portugal | 2046.5 | 2060.3 | 2060.3 | 13.8 |
| Morocco | 1985.1 | 2029.5 | 2048.8 | 44.4 |
| Netherlands | 2011.0 | 2045.6 | 2045.6 | 34.6 |
| Belgium | 1957.0 | 2027.4 | 2027.4 | 70.4 |
| Norway | 1971.9 | 2021.2 | 2021.2 | 49.3 |
| Switzerland | 1956.3 | 2003.5 | 2003.5 | 47.2 |
| Germany | 2004.9 | 1989.5 | 1989.5 | -15.4 |
| Ecuador | 2028.1 | 1964.3 | 1958.5 | -63.8 |
| Mexico | 1980.4 | 2014.0 | 1952.2 | 33.6 |
| Croatia | 1969.6 | 1947.0 | 1947.0 | -22.6 |
| Turkey | 1969.7 | 1923.2 | 1923.2 | -46.5 |
| Uruguay | 1975.5 | 1924.8 | 1919.0 | -50.7 |
| Senegal | 1913.8 | 1880.3 | 1899.6 | -33.5 |
| Paraguay | 1918.7 | 1901.0 | 1895.2 | -17.7 |

Mayores subidas: Spain (+108), England (+108), Belgium (+70), Argentina (+60), Ghana (+57)

Mayores caídas: Uzbekistan (-89), Panama (-80), Tunisia (-77), Ecuador (-64), Iraq (-64)

## 7. Parámetros
| Parámetro | Pre-Mundial | Final (con 2026) |
|---|---|---|
| μ (log goles base) | 0.245 | 0.244 |
| β (efecto diferencia Elo) | 0.745 | 0.746 |
| γ (ataque local) | 0.095 | 0.096 |
| δ (defensa local) | 0.296 | 0.294 |
| ω (ajuste goles en Mundial) | -0.047 | -0.012 |
| ρ (Dixon-Coles) | -0.047 | -0.048 |
| Escala penales | 0.248 | 0.242 |
| Nitidez eliminatorias | 1.519 | 1.664 |
| Corrección UEFA | 0.000 | 0.000 |
| Corrección CONMEBOL | 6.017 | -5.808 |
| Corrección CONCACAF | -65.508 | -61.790 |
| Corrección CAF | 6.019 | 19.265 |
| Corrección AFC | -83.281 | -88.236 |
| Corrección OFC | 7.269 | -9.838 |

Configuración completa: `Output/model_config.json`.
