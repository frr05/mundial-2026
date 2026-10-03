# Simulador del Mundial 2026 (Canadá · USA · México)

Modelo predictivo del Mundial 2026: ratings Elo + modelo de goles Dixon-Coles + simulación
Monte Carlo del formato real de 48 equipos. El modelo está calibrado con datos históricos y
re-entrenado con los **104 resultados reales del Mundial 2026** (campeón: España, 1-0 a Argentina).

📄 **Reporte completo:** [`Output/REPORTE_MUNDIAL_2026.md`](Output/REPORTE_MUNDIAL_2026.md)

### Objetivo

Aprender sobre modelado Elo, simulación de partidos (Poisson) y análisis estadístico aplicado a fútbol.

## Uso rápido

```bash
pip install -r requirements.txt
python simulator.py todo                 # todo: probabilidades, backtest, ratings y reporte (~40 s)
python simulator.py pre --sims 20000     # probabilidades pre-Mundial
python simulator.py live --hasta R16     # simula el resto fijando los resultados reales hasta octavos
python simulator.py backtest             # compara modelos contra los partidos reales
python simulator.py ratings              # modelo final (incluye el Mundial 2026) y ranking
python -m pytest                         # tests
python journalist.py                     # reporte "periodista" + gráficos
```

Notebook principal: [`modelo_mundial_2026.ipynb`](modelo_mundial_2026.ipynb).

## Cómo funciona el modelo (`modelo/`)

| Módulo | Qué hace |
|---|---|
| `elo.py` | Elo estilo eloratings.net sobre toda la historia (1872 → hoy). K según torneo (Mundial 60, continentales 50, eliminatorias/Nations League 40, otros 30, amistosos 20), multiplicador por diferencia de gol y ventaja de local (100) solo si la cancha no es neutral. |
| `goals.py` | Goles Dixon-Coles: `log λ = μ ± β·ΔElo/400 + local`, con corrección ρ para marcadores bajos. Parámetros por máxima verosimilitud (≈16.000 partidos desde 2010). Modelo de penales ajustado con las tandas históricas (casi una moneda al aire). |
| `model.py` | Junta todo y agrega dos correcciones aprendidas de los Mundiales: **por confederación** (AFC ≈ −88 y CONCACAF ≈ −62 puntos Elo respecto de UEFA) y **nitidez en eliminatorias** (el favorito avanza más de lo que predice el modelo de 90'). `build_model(cutoff)` entrena solo con datos previos a `cutoff`. |
| `tournament.py` | Monte Carlo del formato real: desempates FIFA (enfrentamiento directo primero), 8 mejores terceros con reglas de cruce FIFA, Partidos 73-104, prórroga, penales, ventaja de local solo para anfitriones en su país y Elo dinámico dentro de cada simulación. Puede fijar resultados ya jugados. |
| `evaluate.py` | Backtest contra el Mundial 2026 y validación en los Mundiales 2006-2026 sin mirar el futuro. |
| `report.py` | Genera el reporte en Markdown. |

## Errores corregidos del modelo anterior

- **Probabilidades > 100%** en `tournament_probabilities.csv` (p. ej. Francia 134% en dieciseisavos): la cuenta del Monte Carlo usaba los ganadores de la fase siguiente, contando dos veces a los eliminados. Corregido en el notebook y reemplazado por un conteo "alcanzó al menos la fase X".
- **Slots de terceros que se pisaban** en la asignación de Dieciseisavos (detectado al reproducir el bracket real).
- **Rutas `output/`** que no existen en Linux/macOS (la carpeta es `Output/`); `journalist.py` buscaba archivos que nadie generaba.
- **`simulator.py`** tenía solo 31 equipos, un fixture distinto al oficial y otra fórmula de goles que los notebooks.
- **Fórmula de goles heurística** (`λ = 1.35 + diff/200`) sobre el ranking FIFA: predecía 15.6% de empates (real: 28%). Ahora es un modelo calibrado.
- **Elo sin contexto**: amistosos con K≈0, sin ventaja de local ni diferencia de gol, desempates de grupo sin enfrentamiento directo.

## Resultados del backtest (Mundial 2026, 104 partidos)

| Modelo | Grupos log-loss ↓ | Grupos RPS ↓ | KO log-loss ↓ | KO acierto |
|---|---|---|---|---|
| Original (ranking FIFA + λ heurístico) | 1.012 | 0.176 | 0.418 | 75% |
| **Nuevo (pre-Mundial)** | **0.887** | **0.158** | **0.401** | 78% |

Antes del torneo el modelo nuevo tenía a **España como favorita (~28%)**, seguida de Argentina (~23%):
los dos finalistas reales.

## Qué se aprendió del Mundial 2026

Cada cambio se validó en los Mundiales 2006-2026 entrenando solo con datos anteriores a cada uno:

- ✅ **Corrección por confederación**: AFC y CONCACAF vuelven a rendir bajo su Elo (y UEFA sobre él). Mejora 4 de 6 Mundiales.
- ✅ **Nitidez en eliminatorias**: en 2026 el favorito avanzó 84% de las veces vs ~68% predicho (1986-2026: 73% vs 65%). Ajustado con 1986-2022, mejora 2026 fuera de muestra (KO log-loss 0.452 → 0.401).
- ❌ **Capa "más goles / más empates en Mundial"**: aunque 2026 tuvo ≈3 goles y 28% de empates por partido, ajustarla empeora en todos los Mundiales probados → sobreajuste, descartada.

## Archivos de salida (`Output/`)

| Archivo | Contenido |
|---|---|
| `REPORTE_MUNDIAL_2026.md` | Reporte completo |
| `tournament_probabilities.csv` | Probabilidades pre-Mundial por fase y posición en el grupo |
| `backtest_mundial2026_*.csv` | Probabilidades por partido, resumen y mayores sorpresas |
| `backtest_mundiales_historicos.csv` | Validación por Mundial y variante del modelo |
| `evolucion_probabilidad_campeon.csv` | Probabilidad de campeón tras cada fase |
| `ratings_actuales.csv`, `ratings_pre_mundial.csv`, `cambio_ratings_mundial2026.csv` | Ratings |
| `model_config.json` | Parámetros del modelo final |
| `elo_history.csv` | Historial Elo desde 2018 (date, team, rating) |

## Datos

`Data/results.csv`, `shootouts.csv` y `goalscorers.csv` provienen del dataset
[martj42/international_results](https://github.com/martj42/international_results), actualizado
hasta agosto de 2026 (incluye todos los partidos del Mundial).

## Limitaciones

- La tabla oficial de cruces de terceros (Anexo C, 495 combinaciones) no está completa en el
  código: se usa la combinación real de 2026 y, para las demás, una asignación válida según las
  reglas FIFA por backtracking (puede diferir de la oficial en algunos cruces).
- El modelo no conoce lesiones, convocatorias ni el estado de forma más allá de lo que refleja el Elo.
- Los resultados son estocásticos; usar `--seed` para reproducibilidad.
