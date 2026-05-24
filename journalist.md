# Skill: Periodista - Análisis de Simulaciones del Mundial

## Descripción
Este agente analiza las simulaciones del mundial y genera insights, gráficos y reportes.

## Uso
Invocar con: `/journalist`

## Comandos disponibles

### /journalist report
Genera un reporte completo con insights sobre los equipos.

### /journalist evolution [equipos]
Grafica la evolución de los equipos especificados.
Ejemplo: `/journalist evolution Argentina,Brazil,France`

### /journalist tops
Muestra los equipos en ascenso y descenso.

### /journalist compare [equipo1] [equipo2]
Compara dos equipos.

## Datos que analiza
- `output/elo_history.csv` - Evolución histórica de ratings
- `output/simulation_championship_probs.csv` - Probabilidades de campeón
- `output/simulation_group_results.csv` - Resultados de grupos

## Funciones

```python
# En JournalistAgent:
- load_data()           → Carga datos de CSV
- generate_top_insights() → Insights principales
- plot_team_evolution(teams) → Gráfico comparativo
- plot_champion_probs()  → Gráfico probabilidades
- analyze_team_evolution(team) → Análisis individual
- generate_report()     → Reporte completo
```

## Ejemplo de output

```
==================================================
REPORTE DEL PERIODISTA - MUNDIAL 2026
==================================================
Fecha: 2026-05-23

TOP 10 EQUIPOS POR RATING ACTUAL
==================================================
Argentina: 2050.3
Brazil: 2034.2
France: 2018.5
Germany: 1985.1
Spain: 1972.8
...

EQUIPOS EN ASCENSO
==================================================
USA: +45.2
Japan: +38.1
Morocco: +32.5
...

EQUIPOS EN DESCENSO
==================================================
Belgium: -28.3
Portugal: -22.1
...
```

## Archivos que genera
- `output/journalist_report.txt` - Reporte de texto
- `output/journalist_evolution.png` - Gráfico de evolución
- `output/journalist_champions.png` - Gráfico de probabilidades