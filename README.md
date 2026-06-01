# simulador del mundial 2026 Can Usa Mex

Proyecto de simulación y análisis basado en ratings Elo aplicado a selecciones nacionales, orientado a modelar el Mundial 2026. Incluye simulaciones end-to-end (fase de grupos + eliminación directa), actualización dinámica de Elo por partido, y análisis Monte Carlo para estimar probabilidades de avance por fase.

### Objetivo

Mi objetivo con este proyecto es aprender sobre modelado Elo, simulaciones de partidos (Poisson), y análisis estadístico aplicado a fútbol.

### Formato del torneo

- 12 grupos de 4 equipos cada uno.
- Avanzan los dos primeros de cada grupo más los 8 mejores terceros.
- Eliminación directa desde Dieciseisavos a Final siguiendo el fixture oficial (Partidos 73-104).
- Total de partidos simulados en el torneo: 104.

### Características Principales
- **Modelo de goles**: utiliza distribuciones de Poisson para simular marcadores en 90 minutos y tiempo extra (lambda escalado).
- **Elo dinámico**: actualización de ratings partido a partido con K-factor configurable; penales cuentan como victoria para efectos de Elo.
- **Fase de grupos**: simulación completa de todos los encuentros, tablas por grupo y selección de los 8 mejores terceros.
- **Emparejamiento de terceros**: algoritmo recursivo de backtracking que asigna los mejores terceros a los emparejamientos de Dieciseisavos respetando reglas FIFA.
- **Eliminatoria**: simulación de Dieciseisavos a Final (Partidos 73-104), incluye prórroga y tanda de penales (75% prob. base por penal).
- **Monte Carlo**: ejecuciones múltiples (por defecto 1,000) para estimar probabilidades de alcanzar cada fase y probabilidad de campeón.

### Archivos Clave
- **Notebook principal**: [simulacion_eliminacion_directa.ipynb](simulacion_eliminacion_directa.ipynb)
- **Ratings de entrada**: [output/fifa_elo.csv](output/fifa_elo.csv)
- **Salidas principales**:
  - [output/tournament_probabilities.csv](output/tournament_probabilities.csv)
  - [output/group_stage_statistics.csv](output/group_stage_statistics.csv)
  - [output/elo_history.csv](output/elo_history.csv)

### Uso Rápido
- **Abrir el notebook**: `jupyter notebook simulacion_eliminacion_directa.ipynb` y ejecutar las celdas en orden.
- **Ejecutar la simulación en batch**: editar el parámetro `n_sims` en la celda de Monte Carlo y ejecutar para guardar `output/tournament_probabilities.csv`.

### Requisitos
- **Python 3.10+** (recomendado).
- Paquetes principales: `pandas`, `numpy`, `scipy`, `plotly`.

### Salida y resultados
- El flujo de trabajo genera:
  - Probabilidades por equipo para cada fase del torneo (CSV).
  - Estadísticas agregadas de fase de grupos (CSV).
  - Visualizaciones interactivas (Plotly) desde el notebook.

### Notas importantes
- Los resultados son estocásticos; para reproducibilidad ajustar la semilla con `np.random.seed()` en el notebook.
- El modelo contiene supuestos heurísticos (p. ej. fórmulas lambda, K-factors, p_success en penales) que pueden ajustarse según preferencia.

### Siguientes pasos sugeridos
- Crear un script CLI para ejecutar simulaciones sin abrir Jupyter.
- Validar y extender el modelo de lambda (ajuste por confederación o forma reciente).

### Elementos a mejorar el modelo

- Calibrar la función de lambda (esperanza de goles) con datos históricos y/o regresión, en lugar de fórmulas heurísticas.
- Ajustar el K-factor por tipo de partido y/o importancia (amistoso, clasificatorio, eliminatoria).
- Incluir ventaja de local y/o factor de confederación en la predicción de goles.
- Incorporar forma reciente (últimos N partidos) y lesiones/sanciones en el peso del rating.
- Refinar el modelo de penales (probabilidad por lanzador/portero) o usar modelos bayesianos para mayor realismo.
- Añadir validación cruzada y métricas de calibración (Brier score, log-loss) para comparar variantes del modelo.
- Paralelizar las simulaciones Monte Carlo y permitir ajuste de `n_sims` para balance precisión/tiempo.
- Añadir tests unitarios y un conjunto de datos de validación para reproducibilidad.
