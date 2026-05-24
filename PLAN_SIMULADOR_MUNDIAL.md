# Plan: Simulador de Mundial Python (Código Sencillo)

## Contexto
Simulador del Mundial EE.UU. 2026 que predice el ganador usando:
- Elo ratings dinámicos desde partidos históricos (desde 2022)
- Modelo Poisson para goles esperados
- Simulación Monte Carlo del torneo completo
- Actualización manual de partidos (CSV)
- Dashboard visual
- Salida en CSV para agente reportero

**Principio rector**: Código sencillo y legible. Sin abstracciones. Funciones claras.

## Datos de entrada
- `Data/ultimas_clasificatorias.csv`: Partidos desde 2022 hasta marzo 2026
- `Data/shootouts.csv`: Historial de penales (filtrar desde junio 2022)

## Configuración del modelo (exportable)
El modelo guarda su configuración en `output/model_config.json`:
```json
{
  "initial_rating": 1500,
  "k_factor": 32,
  "home_advantage": 100,
  "poisson_intercept": 1.2,
  "poisson_coef": 0.004,
  "shootout_k_factor": 20,
  "data_source": "ultimas_clasificatorias.csv",
  "date_from": "2022-06-01",
  "tournament": "World Cup 2026"
}
```
Esto permite:
- Reutilizar el modelo para otros torneos
- Exportar a otras computadoras
- Versionar experimentos

## CSVs de salida (para reportero)
- `output/elo_history.csv` - Evolución de ratings Elo por fecha
- `output/group_results.csv` - Posiciones de todos los equipos en cada grupo
- `output/knockout_results.csv` - Resultados de eliminación directa
- `output/champion_probs.csv` - Probabilidades de campeón
- `output/manual_matches.csv` - Partidos agregados manualmente

## Archivos a crear

### 1. simulator.py

```python
import pandas as pd
import numpy as np
import json
import os
from collections import defaultdict
from datetime import datetime

# Crear carpeta output si no existe
os.makedirs('output', exist_ok=True)

# === ELO ===
def calculate_elo(matches_df, initial_rating=1500, k=32, home_adv=100):
    """Calcula ratings Elo iterando partidos cronológicamente."""
    ratings = defaultdict(lambda: initial_rating)
    history = []  # Guardar evolución

    for _, row in matches_df.iterrows():
        home = row['home_team']
        away = row['away_team']
        home_goals = row['home_score']
        away_goals = row['away_score']

        r_home = ratings[home]
        r_away = ratings[away]

        e_home = 1 / (1 + 10 ** ((r_away - r_home - home_adv) / 400))
        e_away = 1 / (1 + 10 ** ((r_home + home_adv - r_away) / 400))

        if home_goals > away_goals:
            s_home, s_away = 1, 0
        elif home_goals < away_goals:
            s_home, s_away = 0, 1
        else:
            s_home, s_away = 0.5, 0.5

        ratings[home] += k * (s_home - e_home)
        ratings[away] += k * (s_away - e_away)

        # Guardar estado actual
        date = row.get('date', datetime.now())
        history.append({'date': date, 'team': home, 'rating': ratings[home]})
        history.append({'date': date, 'team': away, 'rating': ratings[away]})

    return dict(ratings), pd.DataFrame(history)


def calculate_shootout_elo(shootouts_df, initial_rating=1500, k=20, date_from='2022-06-01'):
    """Calcula ratings Elo específicos para penales desde fecha específica."""
    # Filtrar por fecha
    shootouts_df = shootouts_df.copy()
    shootouts_df['date'] = pd.to_datetime(shootouts_df['date'])
    shootouts_df = shootouts_df[shootouts_df['date'] >= pd.to_datetime(date_from)]

    ratings = defaultdict(lambda: initial_rating)

    for _, row in shootouts_df.iterrows():
        winner = row['winner']
        loser = row['loser']

        r_winner = ratings[winner]
        r_loser = ratings[loser]

        e_winner = 1 / (1 + 10 ** ((r_loser - r_winner) / 400))

        ratings[winner] += k * (1 - e_winner)
        ratings[loser] += k * (0 - (1 - e_winner))

    return dict(ratings)


# === POISSON ===
def estimate_lambda(rating_home, rating_away, home_adv=100, intercept=1.2, coef=0.004):
    """Calcula λ (goles esperados)."""
    diff = rating_home + home_adv - rating_away
    return np.exp(intercept + coef * diff)


def estimate_lambda_neutral(rating_a, rating_b, intercept=1.2, coef=0.004):
    """Calcula λ sin ventaja de local (campo neutral)."""
    diff = rating_a - rating_b
    return np.exp(intercept + coef * diff)


def simulate_match(team_home, team_away, ratings):
    """Simula un partido con distribución Poisson."""
    lambda_home = estimate_lambda(ratings[team_home], ratings[team_away])
    lambda_away = estimate_lambda(ratings[team_away], ratings[team_home])

    goals_home = np.random.poisson(lambda_home)
    goals_away = np.random.poisson(lambda_away)

    return goals_home, goals_away


def simulate_match_neutral(team_a, team_b, ratings, seed=None):
    """Simula partido en campo neutral (sin home advantage)."""
    if seed is not None:
        np.random.seed(seed)

    lambda_a = estimate_lambda_neutral(ratings[team_a], ratings[team_b])
    lambda_b = estimate_lambda_neutral(ratings[team_b], ratings[team_a])

    # 90 minutos
    goals_a = np.random.poisson(lambda_a)
    goals_b = np.random.poisson(lambda_b)

    # Si empatan, tiempo extra (15 min cada parte)
    if goals_a == goals_b:
        extra_a = np.random.poisson(lambda_a * 0.166)  # 10 min = 1/6 de 90
        extra_b = np.random.poisson(lambda_b * 0.166)
        goals_a += extra_a
        goals_b += extra_b

    return goals_a, goals_b


def simulate_penalties(team_home, team_away, shootout_ratings, seed=None):
    """Simula penales usando Elo de shootouts."""
    if seed is not None:
        np.random.seed(seed)

    # Bonus basado en rating de shootout
    r_home = shootout_ratings.get(team_home, 1500)
    r_away = shootout_ratings.get(team_away, 1500)

    elo_bonus = (r_home - r_away) / 4000  # Normalizado
    p_home = 0.75 + elo_bonus
    p_away = 0.75 - elo_bonus

    # Clamp probabilities
    p_home = max(0.5, min(0.95, p_home))
    p_away = 1 - p_home

    # Simular ronda de penales (5 tiros cada uno)
    home_score = sum(np.random.random() < p_home for _ in range(5))
    away_score = sum(np.random.random() < p_away for _ in range(5))

    # Si empatan, sudden death
    while home_score == away_score:
        if np.random.random() < p_home:
            home_score += 1
        else:
            away_score += 1

    return home_score, away_score


# === TORNEO ===
def simulate_group(teams, ratings, group_name):
    """Simula un grupo. Retorna DataFrame con posiciones."""
    table = defaultdict(lambda: {'pts': 0, 'gf': 0, 'ga': 0, 'gd': 0, 'w': 0, 'd': 0, 'l': 0})

    for i, home in enumerate(teams):
        for j, away in enumerate(teams):
            if i >= j:
                continue

            gh, ga = simulate_match(home, away, ratings)
            table[home]['gf'] += gh
            table[home]['ga'] += ga
            table[away]['gf'] += ga
            table[away]['ga'] += gh

            if gh > ga:
                table[home]['pts'] += 3
                table[home]['w'] += 1
                table[away]['l'] += 1
            elif gh < ga:
                table[away]['pts'] += 3
                table[away]['w'] += 1
                table[home]['l'] += 1
            else:
                table[home]['pts'] += 1
                table[away]['pts'] += 1
                table[home]['d'] += 1
                table[away]['d'] += 1

    # Agregar diferencia de goles
    for t in table:
        table[t]['gd'] = table[t]['gf'] - table[t]['ga']

    # Crear DataFrame
    results = []
    for team in teams:
        results.append({
            'group': group_name,
            'team': team,
            'pts': table[team]['pts'],
            'w': table[team]['w'],
            'd': table[team]['d'],
            'l': table[team]['l'],
            'gf': table[team]['gf'],
            'ga': table[team]['ga'],
            'gd': table[team]['gd']
        })

    df = pd.DataFrame(results)
    df = df.sort_values(['pts', 'gd', 'gf'], ascending=[False, False, False])
    df['position'] = range(1, 5)
    return df


def get_best_third(group_dfs):
    """Obtiene los 8 mejores terceros para repechaje."""
    thirds = []
    for df in group_dfs:
        third = df[df['position'] == 3].iloc[0]
        thirds.append({'team': third['team'], 'pts': third['pts'], 'gd': third['gd'], 'gf': third['gf'], 'group': third['group']})

    thirds = sorted(thirds, key=lambda x: (x['pts'], x['gd'], x['gf']), reverse=True)
    return [t['team'] for t in thirds[:8]]  # 8 mejores terceros


# Fixture de dieciseisavos (Round of 32) según fixture_mundial.md
ROUND_OF_32_FIXTURE = [
    ('1A', '3rd_best'),   # M1
    ('2B', '2F'),        # M2
    ('1C', '3rd_best'),  # M3
    ('1D', '2E'),        # M4
    ('1B', '3rd_best'),  # M5
    ('2A', '2C'),        # M6
    ('1E', '3rd_best'),  # M7
    ('1F', '2D'),        # M8
    ('1G', '3rd_best'),  # M9
    ('2H', '2L'),        # M10
    ('1I', '3rd_best'),  # M11
    ('1J', '2K'),        # M12
    ('1H', '3rd_best'),  # M13
    ('2G', '2I'),        # M14
    ('1K', '3rd_best'),  # M15
    ('1L', '2J'),        # M16
]


def simulate_knockout(teams, ratings, shootout_ratings):
    """Simula eliminación directa con penales."""
    n = len(teams)
    if n == 1:
        return teams[0], []

    results = []
    next_round = []

    for i in range(0, n, 2):
        home = teams[i]
        away = teams[i + 1]

        gh, ga = simulate_match(home, away, ratings)

        if gh > ga:
            winner = home
        elif gh < ga:
            winner = away
        else:
            # Penales
            ph, pa = simulate_penalties(home, away, shootout_ratings)
            winner = home if ph > pa else away

        next_round.append(winner)
        results.append({'home': home, 'away': away, 'home_goals': gh, 'away_goals': ga, 'winner': winner})

    next_winners, next_results = simulate_knockout(next_round, ratings, shootout_ratings)
    return next_winners, results + next_results


def simulate_full_tournament(teams, ratings, shootout_ratings, seed=None):
    """Simula el torneo completo (12 grupos, 8 mejores terceros, dieciseisavos)."""
    # 12 grupos de 4 equipos
    groups = [teams[i:i+4] for i in range(0, 48, 4)]
    group_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']

    all_group_results = []
    group_winners = {}  # { '1A': team, '2A': team, ... }

    for group, name in zip(groups, group_names):
        df = simulate_group(group, ratings, name)
        all_group_results.append(df)
        # Guardar clasificados
        for pos in [1, 2]:
            team = df[df['position'] == pos]['team'].iloc[0]
            group_winners[f'{pos}{name}'] = team

    # Mejores terceros (8 equipos)
    best_thirds = get_best_third(all_group_results)

    # Dieciseisavos: 32 equipos
    # Emparejamientos según fixture
    r32_teams = []
    for match in ROUND_OF_32_FIXTURE:
        # match[0] = '1A', '2B', etc.
        # match[1] = '3rd_best', '2F', etc.
        if match[1] == '3rd_best':
            # Tomar el siguiente mejor tercero
            r32_teams.append(best_thirds.pop(0))
        else:
            group_letter = match[1][1]  # 'F' from '2F'
            pos = match[1][0]  # '2' from '2F'
            r32_teams.append(group_winners[f'{pos}{group_letter}'])

    # Añadir los primeros de grupo que faltan (1A, 1C, etc.)
    for name in group_names:
        r32_teams.append(group_winners[f'1{name}'])

    # Shuffle para randomizar emparejamientos de terceros
    np.random.seed(seed)
    np.random.shuffle(r32_teams)

    # Dieciseisavos → Octavos → Cuartos → Semis → Final
    winner, knockout_results = simulate_knockout_neutral(r32_teams, ratings, shootout_ratings, seed)

    return winner, all_group_results, knockout_results, best_thirds


def simulate_knockout_neutral(teams, ratings, shootout_ratings, seed=None):
    """Simula eliminación directa en campo neutral con tiempo extra y penales."""
    n = len(teams)
    if n == 1:
        return teams[0], []

    results = []
    next_round = []

    for i in range(0, n, 2):
        home = teams[i]
        away = teams[i + 1]

        # Campo neutral + tiempo extra
        gh, ga = simulate_match_neutral(home, away, ratings, seed=seed)

        if gh > ga:
            winner = home
        elif gh < ga:
            winner = away
        else:
            # Penales
            ph, pa = simulate_penalties(home, away, shootout_ratings, seed=seed)
            winner = home if ph > pa else away

        next_round.append(winner)
        results.append({'home': home, 'away': away, 'home_goals': gh, 'away_goals': ga, 'winner': winner})

    next_winners, next_results = simulate_knockout_neutral(next_round, ratings, shootout_ratings, seed)
    return next_winners, results + next_results


def run_tournament(teams, ratings, shootout_ratings, n_sims=10000, seed=42):
    """Monte Carlo con salida en DataFrames."""
    np.random.seed(seed)

    all_winners = []
    all_group_results = []
    all_knockout_results = []

    for _ in range(n_sims):
        winner, group_dfs, knockout, thirds = simulate_full_tournament(teams, ratings, shootout_ratings)
        all_winners.append(winner)

        for df in group_dfs:
            df['simulation'] = len(all_winners)
            all_group_results.append(df)

        for k in knockout:
            k['simulation'] = len(all_winners)
            all_knockout_results.append(k)

    # Agregar terceros
    third_df = pd.DataFrame([{'team': t, 'simulation': i} for i, t in enumerate(thirds) for t in thirds])

    return {
        'champion_probs': pd.Series(all_winners).value_counts(normalize=True).sort_values(ascending=False),
        'group_results': pd.concat(all_group_results, ignore_index=True),
        'knockout_results': pd.DataFrame(all_knockout_results),
        'best_thirds': third_df
    }


# === CONFIGURACIÓN ===
DEFAULT_CONFIG = {
    'initial_rating': 1500,
    'k_factor': 32,
    'home_advantage': 100,
    'poisson_intercept': 1.2,
    'poisson_coef': 0.004,
    'shootout_k_factor': 20,
    'date_from': '2022-06-01',
    'tournament': 'World Cup 2026'
}


def save_config(config=None, filename='output/model_config.json'):
    """Guarda configuración del modelo."""
    if config is None:
        config = DEFAULT_CONFIG
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)


def load_config(filename='output/model_config.json'):
    """Carga configuración del modelo."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return DEFAULT_CONFIG


def export_model(ratings, shootout_ratings, config, folder='output/export'):
    """Exporta todo el modelo a una carpeta."""
    os.makedirs(folder, exist_ok=True)

    # Guardar ratings
    pd.DataFrame(list(ratings.items()), columns=['team', 'rating']).to_csv(f'{folder}/ratings.csv', index=False)
    pd.DataFrame(list(shootout_ratings.items()), columns=['team', 'rating']).to_csv(f'{folder}/shootout_ratings.csv', index=False)

    # Guardar configuración
    save_config(config, f'{folder}/config.json')

    return folder


def import_model(folder='output/export'):
    """Importa el modelo desde una carpeta."""
    ratings = pd.read_csv(f'{folder}/ratings.csv').set_index('team')['rating'].to_dict()
    shootout_ratings = pd.read_csv(f'{folder}/shootout_ratings.csv').set_index('team')['rating'].to_dict()
    config = load_config(f'{folder}/config.json')

    return ratings, shootout_ratings, config


# === PERSISTENCIA ===
def save_to_csv(results, prefix='simulation'):
    """Guarda todos los resultados en CSV."""
    results['champion_probs'].to_csv(f'output/{prefix}_champion_probs.csv', header=['probability'])

    results['group_results'].to_csv(f'output/{prefix}_group_results.csv', index=False)

    if not results['knockout_results'].empty:
        results['knockout_results'].to_csv(f'output/{prefix}_knockout_results.csv', index=False)

    results['best_thirds'].to_csv(f'output/{prefix}_best_thirds.csv', index=False)


def save_elo_history(elo_history_df):
    """Guarda historial de Elo."""
    elo_history_df.to_csv('output/elo_history.csv', index=False)


def load_manual_matches():
    """Carga partidos manuales."""
    try:
        return pd.read_csv('output/manual_matches.csv')
    except FileNotFoundError:
        return pd.DataFrame(columns=['date', 'home_team', 'away_team', 'home_score', 'away_score', 'tournament'])


def save_manual_matches(matches_df):
    """Guarda partidos manuales."""
    matches_df.to_csv('output/manual_matches.csv', index=False)


def add_manual_match(date, home, away, home_goals, away_goals):
    """Agrega un partido manual."""
    matches = load_manual_matches()
    new_match = pd.DataFrame([{
        'date': date,
        'home_team': home,
        'away_team': away,
        'home_score': home_goals,
        'away_score': away_goals,
        'tournament': 'World Cup 2026'
    }])
    matches = pd.concat([matches, new_match], ignore_index=True)
    save_manual_matches(matches)
    return matches


# === MAIN ===
if __name__ == '__main__':
    # Cargar datos
    matches = pd.read_csv('Data/ultimas_clasificatorias.csv')

    # Agregar partidos manuales
    manual = load_manual_matches()
    if not manual.empty:
        matches = pd.concat([matches, manual], ignore_index=True)

    # Calcular Elo
    ratings, elo_history = calculate_elo(matches)

    # Calcular Elo de shootouts
    shootouts = pd.read_csv('Data/shootouts.csv')
    shootout_ratings = calculate_shootout_elo(shootouts)

    # Guardar historial de Elo
    save_elo_history(elo_history)

    # Equipos clasificados (48 equipos = 12 grupos x 4)
    # Por ahora usamos los 48 clasificados已知
    qualified_teams = [
        # Grupos ya definidos o por definir
    ]

    # Monte Carlo
    results = run_tournament(qualified_teams, ratings, shootout_ratings, n_sims=10000)

    print("=== Probabilidades de Campeón ===")
    print(results['champion_probs'].head(10))

    # Guardar a CSV
    save_to_csv(results)
```

### 2. dashboard.py

```python
import streamlit as st
import pandas as pd
from simulator import *

st.set_page_config(page_title="Simulador Mundial 2026", page_icon="🏆")
st.title("🏆 Simulador Mundial 2026")

# Cargar datos
matches = pd.read_csv('Data/ultimas_clasificatorias.csv')
manual = load_manual_matches()
if not manual.empty:
    matches = pd.concat([matches, manual], ignore_index=True)

# Calcular ratings
ratings, elo_history = calculate_elo(matches)

shootouts = pd.read_csv('Data/shootouts.csv')
shootout_ratings = calculate_shootout_elo(shootouts)

# Equipos
qualified_teams = ['Argentina', 'Brazil', 'France', 'Germany', 'Spain', 'England',
                   'Netherlands', 'Portugal', 'Italy', 'Belgium', 'Croatia', 'Uruguay',
                   'Colombia', 'USA', 'Mexico', 'Japan', 'Morocco', 'Senegal', 'South Korea',
                   'Australia', 'Saudi Arabia', 'Iran', 'Qatar', 'Canada', 'Ecuador',
                   'Panama', 'Peru', 'Chile', 'Paraguay', 'Venezuela', 'Nigeria']

# Sidebar
n_sims = st.sidebar.slider("Simulaciones", 1000, 50000, 10000, step=1000)
seed = st.sidebar.number_input("Seed", value=42)

# Resultados
results = run_tournament(qualified_teams, ratings, shootout_ratings, n_sims=n_sims, seed=seed)

# Guardar CSVs
save_to_csv(results)
save_elo_history(elo_history)

#Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Campeón", "Grupos", "Knockout", "Ratings", "Actualizar"])

with tab1:
    st.subheader("Probabilidades de Campeón")
    st.dataframe(results['champion_probs'].head(16).reset_index().rename(columns={'index': 'Equipo', 0: 'Probabilidad'}))
    st.bar_chart(results['champion_probs'].head(10))

with tab2:
    st.subheader("Resultados de Grupos")
    group_results = results['group_results']
    for letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
        st.markdown(f"**Grupo {letter}**")
        st.dataframe(group_results[group_results['group'] == letter].reset_index(drop=True))

with tab3:
    st.subheader("Eliminación Directa")
    if not results['knockout_results'].empty:
        st.dataframe(results['knockout_results'])

with tab4:
    st.subheader("Ratings Elo")
    ratings_df = pd.DataFrame(list(ratings.items()), columns=['Equipo', 'Rating'])
    ratings_df = ratings_df[ratings_df['Equipo'].isin(qualified_teams)].sort_values('Rating', ascending=False)
    st.dataframe(ratings_df.reset_index(drop=True))

with tab5:
    st.subheader("Agregar Partido Manual")
    with st.form("add_match"):
        col1, col2 = st.columns(2)
        date = col1.date_input("Fecha", value=pd.Timestamp.now())
        home = col2.selectbox("Local", qualified_teams, key="home")

        col3, col4 = st.columns(2)
        away = col3.selectbox("Visitante", qualified_teams, key="away")
        col4.write("")  # spacing

        col5, col6 = st.columns(2)
        goals_home = col5.number_input("Goles Local", min_value=0, value=0)
        goals_away = col6.number_input("Goles Visitante", min_value=0, value=0)

        submit = st.form_submit_button("Agregar")

        if submit and home != away:
            add_manual_match(date, home, away, goals_home, goals_away)
            st.success(f"Partido agregado: {home} {goals_home} - {goals_away} {away}")
            st.rerun()
        elif submit and home == away:
            st.error("Equipos diferentes")

    st.subheader("Partidos Agregados")
    st.dataframe(load_manual_matches())
```

## CSVs de salida (para agente reportero)
| Archivo | Descripción |
|---------|-------------|
| `output/elo_history.csv` | Evolución de ratings por fecha |
| `output/simulation_champion_probs.csv` | Probabilidades de campeón |
| `output/simulation_group_results.csv` | Posiciones en todos los grupos |
| `output/simulation_knockout_results.csv` | Resultados elimination |
| `output/simulation_best_thirds.csv` | Mejores terceros clasificados |
| `output/manual_matches.csv` | Partidos agregados manualmente |

## Verificación
1. `python simulator.py` corre sin errores
2. Ratings plausibles (top equipos ~2000)
3. Probabilidades suman ~1
4. CSVs se generan en carpeta `output/`

## Uso
```bash
python simulator.py
streamlit run dashboard.py
```