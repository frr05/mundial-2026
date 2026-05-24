import pandas as pd
import numpy as np
import json
import os
from collections import defaultdict
from datetime import datetime

os.makedirs('output', exist_ok=True)

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
    if config is None:
        config = DEFAULT_CONFIG
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)


def load_config(filename='output/model_config.json'):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return DEFAULT_CONFIG


# === ELO ===
def calculate_elo(matches_df, initial_rating=1500, k=32, home_adv=100):
    ratings = defaultdict(lambda: initial_rating)
    history = []

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

        date = row.get('date', datetime.now())
        history.append({'date': date, 'team': home, 'rating': ratings[home]})
        history.append({'date': date, 'team': away, 'rating': ratings[away]})

    return dict(ratings), pd.DataFrame(history)


def calculate_shootout_elo(shootouts_df, initial_rating=1500, k=20, date_from='2022-06-01'):
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
    diff = rating_home + home_adv - rating_away
    return np.exp(intercept + coef * diff)


def estimate_lambda_neutral(rating_a, rating_b, intercept=1.2, coef=0.004):
    diff = rating_a - rating_b
    return np.exp(intercept + coef * diff)


def simulate_match(team_home, team_away, ratings):
    lambda_home = estimate_lambda(ratings[team_home], ratings[team_away])
    lambda_away = estimate_lambda(ratings[team_away], ratings[team_home])

    goals_home = np.random.poisson(lambda_home)
    goals_away = np.random.poisson(lambda_away)

    return goals_home, goals_away


def simulate_match_neutral(team_a, team_b, ratings, seed=None):
    if seed is not None:
        np.random.seed(seed)

    lambda_a = estimate_lambda_neutral(ratings[team_a], ratings[team_b])
    lambda_b = estimate_lambda_neutral(ratings[team_b], ratings[team_a])

    goals_a = np.random.poisson(lambda_a)
    goals_b = np.random.poisson(lambda_b)

    if goals_a == goals_b:
        extra_a = np.random.poisson(lambda_a * 0.166)
        extra_b = np.random.poisson(lambda_b * 0.166)
        goals_a += extra_a
        goals_b += extra_b

    return goals_a, goals_b


def simulate_penalties(team_home, team_away, shootout_ratings, seed=None):
    if seed is not None:
        np.random.seed(seed)

    r_home = shootout_ratings.get(team_home, 1500)
    r_away = shootout_ratings.get(team_away, 1500)

    elo_bonus = (r_home - r_away) / 4000
    p_home = 0.75 + elo_bonus
    p_away = 0.75 - elo_bonus

    p_home = max(0.5, min(0.95, p_home))
    p_away = 1 - p_home

    home_score = sum(np.random.random() < p_home for _ in range(5))
    away_score = sum(np.random.random() < p_away for _ in range(5))

    while home_score == away_score:
        if np.random.random() < p_home:
            home_score += 1
        else:
            away_score += 1

    return home_score, away_score


# === TORNEO ===
def simulate_group(teams, ratings, group_name):
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

    for t in table:
        table[t]['gd'] = table[t]['gf'] - table[t]['ga']

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
    thirds = []
    for df in group_dfs:
        third = df[df['position'] == 3].iloc[0]
        thirds.append({'team': third['team'], 'pts': third['pts'], 'gd': third['gd'], 'gf': third['gf'], 'group': third['group']})

    thirds = sorted(thirds, key=lambda x: (x['pts'], x['gd'], x['gf']), reverse=True)
    return [t['team'] for t in thirds[:8]]


ROUND_OF_32_FIXTURE = [
    ('1A', '3rd_best'),
    ('2B', '2F'),
    ('1C', '3rd_best'),
    ('1D', '2E'),
    ('1B', '3rd_best'),
    ('2A', '2C'),
    ('1E', '3rd_best'),
    ('1F', '2D'),
    ('1G', '3rd_best'),
    ('2H', '2L'),
    ('1I', '3rd_best'),
    ('1J', '2K'),
    ('1H', '3rd_best'),
    ('2G', '2I'),
    ('1K', '3rd_best'),
    ('1L', '2J'),
]


def simulate_knockout_neutral(teams, ratings, shootout_ratings, seed=None):
    n = len(teams)
    if n == 1:
        return teams[0], []

    results = []
    next_round = []

    for i in range(0, n, 2):
        home = teams[i]
        away = teams[i + 1]

        gh, ga = simulate_match_neutral(home, away, ratings, seed=seed)

        if gh > ga:
            winner = home
        elif gh < ga:
            winner = away
        else:
            ph, pa = simulate_penalties(home, away, shootout_ratings, seed=seed)
            winner = home if ph > pa else away

        next_round.append(winner)
        results.append({'home': home, 'away': away, 'home_goals': gh, 'away_goals': ga, 'winner': winner})

    next_winners, next_results = simulate_knockout_neutral(next_round, ratings, shootout_ratings, seed)
    return next_winners, results + next_results


def simulate_full_tournament(teams, ratings, shootout_ratings, seed=None):
    groups = [teams[i:i+4] for i in range(0, 48, 4)]
    group_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']

    all_group_results = []
    group_winners = {}

    for group, name in zip(groups, group_names):
        df = simulate_group(group, ratings, name)
        all_group_results.append(df)
        for pos in [1, 2]:
            team = df[df['position'] == pos]['team'].iloc[0]
            group_winners[f'{pos}{name}'] = team

    best_thirds = get_best_third(all_group_results)

    r32_teams = []
    thirds_copy = best_thirds.copy()
    for match in ROUND_OF_32_FIXTURE:
        if match[1] == '3rd_best':
            r32_teams.append(thirds_copy.pop(0))
        else:
            group_letter = match[1][1]
            pos = match[1][0]
            r32_teams.append(group_winners[f'{pos}{group_letter}'])

    for name in group_names:
        r32_teams.append(group_winners[f'1{name}'])

    winner, knockout_results = simulate_knockout_neutral(r32_teams, ratings, shootout_ratings, seed)

    return winner, all_group_results, knockout_results, best_thirds


def run_tournament(teams, ratings, shootout_ratings, n_sims=10000, seed=42):
    np.random.seed(seed)

    all_winners = []
    all_group_results = []
    all_knockout_results = []
    all_best_thirds = []

    for sim in range(n_sims):
        winner, group_dfs, knockout, thirds = simulate_full_tournament(teams, ratings, shootout_ratings, seed=seed + sim)
        all_winners.append(winner)

        for df in group_dfs:
            df['simulation'] = sim + 1
            all_group_results.append(df)

        for k in knockout:
            k['simulation'] = sim + 1
            all_knockout_results.append(k)

        for t in thirds:
            all_best_thirds.append({'team': t, 'simulation': sim + 1})

    return {
        'champion_probs': pd.Series(all_winners).value_counts(normalize=True).sort_values(ascending=False),
        'group_results': pd.concat(all_group_results, ignore_index=True),
        'knockout_results': pd.DataFrame(all_knockout_results),
        'best_thirds': pd.DataFrame(all_best_thirds)
    }


# === EXPORTAR MODELO ===
def export_model(ratings, shootout_ratings, config, folder='output/export'):
    os.makedirs(folder, exist_ok=True)

    pd.DataFrame(list(ratings.items()), columns=['team', 'rating']).to_csv(f'{folder}/ratings.csv', index=False)
    pd.DataFrame(list(shootout_ratings.items()), columns=['team', 'rating']).to_csv(f'{folder}/shootout_ratings.csv', index=False)
    save_config(config, f'{folder}/config.json')

    return folder


def import_model(folder='output/export'):
    ratings = pd.read_csv(f'{folder}/ratings.csv').set_index('team')['rating'].to_dict()
    shootout_ratings = pd.read_csv(f'{folder}/shootout_ratings.csv').set_index('team')['rating'].to_dict()
    config = load_config(f'{folder}/config.json')

    return ratings, shootout_ratings, config


# === PERSISTENCIA ===
def save_to_csv(results, prefix='simulation'):
    results['champion_probs'].to_csv(f'output/{prefix}_champion_probs.csv', header=['probability'])
    results['group_results'].to_csv(f'output/{prefix}_group_results.csv', index=False)

    if not results['knockout_results'].empty:
        results['knockout_results'].to_csv(f'output/{prefix}_knockout_results.csv', index=False)

    results['best_thirds'].to_csv(f'output/{prefix}_best_thirds.csv', index=False)


def save_elo_history(elo_history_df):
    elo_history_df.to_csv('output/elo_history.csv', index=False)


def load_manual_matches():
    try:
        return pd.read_csv('output/manual_matches.csv')
    except FileNotFoundError:
        return pd.DataFrame(columns=['date', 'home_team', 'away_team', 'home_score', 'away_score', 'tournament'])


def save_manual_matches(matches_df):
    matches_df.to_csv('output/manual_matches.csv', index=False)


def add_manual_match(date, home, away, home_goals, away_goals):
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
    print("Cargando datos...")
    matches = pd.read_csv('Data/ultimas_clasificatorias.csv')

    manual = load_manual_matches()
    if not manual.empty:
        matches = pd.concat([matches, manual], ignore_index=True)
        print(f"Agregados {len(manual)} partidos manuales")

    print("Calculando Elo...")
    ratings, elo_history = calculate_elo(matches)

    print("Calculando Elo de shootouts...")
    shootouts = pd.read_csv('Data/shootouts.csv')
    shootout_ratings = calculate_shootout_elo(shootouts)

    save_elo_history(elo_history)
    save_config()

    # 48 equipos - AGREGAR DESPUÉS
    qualified_teams = [
        'Argentina', 'Brazil', 'France', 'Germany', 'Spain', 'England',
        'Netherlands', 'Portugal', 'Italy', 'Belgium', 'Croatia', 'Uruguay',
        'Colombia', 'USA', 'Mexico', 'Japan', 'Morocco', 'Senegal', 'South Korea',
        'Australia', 'Saudi Arabia', 'Iran', 'Qatar', 'Canada', 'Ecuador',
        'Panama', 'Peru', 'Chile', 'Paraguay', 'Venezuela', 'Nigeria',
    ]

    if len(qualified_teams) < 48:
        print(f"Advertencia: Solo hay {len(qualified_teams)} equipos definidos. Se necesitan 48.")

    print("Ejecutando Monte Carlo...")
    results = run_tournament(qualified_teams, ratings, shootout_ratings, n_sims=10000, seed=42)

    print("\n=== Probabilidades de Campeón ===")
    print(results['champion_probs'].head(10))

    save_to_csv(results)
    export_model(ratings, shootout_ratings, DEFAULT_CONFIG)

    print("\nArchivos guardados en output/")