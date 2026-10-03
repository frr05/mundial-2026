"""Carga de datos y definición del formato del Mundial 2026."""
import numpy as np
import pandas as pd

from .config import RESULTS_CSV, SHOOTOUTS_CSV, WC2026_START, WC2026_END

WC_TOURNAMENT = 'FIFA World Cup'

WC2026_GROUPS = {
    'A': ['Mexico', 'South Africa', 'South Korea', 'Czech Republic'],
    'B': ['Canada', 'Bosnia and Herzegovina', 'Qatar', 'Switzerland'],
    'C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
    'D': ['United States', 'Paraguay', 'Australia', 'Turkey'],
    'E': ['Germany', 'Curaçao', 'Ivory Coast', 'Ecuador'],
    'F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
    'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'H': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'],
    'I': ['France', 'Senegal', 'Iraq', 'Norway'],
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'K': ['Portugal', 'DR Congo', 'Uzbekistan', 'Colombia'],
    'L': ['England', 'Croatia', 'Ghana', 'Panama'],
}
TEAM_GROUP = {t: g for g, teams in WC2026_GROUPS.items() for t in teams}

# Anfitriones: juegan "de local" cuando la sede está en su país.
HOST_COUNTRY = {'Mexico': 'Mexico', 'United States': 'United States', 'Canada': 'Canada'}

# Grupos de terceros que puede enfrentar cada primero en Dieciseisavos (reglamento FIFA).
THIRD_PLACE_ALLOWED = {
    'A': ['C', 'E', 'F', 'H', 'I'],
    'B': ['E', 'F', 'G', 'I', 'J'],
    'D': ['B', 'E', 'F', 'I', 'J'],
    'E': ['A', 'B', 'C', 'D', 'F'],
    'G': ['A', 'E', 'H', 'I', 'J'],
    'I': ['C', 'D', 'F', 'G', 'H'],
    'K': ['D', 'E', 'I', 'J', 'L'],
    'L': ['E', 'H', 'I', 'J', 'K'],
}

# Asignaciones oficiales (Anexo C FIFA) conocidas. La tabla completa tiene 495
# combinaciones; para las que no están aquí se usa backtracking con las reglas de arriba.
OFFICIAL_THIRDS_ASSIGNMENT = {
    # Combinación real del Mundial 2026 (verificada con los partidos de Dieciseisavos).
    frozenset('BDEFIJKL'): {'A': 'E', 'B': 'J', 'D': 'B', 'E': 'D',
                            'G': 'I', 'I': 'F', 'K': 'L', 'L': 'K'},
}

# Partidos 73-88: (slot_local, slot_visita, país sede). 'TX' = tercero asignado al 1X.
R32_FIXTURE = {
    73: ('2A', '2B', 'United States'),
    74: ('1E', 'TE', 'United States'),
    75: ('1F', '2C', 'Mexico'),
    76: ('1C', '2F', 'United States'),
    77: ('1I', 'TI', 'United States'),
    78: ('2E', '2I', 'United States'),
    79: ('1A', 'TA', 'Mexico'),
    80: ('1L', 'TL', 'United States'),
    81: ('1D', 'TD', 'United States'),
    82: ('1G', 'TG', 'United States'),
    83: ('2K', '2L', 'Canada'),
    84: ('1H', '2J', 'United States'),
    85: ('1B', 'TB', 'Canada'),
    86: ('1J', '2H', 'United States'),
    87: ('1K', 'TK', 'United States'),
    88: ('2D', '2G', 'United States'),
}

# Partidos 89-104: (ganador_de, ganador_de, país sede, etapa)
KNOCKOUT_FIXTURE = {
    89: (74, 77, 'United States', 'R16'),
    90: (73, 75, 'United States', 'R16'),
    91: (76, 78, 'United States', 'R16'),
    92: (79, 80, 'Mexico', 'R16'),
    93: (83, 84, 'United States', 'R16'),
    94: (81, 82, 'United States', 'R16'),
    95: (86, 88, 'United States', 'R16'),
    96: (85, 87, 'Canada', 'R16'),
    97: (89, 90, 'United States', 'QF'),
    98: (93, 94, 'United States', 'QF'),
    99: (91, 92, 'United States', 'QF'),
    100: (95, 96, 'United States', 'QF'),
    101: (97, 98, 'United States', 'SF'),
    102: (99, 100, 'United States', 'SF'),
    104: (101, 102, 'United States', 'FINAL'),
}
THIRD_PLACE_MATCH = (103, 'United States')

STAGES = ['GROUP', 'R32', 'R16', 'QF', 'SF', 'FINAL', 'CHAMPION']


def load_results(path=RESULTS_CSV, played_only=True):
    """Carga results.csv ordenado cronológicamente (estable dentro de cada fecha)."""
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    if played_only:
        df = df.dropna(subset=['home_score', 'away_score'])
        df = df.astype({'home_score': int, 'away_score': int})
    df['neutral'] = df['neutral'].astype(str).str.upper().eq('TRUE')
    return df.sort_values('date', kind='stable').reset_index(drop=True)


def load_shootouts(path=SHOOTOUTS_CSV):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def wc2026_matches(results, shootouts=None):
    """Partidos del Mundial 2026 ya jugados, con etapa y ganador (incluye penales)."""
    mask = ((results['tournament'] == WC_TOURNAMENT)
            & (results['date'] >= WC2026_START) & (results['date'] <= WC2026_END))
    wc = results[mask].copy()

    def stage(d):
        d = d.strftime('%m-%d')
        if d <= '06-27':
            return 'GROUP'
        if d <= '07-03':
            return 'R32'
        if d <= '07-07':
            return 'R16'
        if d <= '07-11':
            return 'QF'
        if d <= '07-15':
            return 'SF'
        if d == '07-18':
            return '3RD'
        return 'FINAL'

    wc['stage'] = wc['date'].apply(stage)
    wc['winner'] = None
    wc.loc[wc['home_score'] > wc['away_score'], 'winner'] = wc['home_team']
    wc.loc[wc['home_score'] < wc['away_score'], 'winner'] = wc['away_team']
    wc['penalties'] = False
    if shootouts is not None:
        so = shootouts.set_index(['date', 'home_team', 'away_team'])['winner']
        for idx, row in wc[wc['stage'] != 'GROUP'].iterrows():
            key = (row['date'], row['home_team'], row['away_team'])
            if row['home_score'] == row['away_score'] and key in so.index:
                wc.at[idx, 'winner'] = so.loc[key]
                wc.at[idx, 'penalties'] = True
    return wc.reset_index(drop=True)


def known_results_from(wc_matches):
    """Diccionario {(etapa, frozenset(equipos)): fila} para fijar resultados reales en simulación."""
    known = {}
    for _, r in wc_matches.iterrows():
        known[(r['stage'], frozenset((r['home_team'], r['away_team'])))] = {
            r['home_team']: int(r['home_score']), r['away_team']: int(r['away_score']),
            'winner': r['winner'],
        }
    return known


# Confederaciones de los equipos mundialistas (para la corrección por confederación).
_CONF = {
    'UEFA': ['Austria', 'Belgium', 'Bosnia and Herzegovina', 'Bulgaria', 'Croatia', 'Czech Republic',
             'Denmark', 'England', 'France', 'Germany', 'Greece', 'Iceland', 'Italy', 'Netherlands',
             'Norway', 'Poland', 'Portugal', 'Republic of Ireland', 'Romania', 'Russia', 'Scotland',
             'Serbia', 'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'Turkey', 'Ukraine',
             'Wales', 'Serbia and Montenegro', 'Yugoslavia'],
    'CONMEBOL': ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Paraguay', 'Peru',
                 'Uruguay', 'Venezuela'],
    'CONCACAF': ['Canada', 'Costa Rica', 'Curaçao', 'Haiti', 'Honduras', 'Jamaica', 'Mexico', 'Panama',
                 'Trinidad and Tobago', 'United States'],
    'CAF': ['Algeria', 'Angola', 'Cameroon', 'Cape Verde', 'DR Congo', 'Egypt', 'Ghana', 'Ivory Coast',
            'Morocco', 'Nigeria', 'Senegal', 'South Africa', 'Togo', 'Tunisia'],
    'AFC': ['Australia', 'China', 'China PR', 'Iran', 'Iraq', 'Japan', 'Jordan', 'North Korea', 'Qatar',
            'Saudi Arabia', 'South Korea', 'United Arab Emirates', 'Uzbekistan'],
    'OFC': ['New Zealand', 'Tahiti'],
}
CONFEDERATION = {team: conf for conf, teams in _CONF.items() for team in teams}


def world_cup_knockout_mask(results, since='1986-01-01'):
    """Marca los partidos de eliminación directa de cada Mundial desde `since`.

    Se identifican por orden: los primeros N partidos de cada edición son fase de grupos
    (36 con 24 equipos, 48 con 32 equipos, 72 con 48 equipos).
    """
    is_wc = (results['tournament'] == WC_TOURNAMENT) & (results['date'] >= since)
    mask = pd.Series(False, index=results.index)
    for year, w in results[is_wc].groupby(results.loc[is_wc, 'date'].dt.year):
        n_group = 72 if year >= 2026 else (48 if year >= 1998 else 36)
        mask.loc[w.index[n_group:]] = True
    return mask.to_numpy()


def home_team_advanced(matches, shootouts):
    """1.0 si avanzó el equipo local, 0.0 si avanzó el visitante, NaN si no se sabe (empate sin penales)."""
    out = np.where(matches['home_score'] > matches['away_score'], 1.0,
                   np.where(matches['home_score'] < matches['away_score'], 0.0, np.nan))
    so = shootouts.set_index(['date', 'home_team', 'away_team'])['winner']
    so = so[~so.index.duplicated()]
    for i, k in enumerate(zip(matches['date'], matches['home_team'], matches['away_team'])):
        if np.isnan(out[i]) and k in so.index:
            out[i] = float(so.loc[k] == k[1])
    return out
