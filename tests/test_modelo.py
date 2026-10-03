import numpy as np
import pandas as pd
import pytest

from modelo import metrics as M
from modelo.data import (OFFICIAL_THIRDS_ASSIGNMENT, THIRD_PLACE_ALLOWED, WC2026_GROUPS,
                         known_results_from, load_results, load_shootouts, wc2026_matches)
from modelo.elo import EloParams, EloSystem, gd_multiplier, tournament_k
from modelo.goals import GoalModel, score_matrix, outcome_probs
from modelo.model import build_model
from modelo.tournament import (TournamentSimulator, assign_thirds, group_fixtures_from_results,
                               rank_group)


@pytest.fixture(scope='module')
def data():
    res = load_results()
    so = load_shootouts()
    return {
        'results': res, 'shootouts': so, 'wc': wc2026_matches(res, so),
        'fixtures': group_fixtures_from_results(load_results(played_only=False)),
        'model': build_model('2026-06-11', results=res, shootouts=so),
    }


# --- Elo ---
def test_tournament_k():
    assert tournament_k('FIFA World Cup') == 60
    assert tournament_k('Copa América') == 50
    assert tournament_k('FIFA World Cup qualification') == 40
    assert tournament_k('Friendly') == 20


def test_gd_multiplier():
    assert gd_multiplier(1) == 1.0 and gd_multiplier(-2) == 1.5 and gd_multiplier(3) == 1.75


def test_elo_is_zero_sum_and_home_adv():
    elo = EloSystem(EloParams(home_adv=100))
    elo.update('A', 'B', 1, 1, 20, neutral=False)  # empate de local: el local pierde puntos
    assert elo.rating('A') < 1500 < elo.rating('B')
    assert elo.rating('A') + elo.rating('B') == pytest.approx(3000)


# --- Goles ---
def test_score_matrix_normalized_and_symmetric():
    m = score_matrix(1.3, 1.3, rho=-0.05)
    assert m.sum() == pytest.approx(1.0)
    pa, pd_, pb = outcome_probs(m)
    assert pa[0] == pytest.approx(pb[0])
    assert pa[0] + pd_[0] + pb[0] == pytest.approx(1.0)


def test_goal_model_recovers_parameters():
    rng = np.random.default_rng(0)
    n = 20000
    ra = rng.normal(1700, 150, n)
    rb = rng.normal(1700, 150, n)
    true = GoalModel(mu=0.3, beta=0.8, gamma=0.1, delta=0.2, omega=0.0, rho=0.0)
    home = rng.integers(0, 2, n).astype(float)
    la, lb = true.lambdas(ra, rb, home)
    fit = GoalModel.fit(ra, rb, home, np.zeros(n), rng.poisson(la), rng.poisson(lb), fit_omega=False)
    assert fit.mu == pytest.approx(0.3, abs=0.03)
    assert fit.beta == pytest.approx(0.8, abs=0.05)


def test_metrics_perfect_and_uniform():
    o = np.array([0, 1, 2])
    assert M.log_loss(np.eye(3), o) == pytest.approx(0, abs=1e-9)
    assert M.rps(np.full((3, 3), 1 / 3), o) > 0


# --- Torneo ---
def test_head_to_head_tiebreak():
    # A y B empatan en puntos; B ganó el directo aunque A tenga mejor diferencia de gol.
    matches = [('A', 'B', 0, 1), ('A', 'C', 5, 0), ('A', 'D', 1, 0),
               ('B', 'C', 0, 1), ('B', 'D', 1, 0), ('C', 'D', 0, 0)]
    ranked, _ = rank_group(['A', 'B', 'C', 'D'], matches, lambda t: 0)
    assert ranked[:2] == ['B', 'A']


def test_thirds_assignment_respects_rules():
    import itertools
    for combo in itertools.islice(itertools.combinations('ABCDEFGHIJKL', 8), 0, 495, 7):
        a = assign_thirds(combo)
        assert sorted(a.values()) == sorted(combo)
        assert all(g in THIRD_PLACE_ALLOWED[w] for w, g in a.items())
    for key, a in OFFICIAL_THIRDS_ASSIGNMENT.items():
        assert all(g in THIRD_PLACE_ALLOWED[w] for w, g in a.items())


def test_wc2026_data_complete(data):
    wc = data['wc']
    assert len(wc) == 104
    assert wc['winner'][wc['stage'] != 'GROUP'].notna().all()
    assert wc[wc['stage'] == 'FINAL']['winner'].iloc[0] == 'Spain'


def test_simulation_with_all_real_results_reproduces_tournament(data):
    known = known_results_from(data['wc'])
    sim = TournamentSimulator(data['model'], group_fixtures=data['fixtures'], known=known, seed=1)
    called = []
    original = sim._play_90
    sim._play_90 = lambda *a, **k: called.append(a) or original(*a, **k)
    s = sim.simulate_once()
    assert not called, 'todos los partidos deberían venir de los resultados reales'
    assert (s['champion'], s['runner_up'], s['third']) == ('Spain', 'Argentina', 'England')


def test_monte_carlo_probabilities_are_consistent(data):
    df = TournamentSimulator(data['model'], group_fixtures=data['fixtures'], seed=3).run(300)
    stages = ['Dieciseisavos (R32) %', 'Octavos (R16) %', 'Cuartos (QF) %', 'Semifinal (SF) %',
              'Final (F) %', 'Campeón %']
    assert (df[stages].diff(axis=1).iloc[:, 1:] <= 1e-9).all().all()  # monótono por fase
    assert df[stages].max().max() <= 100
    assert df['Dieciseisavos (R32) %'].sum() == pytest.approx(3200)
    assert df['Campeón %'].sum() == pytest.approx(100)
    assert len(df) == sum(len(t) for t in WC2026_GROUPS.values())
