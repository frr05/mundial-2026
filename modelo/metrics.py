"""Métricas de evaluación probabilística."""
import numpy as np

EPS = 1e-12


def outcome_index(goals_a, goals_b):
    """0 = gana A, 1 = empate, 2 = gana B."""
    ga, gb = np.asarray(goals_a), np.asarray(goals_b)
    return np.where(ga > gb, 0, np.where(ga == gb, 1, 2))


def log_loss(probs, outcomes):
    probs = np.asarray(probs)
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(outcomes)), outcomes], EPS, 1))))


def brier(probs, outcomes):
    probs = np.asarray(probs)
    onehot = np.eye(probs.shape[1])[outcomes]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def rps(probs, outcomes):
    """Ranked Probability Score (1X2 ordenado); el estándar para pronósticos de fútbol."""
    probs = np.asarray(probs)
    onehot = np.eye(probs.shape[1])[outcomes]
    cum = np.cumsum(probs, axis=1)[:, :-1] - np.cumsum(onehot, axis=1)[:, :-1]
    return float(np.mean(np.sum(cum ** 2, axis=1) / (probs.shape[1] - 1)))


def accuracy(probs, outcomes):
    return float(np.mean(np.argmax(probs, axis=1) == outcomes))


def binary_scores(p, y):
    """Para "quién avanza" en eliminación directa: p = P(A avanza), y = 1 si A avanzó."""
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return {
        'log_loss': float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        'brier': float(np.mean((p - y) ** 2)),
        'accuracy': float(np.mean((p > 0.5) == (y == 1))),
    }


def summary(probs, outcomes):
    return {
        'log_loss': log_loss(probs, outcomes),
        'brier': brier(probs, outcomes),
        'rps': rps(probs, outcomes),
        'accuracy': accuracy(probs, outcomes),
    }
