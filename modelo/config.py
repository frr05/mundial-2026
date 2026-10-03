"""Rutas y parámetros por defecto del modelo."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'Data'
OUTPUT_DIR = ROOT / 'Output'

RESULTS_CSV = DATA_DIR / 'results.csv'
SHOOTOUTS_CSV = DATA_DIR / 'shootouts.csv'
FIFA_RATINGS_CSV = OUTPUT_DIR / 'fifa_elo.csv'
MODEL_CONFIG_JSON = OUTPUT_DIR / 'model_config.json'

# Día anterior al partido inaugural: todo lo previo es información "pre-Mundial".
WC2026_START = '2026-06-11'
WC2026_END = '2026-07-19'

# Ventana usada para ajustar el modelo de goles (partidos con Elo ya estabilizado).
GOAL_MODEL_FROM = '2010-01-01'
