# 🏆 Proyecto: Modelo Elo para el Mundial 2026

## 📌 Descripción general

Este proyecto implementa un sistema de **ratings Elo aplicado al fútbol internacional**, con el objetivo de analizar y predecir el rendimiento de selecciones nacionales en el contexto del Mundial 2026.

El sistema se basa en resultados históricos de partidos internacionales y genera rankings dinámicos que se actualizan partido a partido, incluyendo variantes para distintos contextos (campeonato regular vs. definiciones por penales).

---

## ⚙️ Objetivos del proyecto

* Construir un sistema de ratings tipo Elo adaptado a fútbol internacional.
* Evaluar el rendimiento relativo de selecciones nacionales.
* Generar rankings comparables entre equipos.
* Modelar escenarios de partidos y posibles resultados.
* Analizar el impacto de los penales en el rating (versión shootout).

---

## 📊 Componentes principales

### 1. Dataset de partidos

* Base de datos con resultados históricos de selecciones.
* Incluye goles, fechas, equipos locales/visitantes y contexto del partido.

### 2. Modelo Elo

* Implementación de sistema Elo clásico adaptado a fútbol.
* Actualización de ratings tras cada partido.
* Ajustes por tipo de partido (amistoso, torneo, eliminación directa).

### 3. Variantes del modelo

* **Elo Campeonato**: basado en resultados estándar.
* **Elo Shootout**: incorpora definición por penales.
* **Elo Combinado**: mezcla ponderada de ambos escenarios.

### 4. Exportación de resultados

Se generan archivos CSV con los resultados finales:

* `ratings_championship.csv` → Ratings del modelo de campeonato.
* `ratings_shootout.csv` → Ratings considerando penales.
* `ratings_combined.csv` → Modelo combinado.
* `elo_history.csv` → Evolución histórica del rating.

---

## 📁 Estructura del proyecto (propuesta)

```
📦 proyecto-elo-mundial
├── data/
│   └── matches.csv
├── output/
│   ├── ratings_championship.csv
│   ├── ratings_shootout.csv
│   ├── ratings_combined.csv
│   └── elo_history.csv
├── notebooks/
│   └── analisis_elo.ipynb
├── src/
│   ├── elo_model.py
│   ├── data_processing.py
│   └── simulation.py
├── README.md
└── requirements.txt
```

---

## 🧠 Lógica del modelo Elo

El rating se actualiza según:

* Resultado del partido (victoria, empate, derrota)
* Diferencia de goles (opcional en variantes extendidas)
* K-factor ajustable según importancia del partido

---

## 🧪 Tecnologías usadas

* Python 🐍
* Pandas
* NumPy
* Jupyter Notebooks

---

## 🚀 Posibles mejoras futuras

* Integrar simulación completa del Mundial 2026
* Añadir modelo probabilístico de goles (Poisson)
* Ajustar ratings por confederación (UEFA, CONMEBOL, etc.)
* Visualización interactiva de rankings
* API para consultar ratings en tiempo real

---

## 📌 Notas

Este proyecto está en desarrollo activo y los resultados pueden variar a medida que se ajustan los parámetros del modelo y se incorporan nuevos datos históricos.

---

## 👤 Autor

Proyecto personal de análisis futbolístico basado en modelos Elo aplicados a selecciones nacionales.
