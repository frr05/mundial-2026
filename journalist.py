"""
Agente Periodista - Analiza simulaciones y genera insights
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class JournalistAgent:
    def __init__(self):
        self.insights = []

    def load_data(self):
        """Carga todos los datos disponibles"""
        self.elo_history = pd.read_csv('output/elo_history.csv')
        self.elo_history['date'] = pd.to_datetime(self.elo_history['date'])

        try:
            self.champion_probs = pd.read_csv('output/simulation_champion_probs.csv')
        except:
            pass

        try:
            self.group_results = pd.read_csv('output/simulation_group_results.csv')
        except:
            pass

    def analyze_team_evolution(self, team):
        """Analiza la evolución de un equipo"""
        team_data = self.elo_history[self.elo_history['team'] == team].sort_values('date')

        if len(team_data) == 0:
            return f"No hay datos para {team}"

        start_rating = team_data['rating'].iloc[0]
        end_rating = team_data['rating'].iloc[-1]
        max_rating = team_data['rating'].max()
        min_rating = team_data['rating'].min()

        trend = "al alza" if end_rating > start_rating else "a la baja"

        return {
            'team': team,
            'start': start_rating,
            'end': end_rating,
            'change': end_rating - start_rating,
            'max': max_rating,
            'min': min_rating,
            'trend': trend,
            'matches': len(team_data)
        }

    def generate_top_insights(self, n=10):
        """Genera los principales insights"""
        insights = []

        # Top equipos por rating actual
        latest = self.elo_history.sort_values('date').groupby('team').last().reset_index()
        top = latest.nlargest(n, 'rating')

        insights.append("="*50)
        insights.append("TOP 10 EQUIPOS POR RATING ACTUAL")
        insights.append("="*50)
        for i, row in top.iterrows():
            insights.append(f"{row['team']}: {row['rating']:.1f}")

        # Equipos en ascenso
        first_last = latest.copy()
        first_last = first_last[first_last['rating'] > 1500]  #过滤

        # Calcular cambio
        trends = []
        for team in latest['team'].unique()[:30]:
            team_data = self.elo_history[self.elo_history['team'] == team].sort_values('date')
            if len(team_data) > 10:
                start = team_data['rating'].iloc[:5].mean()
                end = team_data['rating'].iloc[-5:].mean()
                trends.append({'team': team, 'change': end - start})

        trends_df = pd.DataFrame(trends).sort_values('change', ascending=False)

        insights.append("\n" + "="*50)
        insights.append("EQUIPOS EN ASCENSO")
        insights.append("="*50)
        for _, row in trends_df.head(5).iterrows():
            insights.append(f"{row['team']}: +{row['change']:.1f}")

        insights.append("\n" + "="*50)
        insights.append("EQUIPOS EN DESCENSO")
        insights.append("="*50)
        for _, row in trends_df.tail(5).iterrows():
            insights.append(f"{row['team']}: {row['change']:.1f}")

        return "\n".join(insights)

    def plot_team_evolution(self, teams, save_path='output/journalist_evolution.png'):
        """Grafica evolución de varios equipos"""
        plt.figure(figsize=(14, 6))

        for team in teams:
            team_data = self.elo_history[self.elo_history['team'] == team].sort_values('date')
            if len(team_data) > 0:
                plt.plot(team_data['date'], team_data['rating'], marker='.', label=team, linewidth=1.5)

        plt.title('Evolución Elo - Comparación de Equipos')
        plt.xlabel('Fecha')
        plt.ylabel('Rating Elo')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Gráfico guardado: {save_path}")

    def plot_champion_probs(self, save_path='output/journalist_champions.png'):
        """Grafica probabilidades de campeón"""
        try:
            probs = pd.read_csv('output/simulation_championship_probs.csv')
            plt.figure(figsize=(12, 6))
            plt.barh(probs.iloc[:10, 0], probs.iloc[:10, 1])
            plt.xlabel('Probabilidad')
            plt.title('Top 10 Probabilidades de Campeón')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"Gráfico guardado: {save_path}")
        except Exception as e:
            print(f"No se pudo generar gráfico: {e}")

    def analyze_group_performance(self, group):
        """Analiza rendimiento en un grupo"""
        try:
            grp = self.group_results[self.group_results['group'] == group].copy()
            grp = grp.sort_values('position')
            return grp
        except:
            return None

    def generate_report(self):
        """Genera reporte completo"""
        self.load_data()

        report = []
        report.append("="*60)
        report.append("REPORTE DEL PERIODISTA - MUNDIAL 2026")
        report.append("="*60)
        report.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append("")

        # Insights principales
        report.append(self.generate_top_insights())

        # Guardar
        report_text = "\n".join(report)
        with open('output/journalist_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(report_text)
        print("\nReporte guardado: output/journalist_report.txt")

        # Gráficos
        self.plot_team_evolution(['Argentina', 'Brazil', 'France', 'Germany', 'Spain', 'England'])
        self.plot_champion_probs()


if __name__ == '__main__':
    journalist = JournalistAgent()
    journalist.generate_report()