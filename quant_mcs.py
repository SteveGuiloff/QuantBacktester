import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# CONTRATO DE INTERFAZ DE DATOS - SIMULADOR MONTE CARLO (MCS)
# =============================================================================
# Versión: 1.1 (Módulo de Validación Estadística y Robustez)
# 
# 1. ENTRADA AL SIMULADOR (Desde el Motor de Backtest)
# El simulador requiere el DataFrame 'trades_df' resultante de engine.run():
# -----------------------------------------------------------------------------
# Columnas Críticas Requeridas:
# - pnl_usd       (float): Para proyecciones de crecimiento de capital monetario.
# - pnl_r         (float): Para análisis de eficiencia de riesgo (R-Multiples).
#
# 2. PARÁMETROS DE CONFIGURACIÓN (Input del Usuario)
# -----------------------------------------------------------------------------
# - n_iterations  (int):   Cantidad de caminos aleatorios (Sugerido: 1000+).
# - sample_size   (int):   Número de trades por simulación (Default: len(trades)).
# - start_capital (float): Capital inicial de la cuenta.
# - replacement   (bool):  Uso de Bootstrap (muestreo con reemplazo).
#
# 3. SALIDA DE MÉTRICAS (Diccionario o Atributos de Clase)
# -----------------------------------------------------------------------------
# - prob_ruin     (float): % de iteraciones donde el capital cae bajo el umbral.
# - var_95        (float): Value at Risk (Máxima pérdida esperada al 95% conf).
# - cvar_95       (float): Conditional VaR (Promedio de pérdidas extremas).
# - expected_ret  (float): Retorno mediano (percentil 50).
#
# 4. INTERFAZ DE MÉTODOS REQUERIDOS
# -----------------------------------------------------------------------------
# - run_simulation():        Ejecuta el algoritmo de re-muestreo.
# - get_risk_report():       Devuelve tabla de percentiles y métricas VaR/CVaR.
# - plot_equity_lines():     Genera el gráfico de "abanico" de curvas de capital.
# - plot_pnl_dist():         Histograma de resultados con marcadores de riesgo.
# =============================================================================

class MonteCarloSimulator:
    """
    Implementación del Simulador Monte Carlo para validación de estrategias de trading.
    Utiliza técnicas de Bootstrap para estresar los resultados del backtest.
    """
    
    def __init__(self, trades_df):
        """
        Inicializa el simulador con los resultados netos de un backtest.
        """
        if 'pnl_usd' not in trades_df.columns:
            raise ValueError("El DataFrame debe contener la columna 'pnl_usd'")
            
        self.pnl_data = trades_df['pnl_usd'].values
        self.sim_results = None
        self.start_capital = 0
        self.n_iterations = 0

    def run_simulation(self, n_iterations=1000, sample_size=None, start_capital=10000, replacement=True):
        """
        Ejecuta la simulación utilizando vectorización de NumPy.
        """
        self.start_capital = start_capital
        self.n_iterations = n_iterations
        n_trades = sample_size if sample_size else len(self.pnl_data)
        
        # Generar matriz aleatoria: (Filas = Iteraciones, Columnas = Trades)
        draws = np.random.choice(self.pnl_data, size=(n_iterations, n_trades), replace=replacement)
        
        # Calcular curvas acumuladas
        self.sim_results = start_capital + np.cumsum(draws, axis=1)
        
        # Concatenar el capital inicial al inicio de cada fila para visualización completa
        initial_points = np.full((n_iterations, 1), start_capital)
        self.sim_results = np.hstack((initial_points, self.sim_results))
        
        print(f"[MCS] Simulación finalizada: {n_iterations} iteraciones de {n_trades} trades.")

    def get_risk_report(self, confidence_level=0.95):
        """
        Calcula percentiles, Probabilidad de Ruina, VaR y CVaR.
        """
        if self.sim_results is None:
            return "Error: Ejecute run_simulation() primero."
            
        final_values = self.sim_results[:, -1]
        final_returns = final_values - self.start_capital
        
        # Percentiles de Capital Final
        p_levels = [5, 25, 50, 75, 95]
        p_values = np.percentile(final_values, p_levels)
        
        # Métricas de Riesgo (Basadas en Retornos)
        # VaR: El percentil que define el límite de pérdida
        var_pct = (1 - confidence_level) * 100
        var_val = np.percentile(final_returns, var_pct)
        
        # CVaR: Promedio de los retornos que son peores que el VaR
        tail_losses = final_returns[final_returns <= var_val]
        cvar_val = tail_losses.mean() if len(tail_losses) > 0 else var_val
        
        # Probabilidad de pérdida (Ruina operativa)
        prob_loss = np.mean(final_values < self.start_capital) * 100

        report = {
            "Percentiles": dict(zip(p_levels, p_values)),
            "VaR": var_val,
            "CVaR": cvar_val,
            "Prob_Loss": prob_loss
        }
        
        # Impresión formateada
        print("\n" + "="*40)
        print("      REPORTE DE RIESGO MONTE CARLO")
        print("="*40)
        print(f"Capital Inicial:      ${self.start_capital:,.2f}")
        print(f"Prob. Ruina/Pérdida:  {prob_loss:.2f}%")
        print(f"VaR ({confidence_level*100:.0f}%):          ${abs(var_val):,.2f}")
        print(f"CVaR ({confidence_level*100:.0f}%):         ${abs(cvar_val):,.2f}")
        print("-" * 40)
        print(f"Mediana (P50):        ${p_values[2]:,.2f}")
        print(f"Peor Escenario (P5):  ${p_values[0]:,.2f}")
        print("="*40)
        
        return report

    def plot_equity_lines(self, max_lines=100):
        """
        Visualiza las curvas de equidad simuladas.
        """
        plt.figure(figsize=(12, 6))
        
        # Dibujar una muestra de líneas para evitar lentitud
        n_to_plot = min(max_lines, self.n_iterations)
        idx = np.random.choice(range(self.n_iterations), n_to_plot, replace=False)
        
        for i in idx:
            plt.plot(self.sim_results[i], color='gray', alpha=0.1, linewidth=1)
            
        # Resaltar Mediana y Percentiles Extremos
        median_line = np.percentile(self.sim_results, 50, axis=0)
        p5_line = np.percentile(self.sim_results, 5, axis=0)
        p95_line = np.percentile(self.sim_results, 95, axis=0)
        
        plt.plot(median_line, color='blue', label='Mediana (P50)', linewidth=2)
        plt.plot(p5_line, color='red', linestyle='--', label='Riesgo (P5)', linewidth=1.5)
        plt.plot(p95_line, color='green', linestyle='--', label='Optimista (P95)', linewidth=1.5)
        
        plt.axhline(self.start_capital, color='black', linestyle='-', alpha=0.5)
        plt.title(f"Simulación de Caminos Aleatorios (n={self.n_iterations})")
        plt.xlabel("Número de Operaciones")
        plt.ylabel("Capital USD")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_pnl_dist(self, confidence_level=0.95):
        """
        Visualiza la distribución de resultados finales y el riesgo de cola.
        """
        final_returns = self.sim_results[:, -1] - self.start_capital
        var_val = np.percentile(final_returns, (1 - confidence_level) * 100)
        
        plt.figure(figsize=(10, 6))
        n, bins, patches = plt.hist(final_returns, bins=50, color='skyblue', edgecolor='white', alpha=0.7)
        
        # Colorear zona de riesgo de cola
        for i in range(len(patches)):
            if bins[i] <= var_val:
                patches[i].set_facecolor('red')
                patches[i].set_alpha(0.5)
                
        plt.axvline(var_val, color='darkred', linestyle='--', label=f'VaR {confidence_level*100:.0f}%')
        plt.axvline(0, color='black', linewidth=1)
        
        plt.title("Distribución de Retornos Netos Finales")
        plt.xlabel("Ganancia/Pérdida Neta (USD)")
        plt.ylabel("Frecuencia de Iteraciones")
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

# =============================================================================
# EJEMPLO DE USO (MOCK DATA)
# =============================================================================
if __name__ == "__main__":
    # 1. Crear datos sintéticos que simulen la salida de un motor de backtest
    data = {
        'pnl_usd': np.random.normal(loc=50, scale=300, size=100), # Media $50, Volatilidad $300
        'pnl_r': np.random.normal(loc=0.5, scale=2, size=100)
    }
    mock_trades = pd.DataFrame(data)

    # 2. Inicializar y Correr MCS
    mcs = MonteCarloSimulator(mock_trades)
    mcs.run_simulation(n_iterations=2000, start_capital=10000)

    # 3. Reportes y Gráficos
    mcs.get_risk_report(confidence_level=0.95)
    mcs.plot_equity_lines()
    mcs.plot_pnl_dist()
