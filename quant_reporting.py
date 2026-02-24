import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class QuantReporter:
    """
    Módulo de Analítica y Visualización para resultados del QuantEngineV2.
    """
    def __init__(self, trades_df):
        if trades_df.empty:
            print("⚠️ No hay trades para analizar.")
            self.df = pd.DataFrame()
        else:
            self.df = trades_df.copy()
            self._prepare_data()

    def _prepare_data(self):
        """Calcula métricas acumulativas para reporting."""
        self.df['equity_r'] = self.df['pnl_r'].cumsum()
        self.df['equity_usd'] = self.df['pnl_usd'].cumsum()
        self.df['peak_r'] = self.df['equity_r'].cummax()
        self.df['drawdown_r'] = self.df['equity_r'] - self.df['peak_r']

    def get_summary_stats(self):
        """Genera un reporte detallado con métricas de robustez."""
        if self.df.empty: return pd.Series({})
        
        total_trades = len(self.df)
        wins = self.df[self.df['pnl_r'] > 0.1]
        losses = self.df[self.df['pnl_r'] < -0.1]
        
        total_pnl_r = self.df['pnl_r'].sum()
        max_dd_r = abs(self.df['drawdown_r'].min())
        
        # Nuevas Métricas
        recovery_factor = total_pnl_r / max_dd_r if max_dd_r != 0 else np.inf
        # Sharpe Ratio simplificado: Retorno promedio / Desviación estándar del retorno
        sharpe_r = self.df['pnl_r'].mean() / self.df['pnl_r'].std() if self.df['pnl_r'].std() != 0 else 0

        stats = {
            "Total Trades": total_trades,
            "Win Rate (%)": (len(wins) / total_trades) * 100,
            "Expectancy (R)": self.df['pnl_r'].mean(),
            "Profit Factor (R)": abs(self.df[self.df['pnl_r'] > 0]['pnl_r'].sum() / 
                                     self.df[self.df['pnl_r'] < 0]['pnl_r'].sum()) if any(self.df['pnl_r'] < 0) else np.inf,
            "Max Drawdown (R)": -max_dd_r,
            "Recovery Factor": recovery_factor,
            "Sharpe Ratio (R)": sharpe_r,
            "Total PnL (R)": total_pnl_r,
            "Total PnL (USD)": self.df['pnl_usd'].sum(),
            "Avg Trade (USD)": self.df['pnl_usd'].mean()
        }
        return pd.Series(stats)

    def plot_equity_curve(self):
        """Dibuja la curva de capital y el drawdown."""
        if self.df.empty: return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # Curva de Equity
        ax1.plot(self.df.index, self.df['equity_r'], label='Equity Acumulada (R)', color='#2ecc71', lw=2)
        ax1.fill_between(self.df.index, self.df['equity_r'], alpha=0.1, color='#2ecc71')
        ax1.set_title("Curva de Rendimiento (Múltiplos de R)", fontsize=14, fontweight='bold')
        ax1.set_ylabel("R")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Drawdown
        ax2.fill_between(self.df.index, self.df['drawdown_r'], 0, color='#e74c3c', alpha=0.3, label='Drawdown (R)')
        ax2.set_ylabel("DD (R)")
        ax2.set_xlabel("Número de Trades")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_monthly_analysis(self):
        """Análisis de rentabilidad por mes."""
        if self.df.empty: return
        
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['Year'] = self.df['date'].dt.year
        self.df['Month'] = self.df['date'].dt.month
        
        pivot = self.df.pivot_table(index='Year', columns='Month', values='pnl_r', aggfunc='sum').fillna(0)
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", center=0)
        plt.title("Mapa de Calor de Retornos Mensuales (R)")
        plt.show()

    def print_report(self):
        """Imprime un reporte formateado en consola."""
        stats = self.get_summary_stats()
        print("\n" + "="*40)
        print("       INFORME DE RENDIMIENTO QUANT")
        print("="*40)
        for k, v in stats.items():
            print(f"{k:<20}: {v:>10.2f}")
        print("="*40)
    def print_annual_summary(self):
        """Genera una tabla comparativa de rendimiento año tras año."""
        # Usamos self.df que es el nombre correcto del atributo en esta clase
        if self.df is None or self.df.empty:
            print("No hay datos suficientes para el informe anual.")
            return

        df_t = self.df.copy()
        
        # Intentamos usar 'entry_time', si no existe usamos 'date'
        date_col = 'entry_time' if 'entry_time' in df_t.columns else 'date'
        df_t[date_col] = pd.to_datetime(df_t[date_col])
        df_t['year'] = df_t[date_col].dt.year

        # Agrupación por año
        annual = df_t.groupby('year').apply(lambda x: pd.Series({
            'Trades': len(x),
            'WinRate': (x['pnl_r'] > 0).mean() * 100,
            'PnL_R': x['pnl_r'].sum(),
            'PnL_USD': x['pnl_usd'].sum(),
            'PF': abs(x[x['pnl_r'] > 0]['pnl_r'].sum() / x[x['pnl_r'] < 0]['pnl_r'].sum()) if any(x['pnl_r'] < 0) else np.inf,
            'MaxDD_R': self._calculate_max_dd(x['pnl_r'])
        }), include_groups=False)

        print("\n" + "="*75)
        print(f"{'AÑO':<6} | {'TRADES':<6} | {'WR %':<8} | {'PnL (R)':<10} | {'PnL (USD)':<12} | {'PF':<6} | {'DD (R)':<6}")
        print("-" * 75)
        
        for year, row in annual.iterrows():
            print(f"{int(year):<6} | {int(row['Trades']):<6} | {row['WinRate']:>6.1f}% | {row['PnL_R']:>8.2f}R | ${row['PnL_USD']:>10.2f} | {row['PF']:>5.2f} | {row['MaxDD_R']:>6.2f}")
        print("="*75)
    def _calculate_max_dd(self, pnl_series):
        """Cálculo auxiliar de Max Drawdown en R."""
        cum_pnl = pnl_series.cumsum()
        peak = cum_pnl.expanding().max()
        dd = cum_pnl - peak
        return dd.min()
