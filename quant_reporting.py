import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class QuantReporter:
    """
    Módulo de Analítica Estándar para el sistema de trading.
    Cumple con el CONTRATO DE INTERFAZ DE DATOS V1.3.
    Requiere: ['date', 'side', 'pnl_r', 'pnl_usd', 'exit_time']
    """
    def __init__(self, trades_df):
        if trades_df is None or trades_df.empty:
            print("⚠️ Datos de trades vacíos o nulos.")
            self.df = pd.DataFrame()
            return

        # 1. Validación del Contrato de Datos (Estandarización)
        required_cols = ['date', 'side', 'pnl_r', 'pnl_usd']
        missing = [c for c in required_cols if c not in trades_df.columns]
        if missing:
            raise ValueError(f"Faltan columnas obligatorias en el contrato: {missing}")

        self.df = trades_df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # 2. Preparación de métricas acumuladas
        self._prepare_metrics()

    def _prepare_metrics(self):
        """Cálculos base de rendimiento y curvas de equidad."""
        # Ordenar por tiempo de salida para coherencia en la curva
        sort_col = 'exit_time' if 'exit_time' in self.df.columns else 'date'
        self.df = self.df.sort_values(sort_col)
        
        self.df['equity_r'] = self.df['pnl_r'].cumsum()
        self.df['peak_r'] = self.df['equity_r'].cummax()
        self.df['drawdown_r'] = self.df['equity_r'] - self.df['peak_r']

    # --- IMPLEMENTACIÓN DEL CONTRATO V1.3 ---

    def get_summary_stats(self):
        """Imprime métricas clave (WR%, PF, PnL Total)."""
        if self.df.empty: return
        
        wins = self.df[self.df['pnl_r'] > 0]
        total = len(self.df)
        pnl_r = self.df['pnl_r'].sum()
        pnl_usd = self.df['pnl_usd'].sum()
        
        loss_df = self.df[self.df['pnl_r'] < 0]
        total_loss = abs(loss_df['pnl_r'].sum())
        pf = (wins['pnl_r'].sum() / total_loss) if total_loss > 0 else np.inf

        print("\n" + "="*50)
        print("📊 RESUMEN DE MÉTRICAS CLAVE")
        print("="*50)
        print(f"{'Total Trades':<25}: {total}")
        print(f"{'Win Rate':<25}: {(len(wins)/total)*100:>10.2f}%")
        print(f"{'Profit Factor':<25}: {pf:>10.2f}")
        print(f"{'Total PnL (R)':<25}: {pnl_r:>10.2f}R")
        print(f"{'Total PnL (USD)':<25}: ${pnl_usd:>9.2f}")
        print(f"{'Max Drawdown (R)':<25}: {self.df['drawdown_r'].min():>10.2f}R")
        print("="*50)

    def print_report(self):
        """Muestra el informe general detallado (Desglose por Side/Día)."""
        print("\n📈 INFORME DETALLADO POR DIRECCIÓN Y DÍA:")
        
        df_stats = self.df.copy()
        df_stats['day_name'] = df_stats['date'].dt.day_name()
        
        def _calc_stats(g):
            wins = g[g['pnl_r'] > 0]
            return pd.Series({
                'Trades': int(len(g)),
                'WR%': (len(wins) / len(g)) * 100,
                'PnL(R)': g['pnl_r'].sum(),
                'PnL(USD)': g['pnl_usd'].sum()
            })

        breakdown = df_stats.groupby(['day_name', 'side'], group_keys=False).apply(_calc_stats, include_groups=False).reset_index()
        
        # Ordenar días de la semana
        order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        breakdown['day_name'] = pd.Categorical(breakdown['day_name'], categories=order, ordered=True)
        breakdown = breakdown.sort_values(['day_name', 'side'])
        
        print("-" * 75)
        print(breakdown.to_string(index=False))
        print("-" * 75)

    def print_annual_summary(self):
        """Desglose de rendimiento por año calendario."""
        print("\n📅 RESUMEN ANUAL DE RENDIMIENTO:")
        df_ann = self.df.copy()
        df_ann['year'] = df_ann['date'].dt.year
        
        def _annual(x):
            return pd.Series({
                'Trades': int(len(x)),
                'PnL (R)': x['pnl_r'].sum(),
                'PnL (USD)': x['pnl_usd'].sum(),
                'MaxDD (R)': (x['pnl_r'].cumsum() - x['pnl_r'].cumsum().cummax()).min()
            })

        summary = df_ann.groupby('year').apply(_annual, include_groups=False).reset_index()
        print(summary.to_string(index=False))

    def plot_equity_curve(self):
        """Genera gráfico de curva de capital y DD."""
        if self.df.empty: return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # Curva de Equity
        ax1.plot(self.df['equity_r'].values, color='#1a5fb4', lw=2, label="Equity (R)")
        ax1.set_title("Curva de Equity Acumulada (Unidades R)", fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Underwater Chart
        ax2.fill_between(range(len(self.df)), 0, self.df['drawdown_r'], color='#e01b24', alpha=0.3)
        ax2.plot(self.df['drawdown_r'].values, color='#e01b24', lw=1)
        ax2.set_title("Gráfico de Drawdown (Underwater R)", fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def generate_full_report(self):
        """Ejecuta el análisis integral (incluye todos los reportes)."""
        self.get_summary_stats()
        self.print_report()
        self.print_annual_summary()
        self.plot_equity_curve()
