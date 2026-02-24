import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class QuantReporter:
    """
    Módulo de Analítica Estándar para el sistema de trading.
    Cumple estrictamente con el CONTRATO DE INTERFAZ DE DATOS V1.3.
    
    Responsabilidad: Transformar la salida del QuantEngineV2 en métricas 
    validadas y visualizaciones. 
    Ajuste V1.4: Agrupación basada en 'entry_time' para evitar fugas de sesión.
    """
    def __init__(self, trades_df):
        if trades_df is None or trades_df.empty:
            print("⚠️ Datos de trades vacíos o nulos. No se puede generar reporte.")
            self.df = pd.DataFrame()
            return

        # 1. Validación del Contrato de Datos V1.3
        required_cols = ['id', 'date', 'entry_time', 'exit_time', 'side', 'qty', 'entry', 'exit', 'pnl_usd', 'pnl_r', 'reason']
        missing = [c for c in required_cols if c not in trades_df.columns]
        
        if missing:
            critical = ['side', 'pnl_r', 'pnl_usd']
            if any(c in missing for c in critical):
                raise ValueError(f"Faltan columnas CRÍTICAS del contrato V1.3: {missing}")
            print(f"⚠️ Advertencia: Faltan columnas secundarias del contrato: {missing}")

        self.df = trades_df.copy()
        self._standardize_types()
        self._prepare_metrics()

    def _standardize_types(self):
        """Asegura que los tipos de datos sean correctos para el análisis."""
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date']).dt.date
        if 'entry_time' in self.df.columns:
            self.df['entry_time'] = pd.to_datetime(self.df['entry_time'])
        if 'exit_time' in self.df.columns:
            self.df['exit_time'] = pd.to_datetime(self.df['exit_time'])

    def _prepare_metrics(self):
        """Cálculos base de rendimiento y curvas de equidad."""
        # IMPORTANTE: Ordenamos por 'entry_time' para que la curva de equity 
        # siga el orden en que se asumió el riesgo.
        self.df = self.df.sort_values('entry_time').reset_index(drop=True)
        
        self.df['equity_r'] = self.df['pnl_r'].cumsum()
        self.df['peak_r'] = self.df['equity_r'].cummax()
        self.df['drawdown_r'] = self.df['equity_r'] - self.df['peak_r']

    def get_summary_stats(self):
        """Imprime métricas clave de rendimiento."""
        if self.df.empty: return
        
        total_trades = len(self.df)
        wins = self.df[self.df['pnl_r'] > 0]
        losses = self.df[self.df['pnl_r'] <= 0]
        
        total_pnl_r = self.df['pnl_r'].sum()
        total_pnl_usd = self.df['pnl_usd'].sum()
        
        wr = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
        
        gross_profit = wins['pnl_r'].sum()
        gross_loss = abs(losses['pnl_r'].sum())
        pf = (gross_profit / gross_loss) if gross_loss > 0 else np.inf
        
        expectancy = total_pnl_r / total_trades if total_trades > 0 else 0
        max_dd = self.df['drawdown_r'].min()
        recovery_factor = (total_pnl_r / abs(max_dd)) if max_dd != 0 else np.inf
        sharpe_r = (self.df['pnl_r'].mean() / self.df['pnl_r'].std()) if self.df['pnl_r'].std() > 0 else 0
        avg_trade_usd = total_pnl_usd / total_trades if total_trades > 0 else 0

        print("\n" + "="*40)
        print(f"{'INFORME DE RENDIMIENTO QUANT':^40}")
        print("="*40)
        print(f"{'Total Trades':<20} : {total_trades:>10.2f}")
        print(f"{'Win Rate (%)':<20} : {wr:>10.2f}")
        print(f"{'Expectancy (R)':<20} : {expectancy:>10.2f}")
        print(f"{'Profit Factor (R)':<20} : {pf:>10.2f}")
        print(f"{'Max Drawdown (R)':<20} : {max_dd:>10.2f}")
        print(f"{'Recovery Factor':<20} : {recovery_factor:>10.2f}")
        print(f"{'Sharpe Ratio (R)':<20} : {sharpe_r:>10.2f}")
        print(f"{'Total PnL (R)':<20} : {total_pnl_r:>10.2f}")
        print(f"{'Total PnL (USD)':<20} : {total_pnl_usd:>10.2f}")
        print(f"{'Avg Trade (USD)':<20} : {avg_trade_usd:>10.2f}")
        print("="*40)

    def print_report(self):
        """Muestra el informe por Día y Lado usando entry_time."""
        if self.df.empty: return
        
        print("\n📈 DESGLOSE OPERATIVO POR DÍA Y LADO (Basado en Entry Time):")
        
        temp_df = self.df.copy()
        # CAMBIO CLAVE: Usamos entry_time para determinar el nombre del día
        temp_df['day_name'] = temp_df['entry_time'].dt.day_name()
        
        results = []
        for (day, side), group in temp_df.groupby(['day_name', 'side']):
            w = group[group['pnl_r'] > 0]
            l = group[group['pnl_r'] <= 0]
            gp = w['pnl_r'].sum()
            gl = abs(l['pnl_r'].sum())
            
            local_equity = group['pnl_r'].cumsum()
            local_max_dd = (local_equity - local_equity.cummax()).min()
            
            results.append({
                'Día': day,
                'Lado': side,
                'Trades': int(len(group)),
                'WR%': (len(w)/len(group))*100,
                'PF': (gp/gl) if gl > 0 else np.inf,
                'PnL(R)': group['pnl_r'].sum(),
                'MaxDD(R)': local_max_dd
            })
        
        breakdown = pd.DataFrame(results)
        dias = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        breakdown['Día'] = pd.Categorical(breakdown['Día'], categories=dias, ordered=True)
        breakdown = breakdown.sort_values(['Día', 'Lado'])
        
        print("-" * 95)
        print(breakdown.to_string(index=False, formatters={
            'WR%': '{:,.1f}%'.format,
            'PF': '{:,.2f}'.format,
            'PnL(R)': '{:,.2f}R'.format,
            'MaxDD(R)': '{:,.2f}R'.format
        }))
        print("-" * 95)

    def print_annual_summary(self):
        """Desglose anual detallado basado en entry_time."""
        if self.df.empty: return
        
        print("\n📅 RESUMEN ANUAL DE RENDIMIENTO (Basado en Entry Time):")
        temp_df = self.df.copy()
        # CAMBIO CLAVE: El año se define por la entrada
        temp_df['year'] = temp_df['entry_time'].dt.year
        
        annual_results = []
        for year, group in temp_df.groupby('year'):
            wins = group[group['pnl_r'] > 0]
            losses = group[group['pnl_r'] <= 0]
            
            gp = wins['pnl_r'].sum()
            gl = abs(losses['pnl_r'].sum())
            pnl_r = group['pnl_r'].sum()
            
            eq = group['pnl_r'].cumsum()
            dd = eq - eq.cummax()
            max_dd_r = dd.min()
            
            rec_factor = (pnl_r / abs(max_dd_r)) if max_dd_r != 0 else np.inf
            
            annual_results.append({
                'Año': year,
                'Trades': int(len(group)),
                'WR%': (len(wins)/len(group)) * 100,
                'PF': (gp/gl) if gl > 0 else np.inf,
                'PnL(R)': pnl_r,
                'MaxDD(R)': max_dd_r,
                'Rec. Factor': rec_factor
            })
            
        summary = pd.DataFrame(annual_results)
        print("-" * 95)
        print(summary.to_string(index=False, formatters={
            'WR%': '{:,.1f}%'.format,
            'PF': '{:,.2f}'.format,
            'PnL(R)': '{:,.2f}R'.format,
            'MaxDD(R)': '{:,.2f}R'.format,
            'Rec. Factor': '{:,.2f}'.format
        }))
        print("-" * 95)

    def plot_equity_curve(self):
        """Gráfico de curva de capital."""
        if self.df.empty: return
        
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        ax1.plot(self.df.index, self.df['equity_r'], color='#1a5fb4', lw=2, label="Equity Acumulada (R)")
        ax1.fill_between(self.df.index, 0, self.df['equity_r'], color='#1a5fb4', alpha=0.1)
        ax1.set_title("Curva de Rendimiento (Basada en Secuencia de Entrada)", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Múltiplos de R", fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='upper left')
        
        ax2.fill_between(self.df.index, 0, self.df['drawdown_r'], color='#e01b24', alpha=0.3, label="Drawdown (R)")
        ax2.plot(self.df.index, self.df['drawdown_r'], color='#e01b24', lw=1)
        ax2.set_ylabel("Drawdown R", fontsize=10)
        ax2.set_xlabel("Número de Trade (Secuencia Cronológica)", fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()

    def generate_full_report(self):
        """Ejecuta el análisis integral."""
        self.get_summary_stats()
        self.print_report()
        self.print_annual_summary()
        self.plot_equity_curve()
