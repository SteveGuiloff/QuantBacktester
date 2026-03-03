import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# CONTRATO DE INTERFAZ - PERFORMANCE BREAKER (CIRCUIT BREAKER)
# =============================================================================
# Versión: 2.8 (Filtros con Histéresis y Protección contra Lateralización)
#
# Características:
# - Histéresis aplicada a PF y Sharpe para evitar "flickering".
# - Equity EMA con Buffer/Offset para filtrar ruido en lateralización.
# - Parámetros N de M corregidos (m = ventana dinámica).
# - Mantiene inmutabilidad mediante copias locales y prefijo PB_.
# =============================================================================

class PerformanceBreaker:
    """
    Clase para el análisis, filtrado y visualización de performance de estrategias.
    Implementa filtros avanzados para proteger el capital en regímenes adversos.
    """
    def __init__(self, start_capital=0):
        self.start_capital = start_capital
        self.current_strategy = None
        self.current_params = {}

    def calculate_indicators(self, df, window=20):
        """
        Prepara los indicadores base sobre una copia del DataFrame.
        """
        pb_df = df.copy()
        
        # 1. Equity y Equity EMA (USD)
        pb_df['PB_equity'] = self.start_capital + pb_df['pnl_usd'].cumsum()
        pb_df['PB_ema_equity'] = pb_df['PB_equity'].ewm(span=window, adjust=False).mean()
        
        # 2. Rolling Profit Factor (R)
        def _get_pf(x):
            pos = x[x > 0].sum()
            neg = abs(x[x < 0].sum())
            return pos / neg if neg > 0 else 5.0
            
        pb_df['PB_rolling_pf'] = pb_df['pnl_r'].rolling(window=window).apply(_get_pf)
        pb_df['PB_ema_pf'] = pb_df['PB_rolling_pf'].ewm(span=window//2, adjust=False).mean()
        
        # 3. Rolling Sharpe Ratio (R)
        def _get_sharpe(x):
            if len(x) < 2 or x.std() == 0: return 0
            return (x.mean() / x.std()) * np.sqrt(252)
            
        pb_df['PB_rolling_sharpe'] = pb_df['pnl_r'].rolling(window=window).apply(_get_sharpe)
        pb_df['PB_ema_sharpe'] = pb_df['PB_rolling_sharpe'].ewm(span=window//2, adjust=False).mean()
        
        return pb_df

    def apply_strategy(self, df_with_indicators, strategy='pf_hist', params={}):
        """
        Aplica lógica de decisión con histéresis y buffers para evitar inestabilidad.
        """
        self.current_strategy = strategy
        self.current_params = params
        res_df = df_with_indicators.copy()
        res_df['PB_status'] = True
        
        # Lógica de estados con histéresis
        status_list = []
        curr_status = True
        
        if strategy == 'pf_hist':
            # Unificado: pf_on == pf_off elimina la histéresis
            on, off = params.get('pf_on', 1.2), params.get('pf_off', 0.8)
            for val in res_df['PB_ema_pf']:
                if pd.isna(val): status_list.append(True); continue
                if curr_status and val < off: curr_status = False
                elif not curr_status and val > on: curr_status = True
                status_list.append(curr_status)
                
        elif strategy == 'sharpe_hist':
            # Histéresis para Sharpe Ratio (evita ruidos de volatilidad)
            on, off = params.get('s_on', 0.5), params.get('s_off', 0.1)
            for val in res_df['PB_ema_sharpe']:
                if pd.isna(val): status_list.append(True); continue
                if curr_status and val < off: curr_status = False
                elif not curr_status and val > on: curr_status = True
                status_list.append(curr_status)

        elif strategy == 'ema_equity_buffer':
            # Evita inestabilidad en lateral: requiere un offset (buffer)
            offset = params.get('offset_usd', 100)
            for i, row in res_df.iterrows():
                val, ema = row['PB_equity'], row['PB_ema_equity']
                if curr_status and val < (ema - offset): curr_status = False
                elif not curr_status and val > (ema + offset): curr_status = True
                status_list.append(curr_status)
                
        elif strategy == 'n_de_m':
            # n: ganancias requeridas, m: ventana de trades
    pb.plot_breaker(df_final)
