# =============================================================================
# CONTRATO DE INTERFAZ DE DATOS - ESTRATEGIA | MOTOR | REPORTER
# =============================================================================
# Versión: 1.3 (Estándar de Dirección: 'side' & Interfaz de Reportero Ampliada)
#
# 1. ENTRADA AL MOTOR (Desde la Estrategia)
# El DataFrame 'df_signals' debe estar normalizado con estos nombres exactos:
# -----------------------------------------------------------------------------
# Columnas Requeridas:
# - Timestamp_NY (datetime): Índice temporal (Zona horaria New York).
# - Open         (float):    Precio apertura (Ajustado o Crudo).
# - High         (float):    Precio máximo (Ajustado o Crudo).
# - Low          (float):    Precio mínimo (Ajustado o Crudo).
# - Close        (float):    Precio cierre (Ajustado o Crudo).
# - sig_long     (boolean):  Disparador de compra (True/False).
# - sig_short    (boolean):  Disparador de venta (True/False).
# - sl_level     (float):    Nivel de precio para el Stop Loss.
# - tp_level     (float):    Nivel de precio para el Take Profit.
#
# 2. SALIDA DEL MOTOR / ENTRADA AL REPORTER (Resultado del Backtest)
# El DataFrame 'trades_df' generado por engine.run() debe contener:
# -----------------------------------------------------------------------------
# Columnas Generadas:
# - id           (int):      Identificador único del trade.
# - date         (date):     Fecha de cierre (YYYY-MM-DD).
# - entry_time   (datetime): Timestamp exacto de entrada.
# - exit_time    (datetime): Timestamp exacto de salida.
# - side         (string):   Dirección normalizada: "Long" o "Short".
# - qty          (int):      Cantidad de contratos operados.
# - entry        (float):    Precio de ejecución de entrada.
# - exit         (float):    Precio de ejecución de salida.
# - pnl_usd      (float):    Resultado neto en dólares (c/ comisiones).
# - pnl_r        (float):    Resultado en unidades de Riesgo (R).
# - reason       (string):   Motivo: "TP", "SL" o "ForceClose".
#
# 3. INTERFAZ DEL REPORTERO (Métodos Requeridos)
# El objeto QuantReporter debe exponer los siguientes métodos:
# -----------------------------------------------------------------------------
# - get_summary_stats():     Imprime métricas clave (WR%, PF, PnL Total).
# - print_report():          Muestra el informe general detallado.
# - print_annual_summary():  Desglose de rendimiento por año calendario.
# - plot_equity_curve():     Genera gráfico de curva de capital y DD.
# - generate_full_report():  Ejecuta el análisis integral (incluye todos).
# =============================================================================
import pandas as pd
import numpy as np
import pandas_ta as ta
from quant_backtester_core import QuantEngineV2, StrategyConfig
from quant_reporting import QuantReporter

# =============================================================================
# 0. GESTIÓN DE DATOS (DATA LOADING)
# =============================================================================
# Parquets disponibles en quant@quant-lab:~/data/processed$
# - economic_calendar.parquet
# - gc_1m_continuous.parquet  (Oro)
# - nq_1m_continuous.parquet  (Nasdaq)

DATA_PATH = "/home/quant/data/processed/gc_1m_continuous.parquet"

# =============================================================================
# 1. PARÁMETROS DE LA ESTRATEGIA (Archivo de Estado)
# =============================================================================
# Aquí definiremos los inputs de la GEMA una vez proporcionados.
STRATEGY_PARAMS = {
    "param_1": None,
    "param_2": None
}

# =============================================================================
# 2. LÓGICA DE SEÑALES (Estrategia Desacoplada)
# =============================================================================
def apply_strategy_logic(df, params):
    """
    Transforma datos crudos en señales operativas.
    Responsabilidad única: Generar columnas booleanas y niveles de precios.
    """
    df = df.copy()
    
    # --- A. CÁLCULO DE INDICADORES ---
    # Ejemplo: df['indicador'] = ...
    
    # --- B. DEFINICIÓN DE SEÑALES (BOOLEAN) ---
    df['sig_long']  = False
    df['sig_short'] = False
    
    # --- C. DEFINICIÓN DE NIVELES (SL / TP) ---
    # El motor requiere niveles exactos de precio por vela para el backtest.
    df['sl_level'] = 0.0
    df['tp_level'] = 0.0
    
    return df

# =============================================================================
# 3. ORQUESTACIÓN DEL BACKTEST
# =============================================================================
def run_simulation(data_path, config):
    """Carga datos, aplica estrategia y corre el motor."""
    
    print(f"📥 Cargando datos desde: {data_path}")
    df = pd.read_parquet(data_path)
    
    # Estandarización de Timestamps
    if 'Timestamp_NY' in df.columns:
        df['Timestamp_NY'] = pd.to_datetime(df['Timestamp_NY'])
        df = df.sort_values('Timestamp_NY').reset_index(drop=True)
    
    # Aplicar Lógica
    df_with_signals = apply_strategy_logic(df, STRATEGY_PARAMS)
    
    # Inicializar Motor
    engine = QuantEngineV2(df_with_signals, config)
    
    # Ejecución
    trades = engine.run(
        start_date=df_with_signals['Timestamp_NY'].min(),
        end_date=df_with_signals['Timestamp_NY'].max(),
        verbose=False
    )
    
    return trades

# =============================================================================
# BLOQUE DE EJECUCIÓN (Standalone)
# =============================================================================
if __name__ == "__main__":
    # Configuración de Riesgo e Inmuebles
    # Nota: Ajustar asset_name según el parquet cargado ("GC" o "NQ")
    config = StrategyConfig(
        asset_name="GC",         
        risk_usd=1000,           
        reward_ratio=1.0,        
        trading_windows=[("08:00", "16:00")], 
        resolution="Pessimistic" 
    )
    
    # trades = run_simulation(DATA_PATH, config)
    # if not trades.empty:
    #     reporter = QuantReporter(trades)
    #     reporter.print_full_report()
    
    print("✅ Bloque 0 configurado. Esperando lógica técnica de la estrategia para proceder.")
