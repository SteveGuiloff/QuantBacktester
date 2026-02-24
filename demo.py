import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import time
from quant_backtester_core import QuantEngineV2, StrategyConfig
from quant_reporting import QuantReporter

# --- CONFIGURACIÓN DE PARÁMETROS (MODIFICABLES) ---
ADX_THRESHOLD = 23      # Mínima fuerza de tendencia
ATR_MIN       = 1.5     # Mínima volatilidad (Posible culpable de los años vacíos)
ATR_MAX       = 15.0    # Máxima volatilidad (Protección ante noticias)
QC_THRESHOLD  = 0.8     # Calidad de cierre (Cierre cerca del extremo)

# --- 1. CARGA Y LIMPIEZA ---
# (Asegúrate de que la ruta sea la correcta en tu nueva instancia)
FILE_PATH_PARQUET = '/home/quant/data/processed/gc_1m_raw_continuous.parquet'
df = pd.read_parquet(FILE_PATH_PARQUET)

df['Timestamp_NY'] = pd.to_datetime(df['Timestamp_NY'])
df = df.sort_values('Timestamp_NY').reset_index(drop=True)

# --- 2. INDICADORES (Lógica Pine Script) ---
df['ema11'] = ta.ema(df['Close_adj'], length=11)
df['ema75'] = ta.ema(df['Close_adj'], length=75)
df['atr'] = ta.atr(df['High_adj'], df['Low_adj'], df['Close_adj'], length=14)
df['atr_percent'] = (df['atr'] / df['Close_adj']) * 100
ATR_MIN_PCT = 0.09  # El valor que parece funcionar en 2025



# Filtros de Tendencia (OR en Pine Script)
ema_filt_short = (df['Close_adj'] < df['ema11']) | (df['Close_adj'] < df['ema75'])
ema_filt_long  = (df['Close_adj'] > df['ema11']) | (df['Close_adj'] > df['ema75'])

# DMI / ADX
adx_df = ta.adx(df['High_adj'], df['Low_adj'], df['Close_adj'], length=14)
df['adx'] = adx_df['ADX_14']
momentum_ok = (df['adx'] >= 23) & (df['atr_percent'] >= ATR_MIN_PCT)

# Calidad de Cierre (QC)
df['range'] = df['High_adj'] - df['Low_adj']
df['ubicacion_cierre'] = np.where(df['range'] == 0, 0.5, (df['Close_adj'] - df['Low_adj']) / df['range'])

# --- 3. LÓGICA DE MICRO-PAUSA Y ESTRUCTURA ---
df['v1_high'] = df['High_adj'].shift(1)
df['v1_low'] = df['Low_adj'].shift(1)
df['v1_close'] = df['Close_adj'].shift(1)
df['v1_open'] = df['Open_adj'].shift(1)

df['v2_high'] = df['High_adj'].shift(2)
df['v2_low'] = df['Low_adj'].shift(2)

# Condiciones de pausa
long_pause = (df['v1_high'] < df['v2_high']) & (df['v1_low'] < df['v2_low'])
long_v2_ok = df['v1_close'] > df['v1_open']
long_trigger = df['Close_adj'] > df['v1_high']

short_pause = (df['v1_low'] > df['v2_low']) & (df['v1_high'] > df['v2_high'])
short_v2_ok = df['v1_close'] < df['v1_open']
short_trigger = df['Close_adj'] < df['v1_low']

# --- 4. SEÑALES FINALES (Usando constantes) ---
# Filtro de Momentum y Volatilidad
#momentum_ok = (df['adx'] >= ADX_THRESHOLD) & \
#             (df['atr'] >= ATR_MIN) 
momentum_ok = (df['adx'] >= 23) & (df['atr_percent'] >= ATR_MIN_PCT)

# Señales de Compra y Venta
df['sig_long'] = long_pause & \
                 long_v2_ok & \
                 long_trigger & \
                 momentum_ok & \
                 ema_filt_long & \
                 (df['ubicacion_cierre'] >= QC_THRESHOLD)

df['sig_short'] = short_pause & \
                  short_v2_ok & \
                  short_trigger & \
                  momentum_ok & \
                  ema_filt_short & \
                  (df['ubicacion_cierre'] <= (1 - QC_THRESHOLD))

# --- 5. CÁLCULO DE NIVELES DESACOPLADOS (La Clave de la Sincronía) ---
# SL Estructural
df['sl_level'] = np.where(df['sig_long'], df['v2_low'],
                 np.where(df['sig_short'], df['v2_high'], np.nan))

# Riesgo y TP (Calculados sobre el cierre de la señal como en Pine)
df['risk_pts_signal'] = abs(df['Close_adj'] - df['sl_level'])

df['tp_level'] = np.where(df['sig_long'], df['Close_adj'] + df['risk_pts_signal'],
                 np.where(df['sig_short'], df['Close_adj'] - df['risk_pts_signal'], np.nan))

# Redondeo oficial al tick del Oro (0.1)
df['sl_level'] = (df['sl_level'] / 0.1).round() * 0.1
df['tp_level'] = (df['tp_level'] / 0.1).round() * 0.1

# --- 6. FUNCIONES DE EJECUCIÓN ---
def analyze_specific_day(engine, target_date_str):
    print(f"\n--- 🔍 AUDITORÍA DETALLADA: {target_date_str} ---")
    start = pd.to_datetime(target_date_str + " 00:00:00")
    end = pd.to_datetime(target_date_str + " 23:59:59")
    
    # Diagnóstico rápido antes de correr
    check = df[(df['Timestamp_NY'] >= f"{target_date_str} 09:41:00") & (df['Timestamp_NY'] <= f"{target_date_str} 09:43:00")]
    if not check.empty:
        print("Validando niveles de señal técnica:")
        print(check[['Timestamp_NY', 'Close_adj', 'sl_level', 'tp_level', 'sig_short']])
        
    return engine.run(start_date=start, end_date=end, verbose=True)

# --- 7. LANZAMIENTO ---
config = StrategyConfig(
    asset_name="GC",
    risk_usd=2000,
    reward_ratio=1.0,
    be_trigger_r=10.0, 
    be_offset_ticks=0,
    trading_windows=[("08:00", "12:00")],
    max_trades_per_day=-1
)

engine = QuantEngineV2(df, config)
#res = analyze_specific_day(engine, "2025-12-08")
trades = engine.run(start_date='2010-01-01', end_date='2025-12-31')


if trades is not None and not trades.empty:
    reporter = QuantReporter(trades)
    reporter.print_report()         # El informe general que ya tienes
    reporter.print_annual_summary() # La nueva tabla detallada año a año
    
    detalle_trades = trades[['id', 'entry_time', 'exit_time', 'type', 'entry', 'exit', 'qty','pnl_usd','pnl_r', 'reason']]
    print(f"\nLista detallada de trades (Total: {len(trades)}):")
    # Formateamos para que las horas sean legibles
    print(detalle_trades.to_string(index=False))

