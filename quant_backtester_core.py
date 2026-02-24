import pandas as pd
import numpy as np
from datetime import time, datetime

# =============================================================================
# GOBERNANZA DE ACTIVOS Y ESPECIFICACIONES
# =============================================================================
ASSET_SPECS = {
    "NQ": {"tick_size": 0.25, "tick_value": 5.0, "points_full_value": 20.0, "commission": 10.40, "avg_slippage_ticks": 2},
    "ES": {"tick_size": 0.25, "tick_value": 12.5, "points_full_value": 50.0, "commission": 10.40, "avg_slippage_ticks": 1},
    "YM": {"tick_size": 1.0, "tick_value": 5.0, "points_full_value": 5.0, "commission": 10.40, "avg_slippage_ticks": 1},
    "GC": {"tick_size": 0.1, "tick_value": 10.0, "points_full_value": 100.0, "commission": 13.40, "avg_slippage_ticks": 1},
    "CL": {"tick_size": 0.01, "tick_value": 10.0, "points_full_value": 1000.0, "commission": 13.40, "avg_slippage_ticks": 2}
}

class StrategyConfig:
    """Configuración inmutable de la operativa y gestión de riesgo."""
    def __init__(self, asset_name="NQ", risk_usd=2000, reward_ratio=2.0, be_trigger_r=1.0, 
                 be_offset_ticks=0, max_trades_per_day=-1, trading_windows=[("09:30", "15:55")],
                 force_close_time="15:56", direction="Both", execution_mode="Optimista"):
        spec = ASSET_SPECS.get(asset_name, ASSET_SPECS["NQ"])
        self.asset_name = asset_name
        self.risk_usd = risk_usd
        self.reward_ratio = reward_ratio
        self.be_trigger_r = be_trigger_r
        self.tick_size = spec["tick_size"]
        self.tick_value = spec["tick_value"]
        self.points_full_value = spec["points_full_value"]
        self.comm_per_side = spec["commission"] / 2
        self.slippage_points = spec["avg_slippage_ticks"] * self.tick_size
        self.be_offset = be_offset_ticks * self.tick_size
        self.max_trades_per_day = max_trades_per_day
        self.trading_windows = trading_windows
        self.force_close_time = force_close_time
        self.direction = direction
        self.execution_mode = execution_mode

class AuditLogger:
    """Utilidad para impresión de logs detallados y depuración."""
    @staticmethod
    def log_trade_start(t):
        side_str = "LONG" if t['side'] == 1 else "SHORT"
        print(f"\n🚀 [TRADE #{t['id']}] {side_str} ENTERED @ {t['entry']:.2f}")
        print(f"    Initial SL: {t['sl']:.2f} | Initial TP: {t['tp']:.2f} | Qty: {t['qty']}")

class QuantEngineV2:
    """
    Motor de ejecución de Backtesting. 
    ESTRICTO cumplimiento con Contrato de Interfaz V1.3.
    """
    def __init__(self, df, config):
        self.df = df
        self.config = config
        spec = ASSET_SPECS.get(config.asset_name, ASSET_SPECS["GC"])
        self.tick_size = spec["tick_size"]
        self.tick_value = spec["tick_value"]
        self.trades = []
        self.current_day_trades = 0
        self.last_date = None

    def _is_in_trading_window(self, current_dt):
        if not self.config.trading_windows: return True
        t_obj = current_dt.time()
        curr_time = time(t_obj.hour, t_obj.minute, 0)
        for start_str, end_str in self.config.trading_windows:
            if time.fromisoformat(start_str) <= curr_time <= time.fromisoformat(end_str):
                return True
        return False

    def _round_to_tick(self, price):
        if pd.isna(price): return price
        return np.round(price / self.tick_size) * self.tick_size

    def _resolve_intra_candle(self, row, t):
        """Resolución de niveles SL/TP interna con validación >=."""
        high, low = row['High'], row['Low']
        side_val, sl, tp = t['side'], t['sl'], t['tp']
        mode = self.config.execution_mode 

        if mode == "Optimista":
            if side_val == 1: # Long
                if high >= tp: return "TP"
                if low <= sl: return "SL"
            else: # Short
                if low <= tp: return "TP"
                if high >= sl: return "SL"
        else: # Pesimista o estándar
            if side_val == 1: # Long
                if low <= sl: return "SL"
                if high >= tp: return "TP"
            else: # Short
                if high >= sl: return "SL"
                if low <= tp: return "TP"
        return None

    def run(self, start_date=None, end_date=None, verbose=False):
        self.trades = []
        self.current_day_trades = 0
        self.last_date = None
        
        df_proc = self.df.copy()
        if start_date: df_proc = df_proc[df_proc['Timestamp_NY'] >= start_date]
        if end_date: df_proc = df_proc[df_proc['Timestamp_NY'] <= end_date]
        df_proc = df_proc.reset_index(drop=True)
        
        total_rows = len(df_proc)
        in_pos = False
        t = {}
        f_close_t = time.fromisoformat(self.config.force_close_time)

        for i, row in df_proc.iterrows():
            curr_dt = row['Timestamp_NY']
            curr_date = curr_dt.date()
            curr_time = curr_dt.time()
            
            # 2. CIERRE POR CAMBIO DE DÍA
            if self.last_date is None or curr_date != self.last_date:
                if in_pos:
                    self._close_trade(t, row, curr_dt, "Session_Change")
                    in_pos = False
                self.current_day_trades = 0
                self.last_date = curr_date

            if in_pos:
                res = self._resolve_intra_candle(row, t)
                
                # Gestión de Break-Even
                if not t['be_active']:
                    dist = (row['High'] - t['entry']) if t['side'] == 1 else (t['entry'] - row['Low'])
                    if dist >= (t['risk_pts'] * self.config.be_trigger_r):
                        t['be_active'] = True
                        t['sl'] = t['entry'] + (self.config.be_offset * t['side'])

                # 1. CIERRE POR FIN DE DÍA (VENTANA OPERATIVA)
                is_force_close = (curr_time >= f_close_t)
                
                # 3. CIERRE POR TÉRMINO DE SIMULACIÓN (ÚLTIMA VELA)
                is_last_row = (i == total_rows - 1)
                
                if res or is_force_close or is_last_row:
                    reason = res if res else ("ForceClose_EOD" if is_force_close else "ForceClose_EOS")
                    self._close_trade(t, row, curr_dt, reason)
                    in_pos = False
                
                if in_pos: continue

            # Lógica de Apertura
            if not in_pos and self._is_in_trading_window(curr_dt):
                if self.config.max_trades_per_day == -1 or self.current_day_trades < self.config.max_trades_per_day:
                    is_long = row.get('sig_long', False) and self.config.direction in ["Long", "Both"]
                    is_short = row.get('sig_short', False) and self.config.direction in ["Short", "Both"]

                    if is_long or is_short:
                        ptype = 1 if is_long else -1
                        entry_p = self._round_to_tick(row['Close'] + (self.config.slippage_points * ptype))
                        sl_p = self._round_to_tick(row['sl_level'])
                        tp_p = self._round_to_tick(row['tp_level'])
                        risk_pts = abs(entry_p - sl_p)
                        
                        if risk_pts > 0:
                            risk_usd_contract = (risk_pts / self.tick_size) * self.tick_value
                            qty = int(np.floor(self.config.risk_usd / risk_usd_contract))
                            if qty >= 1:
                                in_pos = True
                                self.current_day_trades += 1
                                # CONTRATO V1.3: Usamos 'side' en el objeto interno también
                                t = {
                                    'id': len(self.trades) + 1, 
                                    'side': ptype,
                                    'entry': entry_p, 
                                    'sl': sl_p, 
                                    'tp': tp_p,
                                    'qty': qty, 
                                    'risk_pts': risk_pts, 
                                    'be_active': False, 
                                    'time': curr_dt
                                }
                                if verbose: AuditLogger.log_trade_start(t)

        return pd.DataFrame(self.trades)

    def _close_trade(self, t, row, curr_dt, reason):
        """Ejecuta el cierre de la posición y registra en el historial."""
        exit_raw = t['sl'] if reason == "SL" else (t['tp'] if reason == "TP" else row['Close'])
        slippage = self.config.slippage_points if reason not in ["TP", "ForceClose_EOD", "Session_Change"] else 0
        exit_final = exit_raw - (slippage * t['side'])

        usd_risk_at_stake = (t['risk_pts'] * self.config.points_full_value * t['qty'])
        pnl_usd = ((exit_final - t['entry']) * t['side'] * t['qty'] * self.config.points_full_value) - (t['qty'] * self.config.comm_per_side * 2)
        pnl_r_value = pnl_usd / usd_risk_at_stake if usd_risk_at_stake != 0 else 0

        self.trades.append({
            'id': t['id'],
            'date': self.last_date,
            'entry_time': t['time'],
            'exit_time': curr_dt,
            'side': "Long" if t['side'] == 1 else "Short", # CONTRATO V1.3
            'qty': t['qty'],
            'entry': t['entry'],
            'exit': exit_final,
            'pnl_usd': pnl_usd,
            'pnl_r': pnl_r_value,
            'reason': reason
        })
