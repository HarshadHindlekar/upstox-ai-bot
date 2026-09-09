import time
import json
import threading
from datetime import datetime
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np

import config
from core.model import TradingModel
from core.risk_manager import RiskManager
from core.data_engine import DataEngine
from core.paper_trader import PaperTrader
from core.auth import UpstoxAuth
from core.live_trader import LiveTrader

DEFAULT_20_STOCKS = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
    "SBIN", "BHARTIARTL", "ITC", "LT", "KOTAKBANK",
    "AXISBANK", "TATAMOTORS", "TATASTEEL", "MARUTI", "SUNPHARMA",
    "TITAN", "BAJFINANCE", "ASIANPAINT", "HCLTECH", "WIPRO"
]

STOCK_BASE_PRICES = {
    "RELIANCE": 1305.0, "TCS": 4250.0, "INFY": 1820.0, "HDFCBANK": 1640.0, "ICICIBANK": 1210.0,
    "SBIN": 820.0, "BHARTIARTL": 1580.0, "ITC": 510.0, "LT": 3650.0, "KOTAKBANK": 1810.0,
    "AXISBANK": 1190.0, "TATAMOTORS": 1050.0, "TATASTEEL": 152.0, "MARUTI": 12400.0, "SUNPHARMA": 1840.0,
    "TITAN": 3680.0, "BAJFINANCE": 7150.0, "ASIANPAINT": 3120.0, "HCLTECH": 1780.0, "WIPRO": 540.0
}


class ColabTradingDashboard:
    """
    High-Contrast Dark Theme Multi-Stock Trading Terminal for Google Colab.
    Features:
    - 20+ liquid NSE stocks scanned simultaneously
    - Automated trade execution when AI confidence > 65%
    - Interactive Watchlist management (Add/Remove stocks on the fly)
    - Mode Switcher: Paper Trading (Simulation) vs Live Upstox Execution
    - In-place 100% flicker-free Base64 chart streaming
    - Zero widget dependencies / zero permission popups
    """

    def __init__(
        self,
        symbol: str = "RELIANCE",
        use_sample_data: bool = True,
        update_interval_sec: float = 3.0,
    ):
        self.active_symbol = symbol.upper()
        self.use_sample_data = use_sample_data
        self.update_interval_sec = update_interval_sec
        self.mode = "PAPER"  # "PAPER" or "LIVE"

        self.watchlist: List[str] = list(DEFAULT_20_STOCKS)

        # AI Model & Risk Engine
        self.model = TradingModel()
        if not self.model.load_model():
            sample_df = DataEngine.generate_synthetic_market_data(bars=1200)
            feat_df = DataEngine.calculate_technical_features(sample_df)
            labeled_df = DataEngine.create_labels(feat_df)
            self.model.train(labeled_df)

        self.risk_manager = RiskManager()
        self.paper_trader = PaperTrader(model=self.model, risk_manager=self.risk_manager)
        self.auth = UpstoxAuth()
        self.live_trader = LiveTrader(auth=self.auth, risk_manager=self.risk_manager, dry_run=True)

        # Multi-stock data feeds & scanner state
        self.stock_data: Dict[str, pd.DataFrame] = {}
        self.stock_bar_index: Dict[str, int] = {}
        self.stock_signals: Dict[str, Dict[str, Any]] = {}
        self.positions: Dict[str, Dict[str, Any]] = {}

        self._init_stock_feeds()

        self.is_running = True
        self.is_paused = False
        self.display_handle = None

        # Register Google Colab JS callbacks for interactive controls
        try:
            from google.colab import output
            output.register_callback("colab_add_stock", self.add_stock)
            output.register_callback("colab_remove_stock", self.remove_stock)
            output.register_callback("colab_select_stock", self.select_stock)
            output.register_callback("colab_toggle_mode", self.toggle_mode)
            output.register_callback("colab_square_off", self.emergency_square_off)
            output.register_callback("colab_stop_bot", self.stop)
            output.register_callback("colab_resume_bot", self.resume)
        except Exception:
            pass

    def _init_stock_feeds(self):
        """Initializes data feeds for all stocks in the active watchlist."""
        for sym in self.watchlist:
            if sym not in self.stock_data:
                base = STOCK_BASE_PRICES.get(sym, 1000.0)
                seed_val = abs(hash(sym)) % 10000
                df = DataEngine.generate_synthetic_market_data(bars=350, start_price=base, seed=seed_val)
                self.stock_data[sym] = df
                self.stock_bar_index[sym] = 60
                self.stock_signals[sym] = {
                    "price": float(df["close"].iloc[60]),
                    "change_pct": 0.0,
                    "signal": "HOLD",
                    "confidence": 0.50,
                }

    def _refresh_display(self):
        """Immediately re-renders the dashboard when the user interacts."""
        if hasattr(self, "display_handle") and self.display_handle is not None:
            try:
                from IPython.display import HTML
                self._update_scanner_table()
                self._update_trade_table()
                self.display_handle.update(HTML(self.get_full_dashboard_html()))
            except Exception:
                pass

    # ==========================================
    # Interactive Actions (Add/Remove/Select)
    # ==========================================
    def add_stock(self, symbol: str):
        sym = str(symbol).strip().upper()
        if sym and sym not in self.watchlist:
            self.watchlist.append(sym)
            self._init_stock_feeds()
            self.active_symbol = sym
            print(f"\n[WATCHLIST] Added {sym} to scanner.")
            self._refresh_display()

    def remove_stock(self, symbol: Optional[str] = None):
        sym = str(symbol).strip().upper() if symbol else self.active_symbol
        if sym in self.watchlist and len(self.watchlist) > 1:
            self.watchlist.remove(sym)
            if self.active_symbol == sym:
                self.active_symbol = self.watchlist[0]
            print(f"\n[WATCHLIST] Removed {sym} from scanner.")
            self._refresh_display()

    def select_stock(self, symbol: str):
        sym = str(symbol).strip().upper()
        if sym in self.watchlist:
            self.active_symbol = sym
            self._refresh_display()

    def toggle_mode(self):
        if self.mode == "PAPER":
            tok = self.auth.load_cached_token()
            if not tok or not self.auth.validate_token(tok):
                print("\n[WARN] Valid Upstox token required for Live mode.")
                return
            self.mode = "LIVE"
            self.live_trader.dry_run = False
            print("\n[ALERT] Switched to LIVE REAL-MONEY MODE (NSE)!")
        else:
            self.mode = "PAPER"
            self.live_trader.dry_run = True
            print("\n[INFO] Switched to SAFE PAPER TRADING MODE (Simulation).")
        self._refresh_display()

    def emergency_square_off(self):
        now_str = datetime.now().strftime("%H:%M:%S")
        for sym, pos in list(self.positions.items()):
            cur_price = self.stock_signals.get(sym, {}).get("price", pos["entry_price"])
            pnl = (cur_price - pos["entry_price"]) * pos["quantity"]
            self.risk_manager.update_pnl(pnl)
            self.paper_trader.trade_history.append({
                "symbol": sym,
                "action": "SELL",
                "quantity": pos["quantity"],
                "entry_price": pos["entry_price"],
                "exit_price": cur_price,
                "pnl": round(pnl, 2),
                "reason": "EMERGENCY_SQUARE_OFF",
                "entry_time": pos["entry_time"],
                "exit_time": now_str,
            })
        self.positions.clear()
        print("\n[SAFETY] All open positions squared off immediately.")
        self._refresh_display()

    def stop(self):
        self.is_running = False
        print("\n[INFO] Bot scanning stopped by user.")
        self._refresh_display()

    def resume(self):
        if not self.is_running:
            self.is_running = True
            import threading
            self._bg_thread = threading.Thread(target=self._run_background_loop, daemon=True)
            self._bg_thread.start()
            print("\n[INFO] Bot scanning resumed.")
            self._refresh_display()

    # ==========================================
    # High-Contrast Dark UI HTML Generators
    # ==========================================
    def _get_header_html(self, status: str) -> str:
        status_color = "#3fb950" if status in ["ACTIVE", "RUNNING"] else "#f85149" if status == "KILLED" else "#d29922"
        mode_bg = "#8957e5" if self.mode == "PAPER" else "#da3633"
        mode_label = "🎮 PAPER TRADING (SIMULATION)" if self.mode == "PAPER" else "🔴 LIVE REAL-MONEY (NSE)"

        return f"""
        <div style="display:flex;justify-content:space-between;align-items:center;background:#161b22;border:1px solid #30363d;padding:14px 20px;border-radius:10px;margin-bottom:12px;color:#f0f6fc;">
            <div style="display:flex;align-items:center;gap:12px;">
                <span style="font-size:20px;">🤖</span>
                <div>
                    <div style="font-size:16px;font-weight:bold;color:#f0f6fc;letter-spacing:0.5px;">Upstox AI Algorithmic Trading Terminal</div>
                    <div style="font-size:11px;color:#8b949e;">Automated Multi-Stock Strategy Engine | Upstox API v2</div>
                </div>
            </div>
            <div style="display:flex;gap:10px;align-items:center;font-size:12px;">
                <span style="background:{mode_bg};color:#ffffff;padding:4px 12px;border-radius:14px;font-weight:bold;letter-spacing:0.5px;">{mode_label}</span>
                <span style="background:#21262d;border:1px solid #30363d;color:#58a6ff;padding:4px 10px;border-radius:14px;font-weight:bold;">Active: {self.active_symbol}</span>
                <span style="background:#21262d;border:1px solid #30363d;color:#c9d1d9;padding:4px 10px;border-radius:14px;font-weight:bold;">Scanning: {len(self.watchlist)} Stocks</span>
                <span style="background:{status_color};color:#ffffff;padding:4px 10px;border-radius:14px;font-weight:bold;">{status}</span>
            </div>
        </div>
        """

    def _get_metrics_html(self) -> str:
        pnl = self.risk_manager.daily_realized_pnl
        pnl_color = "#3fb950" if pnl >= 0 else "#f85149"
        sym = self.active_symbol
        price = self.stock_signals.get(sym, {}).get("price", 1000.0)
        signal = self.stock_signals.get(sym, {}).get("signal", "HOLD")
        confidence = self.stock_signals.get(sym, {}).get("confidence", 0.50)

        pos = self.positions.get(sym)
        pos_str = "⚪ Flat (No Open Position)"
        sl_tp_str = "Waiting for high-confidence AI signal"
        if pos:
            pos_str = f"🟢 LONG {pos['quantity']}x @ ₹{pos['entry_price']:.2f}"
            sl_tp_str = f"SL: ₹{pos['stop_loss']:.2f} | TP: ₹{pos['take_profit']:.2f}"

        open_positions_count = len(self.positions)

        return f"""
        <div style="display:grid;grid-template-columns:repeat(4, 1fr);gap:12px;margin-bottom:12px;">
            <div style="background:#161b22;border:1px solid #30363d;padding:12px 16px;border-radius:8px;">
                <div style="font-size:12px;color:#8b949e;font-weight:600;">DAILY REALIZED P&L</div>
                <div style="font-size:24px;font-weight:bold;color:{pnl_color};margin:4px 0;">₹{pnl:+,.2f}</div>
                <div style="font-size:11px;color:#8b949e;">Daily Loss Limit: -₹{self.risk_manager.max_daily_loss:,.2f}</div>
            </div>
            <div style="background:#161b22;border:1px solid #30363d;padding:12px 16px;border-radius:8px;">
                <div id="card-ltp-title" style="font-size:12px;color:#8b949e;font-weight:600;">{sym} CURRENT LTP</div>
                <div id="card-ltp" style="font-size:24px;font-weight:bold;color:#f0f6fc;margin:4px 0;">₹{price:,.2f}</div>
                <div style="font-size:11px;color:#58a6ff;">Total Open Positions: <b>{open_positions_count} / 4</b></div>
            </div>
            <div style="background:#161b22;border:1px solid #30363d;padding:12px 16px;border-radius:8px;">
                <div id="card-pos-title" style="font-size:12px;color:#8b949e;font-weight:600;">{sym} ACTIVE POSITION</div>
                <div id="card-pos" style="font-size:14px;font-weight:bold;color:#58a6ff;margin:6px 0;">{pos_str}</div>
                <div id="card-sl-tp" style="font-size:11px;color:#8b949e;">{sl_tp_str}</div>
            </div>
            <div style="background:#161b22;border:1px solid #30363d;padding:12px 16px;border-radius:8px;">
                <div id="card-signal-title" style="font-size:12px;color:#8b949e;font-weight:600;">{sym} AI SIGNAL</div>
                <div id="card-signal" style="font-size:20px;font-weight:bold;color:{'#3fb950' if signal == 'BUY' else '#8b949e'};margin:4px 0;">
                    {signal} ({confidence * 100:.1f}%)
                </div>
                <div style="font-size:11px;color:#8b949e;">Auto-Buy Threshold: > {self.model.confidence_threshold * 100:.0f}%</div>
            </div>
        </div>
        """

    def _get_watchlist_bar_html(self) -> str:
        """Renders the Watchlist bar with clickable stock pills and Add/Remove controls."""
        pills = ""
        for sym in self.watchlist:
            is_active = (sym == self.active_symbol)
            bg = "#1f6feb" if is_active else "#21262d"
            txt = "#ffffff" if is_active else "#c9d1d9"
            border = "#58a6ff" if is_active else "#30363d"
            has_pos = "🟢 " if sym in self.positions else ""
            pills += f"""<button type="button" class="stock-pill" data-sym="{sym}" onclick="jsSelectStock('{sym}')" style="background:{bg};color:{txt};border:1px solid {border};padding:4px 10px;border-radius:14px;cursor:pointer;font-weight:bold;font-size:11px;margin:2px;">{has_pos}{sym}</button>"""

        mode_btn_txt = "Switch to LIVE (Real Money)" if self.mode == "PAPER" else "Switch to PAPER (Simulation)"
        mode_btn_bg = "#da3633" if self.mode == "PAPER" else "#8957e5"
        stop_btn_html = """<button type="button" onclick="jsStopBot()" style="background:#21262d;color:#8b949e;border:1px solid #30363d;padding:6px 10px;border-radius:6px;font-weight:bold;font-size:12px;cursor:pointer;">⏸️ Pause</button>""" if self.is_running else """<button type="button" onclick="jsResumeBot()" style="background:#238636;color:white;border:none;padding:6px 10px;border-radius:6px;font-weight:bold;font-size:12px;cursor:pointer;">▶️ Resume</button>"""

        return f"""
        <div style="background:#161b22;padding:12px 16px;border-radius:8px;border:1px solid #30363d;margin-bottom:12px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;flex-wrap:wrap;gap:8px;">
                <div style="display:flex;align-items:center;gap:8px;">
                    <span style="color:#8b949e;font-size:12px;font-weight:bold;">⚙️ MANAGE STOCKS:</span>
                    <input id="new-stock-input" placeholder="Ticker (e.g. ZOMATO)" style="background:#0d1117;color:#f0f6fc;border:1px solid #30363d;padding:6px 10px;border-radius:6px;font-size:12px;width:150px;">
                    <button type="button" onclick="jsAddStock()" style="background:#238636;color:white;border:none;padding:6px 12px;border-radius:6px;font-weight:bold;font-size:12px;cursor:pointer;">+ Add Stock</button>
                    <button type="button" id="remove-btn" onclick="jsRemoveStock()" style="background:#30363d;color:#f0f6fc;border:1px solid #484f58;padding:6px 10px;border-radius:6px;font-weight:bold;font-size:12px;cursor:pointer;">- Remove {self.active_symbol}</button>
                </div>
                <div style="display:flex;align-items:center;gap:8px;">
                    <button type="button" onclick="jsToggleMode()" style="background:{mode_btn_bg};color:white;border:none;padding:6px 12px;border-radius:6px;font-weight:bold;font-size:12px;cursor:pointer;">{mode_btn_txt}</button>
                    <button type="button" onclick="jsSquareOff()" style="background:#b62324;color:white;border:none;padding:6px 12px;border-radius:6px;font-weight:bold;font-size:12px;cursor:pointer;">🛑 Square-Off All</button>
                    {stop_btn_html}
                </div>
            </div>
            <div style="display:flex;flex-wrap:wrap;gap:4px;align-items:center;">
                <span style="color:#58a6ff;font-size:11px;font-weight:bold;margin-right:6px;">👆 CLICK STOCK TO SWITCH ({len(self.watchlist)} STOCKS):</span>
                {pills}
            </div>
        </div>
        """

    def _get_chart_html(self) -> str:
        """Renders GPU-accelerated HTML5 Canvas chart with zero CPU overhead and instant switching."""
        return f"""
        <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:12px 16px;margin:8px 0;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                <span id="chart-title" style="font-size:13px;font-weight:bold;color:#f0f6fc;">📈 Live Price & Moving Averages - {self.active_symbol} ({self.mode} MODE)</span>
                <div style="font-size:11px;display:flex;gap:14px;align-items:center;">
                    <span style="color:#58a6ff;font-weight:bold;">— Price</span>
                    <span style="color:#f0883e;font-weight:bold;">--- EMA 9 (Fast)</span>
                    <span style="color:#3fb950;font-weight:bold;">··· EMA 21 (Trend)</span>
                </div>
            </div>
            <div style="position:relative;width:100%;height:220px;">
                <canvas id="terminal-chart" width="1020" height="220" style="width:100%;height:220px;background:#0d1117;border-radius:8px;display:block;border:1px solid #21262d;"></canvas>
            </div>
        </div>
        """

    def _update_scanner_table(self):
        """Builds high-contrast dark table of all 20+ stocks."""
        rows = ""
        for sym in self.watchlist:
            sig_info = self.stock_signals.get(sym, {})
            price = sig_info.get("price", 0.0)
            change = sig_info.get("change_pct", 0.0)
            signal = sig_info.get("signal", "HOLD")
            conf = sig_info.get("confidence", 0.50)

            chg_color = "#3fb950" if change >= 0 else "#f85149"
            is_active = (sym == self.active_symbol)
            row_bg = "#1c2128" if is_active else "#161b22"

            if signal == "BUY":
                sig_badge = '<span style="background:#238636;color:#ffffff;padding:3px 8px;border-radius:10px;font-weight:bold;font-size:11px;">🟢 BUY</span>'
            else:
                sig_badge = '<span style="color:#8b949e;font-weight:bold;font-size:11px;">HOLD</span>'

            if sym in self.positions:
                pos_badge = '<span style="background:#8957e5;color:#ffffff;padding:3px 8px;border-radius:10px;font-weight:bold;font-size:11px;">🟢 LONG</span>'
            else:
                pos_badge = '<span style="color:#6e7681;font-size:11px;">⚪ FLAT</span>'

            rows += f"""
            <tr onclick="jsSelectStock('{sym}')" style="background:{row_bg};border-bottom:1px solid #21262d;cursor:pointer;" title="Click to view {sym} chart and metrics">
                <td style="padding:8px 12px;font-weight:bold;">
                    <span style="color:#58a6ff;font-size:13px;font-weight:bold;">{sym}</span>
                    {' <span style="color:#e3b341;font-size:10px;">★ ACTIVE</span>' if is_active else ''}
                </td>
                <td style="padding:8px 12px;color:#f0f6fc;font-weight:bold;font-size:13px;">₹{price:,.2f}</td>
                <td style="padding:8px 12px;color:{chg_color};font-weight:bold;font-size:12px;">{change:+.2f}%</td>
                <td style="padding:8px 12px;">{sig_badge}</td>
                <td style="padding:8px 12px;color:#e6edf3;font-size:12px;">{conf * 100:.1f}%</td>
                <td style="padding:8px 12px;">{pos_badge}</td>
            </tr>
            """

        self.scanner_table_str = f"""
        <div style="max-height:240px;overflow-y:auto;border:1px solid #30363d;border-radius:8px;">
            <table style="width:100%;border-collapse:collapse;font-size:12px;background:#161b22;color:#f0f6fc;">
                <thead style="position:sticky;top:0;background:#21262d;z-index:2;">
                    <tr style="text-align:left;border-bottom:2px solid #30363d;color:#8b949e;font-size:11px;">
                        <th style="padding:8px 12px;">SYMBOL</th>
                        <th style="padding:8px 12px;">LTP</th>
                        <th style="padding:8px 12px;">1M CHANGE</th>
                        <th style="padding:8px 12px;">AI SIGNAL</th>
                        <th style="padding:8px 12px;">CONFIDENCE</th>
                        <th style="padding:8px 12px;">POSITION</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
        """

    def _update_trade_table(self):
        trades = self.paper_trader.trade_history
        if not trades:
            self.trade_table_str = "<div style='padding:10px;color:#8b949e;background:#161b22;border:1px solid #30363d;border-radius:8px;font-size:12px;'>No completed trades yet. The bot is actively scanning for >65% confidence AI breakouts.</div>"
            return

        rows = ""
        for t in reversed(trades[-5:]):
            pnl = t["pnl"]
            pnl_color = "#3fb950" if pnl >= 0 else "#f85149"
            rows += f"""
            <tr style="border-bottom:1px solid #21262d;background:#161b22;">
                <td style='padding:7px 10px;color:#8b949e;'>{t.get('exit_time', '')}</td>
                <td style='padding:7px 10px;font-weight:bold;color:#58a6ff;'>{t.get('symbol', '')}</td>
                <td style='padding:7px 10px;color:#f0f6fc;'>{t.get('quantity', 1)} shares</td>
                <td style='padding:7px 10px;color:#f0f6fc;'>₹{t.get('entry_price', 0):.2f}</td>
                <td style='padding:7px 10px;color:#f0f6fc;'>₹{t.get('exit_price', 0):.2f}</td>
                <td style='padding:7px 10px;font-weight:bold;color:{pnl_color};'>₹{pnl:+.2f}</td>
                <td style='padding:7px 10px;color:#c9d1d9;'>{t.get('reason', '')}</td>
            </tr>
            """

        self.trade_table_str = f"""
        <div style="border:1px solid #30363d;border-radius:8px;overflow:hidden;">
            <table style='width:100%;border-collapse:collapse;font-size:12px;background:#161b22;color:#f0f6fc;'>
                <thead style='background:#21262d;text-align:left;color:#8b949e;font-size:11px;'>
                    <tr>
                        <th style='padding:8px 10px;'>TIME</th>
                        <th style='padding:8px 10px;'>SYMBOL</th>
                        <th style='padding:8px 10px;'>QUANTITY</th>
                        <th style='padding:8px 10px;'>ENTRY</th>
                        <th style='padding:8px 10px;'>EXIT</th>
                        <th style='padding:8px 10px;'>P&L</th>
                        <th style='padding:8px 10px;'>REASON</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
        """

    def scan_all_stocks_step(self):
        """Advances candles and automatically executes trades across all 20+ stocks."""
        now_str = datetime.now().strftime("%H:%M:%S")

        for sym in list(self.watchlist):
            df = self.stock_data[sym]
            idx = self.stock_bar_index[sym]

            # Generate more candles if reached buffer end
            if idx >= len(df) - 1:
                new_bars = DataEngine.generate_synthetic_market_data(
                    bars=60,
                    start_price=float(df["close"].iloc[-1]),
                    seed=abs(hash(sym + str(idx))) % 10000
                )
                df = pd.concat([df, new_bars]).reset_index(drop=True)
                self.stock_data[sym] = df

            idx += 1
            self.stock_bar_index[sym] = idx

            # Microsecond technical feature slice (last 40 bars)
            subset = df.iloc[max(0, idx - 40) : idx + 1]
            feat_subset = DataEngine.calculate_technical_features(subset)
            current_bar = feat_subset.iloc[-1]
            close_price = float(current_bar["close"])
            prev_close = float(feat_subset.iloc[-2]["close"])
            change_pct = ((close_price - prev_close) / prev_close) * 100.0

            # 1. Automatic Exit Management
            if sym in self.positions:
                pos = self.positions[sym]
                low_price = float(current_bar["low"])
                high_price = float(current_bar["high"])

                exit_price = None
                reason = None

                if low_price <= pos["stop_loss"]:
                    exit_price = pos["stop_loss"]
                    reason = "STOP_LOSS_HIT"
                elif high_price >= pos["take_profit"]:
                    exit_price = pos["take_profit"]
                    reason = "TAKE_PROFIT_HIT"

                if exit_price is not None:
                    pnl = (exit_price - pos["entry_price"]) * pos["quantity"]
                    self.risk_manager.update_pnl(pnl)

                    trade_record = {
                        "symbol": sym,
                        "action": "SELL",
                        "quantity": pos["quantity"],
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_price,
                        "pnl": round(pnl, 2),
                        "reason": reason,
                        "entry_time": pos["entry_time"],
                        "exit_time": now_str,
                    }
                    self.paper_trader.trade_history.append(trade_record)
                    del self.positions[sym]

            # 2. Automatic Entry Detection (AI Threshold > 65%)
            signal = "HOLD"
            proba = 0.50
            if not self.is_paused and not self.risk_manager.kill_switch_triggered:
                signal, proba = self.model.predict_signal(feat_subset)
                # If confidence > 65% and not already holding, AUTOMATICALLY TRADE!
                if signal == "BUY" and sym not in self.positions and len(self.positions) < 4:
                    qty, sl, tp = self.risk_manager.calculate_position_size(close_price)
                    self.positions[sym] = {
                        "symbol": sym,
                        "quantity": max(1, qty // 3),
                        "entry_price": close_price,
                        "stop_loss": sl,
                        "take_profit": tp,
                        "entry_time": now_str,
                    }

            self.stock_signals[sym] = {
                "price": close_price,
                "change_pct": change_pct,
                "signal": signal,
                "confidence": proba,
            }

        self._update_scanner_table()
        self._update_trade_table()

    def get_full_dashboard_html(self) -> str:
        """Generates the unified high-contrast dark dashboard HTML."""
        status = "KILLED" if self.risk_manager.kill_switch_triggered else "PAUSED" if self.is_paused else "RUNNING"
        header = self._get_header_html(status)
        metrics = self._get_metrics_html()
        watchlist_bar = self._get_watchlist_bar_html()
        chart_html = self._get_chart_html()
        scanner = getattr(self, "scanner_table_str", "<div>Scanning stocks...</div>")
        trades = getattr(self, "trade_table_str", "<div>No completed trades yet.</div>")

        stocks_payload = {}
        for sym in self.watchlist:
            df = self.stock_data.get(sym)
            idx = self.stock_bar_index.get(sym, 50)
            subset = df.iloc[max(0, idx - 30) : idx + 1]
            feat = DataEngine.calculate_technical_features(subset)
            prices = [round(float(p), 2) for p in feat["close"].tail(25)]
            e9 = [round(float(p), 2) for p in feat["ema_9"].tail(25)] if "ema_9" in feat else prices
            e21 = [round(float(p), 2) for p in feat["ema_21"].tail(25)] if "ema_21" in feat else prices
            pos = self.positions.get(sym)
            stocks_payload[sym] = {
                "prices": prices,
                "ema9": e9,
                "ema21": e21,
                "ltp": round(float(prices[-1]), 2) if prices else 0.0,
                "signal": self.stock_signals.get(sym, {}).get("signal", "HOLD"),
                "confidence": round(float(self.stock_signals.get(sym, {}).get("confidence", 0.50)) * 100, 1),
                "change_pct": round(float(self.stock_signals.get(sym, {}).get("change_pct", 0.0)), 2),
                "pos": pos,
            }

        data_json = json.dumps(stocks_payload)

        js_code = f"""
        <script>
        window.chartData = {data_json};
        window.currentActive = '{self.active_symbol}';
        window.currentMode = '{self.mode}';

        function renderCanvasChart(sym) {{
            var canvas = document.getElementById('terminal-chart');
            if (!canvas) return;
            var ctx = canvas.getContext('2d');
            var d = window.chartData[sym];
            if (!d || !d.prices || d.prices.length < 2) return;

            var w = canvas.width;
            var h = canvas.height;
            ctx.clearRect(0, 0, w, h);
            ctx.fillStyle = '#0d1117';
            ctx.fillRect(0, 0, w, h);

            var allVals = d.prices.concat(d.ema9).concat(d.ema21);
            var minV = Math.min.apply(null, allVals);
            var maxV = Math.max.apply(null, allVals);
            var pad = (maxV - minV) * 0.1 || 1.0;
            minV -= pad;
            maxV += pad;
            var range = maxV - minV;

            function toY(v) {{ return h - 25 - ((v - minV) / range) * (h - 50); }}
            function toX(i, n) {{ return 65 + (i / (n - 1)) * (w - 90); }}

            // Horizontal Gridlines & Price Labels
            ctx.strokeStyle = '#21262d';
            ctx.lineWidth = 1;
            ctx.fillStyle = '#8b949e';
            ctx.font = '11px sans-serif';
            ctx.textAlign = 'right';
            for (var i = 0; i <= 4; i++) {{
                var pVal = minV + (i / 4) * range;
                var y = toY(pVal);
                ctx.beginPath();
                ctx.moveTo(65, y);
                ctx.lineTo(w - 20, y);
                ctx.stroke();
                ctx.fillText('₹' + pVal.toFixed(2), 58, y + 4);
            }}

            // Draw EMA 21 (Green dotted)
            if (d.ema21 && d.ema21.length > 1) {{
                ctx.strokeStyle = '#3fb950';
                ctx.lineWidth = 1.5;
                ctx.setLineDash([3, 3]);
                ctx.beginPath();
                for (var i = 0; i < d.ema21.length; i++) {{
                    var x = toX(i, d.ema21.length);
                    var y = toY(d.ema21[i]);
                    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                }}
                ctx.stroke();
                ctx.setLineDash([]);
            }}

            // Draw EMA 9 (Orange dashed)
            if (d.ema9 && d.ema9.length > 1) {{
                ctx.strokeStyle = '#f0883e';
                ctx.lineWidth = 1.5;
                ctx.setLineDash([5, 5]);
                ctx.beginPath();
                for (var i = 0; i < d.ema9.length; i++) {{
                    var x = toX(i, d.ema9.length);
                    var y = toY(d.ema9[i]);
                    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                }}
                ctx.stroke();
                ctx.setLineDash([]);
            }}

            // Draw Price Line (Blue solid)
            ctx.strokeStyle = '#58a6ff';
            ctx.lineWidth = 2.2;
            ctx.beginPath();
            for (var i = 0; i < d.prices.length; i++) {{
                var x = toX(i, d.prices.length);
                var y = toY(d.prices[i]);
                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }}
            ctx.stroke();

            // Price pulse dot
            var lastX = toX(d.prices.length - 1, d.prices.length);
            var lastY = toY(d.prices[d.prices.length - 1]);
            ctx.fillStyle = '#58a6ff';
            ctx.beginPath();
            ctx.arc(lastX, lastY, 4, 0, Math.PI * 2);
            ctx.fill();
        }}

        function jsSelectStock(sym) {{
            window.currentActive = sym;
            var pills = document.querySelectorAll('.stock-pill');
            for (var i = 0; i < pills.length; i++) {{
                var btn = pills[i];
                if (btn.getAttribute('data-sym') === sym) {{
                    btn.style.background = '#1f6feb';
                    btn.style.color = '#ffffff';
                    btn.style.borderColor = '#58a6ff';
                }} else {{
                    btn.style.background = '#21262d';
                    btn.style.color = '#c9d1d9';
                    btn.style.borderColor = '#30363d';
                }}
            }}
            var d = window.chartData[sym];
            if (d) {{
                var ltpEl = document.getElementById('card-ltp');
                if (ltpEl) ltpEl.innerText = '₹' + d.ltp.toLocaleString('en-IN', {{minimumFractionDigits:2}});
                var ltpTitle = document.getElementById('card-ltp-title');
                if (ltpTitle) ltpTitle.innerText = sym + ' CURRENT LTP';
                var sigEl = document.getElementById('card-signal');
                if (sigEl) {{
                    sigEl.innerText = d.signal + ' (' + d.confidence + '%)';
                    sigEl.style.color = d.signal === 'BUY' ? '#3fb950' : '#8b949e';
                }}
                var sigTitle = document.getElementById('card-signal-title');
                if (sigTitle) sigTitle.innerText = sym + ' AI SIGNAL';
                var posEl = document.getElementById('card-pos');
                if (posEl) {{
                    if (d.pos) {{
                        posEl.innerText = '🟢 LONG ' + d.pos.quantity + 'x @ ₹' + d.pos.entry_price.toFixed(2);
                        posEl.style.color = '#3fb950';
                    }} else {{
                        posEl.innerText = '⚪ Flat (No Open Position)';
                        posEl.style.color = '#58a6ff';
                    }}
                }}
                var posTitle = document.getElementById('card-pos-title');
                if (posTitle) posTitle.innerText = sym + ' ACTIVE POSITION';
                var headerSym = document.getElementById('header-active-sym');
                if (headerSym) headerSym.innerText = 'Active: ' + sym;
                var chartTitle = document.getElementById('chart-title');
                if (chartTitle) chartTitle.innerText = '📈 Live Price & Moving Averages - ' + sym + ' (' + window.currentMode + ' MODE)';
                var remBtn = document.getElementById('remove-btn');
                if (remBtn) remBtn.innerText = '- Remove ' + sym;
            }}
            renderCanvasChart(sym);
            if (window.google && google.colab && google.colab.kernel) {{
                google.colab.kernel.invokeFunction('colab_select_stock', [sym], {{}});
            }}
        }}

        function jsAddStock() {{
            var el = document.getElementById('new-stock-input');
            if (el && el.value.trim() && window.google && google.colab && google.colab.kernel) {{
                google.colab.kernel.invokeFunction('colab_add_stock', [el.value.trim().toUpperCase()], {{}});
                el.value = '';
            }}
        }}
        function jsRemoveStock() {{
            if (window.google && google.colab && google.colab.kernel) {{
                google.colab.kernel.invokeFunction('colab_remove_stock', [], {{}});
            }}
        }}
        function jsToggleMode() {{
            if (window.google && google.colab && google.colab.kernel) {{
                google.colab.kernel.invokeFunction('colab_toggle_mode', [], {{}});
            }}
        }}
        function jsSquareOff() {{
            if (window.google && google.colab && google.colab.kernel) {{
                google.colab.kernel.invokeFunction('colab_square_off', [], {{}});
            }}
        }}
        function jsStopBot() {{
            if (window.google && google.colab && google.colab.kernel) {{
                google.colab.kernel.invokeFunction('colab_stop_bot', [], {{}});
            }}
        }}
        function jsResumeBot() {{
            if (window.google && google.colab && google.colab.kernel) {{
                google.colab.kernel.invokeFunction('colab_resume_bot', [], {{}});
            }}
        }}

        setTimeout(function() {{
            renderCanvasChart(window.currentActive);
        }}, 60);
        </script>
        """

        return f"""
        <div id="upstox-dashboard-container" style="font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;max-width:1080px;margin:0 auto;background:#0d1117;padding:16px;border-radius:12px;border:1px solid #30363d;color:#f0f6fc;">
            {js_code}
            {header}
            {metrics}
            {watchlist_bar}
            {chart_html}
            <h4 style="margin:14px 0 6px 0;color:#f0f6fc;font-size:14px;">📊 20+ Stocks Real-Time Multi-Scanner:</h4>
            {scanner}
            <h4 style="margin:14px 0 6px 0;color:#f0f6fc;font-size:14px;">📜 Completed Trades History:</h4>
            {trades}
        </div>
        """

    async def _run_async_loop(self):
        """Asynchronous execution loop yielding to Colab event loop so clicks execute instantly."""
        import asyncio
        while self.is_running:
            await asyncio.sleep(self.update_interval_sec)
            if self.is_running:
                try:
                    self.scan_all_stocks_step()
                    if hasattr(self, "display_handle") and self.display_handle is not None:
                        from IPython.display import HTML
                        self.display_handle.update(HTML(self.get_full_dashboard_html()))
                except Exception:
                    pass

    def render(self, run_loop: bool = True):
        """Renders the complete multi-stock dashboard in Google Colab with in-place zero-flicker updates."""
        from IPython.display import display, HTML

        # Initial scan
        self.scan_all_stocks_step()

        # Create persistent display handle with display_id
        full_html = self.get_full_dashboard_html()
        self.display_handle = display(HTML(full_html), display_id="upstox_ai_dashboard")

        if run_loop:
            import asyncio
            try:
                import nest_asyncio
                nest_asyncio.apply()
            except ImportError:
                import subprocess, sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "nest_asyncio"])
                import nest_asyncio
                nest_asyncio.apply()

            self.is_running = True
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.run_until_complete(self._run_async_loop())
                else:
                    asyncio.run(self._run_async_loop())
            except KeyboardInterrupt:
                print("\n[INFO] Dashboard stopped by user.")
            except Exception as e:
                print(f"\n[INFO] Dashboard session ended: {e}")
