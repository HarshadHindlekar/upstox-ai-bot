import time
import threading
from datetime import datetime
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    Multi-Stock Live Trading Terminal for Google Colab.
    Features:
    - Multi-stock scanning & auto-trading across 20+ NSE stocks
    - Add/Remove custom stocks dynamically
    - Mode Switcher: Paper Trading (Simulation) vs Live Real-Money Execution
    - Live Candlestick & EMA charts
    - Active WebSocket keep-alive to prevent Colab idle timeouts
    """

    def __init__(
        self,
        symbol: str = "RELIANCE",
        use_sample_data: bool = True,
        update_interval_sec: float = 2.5,
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
        self._loop_thread: Optional[threading.Thread] = None

        # ipywidgets init
        try:
            import ipywidgets as widgets
            from IPython.display import display, clear_output
            self.widgets = widgets
            self.display = display
            self.clear_output = clear_output
            self._has_widgets = True
        except ImportError:
            self._has_widgets = False

        self._build_widgets()

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

    def _build_widgets(self):
        if not self._has_widgets:
            return

        w = self.widgets

        # Header Status Bar
        self.header_html = w.HTML(value=self._get_header_html("RUNNING"))

        # Top Metric Cards
        self.metrics_html = w.HTML(value=self._get_metrics_html())

        # Control Row 1: Stock Watchlist selector & Add/Remove
        self.stock_dropdown = w.Dropdown(
            options=self.watchlist,
            value=self.active_symbol if self.active_symbol in self.watchlist else self.watchlist[0],
            description="Active Stock:",
            layout=w.Layout(width="220px")
        )
        self.stock_dropdown.observe(self._on_stock_change, names="value")

        self.input_new_stock = w.Text(
            placeholder="e.g. ZOMATO, TRENT",
            description="New Ticker:",
            layout=w.Layout(width="220px")
        )

        self.btn_add_stock = w.Button(
            description="+ Add Stock",
            button_style="primary",
            tooltip="Add stock to active multi-stock automated scanner",
            layout=w.Layout(width="120px")
        )
        self.btn_add_stock.on_click(self._on_add_stock)

        self.btn_remove_stock = w.Button(
            description="- Remove Stock",
            button_style="warning",
            tooltip="Remove currently selected stock from scanner",
            layout=w.Layout(width="130px")
        )
        self.btn_remove_stock.on_click(self._on_remove_stock)

        self.watchlist_controls = w.HBox(
            [self.stock_dropdown, self.input_new_stock, self.btn_add_stock, self.btn_remove_stock],
            layout=w.Layout(margin="5px 0 10px 0", align_items="center")
        )

        # Control Row 2: Operational Action Buttons
        self.btn_auto = w.Button(
            description="⏹ Stop Loop",
            button_style="danger",
            tooltip="Toggle automated bar scanner",
            icon="stop",
            layout=w.Layout(width="150px")
        )
        self.btn_auto.on_click(self._on_toggle_auto)

        self.btn_step = w.Button(
            description="Step Next Tick",
            button_style="info",
            tooltip="Advance 1 minute tick manually",
            icon="step-forward",
            layout=w.Layout(width="140px")
        )
        self.btn_step.on_click(self._on_step_tick)

        self.btn_pause = w.Button(
            description="⏸ Pause AI Signals",
            button_style="warning",
            tooltip="Pause opening new positions",
            layout=w.Layout(width="160px")
        )
        self.btn_pause.on_click(self._on_pause_toggle)

        self.btn_mode_toggle = w.Button(
            description="Toggle: Paper / Live",
            button_style="info",
            tooltip="Switch between Paper Trading and Live Upstox Execution",
            icon="exchange",
            layout=w.Layout(width="180px")
        )
        self.btn_mode_toggle.on_click(self._on_mode_toggle)

        self.btn_square_off = w.Button(
            description="🛑 Square-Off All",
            button_style="danger",
            tooltip="Emergency exit from all open positions immediately",
            icon="times-circle",
            layout=w.Layout(width="160px")
        )
        self.btn_square_off.on_click(self._on_emergency_square_off)

        self.actions_box = w.HBox(
            [self.btn_auto, self.btn_step, self.btn_pause, self.btn_mode_toggle, self.btn_square_off],
            layout=w.Layout(margin="5px 0 15px 0")
        )

        # Live Chart Output
        self.chart_output = w.Output(layout=w.Layout(height="340px", width="100%"))

        # Multi-Stock Scanner Live Table
        self.scanner_table_html = w.HTML(value="<div>Scanning stocks...</div>")

        # Trade History Table
        self.history_html = w.HTML(value="<div style='padding:6px;color:#888;'>No trades executed yet.</div>")

        # Main Layout Container
        self.container = w.VBox([
            self.header_html,
            self.metrics_html,
            self.watchlist_controls,
            self.actions_box,
            self.chart_output,
            w.HTML("<h4 style='margin:12px 0 6px 0;color:#222;'>📊 20+ Stocks Live Multi-Scanner:</h4>"),
            self.scanner_table_html,
            w.HTML("<h4 style='margin:12px 0 6px 0;color:#222;'>📜 Completed Trades History:</h4>"),
            self.history_html,
        ], layout=w.Layout(
            border="1px solid #ddd",
            padding="16px",
            border_radius="10px",
            background_color="#fdfdfd"
        ))

    def _get_header_html(self, status: str) -> str:
        color = "#28a745" if status in ["ACTIVE", "RUNNING"] else "#dc3545" if status == "KILLED" else "#ffc107"
        mode_color = "#6f42c1" if self.mode == "PAPER" else "#dc3545"
        mode_label = "🎮 PAPER TRADING (SIMULATION)" if self.mode == "PAPER" else "🔴 LIVE REAL-MONEY (NSE)"

        return f"""
        <div style="display:flex;justify-content:space-between;align-items:center;background:#1a1d20;color:white;padding:12px 20px;border-radius:8px;margin-bottom:12px;">
            <div style="font-size:18px;font-weight:bold;">
                🤖 Upstox AI Multi-Stock Trading Terminal
            </div>
            <div style="display:flex;gap:12px;font-size:13px;align-items:center;">
                <span style="background:{mode_color};padding:4px 12px;border-radius:12px;font-weight:bold;color:white;">{mode_label}</span>
                <span>Active: <b>{self.active_symbol}</b></span>
                <span>Scanning: <b>{len(self.watchlist)} Stocks</b></span>
                <span style="background:{color};padding:4px 10px;border-radius:12px;font-weight:bold;color:white;">{status}</span>
            </div>
        </div>
        """

    def _get_metrics_html(self) -> str:
        pnl = self.risk_manager.daily_realized_pnl
        pnl_color = "#28a745" if pnl >= 0 else "#dc3545"
        sym = self.active_symbol
        price = self.stock_signals.get(sym, {}).get("price", 1000.0)
        signal = self.stock_signals.get(sym, {}).get("signal", "HOLD")
        confidence = self.stock_signals.get(sym, {}).get("confidence", 0.50)

        pos = self.positions.get(sym)
        pos_str = "None (Flat)"
        sl_tp_str = "N/A"
        if pos:
            pos_str = f"LONG {pos['quantity']}x @ ₹{pos['entry_price']:.2f}"
            sl_tp_str = f"SL: ₹{pos['stop_loss']:.2f} | TP: ₹{pos['take_profit']:.2f}"

        open_positions_count = len(self.positions)

        return f"""
        <div style="display:grid;grid-template-columns:repeat(4, 1fr);gap:10px;margin-bottom:12px;">
            <div style="background:#fff;border:1px solid #e0e0e0;padding:10px 14px;border-radius:8px;">
                <div style="font-size:12px;color:#777;">Daily Realized P&L</div>
                <div style="font-size:20px;font-weight:bold;color:{pnl_color};">₹{pnl:+,.2f}</div>
                <div style="font-size:11px;color:#999;">Daily Limit: -₹{self.risk_manager.max_daily_loss:,.2f}</div>
            </div>
            <div style="background:#fff;border:1px solid #e0e0e0;padding:10px 14px;border-radius:8px;">
                <div style="font-size:12px;color:#777;">{sym} LTP</div>
                <div style="font-size:20px;font-weight:bold;color:#222;">₹{price:,.2f}</div>
                <div style="font-size:11px;color:#007bff;">Open Positions: <b>{open_positions_count}</b></div>
            </div>
            <div style="background:#fff;border:1px solid #e0e0e0;padding:10px 14px;border-radius:8px;">
                <div style="font-size:12px;color:#777;">{sym} Position</div>
                <div style="font-size:14px;font-weight:bold;color:#007bff;margin-top:2px;">{pos_str}</div>
                <div style="font-size:11px;color:#666;">{sl_tp_str}</div>
            </div>
            <div style="background:#fff;border:1px solid #e0e0e0;padding:10px 14px;border-radius:8px;">
                <div style="font-size:12px;color:#777;">{sym} AI Signal</div>
                <div style="font-size:18px;font-weight:bold;color:{'#28a745' if signal == 'BUY' else '#6c757d'};">
                    {signal} ({confidence * 100:.1f}%)
                </div>
                <div style="font-size:11px;color:#999;">Threshold: {self.model.confidence_threshold * 100:.0f}%</div>
            </div>
        </div>
        """

    def _render_chart(self):
        """Renders live chart for the currently active stock."""
        if not self._has_widgets:
            return

        with self.chart_output:
            self.clear_output(wait=True)
            df = self.stock_data.get(self.active_symbol)
            idx = self.stock_bar_index.get(self.active_symbol, 60)
            if df is None or idx < 35:
                return

            subset = df.iloc[max(0, idx - 35) : idx + 1].copy()
            feat_subset = DataEngine.calculate_technical_features(subset)
            plot_df = feat_subset.tail(30).copy()

            fig, ax = plt.subplots(figsize=(10, 3.2), dpi=100)
            ax.plot(plot_df.index, plot_df["close"], label=f"{self.active_symbol} Close", color="#1f77b4", lw=1.8)
            if "ema_9" in plot_df.columns:
                ax.plot(plot_df.index, plot_df["ema_9"], label="EMA 9", color="#ff7f0e", lw=1.2, ls="--")
            if "ema_21" in plot_df.columns:
                ax.plot(plot_df.index, plot_df["ema_21"], label="EMA 21", color="#2ca02c", lw=1.2, ls=":")

            ax.set_title(f"Live Price & EMAs - {self.active_symbol} ({self.mode} MODE)", fontsize=11, fontweight="bold")
            ax.grid(True, alpha=0.3, ls="--")
            ax.legend(loc="upper left", fontsize=8)
            plt.tight_layout()
            plt.show()

    def _update_scanner_table(self):
        """Renders updating multi-stock scanner table."""
        rows = ""
        for sym in self.watchlist:
            sig_info = self.stock_signals.get(sym, {})
            price = sig_info.get("price", 0.0)
            change = sig_info.get("change_pct", 0.0)
            signal = sig_info.get("signal", "HOLD")
            conf = sig_info.get("confidence", 0.50)

            chg_color = "#28a745" if change >= 0 else "#dc3545"
            sig_color = "#28a745" if signal == "BUY" else "#6c757d"
            has_pos = "🟢 LONG" if sym in self.positions else "⚪ FLAT"

            selected_badge = "👈 View" if sym == self.active_symbol else ""

            rows += f"""
            <tr style="border-bottom:1px solid #eee;{'background:#f0f8ff;' if sym == self.active_symbol else ''}">
                <td style="padding:6px 10px;font-weight:bold;">{sym} <span style="font-size:10px;color:#007bff;">{selected_badge}</span></td>
                <td style="padding:6px 10px;">₹{price:,.2f}</td>
                <td style="padding:6px 10px;color:{chg_color};font-weight:bold;">{change:+.2f}%</td>
                <td style="padding:6px 10px;color:{sig_color};font-weight:bold;">{signal}</td>
                <td style="padding:6px 10px;">{conf * 100:.1f}%</td>
                <td style="padding:6px 10px;font-weight:bold;">{has_pos}</td>
            </tr>
            """

        table = f"""
        <div style="max-height:220px;overflow-y:auto;border:1px solid #e5e5e5;border-radius:6px;">
            <table style="width:100%;border-collapse:collapse;font-size:12px;background:white;">
                <thead style="position:sticky;top:0;background:#f8f9fa;z-index:2;">
                    <tr style="text-align:left;border-bottom:2px solid #ddd;">
                        <th style="padding:6px 10px;">Symbol</th>
                        <th style="padding:6px 10px;">LTP</th>
                        <th style="padding:6px 10px;">1m Change</th>
                        <th style="padding:6px 10px;">AI Signal</th>
                        <th style="padding:6px 10px;">Confidence</th>
                        <th style="padding:6px 10px;">Position</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
        """
        self.scanner_table_html.value = table

    def _update_trade_table(self):
        trades = self.paper_trader.trade_history
        if not trades:
            self.history_html.value = "<div style='padding:6px;color:#888;'>No completed trades yet.</div>"
            return

        rows = ""
        for t in reversed(trades[-5:]):
            pnl = t["pnl"]
            pnl_color = "#28a745" if pnl >= 0 else "#dc3545"
            rows += f"""
            <tr style="border-bottom:1px solid #eee;">
                <td style='padding:5px 8px;'>{t.get('exit_time', '')}</td>
                <td style='padding:5px 8px;font-weight:bold;'>{t.get('symbol', '')}</td>
                <td style='padding:5px 8px;'>{t.get('quantity', 1)} shares</td>
                <td style='padding:5px 8px;'>₹{t.get('entry_price', 0):.2f}</td>
                <td style='padding:5px 8px;'>₹{t.get('exit_price', 0):.2f}</td>
                <td style='padding:5px 8px;font-weight:bold;color:{pnl_color};'>₹{pnl:+.2f}</td>
                <td style='padding:5px 8px;'>{t.get('reason', '')}</td>
            </tr>
            """

        self.history_html.value = f"""
        <table style='width:100%;border-collapse:collapse;font-size:12px;background:white;'>
            <thead>
                <tr style='background:#f1f3f5;text-align:left;'>
                    <th style='padding:6px 8px;'>Time</th>
                    <th style='padding:6px 8px;'>Symbol</th>
                    <th style='padding:6px 8px;'>Size</th>
                    <th style='padding:6px 8px;'>Entry</th>
                    <th style='padding:6px 8px;'>Exit</th>
                    <th style='padding:6px 8px;'>P&L</th>
                    <th style='padding:6px 8px;'>Reason</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
        """

    def scan_all_stocks_step(self):
        """Advances candles and scans all 20+ stocks for AI signals."""
        now_str = datetime.now().strftime("%H:%M:%S")

        for sym in list(self.watchlist):
            df = self.stock_data[sym]
            idx = self.stock_bar_index[sym]

            # Generate more candles if reached buffer end
            if idx >= len(df) - 1:
                new_bars = DataEngine.generate_synthetic_market_data(
                    bars=100,
                    start_price=float(df["close"].iloc[-1]),
                    seed=abs(hash(sym + str(idx))) % 10000
                )
                df = pd.concat([df, new_bars]).reset_index(drop=True)
                self.stock_data[sym] = df

            idx += 1
            self.stock_bar_index[sym] = idx

            subset = df.iloc[: idx + 1]
            feat_subset = DataEngine.calculate_technical_features(subset)
            current_bar = feat_subset.iloc[-1]
            close_price = float(current_bar["close"])
            prev_close = float(feat_subset.iloc[-2]["close"])
            change_pct = ((close_price - prev_close) / prev_close) * 100.0

            # 1. Manage open positions for this stock
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

            # 2. Check AI entry signals if not holding & not paused & kill-switch clear
            signal = "HOLD"
            proba = 0.50
            if not self.is_paused and not self.risk_manager.kill_switch_triggered:
                signal, proba = self.model.predict_signal(feat_subset)
                if signal == "BUY" and sym not in self.positions and len(self.positions) < 4:
                    qty, sl, tp = self.risk_manager.calculate_position_size(close_price)
                    self.positions[sym] = {
                        "symbol": sym,
                        "quantity": max(1, qty // 3),  # Divide risk across portfolio
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

        # Update UI
        status = "KILLED" if self.risk_manager.kill_switch_triggered else "PAUSED" if self.is_paused else "RUNNING"
        self.header_html.value = self._get_header_html(status)
        self.metrics_html.value = self._get_metrics_html()
        self._render_chart()
        self._update_scanner_table()
        self._update_trade_table()

    def _loop_worker(self):
        while self.is_running:
            self.scan_all_stocks_step()
            time.sleep(self.update_interval_sec)

    def _on_toggle_auto(self, b):
        if not self.is_running:
            self.is_running = True
            self.btn_auto.description = "⏹ Stop Loop"
            self.btn_auto.button_style = "danger"
            self._loop_thread = threading.Thread(target=self._loop_worker, daemon=True)
            self._loop_thread.start()
        else:
            self.is_running = False
            self.btn_auto.description = "▶ Start Loop"
            self.btn_auto.button_style = "success"

    def _on_step_tick(self, b):
        self.scan_all_stocks_step()

    def _on_pause_toggle(self, b):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.btn_pause.description = "▶ Resume AI Signals"
            self.btn_pause.button_style = "success"
        else:
            self.btn_pause.description = "⏸ Pause AI Signals"
            self.btn_pause.button_style = "warning"
        self.scan_all_stocks_step()

    def _on_mode_toggle(self, b):
        """Toggles between Paper Trading and Live Upstox Execution with safety checks."""
        if self.mode == "PAPER":
            # Check if token is valid before allowing Live mode
            token = self.auth.load_cached_token()
            if not token or not self.auth.validate_token(token):
                print("\033[91m[CANNOT SWITCH TO LIVE] No valid Upstox Access Token found. Staying in Paper mode.\033[0m")
                return
            self.mode = "LIVE"
            self.live_trader.dry_run = False
            self.btn_mode_toggle.button_style = "danger"
            self.btn_mode_toggle.description = "🔴 Mode: Live (Active)"
            print("\033[91m[WARNING] Switched to LIVE REAL-MONEY TRADING MODE (NSE)!\033[0m")
        else:
            self.mode = "PAPER"
            self.live_trader.dry_run = True
            self.btn_mode_toggle.button_style = "info"
            self.btn_mode_toggle.description = "🎮 Mode: Paper (Active)"
            print("\033[92m[INFO] Switched to SAFE PAPER TRADING MODE (Simulation).\033[0m")
        self.scan_all_stocks_step()

    def _on_stock_change(self, change):
        if change.get("new"):
            self.active_symbol = str(change["new"]).upper()
            self.metrics_html.value = self._get_metrics_html()
            self._render_chart()
            self._update_scanner_table()

    def _on_add_stock(self, b):
        new_sym = self.input_new_stock.value.strip().upper()
        if new_sym and new_sym not in self.watchlist:
            self.watchlist.append(new_sym)
            self._init_stock_feeds()
            self.stock_dropdown.options = list(self.watchlist)
            self.stock_dropdown.value = new_sym
            self.input_new_stock.value = ""
            print(f"[WATCHLIST] Added {new_sym} to automated multi-stock scanner.")
            self.scan_all_stocks_step()

    def _on_remove_stock(self, b):
        sym = self.active_symbol
        if len(self.watchlist) > 1 and sym in self.watchlist:
            self.watchlist.remove(sym)
            self.active_symbol = self.watchlist[0]
            self.stock_dropdown.options = list(self.watchlist)
            self.stock_dropdown.value = self.active_symbol
            print(f"[WATCHLIST] Removed {sym} from scanner.")
            self.scan_all_stocks_step()

    def _on_emergency_square_off(self, b):
        """Immediately closes all open positions across all stocks."""
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
        self._update_trade_table()
        self.scan_all_stocks_step()

    def render(self, run_loop: bool = True):
        """Renders the complete multi-stock dashboard in Google Colab."""
        if not self._has_widgets:
            print("[ERROR] ipywidgets is required to render dashboard. Run: pip install ipywidgets")
            return

        self.display(self.container)
        self.scan_all_stocks_step()

        if run_loop:
            print("[INFO] Live multi-stock scanner is running... (Click '⏹ Stop Loop' or stop cell to halt)")
            try:
                while self.is_running:
                    time.sleep(self.update_interval_sec)
                    if self.is_running:
                        self.scan_all_stocks_step()
            except KeyboardInterrupt:
                print("\n[INFO] Dashboard stopped by user.")


