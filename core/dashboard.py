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


class ColabTradingDashboard:
    """
    Interactive Real-Time Trading UI designed for Google Colab.
    Uses ipywidgets to render an active dashboard inside notebook cells,
    keeping the Colab connection alive while providing visual controls & live charts.
    """

    def __init__(
        self,
        symbol: str = config.DEFAULT_SYMBOL,
        use_sample_data: bool = True,
        update_interval_sec: float = 3.0,
    ):
        self.symbol = symbol
        self.use_sample_data = use_sample_data
        self.update_interval_sec = update_interval_sec

        self.model = TradingModel()
        if not self.model.load_model():
            print("[DASHBOARD] Training base AI model for dashboard...")
            sample_df = DataEngine.generate_synthetic_market_data(bars=1200)
            feat_df = DataEngine.calculate_technical_features(sample_df)
            labeled_df = DataEngine.create_labels(feat_df)
            self.model.train(labeled_df)

        self.risk_manager = RiskManager()
        self.paper_trader = PaperTrader(
            model=self.model,
            risk_manager=self.risk_manager,
            symbol=self.symbol,
        )

        # Simulation data buffer
        self.raw_data = DataEngine.generate_synthetic_market_data(bars=350, seed=42)
        self.current_bar_index = 60
        self.is_running = False
        self.is_paused = False
        self._loop_thread: Optional[threading.Thread] = None

        # Try to import ipywidgets & IPython
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

    def _build_widgets(self):
        if not self._has_widgets:
            return

        w = self.widgets

        # Header Card
        self.header_html = w.HTML(
            value=self._get_header_html("INITIALIZING")
        )

        # Metrics Card
        self.metrics_html = w.HTML(
            value=self._get_metrics_html(2500.0, "HOLD", 0.50)
        )

        # Chart Output area
        self.chart_output = w.Output(layout=w.Layout(height="340px", width="100%"))

        # Trade History Table
        self.history_html = w.HTML(
            value="<div style='padding:8px;color:#888;'>No trades executed yet.</div>"
        )

        # Action Buttons
        self.btn_auto = w.Button(
            description="▶ Start Live Loop",
            button_style="success",
            tooltip="Starts the live bar streaming loop to keep Colab alive",
            icon="play",
            layout=w.Layout(width="180px")
        )
        self.btn_auto.on_click(self._on_toggle_auto)

        self.btn_step = w.Button(
            description="Step Next Tick",
            button_style="info",
            tooltip="Processes a single 1-minute candle tick",
            icon="step-forward",
            layout=w.Layout(width="150px")
        )
        self.btn_step.on_click(self._on_step_tick)

        self.btn_pause = w.Button(
            description="Pause AI Signals",
            button_style="warning",
            tooltip="Temporarily pause the bot from opening new positions",
            icon="pause",
            layout=w.Layout(width="160px")
        )
        self.btn_pause.on_click(self._on_pause_toggle)

        self.btn_square_off = w.Button(
            description="🛑 Square-Off All",
            button_style="danger",
            tooltip="Emergency exit: closes all open positions immediately",
            icon="times-circle",
            layout=w.Layout(width="180px")
        )
        self.btn_square_off.on_click(self._on_emergency_square_off)

        # Control Row
        self.controls_box = w.HBox(
            [self.btn_auto, self.btn_step, self.btn_pause, self.btn_square_off],
            layout=w.Layout(margin="10px 0px 10px 0px")
        )

        # Main Layout Container
        self.container = w.VBox([
            self.header_html,
            self.metrics_html,
            self.controls_box,
            self.chart_output,
            w.HTML("<h4 style='margin:10px 0 5px 0;color:#333;'>Recent Trade Executions:</h4>"),
            self.history_html,
        ], layout=w.Layout(
            border="1px solid #ddd",
            padding="15px",
            border_radius="10px",
            background_color="#fcfcfc"
        ))

    def _get_header_html(self, status: str) -> str:
        color = "#28a745" if status in ["ACTIVE", "RUNNING"] else "#dc3545" if status == "KILLED" else "#ffc107"
        return f"""
        <div style="display:flex;justify-content:space-between;align-items:center;background:#1a1d20;color:white;padding:12px 20px;border-radius:8px;margin-bottom:10px;">
            <div style="font-size:18px;font-weight:bold;">
                🤖 Upstox AI Live Trading Terminal
            </div>
            <div style="display:flex;gap:15px;font-size:13px;align-items:center;">
                <span>Instrument: <b>{self.symbol}</b></span>
                <span>Market: <b>NSE (India)</b></span>
                <span style="background:{color};padding:3px 10px;border-radius:12px;font-weight:bold;color:white;">{status}</span>
            </div>
        </div>
        """

    def _get_metrics_html(self, current_price: float, signal: str, confidence: float) -> str:
        pnl = self.risk_manager.daily_realized_pnl
        pnl_color = "#28a745" if pnl >= 0 else "#dc3545"
        pos = self.paper_trader.current_position

        pos_str = "None (Flat)"
        sl_tp_str = "N/A"
        if pos:
            pos_str = f"LONG {pos['quantity']}x @ ₹{pos['entry_price']:.2f}"
            sl_tp_str = f"SL: ₹{pos['stop_loss']:.2f} | TP: ₹{pos['take_profit']:.2f}"

        return f"""
        <div style="display:grid;grid-template-columns:repeat(4, 1fr);gap:10px;margin-bottom:10px;">
            <div style="background:#fff;border:1px solid #e0e0e0;padding:10px 14px;border-radius:8px;">
                <div style="font-size:12px;color:#777;">Daily Realized P&L</div>
                <div style="font-size:20px;font-weight:bold;color:{pnl_color};">₹{pnl:+,.2f}</div>
                <div style="font-size:11px;color:#999;">Limit: -₹{self.risk_manager.max_daily_loss:,.2f}</div>
            </div>
            <div style="background:#fff;border:1px solid #e0e0e0;padding:10px 14px;border-radius:8px;">
                <div style="font-size:12px;color:#777;">Current LTP</div>
                <div style="font-size:20px;font-weight:bold;color:#222;">₹{current_price:,.2f}</div>
                <div style="font-size:11px;color:#999;">Symbol: {self.symbol}</div>
            </div>
            <div style="background:#fff;border:1px solid #e0e0e0;padding:10px 14px;border-radius:8px;">
                <div style="font-size:12px;color:#777;">Active Position</div>
                <div style="font-size:14px;font-weight:bold;color:#007bff;margin-top:3px;">{pos_str}</div>
                <div style="font-size:11px;color:#666;">{sl_tp_str}</div>
            </div>
            <div style="background:#fff;border:1px solid #e0e0e0;padding:10px 14px;border-radius:8px;">
                <div style="font-size:12px;color:#777;">AI Prediction</div>
                <div style="font-size:18px;font-weight:bold;color:{'#28a745' if signal == 'BUY' else '#6c757d'};">
                    {signal} ({confidence * 100:.1f}%)
                </div>
                <div style="font-size:11px;color:#999;">Threshold: {self.model.confidence_threshold * 100:.0f}%</div>
            </div>
        </div>
        """

    def _render_chart(self, featured_subset: pd.DataFrame):
        """Renders live intraday chart with technical indicators."""
        if not self._has_widgets:
            return

        with self.chart_output:
            self.clear_output(wait=True)
            plot_df = featured_subset.tail(35).copy()

            fig, ax = plt.subplots(figsize=(10, 3.2), dpi=100)
            ax.plot(plot_df.index, plot_df["close"], label="Close Price", color="#1f77b4", lw=1.8)
            if "ema_9" in plot_df.columns:
                ax.plot(plot_df.index, plot_df["ema_9"], label="EMA 9", color="#ff7f0e", lw=1.2, ls="--")
            if "ema_21" in plot_df.columns:
                ax.plot(plot_df.index, plot_df["ema_21"], label="EMA 21", color="#2ca02c", lw=1.2, ls=":")

            ax.set_title(f"Live Price & Moving Averages - {self.symbol}", fontsize=11, fontweight="bold")
            ax.grid(True, alpha=0.3, ls="--")
            ax.legend(loc="upper left", fontsize=8)
            plt.tight_layout()
            plt.show()

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
            <tr>
                <td style='padding:5px 8px;border-bottom:1px solid #eee;'>{t['exit_time']}</td>
                <td style='padding:5px 8px;border-bottom:1px solid #eee;'>{t['quantity']}x {t['symbol']}</td>
                <td style='padding:5px 8px;border-bottom:1px solid #eee;'>₹{t['entry_price']:.2f}</td>
                <td style='padding:5px 8px;border-bottom:1px solid #eee;'>₹{t['exit_price']:.2f}</td>
                <td style='padding:5px 8px;border-bottom:1px solid #eee;font-weight:bold;color:{pnl_color};'>₹{pnl:+.2f}</td>
                <td style='padding:5px 8px;border-bottom:1px solid #eee;'>{t['reason']}</td>
            </tr>
            """

        table = f"""
        <table style='width:100%;border-collapse:collapse;font-size:12px;background:white;'>
            <thead>
                <tr style='background:#f1f3f5;text-align:left;'>
                    <th style='padding:6px 8px;'>Time</th>
                    <th style='padding:6px 8px;'>Order</th>
                    <th style='padding:6px 8px;'>Entry</th>
                    <th style='padding:6px 8px;'>Exit</th>
                    <th style='padding:6px 8px;'>P&L</th>
                    <th style='padding:6px 8px;'>Reason</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
        """
        self.history_html.value = table

    def step_tick(self):
        """Advances simulation by 1 bar and updates all widgets."""
        if self.current_bar_index >= len(self.raw_data) - 1:
            # Generate more synthetic bars seamlessly
            new_bars = DataEngine.generate_synthetic_market_data(bars=100, start_price=float(self.raw_data["close"].iloc[-1]))
            self.raw_data = pd.concat([self.raw_data, new_bars]).reset_index(drop=True)

        self.current_bar_index += 1
        subset = self.raw_data.iloc[: self.current_bar_index + 1]
        featured_subset = DataEngine.calculate_technical_features(subset)

        current_bar = featured_subset.iloc[-1]
        current_price = float(current_bar["close"])

        # Check signals if not paused
        signal = "HOLD"
        proba = 0.50
        if not self.is_paused and not self.risk_manager.kill_switch_triggered:
            signal, proba = self.model.predict_signal(featured_subset)
            self.paper_trader.process_new_bar(current_bar, featured_subset)
        elif self.paper_trader.current_position is not None:
            # Manage existing position exits even if paused
            self.paper_trader.process_new_bar(current_bar, featured_subset)

        # Update UI state
        status = "KILLED" if self.risk_manager.kill_switch_triggered else "PAUSED" if self.is_paused else "RUNNING"
        self.header_html.value = self._get_header_html(status)
        self.metrics_html.value = self._get_metrics_html(current_price, signal, proba)
        self._render_chart(featured_subset)
        self._update_trade_table()

    def _on_step_tick(self, b):
        self.step_tick()

    def _on_pause_toggle(self, b):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.btn_pause.description = "▶ Resume AI"
            self.btn_pause.button_style = "success"
        else:
            self.btn_pause.description = "⏸ Pause AI"
            self.btn_pause.button_style = "warning"
        self.step_tick()

    def _on_emergency_square_off(self, b):
        """Immediately closes any active positions."""
        pos = self.paper_trader.current_position
        if pos:
            current_close = float(self.raw_data.iloc[self.current_bar_index]["close"])
            pnl = (current_close - pos["entry_price"]) * pos["quantity"]
            self.risk_manager.update_pnl(pnl)
            record = {
                "symbol": self.symbol,
                "action": "SELL",
                "quantity": pos["quantity"],
                "entry_price": pos["entry_price"],
                "exit_price": current_close,
                "pnl": round(pnl, 2),
                "reason": "EMERGENCY_SQUARE_OFF",
                "entry_time": pos["entry_time"],
                "exit_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "daily_pnl_so_far": round(self.risk_manager.daily_realized_pnl, 2),
            }
            self.paper_trader.trade_history.append(record)
            self.paper_trader.current_position = None
            self._update_trade_table()
            self.step_tick()

    def _loop_worker(self):
        while self.is_running:
            self.step_tick()
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
            self.btn_auto.description = "▶ Start Live Loop"
            self.btn_auto.button_style = "success"

    def render(self):
        """Displays the interactive dashboard in Google Colab."""
        if not self._has_widgets:
            print("[ERROR] ipywidgets is required to render dashboard. Run: pip install ipywidgets")
            return
        self.display(self.container)
        self.step_tick()
