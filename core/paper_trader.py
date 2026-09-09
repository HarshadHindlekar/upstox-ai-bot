import time
from datetime import datetime
from typing import Optional, Dict, Any, List
import pandas as pd
import config
from core.model import TradingModel
from core.risk_manager import RiskManager


class PaperTrader:
    """Simulates live execution bar-by-bar with virtual capital."""

    def __init__(
        self,
        model: TradingModel,
        risk_manager: RiskManager,
        symbol: str = config.DEFAULT_SYMBOL,
    ):
        self.model = model
        self.risk_manager = risk_manager
        self.symbol = symbol
        self.current_position: Optional[Dict[str, Any]] = None
        self.trade_history: List[Dict[str, Any]] = []
        self.journal_file = config.LOGS_DIR / "paper_trading_journal.csv"

    def process_new_bar(self, bar_data: pd.Series, feature_df: pd.DataFrame):
        """Processes each arriving candle in paper trading mode."""
        timestamp = bar_data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        close_price = float(bar_data["close"])
        high_price = float(bar_data["high"])
        low_price = float(bar_data["low"])

        # 1. Manage existing open position
        if self.current_position:
            pos = self.current_position
            entry_price = pos["entry_price"]
            qty = pos["quantity"]
            sl = pos["stop_loss"]
            tp = pos["take_profit"]

            exit_price = None
            reason = None

            if low_price <= sl:
                exit_price = sl
                reason = "STOP_LOSS_HIT"
            elif high_price >= tp:
                exit_price = tp
                reason = "TAKE_PROFIT_HIT"

            if exit_price is not None:
                pnl = (exit_price - entry_price) * qty
                self.risk_manager.update_pnl(pnl)

                record = {
                    "symbol": self.symbol,
                    "action": "SELL",
                    "quantity": qty,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": round(pnl, 2),
                    "reason": reason,
                    "entry_time": pos["entry_time"],
                    "exit_time": str(timestamp),
                    "daily_pnl_so_far": round(self.risk_manager.daily_realized_pnl, 2),
                }
                self.trade_history.append(record)
                self._save_journal()
                print(
                    f"[{timestamp}] [PAPER EXIT] {reason} | Price: {exit_price:.2f} | "
                    f"PnL: ₹{pnl:+.2f} | Daily PnL: ₹{self.risk_manager.daily_realized_pnl:+.2f}"
                )
                self.current_position = None

        # 2. Check for new trade signal if no position is held
        if self.current_position is None:
            if not self.risk_manager.can_open_new_trade():
                return

            signal, proba = self.model.predict_signal(feature_df)
            if signal == "BUY":
                qty, sl, tp = self.risk_manager.calculate_position_size(close_price)
                self.current_position = {
                    "symbol": self.symbol,
                    "action": "BUY",
                    "quantity": qty,
                    "entry_price": close_price,
                    "stop_loss": sl,
                    "take_profit": tp,
                    "entry_time": str(timestamp),
                }
                print(
                    f"[{timestamp}] [PAPER ENTRY] BUY {qty}x {self.symbol} @ {close_price:.2f} | "
                    f"SL: {sl:.2f} | TP: {tp:.2f} | AI Conf: {proba * 100:.1f}%"
                )

    def _save_journal(self):
        """Saves paper trading history to persistent disk / Google Drive."""
        config.init_storage()
        df = pd.DataFrame(self.trade_history)
        df.to_csv(self.journal_file, index=False)

    def get_summary(self) -> Dict[str, Any]:
        """Returns summary metrics for the paper trading session."""
        total_pnl = self.risk_manager.daily_realized_pnl
        trades = len(self.trade_history)
        wins = sum(1 for t in self.trade_history if t["pnl"] > 0)
        win_rate = (wins / trades * 100.0) if trades > 0 else 0.0

        return {
            "total_trades": trades,
            "win_rate_pct": round(win_rate, 2),
            "realized_pnl": round(total_pnl, 2),
            "kill_switch_triggered": self.risk_manager.kill_switch_triggered,
            "journal_file": str(self.journal_file),
        }
