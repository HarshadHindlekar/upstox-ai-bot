from typing import Dict, Any, List
import pandas as pd
import numpy as np
import config
from core.risk_manager import RiskManager
from core.model import TradingModel


class Backtester:
    """Simulates trading strategy over historical candle data with slippage & fees."""

    def __init__(
        self,
        model: TradingModel,
        risk_manager: RiskManager,
        slippage_pct: float = 0.0003,  # 0.03% slippage
        brokerage_per_order: float = 20.0,  # ₹20 flat Upstox intraday brokerage
    ):
        self.model = model
        self.risk_manager = risk_manager
        self.slippage_pct = slippage_pct
        self.brokerage_per_order = brokerage_per_order

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Runs the backtest over the provided DataFrame."""
        trades: List[Dict[str, Any]] = []
        equity_curve: List[float] = [self.risk_manager.max_capital]
        current_capital = self.risk_manager.max_capital

        position = None  # None or dict(entry_price, qty, sl, tp, entry_time)

        for i in range(len(df)):
            row = df.iloc[i]
            timestamp = row["timestamp"]
            current_close = float(row["close"])
            high = float(row["high"])
            low = float(row["low"])

            # Check open position exit first
            if position is not None:
                qty = position["qty"]
                entry_price = position["entry_price"]
                sl = position["sl"]
                tp = position["tp"]

                exit_price = None
                exit_reason = None

                if low <= sl:
                    exit_price = sl * (1 - self.slippage_pct)
                    exit_reason = "STOP_LOSS"
                elif high >= tp:
                    exit_price = tp * (1 - self.slippage_pct)
                    exit_reason = "TAKE_PROFIT"
                elif i == len(df) - 1:  # End of backtest
                    exit_price = current_close * (1 - self.slippage_pct)
                    exit_reason = "MARKET_CLOSE"

                if exit_price is not None:
                    gross_pnl = (exit_price - entry_price) * qty
                    fees = (self.brokerage_per_order * 2) + (exit_price * qty * 0.00025)  # STT & fees
                    net_pnl = gross_pnl - fees
                    current_capital += net_pnl
                    equity_curve.append(current_capital)

                    trades.append({
                        "entry_time": position["entry_time"],
                        "exit_time": timestamp,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "qty": qty,
                        "gross_pnl": round(gross_pnl, 2),
                        "fees": round(fees, 2),
                        "net_pnl": round(net_pnl, 2),
                        "exit_reason": exit_reason,
                        "capital_after": round(current_capital, 2),
                    })
                    position = None

            # Look for new entry if no open position
            if position is None and i < len(df) - 1:
                # Predict signal using features up to this row
                feat_subset = df.iloc[: i + 1]
                signal, proba = self.model.predict_signal(feat_subset)

                if signal == "BUY" and self.risk_manager.can_open_new_trade():
                    qty, sl, tp = self.risk_manager.calculate_position_size(current_close)
                    entry_price = current_close * (1 + self.slippage_pct)
                    position = {
                        "entry_price": entry_price,
                        "qty": qty,
                        "sl": sl,
                        "tp": tp,
                        "entry_time": timestamp,
                    }

        # Analyze performance
        trades_df = pd.DataFrame(trades)
        if trades_df.empty:
            return {"total_trades": 0, "net_pnl": 0.0, "win_rate": 0.0}

        # Save trade log to persistent drive/local logs
        config.init_storage()
        log_path = config.LOGS_DIR / "backtest_journal.csv"
        trades_df.to_csv(log_path, index=False)

        wins = trades_df[trades_df["net_pnl"] > 0]
        losses = trades_df[trades_df["net_pnl"] < 0]
        win_rate = (len(wins) / len(trades_df)) * 100.0

        total_net_pnl = trades_df["net_pnl"].sum()
        gross_profit = wins["net_pnl"].sum()
        gross_loss = abs(losses["net_pnl"].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.nan

        # Max Drawdown
        equity_series = pd.Series(equity_curve)
        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak
        max_drawdown_pct = float(drawdown.min() * 100.0)

        return {
            "total_trades": len(trades_df),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate_pct": round(win_rate, 2),
            "initial_capital": self.risk_manager.max_capital,
            "final_capital": round(current_capital, 2),
            "net_pnl": round(total_net_pnl, 2),
            "profit_factor": round(profit_factor, 2) if not np.isnan(profit_factor) else "N/A",
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "journal_file": str(log_path),
        }
