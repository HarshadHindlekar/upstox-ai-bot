from dataclasses import dataclass
from typing import Optional, Tuple
import config


@dataclass
class TradeOrder:
    symbol: str
    action: str  # BUY or SELL
    quantity: int
    entry_price: float
    stop_loss: float
    take_profit: float
    timestamp: str


class RiskManager:
    """Strict risk controls, position sizing, and maximum daily loss kill-switch."""

    def __init__(
        self,
        max_capital: float = config.MAX_CAPITAL,
        max_daily_loss: float = config.MAX_DAILY_LOSS,
        risk_per_trade_pct: float = config.RISK_PER_TRADE_PERCENT,
        stop_loss_pct: float = config.STOP_LOSS_PERCENT,
        take_profit_pct: float = config.TAKE_PROFIT_PERCENT,
    ):
        self.max_capital = max_capital
        self.max_daily_loss = max_daily_loss
        self.risk_per_trade_pct = risk_per_trade_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self.daily_realized_pnl: float = 0.0
        self.daily_trades_count: int = 0
        self.kill_switch_triggered: bool = False

    def check_kill_switch(self) -> bool:
        """
        Returns True if daily drawdown exceeds threshold.
        Prevents blown accounts due to consecutive losses or sudden market flash crashes.
        """
        if self.daily_realized_pnl <= -self.max_daily_loss:
            if not self.kill_switch_triggered:
                print(
                    f"\n[CRITICAL KILL-SWITCH TRIGGERED] Daily loss ₹{-self.daily_realized_pnl:.2f} "
                    f"exceeded maximum allowed ₹{self.max_daily_loss:.2f}! Halting all trades for today.\n"
                )
                self.kill_switch_triggered = True
            return True
        return False

    def can_open_new_trade(self) -> bool:
        """Determines if the bot is allowed to enter a new position."""
        if self.check_kill_switch():
            return False
        return True

    def calculate_position_size(self, current_price: float) -> Tuple[int, float, float]:
        """
        Calculates safe share quantity, stop-loss price, and take-profit price.
        Risk is capped at risk_per_trade_pct of available capital.
        """
        stop_loss_price = round(current_price * (1.0 - (self.stop_loss_pct / 100.0)), 2)
        take_profit_price = round(current_price * (1.0 + (self.take_profit_pct / 100.0)), 2)

        risk_amount_per_share = max(current_price - stop_loss_price, 0.01)
        total_risk_budget = self.max_capital * (self.risk_per_trade_pct / 100.0)

        # Quantity based on risk budget
        qty = int(total_risk_budget // risk_amount_per_share)

        # Ensure total position value doesn't exceed max available capital
        max_shares_affordable = int(self.max_capital // current_price)
        final_quantity = max(1, min(qty, max_shares_affordable))

        return final_quantity, stop_loss_price, take_profit_price

    def update_pnl(self, trade_pnl: float):
        """Updates realized daily PnL after a closed trade."""
        self.daily_realized_pnl += trade_pnl
        self.daily_trades_count += 1
        self.check_kill_switch()
