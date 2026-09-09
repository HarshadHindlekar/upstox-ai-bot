from typing import Optional, Dict, Any
import requests
import config
from core.risk_manager import RiskManager
from core.auth import UpstoxAuth


class LiveTrader:
    """
    Executes live orders on Upstox API v2 with strict risk management.
    Includes dry_run guard to avoid unintended real-money execution.
    """

    ORDER_URL = "https://api.upstox.com/v2/order/place"

    def __init__(
        self,
        auth: UpstoxAuth,
        risk_manager: RiskManager,
        dry_run: bool = True,
    ):
        self.auth = auth
        self.risk_manager = risk_manager
        self.dry_run = dry_run

    def place_order(
        self,
        instrument_token: str,
        transaction_type: str,  # 'BUY' or 'SELL'
        quantity: int,
        order_type: str = "MARKET",
        price: float = 0.0,
        trigger_price: float = 0.0,
        product: str = "I",  # 'I' for Intraday (MIS), 'D' for Delivery (CNC)
    ) -> Dict[str, Any]:
        """Places an order through Upstox API v2."""
        if not self.risk_manager.can_open_new_trade():
            print("[LIVE TRADER] Blocked: Risk manager or daily kill-switch is active.")
            return {"status": "blocked", "reason": "kill_switch_active"}

        payload = {
            "quantity": quantity,
            "product": product,
            "validity": "DAY",
            "price": price if order_type in ["LIMIT", "SL"] else 0.0,
            "tag": "upstox_ai_bot",
            "instrument_token": instrument_token,
            "order_type": order_type,
            "transaction_type": transaction_type.upper(),
            "disclosed_quantity": 0,
            "trigger_price": trigger_price if order_type in ["SL", "SL-M"] else 0.0,
            "is_amo": False,
        }

        if self.dry_run:
            print(f"[LIVE TRADER (DRY RUN)] Simulated order submission: {payload}")
            return {"status": "dry_run_success", "payload": payload}

        token = self.auth.get_valid_token()
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

        response = requests.post(self.ORDER_URL, json=payload, headers=headers, timeout=10)
        res_data = response.json()

        if response.status_code == 200 and res_data.get("status") == "success":
            print(f"[LIVE TRADER] Real Order Placed! Order ID: {res_data.get('data', {}).get('order_id')}")
            return res_data
        else:
            print(f"[LIVE TRADER ERROR] Failed to place order: {res_data}")
            return res_data
