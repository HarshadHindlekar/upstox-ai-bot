import datetime
from typing import Optional
import numpy as np
import pandas as pd
import requests
import config


class DataEngine:
    """Handles market data acquisition from Upstox API and feature engineering."""

    BASE_URL = "https://api.upstox.com/v2"

    def __init__(self, access_token: Optional[str] = None):
        self.access_token = access_token or config.UPSTOX_ACCESS_TOKEN

    def fetch_historical_candles(
        self,
        instrument_key: str = config.DEFAULT_INSTRUMENT_KEY,
        interval: str = "1minute",
        to_date: Optional[str] = None,
        from_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetches historical OHLCV candle data from Upstox API v2.
        Dates in YYYY-MM-DD format.
        """
        if to_date is None:
            to_date = datetime.datetime.now().strftime("%Y-%m-%d")
        if from_date is None:
            from_date = (datetime.datetime.now() - datetime.timedelta(days=7)).strftime("%Y-%m-%d")

        url = f"{self.BASE_URL}/historical-candle/{instrument_key}/{interval}/{to_date}/{from_date}"
        headers = {"Accept": "application/json"}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"

        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to fetch historical candles (HTTP {response.status_code}): {response.text}"
            )

        data = response.json()
        candles = data.get("data", {}).get("candles", [])
        if not candles:
            return pd.DataFrame()

        # Upstox returns: [timestamp, open, high, low, close, volume, open_interest]
        df = pd.DataFrame(
            candles,
            columns=["timestamp", "open", "high", "low", "close", "volume", "oi"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    @staticmethod
    def generate_synthetic_market_data(
        bars: int = 1500,
        start_price: float = 2500.0,
        freq: str = "1min",
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Generates realistic synthetic OHLCV intraday market data (Geometric Brownian Motion + Jump Diffusion).
        Essential for testing models and running Colab demos without requiring live credentials.
        """
        np.random.seed(seed)
        dt = 1 / (252 * 375)  # Intraday minutes
        mu = 0.05
        sigma = 0.22

        prices = [start_price]
        for _ in range(bars - 1):
            shock = np.random.normal(0, 1)
            # Add occasional market jumps
            jump = np.random.choice([0, 1], p=[0.98, 0.02]) * np.random.normal(0, 0.015)
            price_next = prices[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shock + jump)
            prices.append(max(price_next, 1.0))

        start_time = datetime.datetime.now() - datetime.timedelta(minutes=bars)
        timestamps = [start_time + datetime.timedelta(minutes=i) for i in range(bars)]

        df = pd.DataFrame({"timestamp": timestamps, "close": prices})
        # Build OHLC from synthetic close
        noise_high = np.random.uniform(0.0005, 0.003, size=bars)
        noise_low = np.random.uniform(0.0005, 0.003, size=bars)
        noise_open = np.random.uniform(-0.001, 0.001, size=bars)

        df["open"] = df["close"] * (1 + noise_open)
        df["high"] = np.maximum(df["open"], df["close"]) * (1 + noise_high)
        df["low"] = np.minimum(df["open"], df["close"]) * (1 - noise_low)
        df["volume"] = np.random.randint(5000, 75000, size=bars)
        df["oi"] = 0

        return df[["timestamp", "open", "high", "low", "close", "volume", "oi"]]

    @staticmethod
    def calculate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates alpha indicators and feature representations for Machine Learning:
        - Exponential Moving Averages (EMA 9, 21, 50)
        - RSI (14)
        - MACD & Signal line
        - Bollinger Bands
        - Average True Range (ATR 14)
        - Volume Moving Average Ratio
        - High-Low range and Log Returns
        """
        df = df.copy()

        # Returns
        df["returns_1"] = df["close"].pct_change(1)
        df["returns_5"] = df["close"].pct_change(5)

        # EMAs
        df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
        df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()
        df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()

        # Moving average distance
        df["dist_ema_9"] = (df["close"] - df["ema_9"]) / df["ema_9"]
        df["dist_ema_21"] = (df["close"] - df["ema_21"]) / df["ema_21"]
        df["ema_cross_9_21"] = (df["ema_9"] - df["ema_21"]) / df["ema_21"]

        # RSI (14)
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss.replace(0, np.nan))
        df["rsi_14"] = 100 - (100 / (1 + rs))
        df["rsi_14"] = df["rsi_14"].fillna(50)

        # MACD (12, 26, 9)
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # Bollinger Bands (20, 2)
        bb_middle = df["close"].rolling(window=20).mean()
        bb_std = df["close"].rolling(window=20).std()
        df["bb_upper"] = bb_middle + (2 * bb_std)
        df["bb_lower"] = bb_middle - (2 * bb_std)
        df["bb_pct_b"] = (df["close"] - df["bb_lower"]) / ((df["bb_upper"] - df["bb_lower"]).replace(0, np.nan))
        df["bb_pct_b"] = df["bb_pct_b"].fillna(0.5)

        # Average True Range (ATR 14)
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr_14"] = true_range.rolling(window=14).mean()
        df["atr_pct"] = df["atr_14"] / df["close"]

        # Volume ratio
        vol_ma = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / (vol_ma.replace(0, np.nan))
        df["volume_ratio"] = df["volume_ratio"].fillna(1.0)

        # Drop warm-up NaN rows
        df = df.dropna().reset_index(drop=True)
        return df

    @staticmethod
    def create_labels(
        df: pd.DataFrame,
        lookahead_bars: int = 5,
        profit_target_pct: float = 0.005,
        stop_loss_pct: float = 0.0025,
    ) -> pd.DataFrame:
        """
        Creates trading classification targets:
        1 = Buy signal (future high hits profit target before low hits stop loss)
        0 = Hold / Neutral / Downward movement
        """
        df = df.copy()
        labels = []

        n = len(df)
        for i in range(n):
            if i + lookahead_bars >= n:
                labels.append(0)
                continue

            current_close = df.loc[i, "close"]
            future_highs = df.loc[i + 1 : i + lookahead_bars, "high"]
            future_lows = df.loc[i + 1 : i + lookahead_bars, "low"]

            target_price = current_close * (1 + profit_target_pct)
            stop_price = current_close * (1 - stop_loss_pct)

            hit_target = (future_highs >= target_price).any()
            hit_stop = (future_lows <= stop_price).any()

            if hit_target and not hit_stop:
                labels.append(1)
            else:
                labels.append(0)

        df["target"] = labels
        return df
