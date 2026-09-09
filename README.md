# 🤖 Upstox AI Algorithmic Trading Bot

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HarshadHindlekar/upstox-ai-bot/blob/main/notebooks/upstox_ai_colab.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning algorithmic trading system integrated with **Upstox API v2**. Built for execution on **Google Colab** with persistent model & log storage in **Google Drive**, as well as local machines.

---

## 🌟 Key Highlights

- **⚡ Google Colab Ready (1-Click Run)**: Train, backtest, and run without local GPU or software setup.
- **💾 Google Drive Persistence**: Automatically mounts Google Drive to `/MyDrive/upstox_ai_bot/` to store trained LightGBM models (`.joblib`), datasets, and trade logs permanently across Colab sessions.
- **🧠 Quantitative Machine Learning**: Features engineered across multi-timeframe EMA, RSI, MACD, Bollinger Bands, and ATR volatility with a high-precision **LightGBM** classifier.
- **🛡️ Built-in Risk Management**:
  - Auto Stop-Loss & Take-Profit calculation.
  - Position sizing based on account capital and risk percentage.
  - **Hard Daily Loss Kill-Switch**: Automatically halts trading if daily drawdown crosses threshold.
- **🖥️ In-Colab Live Interactive Dashboard**: Renders real-time candlestick & EMA charts, live P&L counters, and an **Emergency Square-Off** button directly in the notebook cell — actively streaming updates to prevent Colab idle timeouts!
- **🎮 Safe Simulation Mode**: Default Paper Trading engine lets you test strategies with zero real capital risk.

---

## 🏗️ Architecture

```
                                  +-----------------------+
                                  | Upstox API v2 / Data  |
                                  +-----------------------+
                                              |
                                              v
+------------------------+        +-----------------------+
|  Feature Engineering   | -----> |  LightGBM AI Engine   |
| (EMA, RSI, MACD, ATR)  |        |  (Signal Probability) |
+------------------------+        +-----------------------+
                                              |
                                              v
                                  +-----------------------+
                                  |  Risk Manager         |
                                  |  - Position Sizing    |
                                  |  - Daily Kill Switch  |
                                  +-----------------------+
                                              |
                          +-------------------+-------------------+
                          |                                       |
                          v                                       v
              [Paper Trading Engine]                    [Live Upstox API]
              (Simulation / Colab)                      (Guarded by dry_run)
                          |                                       |
                          +-------------------+-------------------+
                                              |
                                              v
                                  +-----------------------+
                                  | Google Drive Storage  |
                                  | (Models & Trade Logs) |
                                  +-----------------------+
```

---

## 🚀 Quickstart: Run in Google Colab (Single Command)

You can either click the **Open in Colab** badge above, or open a blank Google Colab notebook and run:

```bash
!git clone https://github.com/HarshadHindlekar/upstox-ai-bot.git
%cd upstox-ai-bot
!pip install -q -r requirements.txt
!python run.py --mode paper --sample-data
```

### Running Specific Modes in Colab / Terminal:

```bash
# 1. Train the LightGBM model and persist to Google Drive
python run.py --mode train --sample-data --bars 2000

# 2. Backtest with slippage & Upstox brokerage fees
python run.py --mode backtest --sample-data --bars 1500

# 3. Launch Paper Trading Simulation
python run.py --mode paper --sample-data

# 4. Dry-run live order placement test
python run.py --mode live
```

---

## 💻 Local Machine Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/HarshadHindlekar/upstox-ai-bot.git
   cd upstox-ai-bot
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux / macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your environment variables:**
   ```bash
   cp .env.example .env
   ```
   Open `.env` and configure your `UPSTOX_API_KEY`, `UPSTOX_API_SECRET`, and trading preferences.

---

## 📁 Project Structure

```
upstox-ai-bot/
├── .gitignore                      # Prevents committing tokens & keys
├── .env.example                    # Template for Upstox credentials
├── requirements.txt                # Pinned dependencies
├── run.py                          # Unified CLI entrypoint
├── config.py                       # Environment & Colab/Drive detection
├── core/
│   ├── __init__.py
│   ├── auth.py                     # Upstox v2 OAuth & TOTP token manager
│   ├── data_engine.py              # Candle ingestion & technical indicators
│   ├── model.py                    # LightGBM training, evaluation & persistence
│   ├── risk_manager.py             # Position sizing & daily loss kill-switch
│   ├── backtester.py               # Historical backtesting simulation
│   ├── paper_trader.py             # Real-time paper trading engine
│   └── live_trader.py              # Upstox API live order execution
├── notebooks/
│   └── upstox_ai_colab.ipynb       # One-click Google Colab notebook
└── README.md
```

---

## ⚠️ Disclaimer & Risk Notice

This software is developed for educational, research, and algorithmic testing purposes. Financial trading involves substantial risk of capital loss.
- Always validate models using extensive backtesting and paper trading before deploying real capital.
- The authors and contributors assume no responsibility or liability for financial losses incurred through the use of this code.
