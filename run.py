#!/usr/bin/env python3
"""
Upstox AI Trading Bot - Master Unified CLI & Runner
Author: Harshad Hindlekar
Compatible with Google Colab (Google Drive persistence) and local machines.
"""

import sys
import argparse
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

import config
from core.auth import UpstoxAuth
from core.data_engine import DataEngine
from core.model import TradingModel
from core.risk_manager import RiskManager
from core.backtester import Backtester
from core.paper_trader import PaperTrader
from core.live_trader import LiveTrader

console = Console()


def print_banner():
    banner = """
  _   _           _                 _    ___   ____        _   
 | | | |_ __  ___| |_ _____  __    / \  |_ _| | __ )  ___ | |_ 
 | | | | '_ \/ __| __/ _ \ \/ /   / _ \  | |  |  _ \ / _ \| __|
 | |_| | |_) \__ \ || (_) >  <   / ___ \ | |  | |_) | (_) | |_ 
  \___/| .__/|___/\__\___/_/\_\ /_/   \_\___| |____/ \___/ \__|
       |_|                                                      
    [bold cyan]Upstox AI Algorithmic Trading Bot[/bold cyan] | [green]Colab & Google Drive Enabled[/green]
    """
    console.print(banner)


def handle_auth():
    """Handles Upstox authentication."""
    auth = UpstoxAuth()
    cached = auth.load_cached_token()
    if cached and auth.validate_token(cached):
        console.print("[bold green][✓] Upstox Access Token is currently active and valid![/bold green]")
        return

    login_url = auth.get_login_url()
    console.print("\n[bold yellow]Upstox Authorization Required[/bold yellow]")
    console.print(f"1. Open this URL in your browser:\n[link={login_url}]{login_url}[/link]\n")
    console.print("2. Log in with your Upstox credentials.")
    console.print("3. After login, copy the 'code' parameter from the redirected URL in your browser address bar.")

    totp = auth.generate_current_totp()
    if totp:
        console.print(f"[bold cyan]Generated 2FA TOTP Code:[/bold cyan] {totp}")

    code = input("\nEnter the authorization code: ").strip()
    if code:
        try:
            token_data = auth.exchange_code_for_token(code)
            console.print("[bold green][✓] Token acquired and saved successfully![/bold green]")
        except Exception as e:
            console.print(f"[bold red][✗] Token exchange failed:[/bold red] {e}")


def handle_train(use_sample_data: bool, bars: int):
    """Trains the LightGBM predictive model and persists it to Google Drive / local disk."""
    config.init_storage()
    console.print("\n[bold cyan]=== Training LightGBM Signal Model ===[/bold cyan]")

    if use_sample_data or not config.UPSTOX_ACCESS_TOKEN:
        console.print(f"[yellow]Generating {bars} bars of realistic synthetic market data...[/yellow]")
        raw_df = DataEngine.generate_synthetic_market_data(bars=bars)
    else:
        console.print(f"[green]Fetching historical data from Upstox for {config.DEFAULT_SYMBOL}...[/green]")
        engine = DataEngine()
        raw_df = engine.fetch_historical_candles(instrument_key=config.DEFAULT_INSTRUMENT_KEY)

    console.print("[cyan]Calculating technical features (EMA, RSI, MACD, BB, ATR)...[/cyan]")
    featured_df = DataEngine.calculate_technical_features(raw_df)
    labeled_df = DataEngine.create_labels(featured_df)

    model = TradingModel(confidence_threshold=config.AI_CONFIDENCE_THRESHOLD)
    metrics = model.train(labeled_df)

    table = Table(title="Model Training Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in metrics.items():
        table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    console.print(table)
    console.print(f"[bold green][✓] Model successfully saved to persistent path: {model.model_path}[/bold green]\n")


def handle_backtest(use_sample_data: bool, bars: int):
    """Backtests the strategy over historical candle data."""
    config.init_storage()
    console.print("\n[bold cyan]=== Running Backtest Simulation ===[/bold cyan]")

    model = TradingModel()
    if not model.load_model():
        console.print("[yellow]No saved model found. Training initial model first...[/yellow]")
        handle_train(use_sample_data=use_sample_data, bars=bars)
        model.load_model()

    if use_sample_data or not config.UPSTOX_ACCESS_TOKEN:
        raw_df = DataEngine.generate_synthetic_market_data(bars=bars, seed=123)
    else:
        engine = DataEngine()
        raw_df = engine.fetch_historical_candles(instrument_key=config.DEFAULT_INSTRUMENT_KEY)

    featured_df = DataEngine.calculate_technical_features(raw_df)

    risk_mgr = RiskManager()
    backtester = Backtester(model=model, risk_manager=risk_mgr)
    results = backtester.run(featured_df)

    table = Table(title="Backtest Performance Report")
    table.add_column("Metric", style="cyan")
    table.add_column("Result", style="bold green" if results.get("net_pnl", 0) >= 0 else "bold red")

    for k, v in results.items():
        table.add_row(k, f"₹{v:,.2f}" if "pnl" in k or "capital" in k else str(v))
    console.print(table)


def handle_paper_trading(use_sample_data: bool, bars: int):
    """Simulates real-time paper trading bar-by-bar."""
    config.init_storage()
    console.print("\n[bold cyan]=== Starting Paper Trading Session ===[/bold cyan]")
    console.print(f"Instrument: [yellow]{config.DEFAULT_SYMBOL}[/yellow] | Capital: [green]₹{config.MAX_CAPITAL}[/green] | Max Daily Loss: [red]₹{config.MAX_DAILY_LOSS}[/red]")

    model = TradingModel()
    if not model.load_model():
        console.print("[yellow]Training base model before launching paper trading...[/yellow]")
        handle_train(use_sample_data=True, bars=1000)
        model.load_model()

    risk_mgr = RiskManager()
    trader = PaperTrader(model=model, risk_manager=risk_mgr, symbol=config.DEFAULT_SYMBOL)

    if use_sample_data or not config.UPSTOX_ACCESS_TOKEN:
        console.print(f"[yellow]Streaming {min(bars, 300)} simulated live ticks...[/yellow]")
        raw_df = DataEngine.generate_synthetic_market_data(bars=min(bars, 300), seed=999)
        featured_df = DataEngine.calculate_technical_features(raw_df)

        for i in range(50, len(featured_df)):
            if risk_mgr.kill_switch_triggered:
                break
            current_bar = featured_df.iloc[i]
            history_subset = featured_df.iloc[: i + 1]
            trader.process_new_bar(current_bar, history_subset)

    summary = trader.get_summary()
    console.print("\n[bold green]Paper Trading Session Summary:[/bold green]")
    console.print(summary)


def handle_live(dry_run: bool):
    """Executes live trading orders with Upstox API."""
    config.init_storage()
    auth = UpstoxAuth()
    risk_mgr = RiskManager()
    live = LiveTrader(auth=auth, risk_manager=risk_mgr, dry_run=dry_run)

    console.print("\n[bold magenta]=== Upstox Live Execution Module ===[/bold magenta]")
    if dry_run:
        console.print("[bold yellow][SAFETY ON] Running in DRY-RUN mode. No real money orders will be placed.[/bold yellow]")
    else:
        console.print("[bold red][WARNING] LIVE REAL-MONEY MODE ACTIVE! Real orders will be sent to NSE/BSE.[/bold red]")

    # Example order test check
    res = live.place_order(
        instrument_token=config.DEFAULT_INSTRUMENT_KEY,
        transaction_type="BUY",
        quantity=1,
        order_type="MARKET",
    )
    console.print(f"Execution response: {res}")


def main():
    print_banner()

    parser = argparse.ArgumentParser(description="Upstox AI Algorithmic Trading Bot")
    parser.add_argument(
        "--mode",
        choices=["auth", "train", "backtest", "paper", "live"],
        default="paper",
        help="Operating mode: auth, train, backtest, paper, or live (default: paper)",
    )
    parser.add_argument(
        "--sample-data",
        action="store_true",
        default=True,
        help="Use realistic synthetic market data (ideal for Colab demos or off-market hours)",
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=1500,
        help="Number of market bars for training or simulation",
    )
    parser.add_argument(
        "--live-real-money",
        action="store_true",
        default=False,
        help="Explicit flag required to submit REAL MONEY orders via Upstox API",
    )

    args = parser.parse_args()

    # Mount Google Drive if running in Google Colab
    if config.is_google_colab():
        console.print("[cyan]Google Colab environment detected.[/cyan]")
        config.init_storage(mount_drive=True)

    if args.mode == "auth":
        handle_auth()
    elif args.mode == "train":
        handle_train(use_sample_data=args.sample_data, bars=args.bars)
    elif args.mode == "backtest":
        handle_backtest(use_sample_data=args.sample_data, bars=args.bars)
    elif args.mode == "paper":
        handle_paper_trading(use_sample_data=args.sample_data, bars=args.bars)
    elif args.mode == "live":
        handle_live(dry_run=not args.live_real_money)


if __name__ == "__main__":
    main()
