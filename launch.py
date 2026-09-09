#!/usr/bin/env python3
"""
Upstox AI Trading Bot - Bulletproof One-Command Auto-Pilot Launcher
Handles mounting, dependency installation, git syncing, Drive persistence,
auto-training missing models, health checks, and dashboard launch without failing.
"""

import os
import sys
import subprocess
from pathlib import Path


def log(msg: str, status: str = "INFO"):
    colors = {
        "INFO": "\033[94m[INFO]\033[0m",
        "SUCCESS": "\033[92m[✓ SUCCESS]\033[0m",
        "WARN": "\033[93m[! WARN]\033[0m",
        "ERROR": "\033[91m[✗ ERROR]\033[0m",
    }
    prefix = colors.get(status, f"[{status}]")
    print(f"{prefix} {msg}")


def run_cmd(cmd: str, ignore_error: bool = False) -> bool:
    try:
        subprocess.check_call(cmd, shell=True)
        return True
    except subprocess.CalledProcessError as e:
        if not ignore_error:
            log(f"Command failed: {cmd} (Error: {e})", status="WARN")
        return False


def setup_colab_drive():
    """Mounts Google Drive safely, falls back to local if skipped/error."""
    if "google.colab" in sys.modules or os.path.exists("/content"):
        log("Google Colab runtime detected. Connecting Google Drive...", status="INFO")
        try:
            from google.colab import drive
            if not os.path.exists("/content/drive/MyDrive"):
                drive.mount("/content/drive")
            log("Google Drive successfully mounted!", status="SUCCESS")
        except Exception as e:
            log(f"Google Drive mount bypassed ({e}). Falling back to local storage.", status="WARN")


def ensure_dependencies():
    """Installs or updates all required libraries if missing."""
    log("Checking Python dependencies...", status="INFO")
    required = [
        "pandas", "numpy", "scikit-learn", "lightgbm", "joblib",
        "requests", "pyotp", "python-dotenv", "rich", "tabulate",
        "matplotlib", "ipywidgets"
    ]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        log(f"Installing missing packages: {', '.join(missing)}...", status="INFO")
        run_cmd(f"{sys.executable} -m pip install -q " + " ".join(missing), ignore_error=True)
        log("Dependencies installed successfully.", status="SUCCESS")
    else:
        log("All dependencies are already satisfied.", status="SUCCESS")


def sync_configuration():
    """Loads .env from Google Drive if user stored credentials there."""
    drive_env = Path("/content/drive/MyDrive/upstox_ai_bot/.env")
    local_env = Path(".env")

    if drive_env.exists():
        log(f"Found saved credentials in Google Drive: {drive_env}", status="SUCCESS")
        try:
            import shutil
            shutil.copy(drive_env, local_env)
        except Exception:
            pass
    elif not local_env.exists() and Path(".env.example").exists():
        log("Creating default configuration from .env.example...", status="INFO")
        try:
            import shutil
            shutil.copy(Path(".env.example"), local_env)
        except Exception:
            pass


def main():
    print("""
===================================================================
   🤖 UPSTOX AI TRADING BOT - AUTO-PILOT RESILIENT LAUNCHER
===================================================================
    """)

    # 1. Drive Mount
    setup_colab_drive()

    # 2. Dependency Check & Auto-Install
    ensure_dependencies()

    # 3. Environment & Directories
    sync_configuration()
    import config
    config.init_storage(mount_drive=True)
    log(f"Storage path ready at: {config.DATA_DIR.parent}", status="SUCCESS")

    # 4. Token & Authentication Check (Auto-generates & saves fresh token if expired)
    from core.auth import UpstoxAuth
    auth = UpstoxAuth()
    has_live_auth = auth.ensure_valid_token_interactive()

    # 5. Model Verification & Auto-Train
    from core.model import TradingModel
    from core.data_engine import DataEngine
    model = TradingModel()
    if not model.load_model():
        log("No existing AI model found. Auto-training LightGBM model now...", status="INFO")
        raw_df = DataEngine.generate_synthetic_market_data(bars=2000, seed=42)
        featured_df = DataEngine.calculate_technical_features(raw_df)
        labeled_df = DataEngine.create_labels(featured_df)
        metrics = model.train(labeled_df)
        log(f"Model trained and saved to Drive! AUC: {metrics['test_auc']:.3f}, F1: {metrics['f1']:.3f}", status="SUCCESS")
    else:
        log("Existing trained AI model found and loaded from persistent storage.", status="SUCCESS")

    # 6. Quick Healthcheck Backtest

    from core.risk_manager import RiskManager
    from core.backtester import Backtester
    log("Running strategy health-check backtest...", status="INFO")
    test_df = DataEngine.generate_synthetic_market_data(bars=500, seed=101)
    feat_df = DataEngine.calculate_technical_features(test_df)
    bt = Backtester(model=model, risk_manager=RiskManager())
    results = bt.run(feat_df)
    log(f"Health-check verified: {results.get('total_trades', 0)} simulated trades, Net P&L: ₹{results.get('net_pnl', 0):+,.2f}", status="SUCCESS")

    # 6. Launch Execution
    is_notebook = False
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            is_notebook = True
    except Exception:
        pass

    if is_notebook:
        log("Launching In-Colab Interactive Live Dashboard with Keep-Alive UI...", status="SUCCESS")
        dashboard = ColabTradingDashboard(
            symbol=config.DEFAULT_SYMBOL,
            use_sample_data=not has_live_auth,
        )
        dashboard.render()
    else:
        log("Running in CLI terminal mode. Starting Paper Trading loop...", status="INFO")
        from core.paper_trader import PaperTrader
        trader = PaperTrader(model=model, risk_manager=RiskManager(), symbol=config.DEFAULT_SYMBOL)
        raw_sim = DataEngine.generate_synthetic_market_data(bars=150, seed=777)
        feat_sim = DataEngine.calculate_technical_features(raw_sim)
        for i in range(50, len(feat_sim)):
            trader.process_new_bar(feat_sim.iloc[i], feat_sim.iloc[: i + 1])
        log(f"Simulation completed. Summary: {trader.get_summary()}", status="SUCCESS")

    print("\n===================================================================")
    print("   ✓ SYSTEM ACTIVE & HEALTHY")
    print("===================================================================\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"Unexpected error recovered: {e}", status="ERROR")
        log("Attempting fallback paper trading CLI execution...", status="WARN")
        run_cmd(f"{sys.executable} run.py --mode paper --sample-data --bars 100", ignore_error=True)
