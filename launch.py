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
    """Detects Google Colab environment and logs active storage mode."""
    if is_colab():
        if Path("/content/drive/MyDrive").exists():
            log("Google Drive is mounted and active.", status="SUCCESS")
        else:
            log("Running in Google Colab with Secrets (🔑) and high-speed local container storage.", status="SUCCESS")


def ensure_dependencies():
    """Installs or updates all required libraries if missing."""
    log("Checking Python dependencies...", status="INFO")
    required = [
        "pandas", "numpy", "scikit-learn", "lightgbm", "joblib",
        "requests", "pyotp", "python-dotenv", "rich", "tabulate",
        "matplotlib", "ipywidgets", "nest_asyncio"
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
    """Loads credentials with Colab Secrets (🔑) as primary source, falling back to local .env and Drive."""
    drive_mount = Path("/content/drive/MyDrive")
    local_env = Path(".env")
    found_creds = {}
    found_file = None
    dummy_vals = ["", "your_api_key_here", "your_api_secret_here", "none", "your_token_here", "null"]

    # 1. PRIMARY SOURCE: Google Colab Secrets (🔑 userdata)
    if is_colab():
        try:
            from google.colab import userdata
            for k in ["UPSTOX_API_KEY", "UPSTOX_API_SECRET", "UPSTOX_REDIRECT_URI", "UPSTOX_ACCESS_TOKEN"]:
                try:
                    v = userdata.get(k)
                    if v and str(v).strip().lower() not in dummy_vals:
                        found_creds[k] = str(v).strip().strip('"').strip("'")
                except Exception:
                    pass
            if "UPSTOX_API_KEY" in found_creds:
                log(f"Successfully loaded credentials from Google Colab Secrets (🔑). (API Key: {found_creds['UPSTOX_API_KEY'][:6]}...)", status="SUCCESS")
                env_lines = [f"{k}={v}\n" for k, v in found_creds.items()]
                for k, v in found_creds.items():
                    os.environ[k] = v
                try:
                    local_env.write_text("".join(env_lines), encoding="utf-8")
                except Exception:
                    pass
                return
        except Exception:
            pass

    def try_parse_file(file_path: Path) -> bool:
        nonlocal found_file, found_creds
        for enc in ["utf-8", "utf-8-sig", "utf-16", "latin-1"]:
            try:
                content = None
                try:
                    with open(file_path, "r", encoding=enc) as f:
                        content = f.read()
                except Exception:
                    content = file_path.read_text(encoding=enc, errors="ignore")

                if not content or len(content.strip()) < 5:
                    continue

                log(f"Reading Drive file: {file_path} ({len(content)} chars)", status="INFO")

                parsed = {}
                try:
                    import json
                    json_data = json.loads(content)
                    if isinstance(json_data, dict):
                        for k, v in json_data.items():
                            parsed[str(k).strip().upper()] = str(v).strip().strip('"').strip("'")
                except Exception:
                    pass

                for line in content.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("export "):
                        line = line[7:].strip()
                    if "=" in line:
                        k, v = line.split("=", 1)
                        v = v.split("#")[0].strip()
                        parsed[k.strip().upper()] = v.strip().strip('"').strip("'")
                    elif ":" in line and not line.startswith("http"):
                        k, v = line.split(":", 1)
                        v = v.split("#")[0].strip()
                        parsed[k.strip().upper()] = v.strip().strip('"').strip("'")

                api_key = (
                    parsed.get("UPSTOX_API_KEY")
                    or parsed.get("API_KEY")
                    or parsed.get("CLIENT_ID")
                    or parsed.get("UPSTOX_CLIENT_ID")
                )
                api_secret = (
                    parsed.get("UPSTOX_API_SECRET")
                    or parsed.get("API_SECRET")
                    or parsed.get("SECRET")
                    or parsed.get("UPSTOX_SECRET")
                )
                redirect_uri = (
                    parsed.get("UPSTOX_REDIRECT_URI")
                    or parsed.get("UPSTOX_REDIRECT_URL")
                    or parsed.get("UPSTOX_REDTRECT_URL")
                    or parsed.get("REDIRECT_URI")
                    or parsed.get("REDIRECT_URL")
                )
                access_token = (
                    parsed.get("UPSTOX_ACCESS_TOKEN")
                    or parsed.get("ACCESS_TOKEN")
                )

                if not api_key:
                    log(f"In '{file_path.name}': No UPSTOX_API_KEY line found. Keys present: {list(parsed.keys())}", status="WARN")
                    return False
                elif api_key.lower() in dummy_vals:
                    log(f"In '{file_path.name}': UPSTOX_API_KEY is dummy placeholder '{api_key}'. Real key needed.", status="WARN")
                    return False
                else:
                    found_creds["UPSTOX_API_KEY"] = api_key
                    if api_secret and api_secret.lower() not in dummy_vals:
                        found_creds["UPSTOX_API_SECRET"] = api_secret
                    if redirect_uri:
                        found_creds["UPSTOX_REDIRECT_URI"] = redirect_uri
                    if access_token and access_token.lower() not in dummy_vals:
                        found_creds["UPSTOX_ACCESS_TOKEN"] = access_token
                    found_file = file_path
                    return True
            except Exception:
                continue
        return False

    # 2. Check local .env file
    if "UPSTOX_API_KEY" not in found_creds and local_env.exists():
        if try_parse_file(local_env):
            log(f"Directly loaded credentials from local repository: {local_env}", status="SUCCESS")

    # 3. If still not found and Drive is mounted, check Drive candidates
    if "UPSTOX_API_KEY" not in found_creds and drive_mount.exists():
        # Invalidate Colab Drive FUSE cache by listing with hidden files (-a)
        for folder_name in ["upstox_ai_bot", "upstox ai bot"]:
            target_dir = drive_mount / folder_name
            if target_dir.exists():
                try:
                    subprocess.run(["ls", "-la", str(target_dir)], capture_output=True, timeout=5)
                except Exception:
                    pass

        direct_candidates = [
            drive_mount / "upstox_ai_bot" / ".env",
            drive_mount / "upstox ai bot" / ".env",
            drive_mount / "upstox_ai_bot" / "env",
            drive_mount / "upstox ai bot" / "env",
            drive_mount / "upstox_ai_bot" / ".env.txt",
            drive_mount / "upstox ai bot" / ".env.txt",
            drive_mount / "upstox_ai_bot" / "env.txt",
            drive_mount / "upstox ai bot" / "env.txt",
            drive_mount / ".env",
            drive_mount / "env",
        ]

        for candidate in direct_candidates:
            if candidate.exists() or candidate.is_file():
                if try_parse_file(candidate):
                    log(f"Directly loaded credentials from: {candidate}", status="SUCCESS")
                    break
            else:
                try:
                    if try_parse_file(candidate):
                        log(f"Directly loaded credentials from: {candidate}", status="SUCCESS")
                        break
                except Exception:
                    pass

        # If not found, try force-remounting Google Drive once to refresh web UI uploads
        if not found_file and is_colab():
            try:
                from google.colab import drive
                log("Refreshing Google Drive cache for newly uploaded files...", status="INFO")
                drive.mount("/content/drive", force_remount=True)
                for candidate in direct_candidates:
                    if try_parse_file(candidate):
                        log(f"Loaded credentials after Drive refresh from: {candidate}", status="SUCCESS")
                        break
            except Exception:
                pass

    # 2. If not found via direct paths, do broader folder search
    if not found_file:
        candidate_dirs = [
            drive_mount / "upstox_ai_bot",
            drive_mount / "upstox ai bot",
            drive_mount,
        ]
        try:
            for entry in drive_mount.iterdir():
                if entry.is_dir():
                    name_lower = entry.name.lower()
                    if any(kw in name_lower for kw in ["upstox", "trading", "bot"]) and entry not in candidate_dirs:
                        candidate_dirs.append(entry)
        except Exception:
            pass

        for d in candidate_dirs:
            if not d.exists():
                continue
            try:
                for item in d.iterdir():
                    if item.is_file() and item.stat().st_size < 100_000:
                        if try_parse_file(item):
                            break
            except Exception:
                pass
            if found_file:
                break

    if found_file and "UPSTOX_API_KEY" in found_creds:
        log(f"Found saved credentials in Google Drive: {found_file}", status="SUCCESS")
        env_lines = []
        for k, v in found_creds.items():
            env_lines.append(f"{k}={v}\n")
            os.environ[k] = v

        try:
            local_env.write_text("".join(env_lines), encoding="utf-8")
            log(f"Loaded credentials into environment (API Key: {found_creds['UPSTOX_API_KEY'][:6]}...).", status="SUCCESS")
        except Exception as e:
            log(f"Could not write local .env: {e}", status="WARN")
    else:
        log("No valid credential file found in Drive. Checked .env / env files.", status="WARN")


def main():
    print("""
===================================================================
   🤖 UPSTOX AI TRADING BOT - AUTO-PILOT RESILIENT LAUNCHER
===================================================================
    """)

    # Enable Google Colab widget manager immediately so widgets render without prompts
    try:
        from google.colab import output
        output.enable_custom_widget_manager()
    except Exception:
        pass

    # 0. Auto-sync repository to latest git code
    try:
        subprocess.run("git pull origin main", shell=True, capture_output=True)
    except Exception:
        pass


    # 1. Drive Mount
    setup_colab_drive()

    # 2. Dependency Check & Auto-Install
    ensure_dependencies()

    # 3. Environment & Directories
    sync_configuration()
    import importlib
    from dotenv import load_dotenv
    load_dotenv(Path(".env"), override=True)
    import config
    importlib.reload(config)
    config.init_storage(mount_drive=False)
    log(f"Storage path ready at: {config.DATA_DIR}", status="SUCCESS")

    # 4. Token & Authentication Check (Auto-generates & saves fresh token if expired)
    import importlib
    import core.auth
    importlib.reload(core.auth)
    from core.auth import UpstoxAuth
    auth = UpstoxAuth()
    has_live_auth = auth.ensure_valid_token_interactive()

    # 5. Model Verification & Auto-Train
    import core.model
    import core.data_engine
    importlib.reload(core.model)
    importlib.reload(core.data_engine)
    from core.model import TradingModel
    from core.data_engine import DataEngine
    model = TradingModel()
    if not model.load_model():
        log("No existing AI model found. Auto-training LightGBM model now...", status="INFO")
        raw_df = DataEngine.generate_synthetic_market_data(bars=2000, seed=42)
        featured_df = DataEngine.calculate_technical_features(raw_df)
        labeled_df = DataEngine.create_labels(featured_df)
        metrics = model.train(labeled_df)
        acc = metrics.get('accuracy', 0.0)
        bull_p = metrics.get('bull_precision', 0.0)
        bear_p = metrics.get('bear_precision', 0.0)
        log(f"Model trained and saved to Drive! Accuracy: {acc:.1%}, Bull Precision: {bull_p:.1%}, Bear Precision: {bear_p:.1%}", status="SUCCESS")
    else:
        log("Existing trained AI model found and loaded from persistent storage.", status="SUCCESS")

    # 6. Quick Healthcheck Backtest
    import core.risk_manager
    import core.backtester
    importlib.reload(core.risk_manager)
    importlib.reload(core.backtester)
    from core.risk_manager import RiskManager
    from core.backtester import Backtester
    log("Running strategy health-check backtest...", status="INFO")
    test_df = DataEngine.generate_synthetic_market_data(bars=500, seed=101)
    feat_df = DataEngine.calculate_technical_features(test_df)
    bt = Backtester(model=model, risk_manager=RiskManager())
    results = bt.run(feat_df)
    log(f"Health-check verified: {results.get('total_trades', 0)} simulated trades, Net P&L: ₹{results.get('net_pnl', 0):+,.2f}", status="SUCCESS")

    # 7. Launch Execution
    is_notebook = False
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            is_notebook = True
    except Exception:
        pass

    if is_notebook:
        log("Launching In-Colab Interactive Live Dashboard with Keep-Alive UI...", status="SUCCESS")
        import core.dashboard
        importlib.reload(core.dashboard)
        from core.dashboard import ColabTradingDashboard
        dashboard = ColabTradingDashboard(
            symbol=config.DEFAULT_SYMBOL,
            use_sample_data=not has_live_auth,
        )
        dashboard.render(run_loop=True)
    else:
        log("Running in CLI terminal mode. Starting Paper Trading loop...", status="INFO")
        import core.paper_trader
        importlib.reload(core.paper_trader)
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
