import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load local .env if available
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def is_google_colab() -> bool:
    """Checks whether the environment is Google Colab."""
    return "google.colab" in sys.modules or os.path.exists("/content")


# Base directory configuration
if is_google_colab():
    # If in Colab, persistent paths point to Google Drive ONLY IF already mounted
    DRIVE_MOUNT_POINT = Path("/content/drive")
    if (DRIVE_MOUNT_POINT / "MyDrive").exists():
        custom_dir = os.getenv("DRIVE_PROJECT_DIR")
        if custom_dir:
            folder_name = custom_dir
        elif (DRIVE_MOUNT_POINT / "MyDrive" / "upstox ai bot").exists():
            folder_name = "upstox ai bot"
        else:
            folder_name = "upstox_ai_bot"
        DRIVE_BASE_DIR = DRIVE_MOUNT_POINT / "MyDrive" / folder_name
        DATA_DIR = DRIVE_BASE_DIR / "data"
        MODELS_DIR = DRIVE_BASE_DIR / "models"
        LOGS_DIR = DRIVE_BASE_DIR / "logs"
    else:
        # Seamless local fallback in Colab container
        DRIVE_BASE_DIR = BASE_DIR
        DATA_DIR = BASE_DIR / "data"
        MODELS_DIR = BASE_DIR / "models"
        LOGS_DIR = BASE_DIR / "logs"
else:
    # Local paths
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"


def init_storage(mount_drive: bool = False):
    """Initializes storage directories without forcing Google Drive mount."""
    if is_google_colab() and mount_drive:
        try:
            from google.colab import drive
            if not os.path.exists("/content/drive/MyDrive"):
                drive.mount("/content/drive")
        except Exception:
            pass

    # Create directories if they do not exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


# 1. Primary Source for Colab: Google Colab Secrets (🔑 userdata)
if is_google_colab():
    try:
        from google.colab import userdata
        for key in ["UPSTOX_API_KEY", "UPSTOX_API_SECRET", "UPSTOX_REDIRECT_URI", "UPSTOX_ACCESS_TOKEN"]:
            val = userdata.get(key)
            if val and str(val).strip():
                os.environ[key] = str(val).strip().strip('"').strip("'")
    except Exception:
        pass

# 2. Upstox API credentials from environment
UPSTOX_API_KEY = (os.getenv("UPSTOX_API_KEY") or "").strip().strip('"').strip("'")
UPSTOX_API_SECRET = (os.getenv("UPSTOX_API_SECRET") or "").strip().strip('"').strip("'")
UPSTOX_REDIRECT_URI = (
    os.getenv("UPSTOX_REDIRECT_URI")
    or os.getenv("UPSTOX_REDIRECT_URL")
    or os.getenv("UPSTOX_REDTRECT_URL")
    or "http://127.0.0.1:8000/auth/callback"
).strip().strip('"').strip("'")
UPSTOX_ACCESS_TOKEN = (os.getenv("UPSTOX_ACCESS_TOKEN") or "").strip().strip('"').strip("'")

# Automated login (optional)
UPSTOX_USER_ID = os.getenv("UPSTOX_USER_ID", "")
UPSTOX_PIN = os.getenv("UPSTOX_PIN", "")
UPSTOX_TOTP_SECRET = os.getenv("UPSTOX_TOTP_SECRET", "")

# Trading & Risk Parameters
DEFAULT_INSTRUMENT_KEY = os.getenv("DEFAULT_INSTRUMENT_KEY", "NSE_EQ|INE002A01018")
DEFAULT_SYMBOL = os.getenv("DEFAULT_SYMBOL", "RELIANCE")
TIMEFRAME = os.getenv("TIMEFRAME", "1minute")

MAX_CAPITAL = float(os.getenv("MAX_CAPITAL", "50000.0"))
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", "2000.0"))
RISK_PER_TRADE_PERCENT = float(os.getenv("RISK_PER_TRADE_PERCENT", "1.0"))
STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", "0.8"))
TAKE_PROFIT_PERCENT = float(os.getenv("TAKE_PROFIT_PERCENT", "1.6"))

# AI Model settings
AI_CONFIDENCE_THRESHOLD = float(os.getenv("AI_CONFIDENCE_THRESHOLD", "0.65"))
MODEL_FILE_NAME = "upstox_lightgbm_model.joblib"
