import os
import re
import json
import urllib.parse
from pathlib import Path
from typing import Optional, Dict, Any
import requests
import pyotp
import config


class UpstoxAuth:
    """Handles Upstox API v2 OAuth authentication and token lifecycle."""

    AUTH_URL = "https://api.upstox.com/v2/login/authorization/dialog"
    TOKEN_URL = "https://api.upstox.com/v2/login/authorization/token"
    PROFILE_URL = "https://api.upstox.com/v2/user/profile"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        redirect_uri: Optional[str] = None,
    ):
        self.api_key = api_key or config.UPSTOX_API_KEY
        self.api_secret = api_secret or config.UPSTOX_API_SECRET
        self.redirect_uri = redirect_uri or config.UPSTOX_REDIRECT_URI
        self.token_file: Path = config.LOGS_DIR / "access_token.json"
        self._access_token: Optional[str] = config.UPSTOX_ACCESS_TOKEN or None

    def get_login_url(self) -> str:
        """Returns the login URL for user authorization."""
        params = {
            "client_id": self.api_key,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
        }
        return f"{self.AUTH_URL}?{urllib.parse.urlencode(params)}"

    @staticmethod
    def extract_code(raw_input: str) -> str:
        """
        Extracts authorization code from raw string or full redirect URL.
        Handles:
          - 'http://127.0.0.1:8000/auth/callback?code=abc12345'
          - '?code=abc12345'
          - 'abc12345'
        """
        raw = raw_input.strip()
        if "code=" in raw:
            parsed = urllib.parse.urlparse(raw)
            qs = urllib.parse.parse_qs(parsed.query)
            if "code" in qs and qs["code"]:
                return qs["code"][0]
            # Fallback regex
            match = re.search(r"code=([^&]+)", raw)
            if match:
                return match.group(1)
        return raw

    def generate_current_totp(self, secret: Optional[str] = None) -> Optional[str]:
        """Generates current 6-digit TOTP code if secret is provided."""
        totp_secret = secret or config.UPSTOX_TOTP_SECRET
        if not totp_secret:
            return None
        totp = pyotp.TOTP(totp_secret)
        return totp.now()

    def exchange_code_for_token(self, auth_input: str) -> Dict[str, Any]:
        """Exchanges authorization code received after login for an access token."""
        code = self.extract_code(auth_input)
        headers = {
            "accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {
            "code": code,
            "client_id": self.api_key,
            "client_secret": self.api_secret,
            "redirect_uri": self.redirect_uri,
            "grant_type": "authorization_code",
        }

        response = requests.post(self.TOKEN_URL, headers=headers, data=data, timeout=15)
        res_data = response.json()

        if response.status_code == 200 and "access_token" in res_data:
            self._access_token = res_data["access_token"]
            self._save_token(res_data)
            self._update_env_files(self._access_token)
            return res_data
        else:
            raise RuntimeError(f"Failed to fetch Upstox access token: {res_data}")

    def _save_token(self, token_data: Dict[str, Any]):
        """Saves access token to disk / Google Drive for reuse."""
        config.init_storage()
        try:
            with open(self.token_file, "w", encoding="utf-8") as f:
                json.dump(token_data, f, indent=2)
            print(f"[AUTH] Access token cached to: {self.token_file}")
        except Exception as e:
            print(f"[AUTH] Could not save token file: {e}")

    def _update_env_files(self, new_token: str):
        """Updates .env file locally and on Google Drive with the new active token."""
        os.environ["UPSTOX_ACCESS_TOKEN"] = new_token
        config.UPSTOX_ACCESS_TOKEN = new_token

        target_paths = [
            Path(".env"),
            Path("/content/drive/MyDrive/upstox_ai_bot/.env"),
        ]

        for env_path in target_paths:
            try:
                if env_path.parent.exists():
                    lines = []
                    token_written = False
                    if env_path.exists():
                        with open(env_path, "r", encoding="utf-8") as f:
                            for line in f:
                                if line.startswith("UPSTOX_ACCESS_TOKEN="):
                                    lines.append(f"UPSTOX_ACCESS_TOKEN={new_token}\n")
                                    token_written = True
                                else:
                                    lines.append(line)

                    if not token_written:
                        lines.append(f"UPSTOX_ACCESS_TOKEN={new_token}\n")

                    with open(env_path, "w", encoding="utf-8") as f:
                        f.writelines(lines)
                    print(f"[AUTH] Updated token in: {env_path}")
            except Exception as e:
                print(f"[AUTH] Could not update {env_path}: {e}")

    def load_cached_token(self) -> Optional[str]:
        """Loads cached token if present."""
        if self._access_token and self.validate_token(self._access_token):
            return self._access_token

        if self.token_file.exists():
            try:
                with open(self.token_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    token = data.get("access_token")
                    if token and self.validate_token(token):
                        self._access_token = token
                        return token
            except Exception:
                pass
        return None

    def validate_token(self, token: Optional[str] = None) -> bool:
        """Checks whether the token is currently valid by pinging user profile API."""
        tok = token or self._access_token
        if not tok:
            return False

        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {tok}",
        }
        try:
            res = requests.get(self.PROFILE_URL, headers=headers, timeout=5)
            if res.status_code == 200:
                data = res.json()
                return data.get("status") == "success"
        except Exception:
            return False
        return False

    def ensure_valid_token_interactive(self) -> bool:
        """
        Self-healing authentication check:
        1. Tests if current token is valid.
        2. If expired or missing, displays login URL and prompts user to paste code/redirect URL.
        3. Exchanges for fresh token and auto-updates .env & Google Drive.
        4. If user presses Enter without input, gracefully continues in Paper/Simulation mode without failing.
        """
        # Step 1: Check existing token
        tok = self.load_cached_token()
        if tok and self.validate_token(tok):
            print("\033[92m[✓ SUCCESS] Upstox Access Token is active, valid, and connected!\033[0m")
            return True

        if not self.api_key or not self.api_secret:
            print("\033[93m[! WARN] UPSTOX_API_KEY or SECRET not set. Running in Paper Trading / Simulation mode.\033[0m")
            return False

        login_url = self.get_login_url()

        print("\n" + "=" * 65)
        print("  🔑 UPSTOX TOKEN EXPIRED OR MISSING - GENERATE FRESH TOKEN")
        print("=" * 65)
        print("\n1. Click or open this login URL in your browser:")
        print(f"\n   \033[94m{login_url}\033[0m\n")

        # In Colab/Jupyter, display rich HTML button
        try:
            from IPython.display import display, HTML
            display(HTML(f"""
            <div style="margin:10px 0;">
                <a href="{login_url}" target="_blank" style="background:#007bff;color:white;padding:8px 16px;text-decoration:none;border-radius:6px;font-weight:bold;">
                    👉 Click Here to Login to Upstox
                </a>
            </div>
            """))
        except Exception:
            pass

        print("2. Log into Upstox and authorize the app.")
        print("3. After login, copy the full redirected URL from your browser address bar.")
        print("   (e.g., http://127.0.0.1:8000/auth/callback?code=XXXXXX)")
        print("-" * 65)

        try:
            auth_input = input("\nPaste the authorization code or full redirected URL (or press Enter to skip for demo): ").strip()
            if not auth_input:
                print("\033[93m[! INFO] Skipped token generation. Continuing in Paper/Demo mode.\033[0m\n")
                return False

            self.exchange_code_for_token(auth_input)
            print("\033[92m[✓ SUCCESS] Fresh token acquired, validated, and saved to .env & Google Drive!\033[0m\n")
            return True
        except Exception as e:
            print(f"\033[91m[✗ ERROR] Failed to exchange token: {e}\033[0m")
            print("\033[93m[! INFO] Falling back to Paper Trading / Simulation mode.\033[0m\n")
            return False

    def get_valid_token(self) -> str:
        """Returns valid token or prompts / raises error."""
        token = self.load_cached_token()
        if token and self.validate_token(token):
            return token
        raise ValueError(
            "No valid Upstox access token found. Please run authentication first via login URL."
        )
