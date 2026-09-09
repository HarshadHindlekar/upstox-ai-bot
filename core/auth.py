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

    def generate_current_totp(self, secret: Optional[str] = None) -> Optional[str]:
        """Generates current 6-digit TOTP code if secret is provided."""
        totp_secret = secret or config.UPSTOX_TOTP_SECRET
        if not totp_secret:
            return None
        totp = pyotp.TOTP(totp_secret)
        return totp.now()

    def exchange_code_for_token(self, auth_code: str) -> Dict[str, Any]:
        """Exchanges authorization code received after login for an access token."""
        headers = {
            "accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {
            "code": auth_code.strip(),
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
            return res_data
        else:
            raise RuntimeError(f"Failed to fetch Upstox access token: {res_data}")

    def _save_token(self, token_data: Dict[str, Any]):
        """Saves access token to disk / Google Drive for reuse."""
        config.init_storage()
        try:
            with open(self.token_file, "w", encoding="utf-8") as f:
                json.dump(token_data, f, indent=2)
            print(f"[AUTH] Access token saved to: {self.token_file}")
        except Exception as e:
            print(f"[AUTH] Could not save token file: {e}")

    def load_cached_token(self) -> Optional[str]:
        """Loads cached token if present."""
        if self._access_token:
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

    def get_valid_token(self) -> str:
        """Returns valid token or prompts / raises error."""
        token = self.load_cached_token()
        if token and self.validate_token(token):
            return token
        raise ValueError(
            "No valid Upstox access token found. Please run authentication first via login URL."
        )
