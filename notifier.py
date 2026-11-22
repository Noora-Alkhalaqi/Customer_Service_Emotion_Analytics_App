"""Telegram helper (free Bot API). The token and chat id are read from environment variables that YOU set on your
own computer; they are never written in the code. Put them once in a file named '.env' next to app.py:

    TELEGRAM_BOT_TOKEN=...
    TELEGRAM_CHAT_ID=...
    LIVE_PUBLIC_URL=https://your-tunnel-address
"""
import os
from pathlib import Path

import requests


def load_env_file(path=None):
    """Read KEY=value lines from a local '.env' file (next to this file) into the environment.
    Variables that are already set on the computer win. Lines starting with # are ignored."""
    path = Path(path) if path else Path(__file__).resolve().parent / ".env"
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
    return True


load_env_file()


def telegram_configured():
    return bool(os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"))


def send_telegram_message(text, chat_id=None, timeout=15):
    """Send a text message. Returns (ok, message). The token never appears in the returned message."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False, "Telegram is not configured (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)."
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          data={"chat_id": chat_id, "text": text, "disable_web_page_preview": "true"}, timeout=timeout)
    except requests.RequestException as exc:
        return False, f"Could not reach Telegram: {type(exc).__name__}"
    if r.status_code != 200:
        try:
            reason = r.json().get("description", "")
        except ValueError:
            reason = ""
        return False, f"Telegram refused the message (HTTP {r.status_code}). {reason}".strip()
    return True, "Message sent."


def send_telegram_document(data, filename, caption="", chat_id=None, timeout=60):
    """Send a file (bytes) as a Telegram document. Returns (ok, message). The token never appears in the message."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False, "Telegram is not configured (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)."
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendDocument",
                          data={"chat_id": chat_id, "caption": caption[:1000]},
                          files={"document": (filename, data, "text/html")}, timeout=timeout)
    except requests.RequestException as exc:
        return False, f"Could not reach Telegram: {type(exc).__name__}"
    if r.status_code != 200:
        try:
            reason = r.json().get("description", "")
        except ValueError:
            reason = ""
        return False, f"Telegram refused the file (HTTP {r.status_code}). {reason}".strip()
    return True, "File sent."