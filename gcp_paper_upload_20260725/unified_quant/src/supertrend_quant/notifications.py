from __future__ import annotations

import os

import requests

from .env import load_env


class TelegramNotifier:
    def __init__(self, token: str | None = None, chat_id: str | None = None):
        load_env()
        self.token = token if token is not None else os.getenv("TELEGRAM_TOKEN")
        self.chat_id = chat_id if chat_id is not None else os.getenv("TELEGRAM_CHAT_ID")

    def send(self, message: str) -> bool:
        print(f"Telegram: {message.replace('*', '')}")
        if not self.token or not self.chat_id or "YOUR_" in self.token:
            return False
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": message, "parse_mode": "Markdown"}
        try:
            res = requests.post(url, json=payload, timeout=5)
            return res.status_code == 200
        except Exception as exc:
            print(f"Telegram send failed: {exc}")
            return False
