"""Spend guard for the public demo.

A public URL wired to a real OpenAI key is an open tap on someone's budget.
This caps total spend per UTC day; once the cap is hit the API stops calling the
model and returns a friendly "demo limit reached" message instead. Cheap
insurance so a curious visitor (or a bot) can't run up the bill.

In-process and best-effort — fine for a single free-tier instance. For multiple
replicas, back it with the agentcost SQLite/Postgres sink instead.
"""
from __future__ import annotations

import os
import threading
from datetime import datetime, timezone


class DailySpendGuard:
    def __init__(self, cap_usd: float):
        self.cap_usd = cap_usd
        self._day = self._today()
        self._spent = 0.0
        self._lock = threading.Lock()

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _roll(self) -> None:
        today = self._today()
        if today != self._day:
            self._day, self._spent = today, 0.0

    def allowed(self) -> bool:
        """True if there's budget left for another query today."""
        if self.cap_usd <= 0:
            return True
        with self._lock:
            self._roll()
            return self._spent < self.cap_usd

    def add(self, cost_usd: float) -> None:
        with self._lock:
            self._roll()
            self._spent += max(0.0, cost_usd)

    def status(self) -> dict:
        with self._lock:
            self._roll()
            return {"day": self._day, "spent_usd": round(self._spent, 4),
                    "cap_usd": self.cap_usd, "remaining_usd": round(max(0.0, self.cap_usd - self._spent), 4)}


GUARD = DailySpendGuard(float(os.environ.get("REGAGENT_DAILY_USD_CAP", "0") or 0))
