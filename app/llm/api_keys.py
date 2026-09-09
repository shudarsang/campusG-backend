"""
api_keys.py

Rotating pool of Gemini API keys.

Every environment variable whose name starts with GOOGLE_API_KEY is
treated as a usable key (GOOGLE_API_KEY, GOOGLE_API_KEYdg,
GOOGLE_API_KEY_2, ...), so adding a key to .env is enough to put it
into rotation.

When a key runs out of quota or is rejected, it is parked on a
cooldown and the next key takes over immediately. The pool stays on
that key for subsequent requests instead of retrying the dead one
every time, so a single exhausted key costs one failed call, not one
per message.
"""

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from app.utils.logger import setup_logger

# The pool is built at import time, so .env has to be loaded here
# rather than relying on whichever module imports this one first.
BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

logger = setup_logger(__name__)

# Env vars holding a Gemini key. The suffix is free-form and only
# used as a display label in the logs.
KEY_PATTERN = re.compile(r"^GOOGLE_API_KEY(.*)$")

# How long an exhausted key is parked when the API does not tell us
# to wait for a specific number of seconds (daily caps report a
# retry delay that is often shorter than the real reset).
DEFAULT_COOLDOWN_SECONDS = 15 * 60


@dataclass
class ApiKey:
    """A single Gemini credential and its current health."""

    label: str
    value: str
    cooldown_until: float = 0.0
    failures: int = field(default=0)

    @property
    def available(self) -> bool:
        return time.time() >= self.cooldown_until

    @property
    def cooldown_remaining(self) -> int:
        return max(0, int(self.cooldown_until - time.time()))


class ApiKeyPool:
    """
    Holds every configured key and decides which one to use next.
    """

    def __init__(self):

        self._keys: List[ApiKey] = self._discover()
        self._index: int = 0

        if not self._keys:
            logger.warning(
                "No GOOGLE_API_KEY* entries found in .env; "
                "Gemini calls will be unavailable."
            )
        else:
            logger.info(
                f"Loaded {len(self._keys)} Gemini API key(s): "
                f"{', '.join(k.label for k in self._keys)}"
            )

    @staticmethod
    def _discover() -> List[ApiKey]:
        """
        Collect GOOGLE_API_KEY* values from the environment.
        """

        found = []

        for name, raw in os.environ.items():

            match = KEY_PATTERN.match(name)

            if not match:
                continue

            # .env values may carry surrounding quotes
            value = (raw or "").strip().strip('"').strip("'")

            if not value:
                continue

            label = match.group(1) or "default"

            found.append(ApiKey(label=label, value=value))

        # Deterministic order so restarts behave the same way
        found.sort(key=lambda k: k.label)

        return found

    @property
    def size(self) -> int:
        return len(self._keys)

    @property
    def labels(self) -> List[str]:
        return [k.label for k in self._keys]

    def all_keys(self) -> List[ApiKey]:
        return list(self._keys)

    def current(self) -> Optional[ApiKey]:
        """
        The key to use right now, or None if the pool is empty.

        Prefers an available key. If every key is cooling down, the
        one that recovers soonest is returned so the request still
        gets a chance rather than failing outright.
        """

        if not self._keys:
            return None

        for offset in range(len(self._keys)):

            candidate = self._keys[(self._index + offset) % len(self._keys)]

            if candidate.available:
                self._index = (self._index + offset) % len(self._keys)
                return candidate

        soonest = min(self._keys, key=lambda k: k.cooldown_until)

        logger.warning(
            f"All {len(self._keys)} keys are cooling down; "
            f"retrying '{soonest.label}' "
            f"(ready in {soonest.cooldown_remaining}s)."
        )

        return soonest

    def penalise(
        self,
        key: ApiKey,
        retry_after: Optional[float] = None
    ) -> None:
        """
        Park a key that failed and move the pointer to the next one.
        """

        wait = retry_after if retry_after else DEFAULT_COOLDOWN_SECONDS

        key.cooldown_until = time.time() + wait
        key.failures += 1

        self._index = (self._keys.index(key) + 1) % len(self._keys)

        healthy = [k.label for k in self._keys if k.available]

        logger.warning(
            f"Key '{key.label}' parked for {int(wait)}s. "
            f"Still usable: {healthy or 'none'}"
        )

    def report_success(self, key: ApiKey) -> None:
        """
        Clear a key's cooldown after it serves a request.
        """

        key.cooldown_until = 0.0

    def status(self) -> List[dict]:
        """
        Snapshot of the pool, used by the /health endpoint.
        """

        return [
            {
                "key": k.label,
                "available": k.available,
                "cooldown_seconds": k.cooldown_remaining,
                "failures": k.failures,
            }
            for k in self._keys
        ]


key_pool = ApiKeyPool()
