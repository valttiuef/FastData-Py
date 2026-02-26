from __future__ import annotations
"""Lightweight wrapper around the system keyring for sensitive values.

The OpenAI API key should not be stored in plaintext alongside regular
settings.  This module centralizes access to the system keyring so we can
persist secrets without writing them to disk ourselves.
"""


import logging
from typing import Optional

import keyring
from keyring.errors import KeyringError

_log = logging.getLogger(__name__)


def save_secret(service: str, name: str, value: Optional[str]) -> None:
    """Persist *value* in the system keyring under (*service*, *name*).

    If *value* is falsy the secret will be deleted instead.  Any keyring
    errors are logged but not raised so callers can fail gracefully while
    keeping the secret in memory for the current session.
    """

    try:
        if value:
            keyring.set_password(service, name, value)
        else:
            try:
                keyring.delete_password(service, name)
            except KeyringError:
                # Deleting a missing secret should not be fatal
                pass
    except KeyringError as exc:  # pragma: no cover - environment dependent
        _log.warning("Failed to store secret in keyring: %s", exc)


def load_secret(service: str, name: str) -> Optional[str]:
    """Retrieve a secret from the system keyring.

    Returns ``None`` if the secret is unavailable or the keyring backend is
    inaccessible.  Errors are logged for visibility but intentionally not
    re-raised.
    """

    try:
        return keyring.get_password(service, name)
    except KeyringError as exc:  # pragma: no cover - environment dependent
        _log.warning("Failed to load secret from keyring: %s", exc)
        return None
