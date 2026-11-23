"""Particle lifecycle codes and meanings shared by execution and result clients.

These describe numerical particle outcomes. Field support, accuracy and run
integrity use their own statuses.
"""

from types import MappingProxyType
from typing import Final

PENDING: Final = 0
ACTIVE: Final = 1
STUCK: Final = 2
ESCAPED: Final = 3
FAILED: Final = 4
HELD: Final = 5

NAMES = MappingProxyType(
    {
        PENDING: "pending",
        ACTIVE: "active",
        STUCK: "stuck",
        ESCAPED: "escaped",
        FAILED: "failed",
        HELD: "held",
    }
)
TERMINAL: Final = (STUCK, ESCAPED, FAILED, HELD)
