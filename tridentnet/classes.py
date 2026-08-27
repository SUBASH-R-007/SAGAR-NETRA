"""Canonical class map for all TridentNet brains.

The last three classes are *hard negatives*: natural seabed features the
detector must learn to tell apart from man-made debris. They absorb
confusions during training and are never surfaced to the user.
"""

from __future__ import annotations

from typing import Final

#: Index order is frozen — it defines label ids in every trained model.
CLASS_NAMES: Final[tuple[str, ...]] = (
    "ghost_net",
    "wreck",
    "aircraft",
    "pipeline",
    "cylinder_drum",
    "tire",
    "container",
    "human_body",
    "mine_like",
    "rock_cluster",  # hard negative
    "sand_ripple",  # hard negative
    "reef",  # hard negative
)

HARD_NEGATIVES: Final[frozenset[str]] = frozenset({"rock_cluster", "sand_ripple", "reef"})

#: Ensemble-level label for open-set finds (Brain C reconstruction-error blobs
#: with no detector class). Not a detector training class.
ANOMALY_CLASS: Final[str] = "unknown_anomaly"

#: Classes that may be shown to the user.
REPORTABLE: Final[tuple[str, ...]] = tuple(
    name for name in CLASS_NAMES if name not in HARD_NEGATIVES
) + (ANOMALY_CLASS,)

CLASS_TO_ID: Final[dict[str, int]] = {name: i for i, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS: Final[dict[int, str]] = dict(enumerate(CLASS_NAMES))


def is_reportable(name: str) -> bool:
    return name in REPORTABLE
