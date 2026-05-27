"""Planting management module.

Translates ``AUTPLT.for`` automatic-planting and the planting section of
``MgmtOps.for``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlantingEvent:
    """A single planting event.

    Attributes:
        date: Sowing date (YYYYDDD).
        pltpop: Plant population (plants m⁻²).
        sdepth: Sowing depth (cm).
        rowspc: Row spacing (cm).
        cultivar: Cultivar identifier string (matches :attr:`MaizeCultivar.varno`
            or equivalent).
    """

    date: int
    pltpop: float = 7.0
    sdepth: float = 5.0
    rowspc: float = 75.0
    cultivar: str = "GENERIC"
