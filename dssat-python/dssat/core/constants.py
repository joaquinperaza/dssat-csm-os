"""Global constants used throughout the DSSAT Cropping System Model.

Mirrors the constant declarations in ``ModuleDefs.for``.
"""

import math

# Soil layer limits
NL: int = 20
"""Maximum number of soil layers."""

TS: int = 24
"""Number of hourly time steps per day."""

NAPPL: int = 9000
"""Maximum number of management applications or operations."""

NCOHORTS: int = 300
"""Maximum number of plant cohorts."""

NELEM: int = 3
"""Number of nutrient elements modelled (N, P, K)."""

# Nutrient array indices (1-based, converted to 0-based for Python)
N: int = 0
"""Nitrogen array index."""

P: int = 1
"""Phosphorus array index."""

K: int = 2
"""Potassium array index."""

PI: float = math.pi
RAD: float = math.pi / 180.0

# Dynamic (phase) codes passed to every module each time step
RUNINIT: int = 1
"""Run initialisation — called once at the very start of a run."""

SEASINIT: int = 2
"""Seasonal initialisation — called once per season/crop cycle."""

RATE: int = 3
"""Rate calculation phase — compute daily rates."""

INTEGR: int = 4
"""Integration phase — update state variables from rates."""

OUTPUT: int = 5
"""Output phase — write daily output."""

SEASEND: int = 6
"""End-of-season phase — final season summaries."""

ENDRUN: int = 7
"""End-of-run phase — final cleanup."""

# Crop status codes
STATUS_OK: int = 0
STATUS_GERMINATION_FAILURE: int = 12
STATUS_WATER_STRESS_DEATH: int = 6
STATUS_COLD_STRESS_DEATH: int = 5
STATUS_MATURE: int = 1

MONTH_TXT = [
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
]
