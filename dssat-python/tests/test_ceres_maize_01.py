"""Integration test: CERES-Maize with real EJEMPLO1 fortran outputs.

Run with::

    pytest tests/test_ceres_maize_01.py -v
"""
from pathlib import Path


# ---------------------------------------------------------------------------
# CERES Maize experiment using Fortran
# Verify that this implementation matches the Fortran output
# ---------------------------------------------------------------------------

YIELD_TOLERANCE = 10.0
DAYS_TOLERANCE = 1
DIR = Path("test_data/EJEMPLO1")
EXP_MZX = DIR / "EJEMPLO1.MZX"
FORTRAN_OUTPUT = DIR / "summary.fortran"

#  ------------- Do not modify previous lines ------------------

