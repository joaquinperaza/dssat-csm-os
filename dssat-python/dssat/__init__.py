"""DSSAT Cropping System Model — pure Python implementation.

This package is a faithful Python translation of the DSSAT CSM Fortran
source (``dssat-csm-os``).  It exposes the same physical models and
algorithms in idiomatic Python with JSON-based I/O and full
:mod:`mkdocstrings`-compatible API documentation.

Quick start
-----------

.. code-block:: python

    from dssat import Simulation

    # Build from a JSON experiment file
    sim = Simulation.from_json("my_experiment.json")
    results = sim.run()

    # End-of-season summary
    print(results["summary"])

    # Daily time series as a pandas DataFrame
    print(results["daily"].head())

Modules
-------

:mod:`dssat.core`
    Data types (:class:`~dssat.core.types.SoilType`,
    :class:`~dssat.core.types.WeatherType`, …), constants, date utilities.

:mod:`dssat.weather`
    Solar geometry, hourly disaggregation, CO₂ handling.

:mod:`dssat.soil`
    Soil water balance, N/P/K, organic matter.

:mod:`dssat.crop`
    All crop models (CERES-Maize implemented; others as documented stubs).

:mod:`dssat.management`
    Planting, irrigation, fertiliser, tillage, harvest.

:mod:`dssat.io`
    JSON readers/writers and JSON-Schema definitions.
"""

__version__ = "4.8.5"
__author__ = "DSSAT Foundation"
__license__ = "BSD-3-Clause"

from dssat.core.engine import Simulation
from dssat.core.types import (
    ControlType, SwitchType, SoilType, WeatherType,
)
from dssat.crop import (
    CropModel, CeresMaize, MaizeCultivar,
    CeresSorghum, SorghumCultivar,
    CeresMillet, MilletCultivar,
    CeresWheat, WheatCultivar,
)

__all__ = [
    "Simulation",
    "ControlType",
    "SwitchType",
    "SoilType",
    "WeatherType",
    "CropModel",
    "CeresMaize", "MaizeCultivar",
    "CeresSorghum", "SorghumCultivar",
    "CeresMillet", "MilletCultivar",
    "CeresWheat", "WheatCultivar",
]
