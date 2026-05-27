"""Abstract base class for all DSSAT crop models.

Every crop model (CERES-Maize, CERES-Wheat, CROPGRO-Soybean, …) inherits
from :class:`CropModel` and implements the :meth:`run` and :meth:`summary`
methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from dssat.core.types import (
    ControlType, SwitchType, SoilType, WeatherType,
)


class CropModel(ABC):
    """Abstract interface for a DSSAT crop model.

    Attributes:
        crop_code: Two-letter DSSAT crop code (e.g. ``"MZ"``, ``"WH"``).
        model_name: Eight-character model name (e.g. ``"MZCER048"``).
    """

    crop_code: str = "XX"
    model_name: str = "XXXXXXXX"

    @abstractmethod
    def run(
        self,
        control: ControlType,
        iswitch: SwitchType,
        soilprop: SoilType,
        weather: WeatherType,
        sw: np.ndarray,
        nh4: np.ndarray,
        no3: np.ndarray,
        es: float,
        eop: float,
    ) -> None:
        """Advance the crop model by one simulation step.

        Args:
            control: Simulation control block (provides ``yrdoy``,
                ``dynamic``, etc.).
            iswitch: Simulation switches.
            soilprop: Soil profile properties.
            weather: Today's weather.
            sw: Volumetric soil water content by layer.
            nh4: Ammonium concentration by layer (µg g⁻¹).
            no3: Nitrate concentration by layer (µg g⁻¹).
            es: Potential soil evaporation (mm d⁻¹).
            eop: Potential transpiration (mm d⁻¹).
        """

    @abstractmethod
    def summary(self) -> dict:
        """Return end-of-season summary variables.

        Returns:
            Dictionary whose keys map to standard DSSAT Summary.OUT columns.
        """
