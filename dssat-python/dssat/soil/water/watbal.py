"""Soil water balance module.

Translated from ``WATBAL.for``, ``WBSUBS.for``, ``INFIL.for``, and
``SATFLO.for``.

The tipping-bucket approach is used: each layer fills from the top until
DUL, then excess drains to the layer below.  Upward capillary flow is
computed from a simplified gradient between layers.

References:
    Ritchie, J.T. (1998) Soil water balance and plant stress. In:
    *Understanding Options for Agricultural Production*, Tsuji et al. (eds.),
    Kluwer Academic Publishers.
"""

from __future__ import annotations

import math

import numpy as np

from dssat.core.constants import (
    NL, RUNINIT, SEASINIT, RATE, INTEGR, OUTPUT,
)
from dssat.core.types import (
    ControlType, SwitchType, SoilType, WeatherType,
    TillType, MulchType, FloodWatType,
)


class WaterBalance:
    """Daily soil water balance.

    This class corresponds to the ``WATBAL`` subroutine.  It is initialised
    once per season and stepped forward one day at a time.

    Attributes:
        sw: Volumetric soil water content by layer (cm³ cm⁻³), shape ``(NL,)``.
        drain: Drainage from bottom of profile (mm d⁻¹).
        drn: Drainage flux by layer (mm d⁻¹), shape ``(NL,)``.
        runoff: Surface runoff (mm d⁻¹).
        winf: Water infiltrated into soil surface (mm d⁻¹).
        snow: Snow pack depth (mm water equivalent).
        upflow: Upward capillary flux by layer (cm d⁻¹), shape ``(NL,)``.
        swdelts: SW change due to drainage/infiltration, shape ``(NL,)``.
        swdeltu: SW change due to upward flow / evaporation, shape ``(NL,)``.
        tswini: Total profile water at season start (cm).
        tsw: Current total profile water (cm).
        crain: Cumulative rainfall (mm).
        tdrain: Cumulative drainage (mm).
        trunof: Cumulative runoff (mm).
    """

    def __init__(self) -> None:
        self.sw = np.zeros(NL)
        self.drain = 0.0
        self.drn = np.zeros(NL)
        self.runoff = 0.0
        self.winf = 0.0
        self.snow = 0.0
        self.upflow = np.zeros(NL)
        self.swdelts = np.zeros(NL)
        self.swdeltu = np.zeros(NL)
        self.tswini = 0.0
        self.tsw = 0.0
        self.crain = 0.0
        self.tdrain = 0.0
        self.trunof = 0.0
        self._swcon = 0.0   # profile drainage constant (0–1)
        self._cn = 75.0     # SCS curve number

    # ------------------------------------------------------------------
    def run(
        self,
        control: ControlType,
        iswitch: SwitchType,
        soilprop: SoilType,
        weather: WeatherType,
        es: float,
        irramt: float,
        swdeltx: np.ndarray,
        tillvals: TillType,
        mulch: MulchType,
        floodwat: FloodWatType,
    ) -> None:
        """Advance the water balance one simulation step.

        Args:
            control: Simulation control block.
            iswitch: Simulation switches.
            soilprop: Current soil profile properties.
            weather: Today's weather.
            es: Potential soil evaporation (mm d⁻¹).
            irramt: Irrigation amount (mm).
            swdeltx: SW change due to root extraction, shape ``(NL,)`` (cm³ cm⁻³).
            tillvals: Tillage state.
            mulch: Mulch state.
            floodwat: Flooded-field water state.
        """
        dynamic = control.dynamic
        if dynamic == RUNINIT:
            pass
        elif dynamic == SEASINIT:
            self._season_init(soilprop)
        elif dynamic == RATE:
            if iswitch.iswwat == "Y":
                self._daily_rate(
                    soilprop, weather, es, irramt, swdeltx, mulch, floodwat, iswitch
                )
        elif dynamic == INTEGR:
            if iswitch.iswwat == "Y":
                self._integrate(soilprop)

    # ------------------------------------------------------------------
    def _season_init(self, soilprop: SoilType) -> None:
        """Seasonal initialisation."""
        nlayr = soilprop.nlayr
        self.snow = 0.0
        self.drain = 0.0
        self.drn[:] = 0.0
        self.runoff = 0.0
        self.upflow[:] = 0.0
        self.swdelts[:] = 0.0
        self.swdeltu[:] = 0.0
        self.winf = 0.0
        self.crain = 0.0
        self.tdrain = 0.0
        self.trunof = 0.0

        # derive drainage constant from SWCN (layer 1)
        self._swcon = min(max(soilprop.swcn[0] / 24.0, 0.0), 1.0)

        # SCS curve number (simplified: function of LL and DUL)
        avg_awc = float(np.mean(soilprop.dul[:nlayr] - soilprop.ll[:nlayr]))
        self._cn = max(40.0, min(95.0, 100.0 - avg_awc * 200.0))

        self.tswini = float(
            np.sum(self.sw[:nlayr] * soilprop.dlayr[:nlayr])
        )
        self.tsw = self.tswini

    # ------------------------------------------------------------------
    def _daily_rate(
        self,
        soilprop: SoilType,
        weather: WeatherType,
        es: float,
        irramt: float,
        swdeltx: np.ndarray,
        mulch: MulchType,
        floodwat: FloodWatType,
        iswitch: SwitchType,
    ) -> None:
        """Compute daily water-balance rate terms."""
        nlayr = soilprop.nlayr
        tmax = weather.tmax
        rain = weather.rain

        # ---- 1. Snow accumulation and melt
        watavl = _snowfall(tmax, rain, self.snow)
        snow_new = self.snow
        if tmax > 1.0:
            snomlt = min(tmax + rain * 0.4, self.snow)
            snow_new = max(self.snow - snomlt, 0.0)
            watavl = rain + snomlt
        elif tmax <= 1.0:
            snow_new = self.snow + rain
            watavl = 0.0
        if snow_new < 0.001:
            snow_new = 0.0
        self.snow = snow_new

        # ---- 2. Runoff (SCS method)
        self.runoff = _scs_runoff(watavl, self._cn, soilprop, nlayr, self.sw)
        self.winf = max(watavl - self.runoff + irramt, 0.0)
        pinf = self.winf * 0.1  # mm → cm

        # ---- 3. Infiltration and saturated flow
        if pinf > 0.0001:
            self.swdelts, self.drn, self.drain = _infil(
                soilprop, nlayr, pinf, self.sw
            )
        else:
            self.swdelts, self.drn, self.drain = _satflo(
                soilprop, nlayr, self.sw, self._swcon
            )

        # ---- 4. Upward flow (capillary rise for evaporation)
        sw_avail = np.maximum(0.0, self.sw + self.swdelts)
        self.upflow, self.swdeltu = _up_flow(
            soilprop, nlayr, self.sw, sw_avail
        )

        # ---- 5. Cumulative summaries
        self.crain += rain
        self.tdrain += self.drain
        self.trunof += self.runoff

    # ------------------------------------------------------------------
    def _integrate(self, soilprop: SoilType) -> None:
        """Apply rate changes to state variables."""
        nlayr = soilprop.nlayr
        self.sw[:nlayr] = np.clip(
            self.sw[:nlayr]
            + self.swdelts[:nlayr]
            + self.swdeltu[:nlayr],
            0.0,
            soilprop.sat[:nlayr],
        )
        self.tsw = float(
            np.sum(self.sw[:nlayr] * soilprop.dlayr[:nlayr])
        )

    # ------------------------------------------------------------------
    def initialise_sw(self, sw_init: np.ndarray, nlayr: int, ll: np.ndarray) -> None:
        """Set initial soil water content from experiment specification.

        Args:
            sw_init: Initial volumetric water content by layer (cm³ cm⁻³).
            nlayr: Number of active layers.
            ll: Lower limit (wilting point) by layer.
        """
        for L in range(nlayr):
            if sw_init[L] < ll[L]:
                if L == 0:
                    air_dry = 0.30 * ll[L]
                    self.sw[L] = max(sw_init[L], air_dry)
                else:
                    self.sw[L] = ll[L]
            else:
                self.sw[L] = sw_init[L]


# ---------------------------------------------------------------------------
# Free functions — individual process algorithms
# ---------------------------------------------------------------------------

def _snowfall(tmax: float, rain: float, snow: float) -> float:
    """Return water available for infiltration after snow accounting."""
    if tmax > 1.0:
        snomlt = min(tmax + rain * 0.4, snow)
        return rain + snomlt
    else:
        return 0.0


def _scs_runoff(
    watavl: float,
    cn: float,
    soilprop: SoilType,
    nlayr: int,
    sw: np.ndarray,
) -> float:
    """SCS curve-number runoff (mm).

    Soil water content is used to adjust the CN for antecedent moisture.

    Args:
        watavl: Water available at surface (mm).
        cn: Base SCS curve number.
        soilprop: Soil properties.
        nlayr: Number of active layers.
        sw: Current water content by layer.

    Returns:
        Surface runoff depth (mm).
    """
    if watavl < 0.001:
        return 0.0

    # Antecedent moisture correction
    theta = float(
        np.mean(
            (sw[:nlayr] - soilprop.ll[:nlayr])
            / np.maximum(soilprop.dul[:nlayr] - soilprop.ll[:nlayr], 0.01)
        )
    )
    theta = min(max(theta, 0.0), 1.0)
    cn_adj = cn * (0.5 + theta * 0.5)  # dry=0.5×CN, wet=CN
    s = 25400.0 / cn_adj - 254.0       # retention parameter (mm)
    ia = 0.2 * s                       # initial abstraction
    if watavl <= ia:
        return 0.0
    return (watavl - ia) ** 2 / (watavl - ia + s)


def _infil(
    soilprop: SoilType,
    nlayr: int,
    pinf: float,  # potential infiltration (cm)
    sw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Tipping-bucket infiltration and drainage.

    Water fills each layer to DUL; excess percolates to the layer below.
    Below the last layer, excess becomes drainage.

    Args:
        soilprop: Soil profile properties.
        nlayr: Number of active layers.
        pinf: Potential infiltration at surface (cm).
        sw: Current soil water content (cm³ cm⁻³).

    Returns:
        Tuple ``(swdelts, drn, drain)`` where *swdelts* and *drn* are
        arrays of shape ``(NL,)`` and *drain* is a scalar (mm d⁻¹).
    """
    swdelts = np.zeros(NL)
    drn = np.zeros(NL)
    excess = pinf

    for L in range(nlayr):
        capacity = (soilprop.dul[L] - sw[L]) * soilprop.dlayr[L]
        capacity = max(capacity, 0.0)
        fill = min(excess, capacity)
        swdelts[L] = fill / soilprop.dlayr[L]
        excess = excess - fill
        # Allow some fast drainage by hydraulic conductivity
        drn[L] = excess

    drain = excess * 10.0  # cm → mm
    return swdelts, drn, drain


def _satflo(
    soilprop: SoilType,
    nlayr: int,
    sw: np.ndarray,
    swcon: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Saturated-flow drainage on dry days (no rain/irrigation).

    Uses the SWCON approach: excess above DUL drains at rate SWCON.

    Args:
        soilprop: Soil properties.
        nlayr: Number of active layers.
        sw: Current soil water (cm³ cm⁻³).
        swcon: Drainage constant (d⁻¹).

    Returns:
        Tuple ``(swdelts, drn, drain)``.
    """
    swdelts = np.zeros(NL)
    drn = np.zeros(NL)
    drain = 0.0

    for L in range(nlayr):
        excess = max(sw[L] - soilprop.dul[L], 0.0)
        if excess > 0.0:
            flow = excess * swcon
            swdelts[L] -= flow
            drn[L] = flow * soilprop.dlayr[L] * 10.0
            drain += drn[L]
            # Move drainage to layer below
            if L + 1 < nlayr:
                capacity = max(soilprop.dul[L + 1] - sw[L + 1], 0.0)
                inflow = min(flow * soilprop.dlayr[L] / soilprop.dlayr[L + 1], capacity)
                swdelts[L + 1] += inflow

    return swdelts, drn, drain


def _up_flow(
    soilprop: SoilType,
    nlayr: int,
    sw: np.ndarray,
    sw_avail: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Upward capillary flow driven by evaporation demand.

    Moves water upward when upper layers are drier than lower layers,
    limited by the hydraulic gradient (water content difference).

    Args:
        soilprop: Soil properties.
        nlayr: Number of active layers.
        sw: Soil water content (cm³ cm⁻³).
        sw_avail: Adjusted soil water including today's drainage.

    Returns:
        Tuple ``(upflow, swdeltu)`` both shape ``(NL,)`` (cm d⁻¹).
    """
    upflow = np.zeros(NL)
    swdeltu = np.zeros(NL)

    for L in range(nlayr - 1, 0, -1):
        thet1 = max(sw_avail[L - 1] - soilprop.ll[L - 1], 0.0)
        thet2 = max(sw_avail[L] - soilprop.ll[L], 0.0)
        if thet2 > thet1:
            grad = thet2 - thet1
            flow = min(grad * 0.1, thet2 * 0.5)
            flow = max(flow, 0.0)
            upflow[L] = flow
            swdeltu[L] -= flow
            swdeltu[L - 1] += flow

    return upflow, swdeltu
