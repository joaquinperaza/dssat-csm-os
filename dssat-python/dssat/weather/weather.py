"""Weather module — daily and hourly meteorological processing.

Translated from ``weathr.for``, ``SOLAR.for``, and ``HMET.for``.

The :class:`WeatherModule` reads a sequence of daily weather records
(provided as a list of dicts from the JSON I/O layer), computes solar
geometry, disaggregates to hourly values, and populates a
:class:`~dssat.core.types.WeatherType` object for each simulation day.
"""

from __future__ import annotations

import math
from typing import Sequence

from dssat.core.constants import RUNINIT, SEASINIT, RATE, INTEGR, OUTPUT
from dssat.core.types import ControlType, SwitchType, WeatherType
from dssat.core.date_utils import yr_doy
from dssat.weather.solar import day_length, solar_params, generate_hourly

PI = math.pi
CO2_DEFAULT = 380.0


class WeatherModule:
    """Daily weather processing, solar geometry, and hourly disaggregation.

    This class mirrors the ``WEATHR`` subroutine in ``weathr.for``.  On each
    call to :meth:`run`, it reads the daily record for ``CONTROL.yrdoy``,
    computes derived quantities (PAR, cloudiness, hourly profiles), and
    populates the shared :class:`WeatherType` state object.

    Attributes:
        weather: The :class:`WeatherType` output populated each day.

    Args:
        daily_records: List of daily weather dicts keyed by YYYYDDD integer.
            Each dict must supply ``tmax``, ``tmin``, ``srad``, ``rain`` and
            optionally ``rhum``, ``windsp``, ``tdew``, ``co2``.
        xlat: Station latitude (decimal degrees, negative = south).
        xlong: Station longitude (decimal degrees, negative = west).
        xelev: Station elevation (m).
        refht: Reference height for temperature/humidity (m).
        windht: Instrument height for wind (m).
        tav: Annual average temperature (°C) — used for soil temperature.
        tamp: Annual temperature amplitude (°C).
        co2: Background CO₂ concentration (ppm).
    """

    def __init__(
        self,
        daily_records: dict[int, dict],
        xlat: float,
        xlong: float = 0.0,
        xelev: float = 0.0,
        refht: float = 2.0,
        windht: float = 10.0,
        tav: float = 20.0,
        tamp: float = 10.0,
        co2: float = CO2_DEFAULT,
    ) -> None:
        self._records = daily_records
        self.weather = WeatherType()
        self.weather.xlat = xlat
        self.weather.xlong = xlong
        self.weather.xelev = xelev
        self.weather.refht = refht
        self.weather.windht = windht
        self.weather.tav = tav
        self.weather.tamp = tamp
        self.weather.co2 = co2

    # ------------------------------------------------------------------
    def run(
        self,
        control: ControlType,
        iswitch: SwitchType,
    ) -> None:
        """Process one simulation step.

        Args:
            control: Simulation control block (provides ``yrdoy`` and ``dynamic``).
            iswitch: Simulation switches (not currently used but kept for API
                consistency with the Fortran interface).
        """
        dynamic = control.dynamic
        if dynamic in (RUNINIT, SEASINIT):
            self._init(control)
        if dynamic in (RATE, INTEGR, OUTPUT):
            self._daily_update(control)

    # ------------------------------------------------------------------
    def _init(self, control: ControlType) -> None:
        """Initialise the weather module for a new season."""
        w = self.weather
        # Carry station metadata; CO2 initialised from file or default
        w.co2 = self.weather.co2

    # ------------------------------------------------------------------
    def _daily_update(self, control: ControlType) -> None:
        """Read today's record and compute all derived weather variables."""
        yrdoy = control.yrdoy
        rec = self._records.get(yrdoy)
        if rec is None:
            raise KeyError(f"No weather record for YRDOY={yrdoy}")

        w = self.weather
        year, doy = yr_doy(yrdoy)

        # -- mandatory daily inputs
        w.tmax = float(rec["tmax"])
        w.tmin = float(rec["tmin"])
        w.srad = float(rec["srad"])
        w.rain = float(rec.get("rain", 0.0))

        # -- optional daily inputs with fallbacks
        w.windsp = float(rec.get("windsp", 150.0))
        w.rhum = float(rec.get("rhum", -99.0))
        w.co2 = float(rec.get("co2", w.co2))

        # -- solar geometry
        w.dayl, dec, w.snup, w.sndn = day_length(doy, w.xlat)
        w.twilen = w.dayl  # twilight = dayl (approximate)

        # -- dew point
        if "tdew" in rec:
            w.tdew = float(rec["tdew"])
        elif w.rhum > 0:
            w.tdew = _rh_to_tdew(w.tmin, w.rhum)
        else:
            w.tdew = w.tmin - 2.0  # fallback: assume Tdew ~ Tmin - 2

        # -- vapour pressure
        w.vapr = _vpsat(w.tdew) / 1000.0  # Pa → kPa
        vpsat_air = _vpsat((w.tmax + w.tmin) / 2.0) / 1000.0
        w.vpdf = max(vpsat_air - w.vapr, 0.0)
        w.vpd_transp = w.vpdf

        # -- PAR (MJ m⁻² d⁻¹); default 0.5 * SRAD if not supplied
        w.par = float(rec.get("par", 0.5 * w.srad))

        # -- cloudiness and Spitters integral
        w.clouds, isinb, s0n = solar_params(w.dayl, dec, w.srad, w.xlat)

        # -- hourly disaggregation (HMET)
        hourly = generate_hourly(
            clouds=w.clouds,
            dayl=w.dayl,
            dec=dec,
            isinb=isinb,
            par=w.par,
            refht=w.refht,
            sndn=w.sndn,
            snup=w.snup,
            s0n=s0n,
            srad=w.srad,
            tdew=w.tdew,
            tmax=w.tmax,
            tmin=w.tmin,
            windht=w.windht,
            windsp=w.windsp,
            xlat=w.xlat,
        )
        w.tairhr[:] = hourly["tairhr"]
        w.tgro[:] = hourly["tgro"]
        w.radhr[:] = hourly["radhr"]
        w.parhr[:] = hourly["parhr"]
        w.rhumhr[:] = hourly["rhumhr"]
        w.windhr[:] = hourly["windhr"]
        w.amtrh[:] = hourly["amtrh"]
        w.azzon[:] = hourly["azzon"]
        w.beta[:] = hourly["beta"]
        w.frdifp[:] = hourly["frdifp"]
        w.frdifr[:] = hourly["frdifr"]
        w.tavg = hourly["tavg"]
        w.tday = hourly["tday"]
        w.tgroav = hourly["tgroav"]
        w.tgrody = hourly["tgrody"]

        # wind run (km d⁻¹ → already in windsp)
        w.windrun = w.windsp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vpsat(temp: float) -> float:
    """Saturated vapour pressure (Pa) — Tetens formula."""
    return 610.78 * math.exp(17.269 * temp / (temp + 237.3))


def _rh_to_tdew(tmin: float, rh: float) -> float:
    """Estimate dew point from Tmin and RH (%).

    Simple approximation: dew point ≈ Tmin when RH is high.
    """
    if rh >= 100.0:
        return tmin
    # Magnus approximation
    a, b = 17.625, 243.04
    gamma = math.log(rh / 100.0) + a * tmin / (b + tmin)
    return b * gamma / (a - gamma)
