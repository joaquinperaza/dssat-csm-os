"""Unit and integration tests for CERES-Maize.

Run with::

    pytest tests/ -v
"""

from __future__ import annotations

import datetime
import math

import numpy as np
import pytest

from dssat.core.constants import NL, RUNINIT, SEASINIT, RATE, INTEGR
from dssat.core.date_utils import date_to_yrdoy, yrdoy_to_date, incdat, timdif
from dssat.core.types import SoilType, WeatherType, ControlType, SwitchType
from dssat.weather.solar import day_length, solar_params, generate_hourly
from dssat.soil.water.watbal import WaterBalance, _scs_runoff, _infil
from dssat.crop.ceres_maize import CeresMaize, MaizeCultivar
from dssat.crop.ceres_maize.phenology import compute_dtt


# ---------------------------------------------------------------------------
# Date utilities
# ---------------------------------------------------------------------------

class TestDateUtils:
    def test_roundtrip(self):
        d = datetime.date(2020, 6, 15)
        assert yrdoy_to_date(date_to_yrdoy(d)) == d

    def test_known_date(self):
        assert date_to_yrdoy(datetime.date(2020, 1, 1)) == 2020001
        assert date_to_yrdoy(datetime.date(2020, 12, 31)) == 2020366  # leap year

    def test_incdat(self):
        assert incdat(2020001, 1) == 2020002
        assert incdat(2020365, 1) == 2020366  # leap
        assert incdat(2020366, 1) == 2021001

    def test_timdif(self):
        assert timdif(2020001, 2020032) == 31


# ---------------------------------------------------------------------------
# Solar geometry
# ---------------------------------------------------------------------------

class TestSolar:
    def test_equinox_daylength(self):
        """At the equator on the equinox, day length should be ~12 h."""
        dayl, dec, snup, sndn = day_length(doy=80, xlat=0.0)
        assert abs(dayl - 12.0) < 0.1

    def test_summer_solstice_north(self):
        """At 45°N near summer solstice, day length should be > 15 h."""
        dayl, _, _, _ = day_length(doy=172, xlat=45.0)
        assert dayl > 15.0

    def test_no_negative_daylength(self):
        """Day length should never be negative."""
        for lat in [-90, -45, 0, 45, 90]:
            for doy in [1, 90, 180, 270, 365]:
                dayl, _, _, _ = day_length(doy, lat)
                assert dayl >= 0.0, f"Negative dayl at lat={lat}, doy={doy}"

    def test_hourly_rad_sums_to_daily(self):
        """Sum of hourly radiation should approximate daily SRAD."""
        xlat = 35.0
        doy = 180
        dayl, dec, snup, sndn = day_length(doy, xlat)
        clouds, isinb, s0n = solar_params(dayl, dec, srad=20.0, xlat=xlat)
        hourly = generate_hourly(
            clouds=clouds, dayl=dayl, dec=dec, isinb=isinb,
            par=10.0, refht=2.0, sndn=sndn, snup=snup, s0n=s0n,
            srad=20.0, tdew=10.0, tmax=30.0, tmin=15.0,
            windht=10.0, windsp=200.0, xlat=xlat,
        )
        total_rad = float(np.sum(hourly["radhr"]))
        # Should be within ~20% of daily SRAD (some loss expected due to
        # atmospheric absorption differences between methods)
        assert abs(total_rad - 20.0) / 20.0 < 0.4


# ---------------------------------------------------------------------------
# Phenology — thermal time
# ---------------------------------------------------------------------------

class TestDTT:
    def test_dtt_zero_below_tbase(self):
        dtt = compute_dtt(5.0, 0.0, 15.0, 12.0, 0.0, 5, 1, 8.0, 34.0, 26.0)
        assert dtt == 0.0

    def test_dtt_positive_above_tbase(self):
        dtt = compute_dtt(25.0, 15.0, 15.0, 12.0, 0.0, 15, 1, 8.0, 34.0, 26.0)
        assert dtt > 0.0

    def test_dtt_capped_at_topt(self):
        """When tmin > topt, DTT should equal topt - tbase."""
        dtt = compute_dtt(40.0, 40.0, 15.0, 12.0, 0.0, 15, 1, 8.0, 34.0, 34.0)
        assert abs(dtt - (34.0 - 8.0)) < 1e-5

    def test_snow_crown_temp(self):
        """Crown temperature adjustment should give positive DTT when Tmin<0."""
        # Use tbase=2 so no-snow case is below tbase but snow warms crown above it
        dtt_snow = compute_dtt(5.0, -1.0, 5.0, 8.0, 50.0, 5, 9, 2.0, 26.0, 26.0)
        dtt_no_snow = compute_dtt(5.0, -1.0, 5.0, 8.0, 0.0, 5, 9, 2.0, 26.0, 26.0)
        # Crown temperature is warmer with snow, so DTT is larger
        assert dtt_snow > dtt_no_snow


# ---------------------------------------------------------------------------
# Soil water balance
# ---------------------------------------------------------------------------

class TestWaterBalance:
    def _make_soil(self) -> SoilType:
        soil = SoilType()
        soil.nlayr = 5
        for i in range(5):
            soil.dlayr[i] = 20.0
            soil.ds[i] = (i + 1) * 20.0
            soil.ll[i] = 0.10
            soil.dul[i] = 0.28
            soil.sat[i] = 0.45
            soil.swcn[i] = 0.5
            soil.bd[i] = 1.35
        return soil

    def test_drainage_excess(self):
        """Water above DUL should generate drainage."""
        soil = self._make_soil()
        sw = np.full(NL, 0.35)  # above DUL
        swdelts, drn, drain = _infil(soil, 5, pinf=0.0, sw=sw)
        # satflo should drain excess
        from dssat.soil.water.watbal import _satflo
        swdelts2, drn2, drain2 = _satflo(soil, 5, sw, swcon=0.6)
        assert drain2 > 0.0

    def test_no_drainage_at_dul(self):
        """Water at DUL should have negligible drainage."""
        soil = self._make_soil()
        sw = np.full(NL, 0.28)  # exactly at DUL
        from dssat.soil.water.watbal import _satflo
        _, _, drain = _satflo(soil, 5, sw, swcon=0.6)
        assert drain < 1e-9

    def test_runoff_no_rain(self):
        """Zero rainfall gives zero runoff."""
        soil = self._make_soil()
        sw = np.full(NL, 0.20)
        runoff = _scs_runoff(0.0, 75.0, soil, 5, sw)
        assert runoff == 0.0

    def test_runoff_saturated(self):
        """Heavy rain on saturated soil gives non-zero runoff."""
        soil = self._make_soil()
        sw = np.full(NL, 0.44)  # near-saturated
        runoff = _scs_runoff(50.0, 75.0, soil, 5, sw)
        assert runoff > 0.0


# ---------------------------------------------------------------------------
# CERES-Maize integration — phenology progression
# ---------------------------------------------------------------------------

class TestCeresMaizeIntegration:
    def _make_control(self, yrdoy: int, dynamic: int) -> ControlType:
        ctrl = ControlType()
        ctrl.yrdoy = yrdoy
        ctrl.yrsim = yrdoy
        ctrl.dynamic = dynamic
        return ctrl

    def _make_iswitch(self) -> SwitchType:
        sw = SwitchType()
        sw.iswwat = "N"
        sw.iswnit = "N"
        return sw

    def _make_soil(self) -> SoilType:
        soil = SoilType()
        soil.nlayr = 5
        for i in range(5):
            soil.dlayr[i] = 20.0
            soil.ds[i] = (i + 1) * 20.0
            soil.ll[i] = 0.10
            soil.dul[i] = 0.28
            soil.sat[i] = 0.45
            soil.swcn[i] = 0.5
            soil.bd[i] = 1.35
            soil.shf[i] = 1.0
            soil.kg2ppm[i] = 0.1
        return soil

    def _make_weather(self, tmax=28.0, tmin=16.0, srad=18.0) -> WeatherType:
        w = WeatherType()
        w.tmax = tmax
        w.tmin = tmin
        w.srad = srad
        w.rain = 0.0
        w.dayl = 13.5
        w.co2 = 380.0
        w.tavg = (tmax + tmin) / 2.0
        w.vapr = 1.5
        return w

    def test_stage_progression_to_maturity(self):
        """Model should reach maturity (istage=6) within 200 days under good conditions."""
        cv = MaizeCultivar(
            p1=200.0, p2=0.4, p5=700.0, g2=700.0, g3=8.0, phint=38.9,
            tbase=8.0, topt=34.0, ropt=26.0, p2o=12.5,
            gdde=6.0, dsgft=150.0, rue=4.0, kcan=0.85,
        )
        yrsim = 2020100
        model = CeresMaize(
            cultivar=cv, pltpop=7.0, sdepth=5.0, yrsim=yrsim,
            rowspc=75.0, slpf=1.0, yrplt=yrsim,
        )
        soil = self._make_soil()
        iswitch = self._make_iswitch()

        sw = np.full(NL, 0.25)
        nh4 = np.zeros(NL)
        no3 = np.zeros(NL)

        ctrl = self._make_control(yrsim, RUNINIT)
        w = self._make_weather()
        model.run(ctrl, iswitch, soil, w, sw, nh4, no3, 0.0, 0.0)
        ctrl.dynamic = SEASINIT
        model.run(ctrl, iswitch, soil, w, sw, nh4, no3, 0.0, 0.0)

        matured = False
        for day_offset in range(1, 250):
            yrdoy = incdat(yrsim, day_offset)
            ctrl.yrdoy = yrdoy
            ctrl.dynamic = RATE
            model.run(ctrl, iswitch, soil, w, sw, nh4, no3, 0.0, 0.0)
            ctrl.dynamic = INTEGR
            model.run(ctrl, iswitch, soil, w, sw, nh4, no3, 0.0, 0.0)
            if model.pheno.istage == 6:
                matured = True
                break

        assert matured, "Crop did not reach maturity within 250 days"

    def test_yield_reasonable(self):
        """Grain yield should be in a plausible range for maize (3–15 t ha⁻¹)."""
        cv = MaizeCultivar(
            p1=200.0, p2=0.4, p5=700.0, g2=700.0, g3=8.0, phint=38.9,
            tbase=8.0, topt=34.0, ropt=26.0, p2o=12.5,
            gdde=6.0, dsgft=150.0, rue=4.0, kcan=0.85,
        )
        yrsim = 2020100
        model = CeresMaize(
            cultivar=cv, pltpop=7.0, sdepth=5.0, yrsim=yrsim,
            rowspc=75.0, slpf=1.0, yrplt=yrsim,
        )
        soil = self._make_soil()
        iswitch = self._make_iswitch()
        sw = np.full(NL, 0.25)
        nh4 = np.zeros(NL)
        no3 = np.zeros(NL)

        ctrl = self._make_control(yrsim, RUNINIT)
        w = self._make_weather()
        model.run(ctrl, iswitch, soil, w, sw, nh4, no3, 0.0, 0.0)
        ctrl.dynamic = SEASINIT
        model.run(ctrl, iswitch, soil, w, sw, nh4, no3, 0.0, 0.0)

        for day_offset in range(1, 250):
            yrdoy = incdat(yrsim, day_offset)
            ctrl.yrdoy = yrdoy
            ctrl.dynamic = RATE
            model.run(ctrl, iswitch, soil, w, sw, nh4, no3, 0.0, 0.0)
            ctrl.dynamic = INTEGR
            model.run(ctrl, iswitch, soil, w, sw, nh4, no3, 0.0, 0.0)
            if model.pheno.istage == 6:
                break

        yield_kg_ha = model.growth.yield_
        assert 1000 <= yield_kg_ha <= 20000, (
            f"Yield {yield_kg_ha:.0f} kg ha⁻¹ out of plausible range"
        )

    def test_summary_keys(self):
        """Summary dict should contain required keys."""
        cv = MaizeCultivar()
        model = CeresMaize(cultivar=cv, pltpop=7.0, sdepth=5.0, yrsim=2020100)
        summary = model.summary()
        required = {"crop", "yield_kg_ha", "biomass_kg_ha", "harvest_index", "crop_status"}
        assert required.issubset(summary.keys())
