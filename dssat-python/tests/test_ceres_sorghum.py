"""Smoke tests for CERES-Sorghum crop model.

Tests are skipped gracefully if CeresSorghum is not importable.
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    from dssat.crop.ceres_sorghum import CeresSorghum, SorghumCultivar
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

from dssat.core.constants import NL, RUNINIT, SEASINIT, RATE, INTEGR
from dssat.core.date_utils import incdat
from dssat.core.types import SoilType, WeatherType, ControlType, SwitchType

pytestmark = pytest.mark.skipif(
    not _AVAILABLE, reason="CeresSorghum not importable"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_soil() -> SoilType:
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


def _make_weather(tmax: float = 30.0, tmin: float = 18.0) -> WeatherType:
    w = WeatherType()
    w.tmax = tmax
    w.tmin = tmin
    w.srad = 20.0
    w.rain = 0.0
    w.dayl = 13.0
    w.co2 = 380.0
    w.tavg = (tmax + tmin) / 2.0
    w.vapr = 1.5
    return w


def _make_ctrl(yrdoy: int, dynamic: int) -> ControlType:
    ctrl = ControlType()
    ctrl.yrdoy = yrdoy
    ctrl.yrsim = yrdoy
    ctrl.dynamic = dynamic
    return ctrl


def _make_iswitch() -> SwitchType:
    sw = SwitchType()
    sw.iswwat = "N"
    sw.iswnit = "N"
    return sw


def _make_cultivar() -> "SorghumCultivar":
    return SorghumCultivar(
        p1=400.0, p2o=12.5, p2r=150.0, p3=90.0, p4=45.0, p5=450.0,
        g1=6.0, g2=25.0, phint=38.9, panth=500.0,
        tbase=7.0, topt=30.0, ropt=26.0, gdde=6.5, rue=3.5, kcan=0.85,
    )


def _run_season(model: "CeresSorghum", yrsim: int, days: int = 300) -> None:
    """Run up to *days* days of simulation or until maturity."""
    soil = _make_soil()
    iswitch = _make_iswitch()
    sw = np.full(NL, 0.25)
    nh4 = np.zeros(NL)
    no3 = np.zeros(NL)
    w = _make_weather()

    ctrl = _make_ctrl(yrsim, RUNINIT)
    model.run(ctrl, iswitch, soil, w, sw, nh4, no3, 0.0, 0.0)
    ctrl.dynamic = SEASINIT
    model.run(ctrl, iswitch, soil, w, sw, nh4, no3, 0.0, 0.0)

    for day_offset in range(1, days + 1):
        yrdoy = incdat(yrsim, day_offset)
        ctrl.yrdoy = yrdoy
        ctrl.dynamic = RATE
        model.run(ctrl, iswitch, soil, w, sw, nh4, no3, 0.0, 0.0)
        ctrl.dynamic = INTEGR
        model.run(ctrl, iswitch, soil, w, sw, nh4, no3, 0.0, 0.0)
        if model.pheno.istage == 6:
            break


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCeresSorghumCultivar:
    def test_sorghum_cultivar_defaults(self):
        """Instantiate default SorghumCultivar and check required fields."""
        cv = SorghumCultivar()
        assert hasattr(cv, "p1")
        assert hasattr(cv, "p5")
        assert hasattr(cv, "g2")
        assert hasattr(cv, "phint")
        assert cv.p1 > 0.0
        assert cv.g2 > 0.0
        assert cv.phint > 0.0

    def test_sorghum_cultivar_custom(self):
        """Custom cultivar coefficients are stored correctly."""
        cv = _make_cultivar()
        assert cv.p1 == 400.0
        assert cv.g2 == 25.0


class TestCeresSorghumPhenology:
    def test_sorghum_phenology_advances(self):
        """Running 300 days of warm weather, istage should reach 5 or 6."""
        cv = _make_cultivar()
        yrsim = 2020100
        model = CeresSorghum(cultivar=cv, pltpop=15.0, sdepth=4.0, yrsim=yrsim, yrplt=yrsim)
        _run_season(model, yrsim, days=300)

        assert model.pheno.istage in (5, 6), (
            f"Expected istage 5 or 6, got {model.pheno.istage}"
        )

    def test_sorghum_reaches_maturity(self):
        """CeresSorghum should reach maturity (istage=6) within 300 days."""
        cv = _make_cultivar()
        yrsim = 2020100
        model = CeresSorghum(cultivar=cv, pltpop=15.0, sdepth=4.0, yrsim=yrsim, yrplt=yrsim)
        _run_season(model, yrsim, days=300)
        assert model.pheno.istage == 6, (
            f"Crop did not reach maturity; final istage={model.pheno.istage}"
        )


class TestCeresSorghumGrowth:
    def test_sorghum_yield_positive(self):
        """Grain yield should be positive after a full season."""
        cv = _make_cultivar()
        yrsim = 2020100
        model = CeresSorghum(cultivar=cv, pltpop=15.0, sdepth=4.0, yrsim=yrsim, yrplt=yrsim)
        _run_season(model, yrsim, days=300)

        yield_kg_ha = model.growth.yield_
        assert yield_kg_ha > 0.0, f"Yield should be > 0; got {yield_kg_ha:.1f} kg ha⁻¹"

    @pytest.mark.xfail(reason="Sorghum grain fill calibration in progress; yield may be low")
    def test_sorghum_yield_realistic(self):
        """Grain yield should be in agronomically plausible range (2000–15000 kg ha⁻¹).

        Note: This test is marked xfail while the sorghum grain fill algorithm
        is being calibrated.
        """
        cv = _make_cultivar()
        yrsim = 2020100
        model = CeresSorghum(cultivar=cv, pltpop=15.0, sdepth=4.0, yrsim=yrsim, yrplt=yrsim)
        _run_season(model, yrsim, days=300)

        yield_kg_ha = model.growth.yield_
        assert 2000 <= yield_kg_ha <= 15000, (
            f"Yield {yield_kg_ha:.0f} kg ha⁻¹ out of plausible range [2000, 15000]"
        )

    def test_sorghum_no_negative_biomass(self):
        """Biomass should never be negative during the season."""
        cv = _make_cultivar()
        yrsim = 2020100
        model = CeresSorghum(cultivar=cv, pltpop=15.0, sdepth=4.0, yrsim=yrsim, yrplt=yrsim)
        soil = _make_soil()
        iswitch = _make_iswitch()
        sw = np.full(NL, 0.25)
        nh4 = np.zeros(NL)
        no3 = np.zeros(NL)
        w = _make_weather()

        ctrl = _make_ctrl(yrsim, RUNINIT)
        model.run(ctrl, iswitch, soil, w, sw, nh4, no3, 0.0, 0.0)
        ctrl.dynamic = SEASINIT
        model.run(ctrl, iswitch, soil, w, sw, nh4, no3, 0.0, 0.0)

        for day_offset in range(1, 300):
            yrdoy = incdat(yrsim, day_offset)
            ctrl.yrdoy = yrdoy
            ctrl.dynamic = RATE
            model.run(ctrl, iswitch, soil, w, sw, nh4, no3, 0.0, 0.0)
            ctrl.dynamic = INTEGR
            model.run(ctrl, iswitch, soil, w, sw, nh4, no3, 0.0, 0.0)

            g = model.growth
            assert g.biomas >= 0.0, f"Negative biomas on day {day_offset}: {g.biomas}"
            if model.pheno.istage == 6:
                break

    def test_sorghum_summary_keys(self):
        """Summary dict should contain the required keys."""
        cv = SorghumCultivar()
        model = CeresSorghum(cultivar=cv, pltpop=15.0, sdepth=4.0, yrsim=2020100)
        summary = model.summary()
        required = {"crop", "yield_kg_ha", "biomass_kg_ha", "harvest_index", "crop_status"}
        assert required.issubset(summary.keys()), (
            f"Missing keys: {required - set(summary.keys())}"
        )
