"""Smoke tests for CERES-Wheat crop model.

Tests are skipped gracefully if CeresWheat is not importable.
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    from dssat.crop.ceres_wheat import CeresWheat, WheatCultivar
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

from dssat.core.constants import NL, RUNINIT, SEASINIT, RATE, INTEGR
from dssat.core.date_utils import incdat
from dssat.core.types import SoilType, WeatherType, ControlType, SwitchType

pytestmark = pytest.mark.skipif(
    not _AVAILABLE, reason="CeresWheat not importable"
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


def _make_weather(tmax: float = 22.0, tmin: float = 10.0) -> WeatherType:
    """Cool-season weather appropriate for wheat."""
    w = WeatherType()
    w.tmax = tmax
    w.tmin = tmin
    w.srad = 16.0
    w.rain = 0.0
    w.dayl = 14.0
    w.co2 = 380.0
    w.tavg = (tmax + tmin) / 2.0
    w.vapr = 1.0
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


def _make_cultivar() -> "WheatCultivar":
    """Spring wheat cultivar (p1v=0) for faster testing."""
    return WheatCultivar(
        p1v=0.0, p1d=3.675, p5=500.0,
        g1=6.24, g2=32.0, g3=1.15, phint=95.0,
        tbase=0.0, topt=26.0,
    )


def _run_season(model: "CeresWheat", yrsim: int, days: int = 350) -> None:
    """Run up to *days* days or until maturity."""
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

class TestCeresWheatCultivar:
    def test_wheat_cultivar_defaults(self):
        """Instantiate default WheatCultivar and check required fields."""
        cv = WheatCultivar()
        assert hasattr(cv, "p1v")
        assert hasattr(cv, "p5")
        assert hasattr(cv, "g2")
        assert hasattr(cv, "phint")
        assert cv.p5 > 0.0
        assert cv.g2 > 0.0
        assert cv.phint > 0.0

    def test_wheat_cultivar_varno_alias(self):
        """WheatCultivar.varno should equal .id."""
        cv = WheatCultivar(id="TEST01")
        assert cv.varno == "TEST01"

    def test_wheat_cultivar_custom(self):
        """Custom cultivar coefficients are stored correctly."""
        cv = _make_cultivar()
        assert cv.p1v == 0.0
        assert cv.g2 == 32.0


class TestCeresWheatPhenology:
    def test_wheat_phenology_advances(self):
        """Running 350 days of suitable weather, istage should reach 5 or 6."""
        cv = _make_cultivar()
        yrsim = 2020100
        model = CeresWheat(
            cultivar=cv, pltpop=300.0, sdepth=3.0, rowspc=20.0,
            yrsim=yrsim, yrplt=yrsim,
        )
        _run_season(model, yrsim, days=350)

        assert model.pheno.istage in (5, 6), (
            f"Expected istage 5 or 6, got {model.pheno.istage}"
        )

    def test_wheat_reaches_maturity(self):
        """CeresWheat spring type should reach maturity within 350 days."""
        cv = _make_cultivar()
        yrsim = 2020100
        model = CeresWheat(
            cultivar=cv, pltpop=300.0, sdepth=3.0, rowspc=20.0,
            yrsim=yrsim, yrplt=yrsim,
        )
        _run_season(model, yrsim, days=350)
        assert model.pheno.istage == 6, (
            f"Crop did not reach maturity; final istage={model.pheno.istage}"
        )


class TestCeresWheatGrowth:
    def test_wheat_yield_reasonable(self):
        """Grain yield should be in plausible range (1000–10000 kg ha⁻¹)."""
        cv = _make_cultivar()
        yrsim = 2020100
        model = CeresWheat(
            cultivar=cv, pltpop=300.0, sdepth=3.0, rowspc=20.0,
            yrsim=yrsim, yrplt=yrsim,
        )
        _run_season(model, yrsim, days=350)

        yield_kg_ha = model.growth.yield_
        assert 1000 <= yield_kg_ha <= 10000, (
            f"Yield {yield_kg_ha:.0f} kg ha⁻¹ out of plausible range [1000, 10000]"
        )

    def test_wheat_no_negative_biomass(self):
        """Biomass should never be negative during the season."""
        cv = _make_cultivar()
        yrsim = 2020100
        model = CeresWheat(
            cultivar=cv, pltpop=300.0, sdepth=3.0, rowspc=20.0,
            yrsim=yrsim, yrplt=yrsim,
        )
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

        for day_offset in range(1, 350):
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

    def test_wheat_summary_keys(self):
        """Summary dict should contain the required keys."""
        cv = WheatCultivar()
        model = CeresWheat(cultivar=cv, pltpop=300.0, sdepth=3.0)
        summary = model.summary()
        required = {"crop", "yield_kg_ha", "biomass_kg_ha", "harvest_index", "crop_status"}
        assert required.issubset(summary.keys()), (
            f"Missing keys: {required - set(summary.keys())}"
        )
