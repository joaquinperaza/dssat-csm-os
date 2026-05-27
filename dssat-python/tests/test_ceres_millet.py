"""Smoke tests for CERES-Millet crop model.

Tests are skipped gracefully if CeresMillet is not importable.
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    from dssat.crop.ceres_millet import CeresMillet, MilletCultivar
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

from dssat.core.constants import NL, RUNINIT, SEASINIT, RATE, INTEGR
from dssat.core.date_utils import incdat
from dssat.core.types import SoilType, WeatherType, ControlType, SwitchType

pytestmark = pytest.mark.skipif(
    not _AVAILABLE, reason="CeresMillet not importable"
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


def _make_weather(tmax: float = 32.0, tmin: float = 20.0) -> WeatherType:
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


def _make_cultivar() -> "MilletCultivar":
    # New API: p3/p4 are dynamically computed in the Fortran model;
    # g4 is the partitioning coefficient, g5 is kernel size (mg).
    return MilletCultivar(
        p1=350.0, p2o=12.5, p2r=100.0, p5=400.0,
        g4=0.9, g5=8.0, phint=38.9,
        tbase=8.0, rue=3.0, kcan=0.85,
    )


def _run_season(model: "CeresMillet", yrsim: int, days: int = 300) -> None:
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

class TestCeresMilletCultivar:
    def test_millet_cultivar_defaults(self):
        """Instantiate default MilletCultivar and check required fields."""
        cv = MilletCultivar()
        assert hasattr(cv, "p1")
        assert hasattr(cv, "p5")
        assert hasattr(cv, "g5")   # g5 = kernel size (mg)
        assert hasattr(cv, "phint")
        assert cv.p1 > 0.0
        assert cv.g5 > 0.0
        assert cv.phint > 0.0

    def test_millet_cultivar_custom(self):
        """Custom cultivar coefficients are stored correctly."""
        cv = _make_cultivar()
        assert cv.p1 == 350.0
        assert cv.g5 == 8.0


class TestCeresMilletPhenology:
    def test_millet_phenology_advances(self):
        """Running 300 days of warm weather, istage should reach 5 or 6."""
        cv = _make_cultivar()
        yrsim = 2020100
        model = CeresMillet(cultivar=cv, pltpop=25.0, sdepth=3.0, yrsim=yrsim, yrplt=yrsim)
        _run_season(model, yrsim, days=300)

        assert model.pheno.istage in (5, 6), (
            f"Expected istage 5 or 6, got {model.pheno.istage}"
        )

    def test_millet_reaches_maturity(self):
        """CeresMillet should reach maturity (istage=6) within 300 days."""
        cv = _make_cultivar()
        yrsim = 2020100
        model = CeresMillet(cultivar=cv, pltpop=25.0, sdepth=3.0, yrsim=yrsim, yrplt=yrsim)
        _run_season(model, yrsim, days=300)
        assert model.pheno.istage == 6, (
            f"Crop did not reach maturity; final istage={model.pheno.istage}"
        )


class TestCeresMilletGrowth:
    def test_millet_yield_positive(self):
        """Grain yield should be positive after a full season."""
        cv = _make_cultivar()
        yrsim = 2020100
        model = CeresMillet(cultivar=cv, pltpop=25.0, sdepth=3.0, yrsim=yrsim, yrplt=yrsim)
        _run_season(model, yrsim, days=300)

        yield_kg_ha = model.growth.yield_
        assert yield_kg_ha > 0.0, f"Yield should be > 0; got {yield_kg_ha:.1f} kg ha⁻¹"

    @pytest.mark.xfail(reason="Millet grain fill calibration in progress; yield may be low")
    def test_millet_yield_realistic(self):
        """Grain yield should be in agronomically plausible range (1000–8000 kg ha⁻¹).

        Note: This test is marked xfail while the millet grain fill algorithm
        is being calibrated.
        """
        cv = _make_cultivar()
        yrsim = 2020100
        model = CeresMillet(cultivar=cv, pltpop=25.0, sdepth=3.0, yrsim=yrsim, yrplt=yrsim)
        _run_season(model, yrsim, days=300)

        yield_kg_ha = model.growth.yield_
        assert 1000 <= yield_kg_ha <= 8000, (
            f"Yield {yield_kg_ha:.0f} kg ha⁻¹ out of plausible range [1000, 8000]"
        )

    def test_millet_no_negative_biomass(self):
        """Biomass should never be negative during the season."""
        cv = _make_cultivar()
        yrsim = 2020100
        model = CeresMillet(cultivar=cv, pltpop=25.0, sdepth=3.0, yrsim=yrsim, yrplt=yrsim)
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

    def test_millet_summary_keys(self):
        """Summary dict should contain the required keys."""
        cv = MilletCultivar()
        model = CeresMillet(cultivar=cv, pltpop=25.0, sdepth=3.0, yrsim=2020100)
        summary = model.summary()
        required = {"crop", "yield_kg_ha", "biomass_kg_ha", "harvest_index", "crop_status"}
        assert required.issubset(summary.keys()), (
            f"Missing keys: {required - set(summary.keys())}"
        )
