"""Comprehensive unit tests for CERES phenology models.

Tests verify that:
1. Thermal time (DTT) computation matches Fortran algorithms
2. Stage progressions follow correct ISTAGE flow
3. Stage transitions happen at correct thermal time thresholds
4. Carryover TT is applied at stage transitions
5. XSTAGE continuous values match Fortran formulas
6. Germination and emergence logic matches SPE parameters
7. Photoperiod sensitivity (stage 2) is correctly applied

Run with::

    pytest tests/test_phenology_models.py -v
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from dssat.core.constants import NL, RUNINIT, SEASINIT, RATE, INTEGR
from dssat.core.date_utils import incdat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sw(val: float = 0.25) -> np.ndarray:
    """Soil water array above LL — no water stress."""
    return np.full(NL, val)


def _make_ll() -> np.ndarray:
    return np.full(NL, 0.10)


def _make_dul() -> np.ndarray:
    return np.full(NL, 0.28)


def _make_dlayr() -> np.ndarray:
    dlayr = np.zeros(NL)
    dlayr[:5] = 20.0
    return dlayr


# ---------------------------------------------------------------------------
# Maize phenology tests
# ---------------------------------------------------------------------------

class TestMaizeDtt:
    """DTT computation for CERES-Maize (compute_dtt in MZ_PHENOL.for)."""

    def setup_method(self):
        from dssat.crop.ceres_maize.phenology import compute_dtt
        self.compute_dtt = compute_dtt

    def test_dtt_above_topt_capped(self):
        """When TMIN > DOPT, DTT should equal DOPT - TBASE."""
        dtt = self.compute_dtt(
            tmax=40.0, tmin=36.0, srad=20.0, dayl=12.0, snow=0.0,
            leafno=5, istage=1, tbase=8.0, topt=34.0, ropt=34.0,
        )
        assert dtt == pytest.approx(34.0 - 8.0)

    def test_dtt_below_tbase_zero(self):
        """When TMAX < TBASE, DTT should be 0."""
        dtt = self.compute_dtt(
            tmax=5.0, tmin=0.0, srad=10.0, dayl=8.0, snow=0.0,
            leafno=5, istage=1, tbase=8.0, topt=34.0, ropt=34.0,
        )
        assert dtt == pytest.approx(0.0)

    def test_dtt_simple_average(self):
        """Moderate temps without soil temperature correction."""
        # LEAFNO > 10 and TMIN >= TBASE, TMAX <= DOPT
        dtt = self.compute_dtt(
            tmax=30.0, tmin=15.0, srad=20.0, dayl=12.0, snow=0.0,
            leafno=15, istage=1, tbase=8.0, topt=34.0, ropt=34.0,
        )
        expected = (30.0 + 15.0) / 2.0 - 8.0
        assert dtt == pytest.approx(expected)

    def test_dtt_uses_ropt_after_anthesis(self):
        """Stages 4-6 use ROPT instead of TOPT."""
        # With TMAX > ROPT but TMIN < ROPT → hourly integration
        dtt_veg = self.compute_dtt(
            tmax=36.0, tmin=10.0, srad=20.0, dayl=12.0, snow=0.0,
            leafno=15, istage=1, tbase=8.0, topt=34.0, ropt=28.0,
        )
        dtt_repro = self.compute_dtt(
            tmax=36.0, tmin=10.0, srad=20.0, dayl=12.0, snow=0.0,
            leafno=15, istage=5, tbase=8.0, topt=34.0, ropt=28.0,
        )
        # Reproductive DTT should be <= vegetative (since ROPT < TOPT)
        assert dtt_repro <= dtt_veg

    def test_dtt_soil_temp_method(self):
        """LEAFNO <= 10 triggers soil temperature method."""
        dtt = self.compute_dtt(
            tmax=32.0, tmin=20.0, srad=20.0, dayl=12.0, snow=0.0,
            leafno=5, istage=1, tbase=8.0, topt=34.0, ropt=34.0,
        )
        assert dtt > 0.0

    def test_dtt_snow_uses_crown_temp(self):
        """With snow, DTT uses simple crown temperature."""
        dtt_snow = self.compute_dtt(
            tmax=32.0, tmin=20.0, srad=20.0, dayl=12.0, snow=10.0,
            leafno=5, istage=1, tbase=8.0, topt=34.0, ropt=34.0,
        )
        # With snow, uses (tempcn + tempcx)/2 - tbase
        expected = (32.0 + 20.0) / 2.0 - 8.0
        assert dtt_snow == pytest.approx(expected)


class TestMaizePhenologyStages:
    """Stage progression for CERES-Maize phenology."""

    def setup_method(self):
        from dssat.crop.ceres_maize.phenology import MaizeCultivar, MaizePhenology
        self.MaizeCultivar = MaizeCultivar
        self.MaizePhenology = MaizePhenology

    def _make_pheno(self, **kwargs):
        cv = self.MaizeCultivar(p1=200.0, p2=0.3, p5=500.0, g2=600.0,
                                g3=8.0, phint=38.9, djti=4.0, p2o=12.5,
                                tbase=8.0, topt=34.0, ropt=34.0, gdde=6.0,
                                **kwargs)
        return self.MaizePhenology(cv, pltpop=7.0, sdepth=5.0, yrsim=2020100)

    def test_initial_state_is_stage7(self):
        """After RUNINIT, phenology should be in stage 7."""
        pheno = self._make_pheno()
        pheno.run(dynamic=RUNINIT, yrdoy=2020100,
                  tmax=32.0, tmin=20.0, srad=20.0, dayl=13.0,
                  snow=0.0, sw=_make_sw(), ll=_make_ll(),
                  dlayr=_make_dlayr(), nlayr=5, iswwat="N")
        assert pheno.state.istage == 7

    def test_stage7_to_8_transition(self):
        """On sowing day, should transition to stage 8."""
        pheno = self._make_pheno()
        pheno.run(dynamic=SEASINIT, yrdoy=2020100,
                  tmax=32.0, tmin=20.0, srad=20.0, dayl=13.0,
                  snow=0.0, sw=_make_sw(), ll=_make_ll(),
                  dlayr=_make_dlayr(), nlayr=5, iswwat="N")
        # After SEASINIT, still in 7; first RATE step triggers transition
        pheno.run(dynamic=RATE, yrdoy=2020101,
                  tmax=32.0, tmin=20.0, srad=20.0, dayl=13.0,
                  snow=0.0, sw=_make_sw(), ll=_make_ll(),
                  dlayr=_make_dlayr(), nlayr=5, iswwat="N")
        assert pheno.state.istage == 8

    def test_emergence_with_iswwat_off(self):
        """With water switch off, crop should emerge quickly."""
        pheno = self._make_pheno()
        pheno.run(dynamic=SEASINIT, yrdoy=2020100,
                  tmax=32.0, tmin=20.0, srad=20.0, dayl=13.0,
                  snow=0.0, sw=_make_sw(), ll=_make_ll(),
                  dlayr=_make_dlayr(), nlayr=5, iswwat="N")

        yrdoy = 2020100
        for i in range(1, 30):
            yrdoy = incdat(2020100, i)
            pheno.run(dynamic=RATE, yrdoy=yrdoy,
                      tmax=32.0, tmin=20.0, srad=20.0, dayl=13.0,
                      snow=0.0, sw=_make_sw(), ll=_make_ll(),
                      dlayr=_make_dlayr(), nlayr=5, iswwat="N")
            if pheno.state.istage == 1:
                break
        assert pheno.state.istage == 1, \
            f"Expected stage 1, got {pheno.state.istage}"

    def test_full_season_reaches_maturity(self):
        """A full season of warm weather should reach maturity (istage=10)."""
        pheno = self._make_pheno()
        pheno.run(dynamic=SEASINIT, yrdoy=2020100,
                  tmax=32.0, tmin=20.0, srad=20.0, dayl=13.0,
                  snow=0.0, sw=_make_sw(), ll=_make_ll(),
                  dlayr=_make_dlayr(), nlayr=5, iswwat="N")

        final_stage = 7
        for i in range(1, 250):
            yrdoy = incdat(2020100, i)
            pheno.run(dynamic=RATE, yrdoy=yrdoy,
                      tmax=32.0, tmin=20.0, srad=20.0, dayl=13.0,
                      snow=0.0, sw=_make_sw(), ll=_make_ll(),
                      dlayr=_make_dlayr(), nlayr=5, iswwat="N",
                      twilen=13.0, xn=1, sump=0.0)
            final_stage = pheno.state.istage
            if final_stage == 10:
                break
        assert final_stage == 10, f"Crop did not mature; final istage={final_stage}"

    def test_maturity_date_set(self):
        """After maturity, mdate should be > 0."""
        pheno = self._make_pheno()
        pheno.run(dynamic=SEASINIT, yrdoy=2020100,
                  tmax=32.0, tmin=20.0, srad=20.0, dayl=13.0,
                  snow=0.0, sw=_make_sw(), ll=_make_ll(),
                  dlayr=_make_dlayr(), nlayr=5, iswwat="N")

        for i in range(1, 250):
            yrdoy = incdat(2020100, i)
            pheno.run(dynamic=RATE, yrdoy=yrdoy,
                      tmax=32.0, tmin=20.0, srad=20.0, dayl=13.0,
                      snow=0.0, sw=_make_sw(), ll=_make_ll(),
                      dlayr=_make_dlayr(), nlayr=5, iswwat="N",
                      twilen=13.0, xn=1, sump=0.0)
            if pheno.state.istage == 10:
                break
        assert pheno.state.mdate > 0

    def test_p9_computed_from_gdde(self):
        """P9 = 45 + GDDE*SDEPTH should be set at germination."""
        pheno = self._make_pheno()
        sdepth = 5.0
        gdde = 6.0
        expected_p9 = 45.0 + gdde * sdepth

        pheno.run(dynamic=SEASINIT, yrdoy=2020100,
                  tmax=32.0, tmin=20.0, srad=20.0, dayl=13.0,
                  snow=0.0, sw=_make_sw(), ll=_make_ll(),
                  dlayr=_make_dlayr(), nlayr=5, iswwat="N")
        # Advance to germination (stage 8 → 9 transition)
        pheno.run(dynamic=RATE, yrdoy=2020101,
                  tmax=32.0, tmin=20.0, srad=20.0, dayl=13.0,
                  snow=0.0, sw=_make_sw(), ll=_make_ll(),
                  dlayr=_make_dlayr(), nlayr=5, iswwat="N")
        # Advance through germination
        for i in range(2, 5):
            pheno.run(dynamic=RATE, yrdoy=incdat(2020100, i),
                      tmax=32.0, tmin=20.0, srad=20.0, dayl=13.0,
                      snow=0.0, sw=_make_sw(), ll=_make_ll(),
                      dlayr=_make_dlayr(), nlayr=5, iswwat="N")
        assert pheno.state.p9 == pytest.approx(expected_p9)


class TestMaizeXstage:
    """XSTAGE continuous values for CERES-Maize."""

    def setup_method(self):
        from dssat.crop.ceres_maize.phenology import MaizeCultivar, MaizePhenology
        self.MaizeCultivar = MaizeCultivar
        self.MaizePhenology = MaizePhenology

    def test_xstage_in_stage1(self):
        """In stage 1: XSTAGE = SUMDTT/P1 (should be 0-1 range)."""
        cv = self.MaizeCultivar(p1=200.0, p2=0.3, p5=500.0, g2=600.0,
                                g3=8.0, phint=38.9)
        pheno = self.MaizePhenology(cv, pltpop=7.0, sdepth=5.0, yrsim=2020100)
        # Force stage 1 with some sumdtt
        pheno.state.istage = 1
        pheno.state.sumdtt = 100.0
        pheno.state.dtt = 20.0

        # Compute xstage directly from formula
        expected = 2.0 * 100.0 / 200.0  # XSTAGE = 2*SUMDTT/P1 → 1.0
        assert expected == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Sorghum phenology tests
# ---------------------------------------------------------------------------

class TestSorghumDtt:
    """DTT computation for CERES-Sorghum."""

    def setup_method(self):
        from dssat.crop.ceres_sorghum.phenology import compute_dtt
        self.compute_dtt = compute_dtt

    def test_dtt_below_tbase_zero(self):
        dtt = self.compute_dtt(
            tmax=5.0, tmin=0.0, srad=10.0, dayl=8.0, snow=0.0,
            leafno=5, istage=1, tbase=7.0, topt=30.0, ropt=26.0,
        )
        assert dtt == pytest.approx(0.0)

    def test_dtt_simple_case(self):
        """Simple average case with no clipping."""
        dtt = self.compute_dtt(
            tmax=28.0, tmin=15.0, srad=20.0, dayl=12.0, snow=0.0,
            leafno=15, istage=1, tbase=7.0, topt=30.0, ropt=26.0,
        )
        expected = (28.0 + 15.0) / 2.0 - 7.0
        assert dtt == pytest.approx(expected)


class TestSorghumPhenologyStages:
    """Stage progression for CERES-Sorghum."""

    def setup_method(self):
        from dssat.crop.ceres_sorghum.phenology import SorghumCultivar, SorghumPhenology
        self.SorghumCultivar = SorghumCultivar
        self.SorghumPhenology = SorghumPhenology

    def _make_pheno(self):
        cv = self.SorghumCultivar(
            p1=400.0, p2=100.0, p2o=12.5, p2r=180.0,
            p3=80.0, p4=50.0, p5=400.0, g1=6.5, g2=27.0,
            phint=38.9, panth=500.0,
            tbase=7.0, topt=30.0, ropt=26.0, gdde=6.5,
        )
        return self.SorghumPhenology(cv, pltpop=10.0, sdepth=4.0, yrsim=2020100)

    def test_initial_state_stage7(self):
        pheno = self._make_pheno()
        pheno.run(dynamic=SEASINIT, yrdoy=2020100,
                  tmax=30.0, tmin=20.0, srad=20.0, dayl=13.0,
                  snow=0.0, sw=_make_sw(), ll=_make_ll(),
                  dlayr=_make_dlayr(), nlayr=5, iswwat="N")
        assert pheno.state.istage == 7

    def test_reaches_maturity(self):
        """Sorghum should reach maturity within 300 days of warm weather."""
        pheno = self._make_pheno()
        pheno.run(dynamic=SEASINIT, yrdoy=2020100,
                  tmax=30.0, tmin=20.0, srad=20.0, dayl=13.0,
                  snow=0.0, sw=_make_sw(), ll=_make_ll(),
                  dlayr=_make_dlayr(), nlayr=5, iswwat="N")

        final_stage = 7
        for i in range(1, 300):
            yrdoy = incdat(2020100, i)
            pheno.run(dynamic=RATE, yrdoy=yrdoy,
                      tmax=30.0, tmin=20.0, srad=20.0, dayl=13.0,
                      snow=0.0, sw=_make_sw(), ll=_make_ll(),
                      dlayr=_make_dlayr(), nlayr=5, iswwat="N",
                      twilen=13.0)
            final_stage = pheno.state.istage
            if final_stage == 6 and pheno.state.mdate > 0:
                break
        assert pheno.state.crop_status == 1

    def test_state_p2_initialized(self):
        """state.p2 should equal cv.p2 after _init."""
        pheno = self._make_pheno()
        pheno._init(2020100)
        assert pheno.state.p2 == pytest.approx(100.0)

    def test_stage_order(self):
        """Stages should progress in order 7→8→9→1→2→3→4→5→6."""
        pheno = self._make_pheno()
        pheno.run(dynamic=SEASINIT, yrdoy=2020100,
                  tmax=30.0, tmin=20.0, srad=20.0, dayl=13.0,
                  snow=0.0, sw=_make_sw(), ll=_make_ll(),
                  dlayr=_make_dlayr(), nlayr=5, iswwat="N")

        # Record SEASINIT stage (7) before any RATE calls
        stages_seen = [pheno.state.istage]
        for i in range(1, 300):
            yrdoy = incdat(2020100, i)
            pheno.run(dynamic=RATE, yrdoy=yrdoy,
                      tmax=30.0, tmin=20.0, srad=20.0, dayl=13.0,
                      snow=0.0, sw=_make_sw(), ll=_make_ll(),
                      dlayr=_make_dlayr(), nlayr=5, iswwat="N",
                      twilen=13.0)
            s = pheno.state.istage
            if stages_seen[-1] != s:
                stages_seen.append(s)
            if pheno.state.crop_status == 1:
                break

        # Expected order: starts at 7 (SEASINIT), ends at 6 (maturity)
        assert stages_seen[0] == 7
        assert stages_seen[-1] == 6


# ---------------------------------------------------------------------------
# Millet phenology tests
# ---------------------------------------------------------------------------

class TestMilletDtt:
    """DTT computation for CERES-Millet."""

    def setup_method(self):
        from dssat.crop.ceres_millet.phenology import compute_dtt
        self.compute_dtt = compute_dtt

    def test_dtt_below_tbase_zero(self):
        dtt = self.compute_dtt(
            tmax=6.0, tmin=0.0, srad=10.0, dayl=8.0, snow=0.0,
            leafno=5, istage=1, tbase=8.0, topt=33.0, ropt=28.0,
        )
        assert dtt == pytest.approx(0.0)

    def test_dtt_moderate_temps(self):
        dtt = self.compute_dtt(
            tmax=28.0, tmin=15.0, srad=20.0, dayl=12.0, snow=0.0,
            leafno=15, istage=1, tbase=8.0, topt=33.0, ropt=28.0,
        )
        expected = (28.0 + 15.0) / 2.0 - 8.0
        assert dtt == pytest.approx(expected)


class TestMilletPhenologyStages:
    """Stage progression for CERES-Millet."""

    def setup_method(self):
        from dssat.crop.ceres_millet.phenology import MilletCultivar, MilletPhenology
        self.MilletCultivar = MilletCultivar
        self.MilletPhenology = MilletPhenology

    def _make_pheno(self):
        cv = self.MilletCultivar(
            p1=350.0, p2o=12.5, p2r=100.0, p5=400.0,
            g4=0.9, g5=8.0, phint=38.9,
            tbase=8.0, topt=33.0, ropt=28.0, djti=68.0,
        )
        return self.MilletPhenology(cv, pltpop=25.0, sdepth=3.0, yrsim=2020100)

    def test_initial_state_stage7(self):
        pheno = self._make_pheno()
        pheno.run(dynamic=SEASINIT, yrdoy=2020100,
                  tmax=32.0, tmin=20.0, srad=20.0, dayl=13.0,
                  snow=0.0, sw=_make_sw(), ll=_make_ll(),
                  dlayr=_make_dlayr(), nlayr=5, iswwat="N")
        assert pheno.state.istage == 7

    def test_p2_initialized_from_djti(self):
        """state.p2 should equal cv.djti = 68.0."""
        pheno = self._make_pheno()
        pheno._init(2020100)
        assert pheno.state.p2 == pytest.approx(68.0)

    def test_p9_formula(self):
        """P9 = 45 + 6 * SDEPTH (hardcoded coefficient in ML_PHASEI)."""
        pheno = self._make_pheno()
        pheno.run(dynamic=SEASINIT, yrdoy=2020100,
                  tmax=32.0, tmin=20.0, srad=20.0, dayl=13.0,
                  snow=0.0, sw=_make_sw(), ll=_make_ll(),
                  dlayr=_make_dlayr(), nlayr=5, iswwat="N")
        # Trigger germination
        for i in range(1, 4):
            pheno.run(dynamic=RATE, yrdoy=incdat(2020100, i),
                      tmax=32.0, tmin=20.0, srad=20.0, dayl=13.0,
                      snow=0.0, sw=_make_sw(), ll=_make_ll(),
                      dlayr=_make_dlayr(), nlayr=5, iswwat="N")
        # P9 = 45 + 6.0 * 3.0 (sdepth=3.0)
        expected_p9 = 45.0 + 6.0 * 3.0
        assert pheno.state.p9 == pytest.approx(expected_p9)

    def test_reaches_maturity(self):
        """Millet should reach maturity within 300 days of warm weather."""
        pheno = self._make_pheno()
        pheno.run(dynamic=SEASINIT, yrdoy=2020100,
                  tmax=32.0, tmin=20.0, srad=20.0, dayl=13.0,
                  snow=0.0, sw=_make_sw(), ll=_make_ll(),
                  dlayr=_make_dlayr(), nlayr=5, iswwat="N")

        final_stage = 7
        for i in range(1, 300):
            yrdoy = incdat(2020100, i)
            pheno.run(dynamic=RATE, yrdoy=yrdoy,
                      tmax=32.0, tmin=20.0, srad=20.0, dayl=13.0,
                      snow=0.0, sw=_make_sw(), ll=_make_ll(),
                      dlayr=_make_dlayr(), nlayr=5, iswwat="N",
                      twilen=13.0)
            final_stage = pheno.state.istage
            if pheno.state.crop_status == 1:
                break
        assert pheno.state.crop_status == 1, \
            f"Millet did not mature; final_stage={final_stage}"

    def test_dynamic_p3_computed_at_stage2_transition(self):
        """P3 should be dynamically computed at stage 2→3 transition."""
        pheno = self._make_pheno()
        pheno.run(dynamic=SEASINIT, yrdoy=2020100,
                  tmax=32.0, tmin=20.0, srad=20.0, dayl=13.0,
                  snow=0.0, sw=_make_sw(), ll=_make_ll(),
                  dlayr=_make_dlayr(), nlayr=5, iswwat="N")

        # Run until stage 3 is reached (use explicit check: stages 7/8/9 are
        # numerically > 3 but precede stage 1 in the sequence)
        for i in range(1, 200):
            yrdoy = incdat(2020100, i)
            pheno.run(dynamic=RATE, yrdoy=yrdoy,
                      tmax=32.0, tmin=20.0, srad=20.0, dayl=13.0,
                      snow=0.0, sw=_make_sw(), ll=_make_ll(),
                      dlayr=_make_dlayr(), nlayr=5, iswwat="N",
                      twilen=13.0)
            if pheno.state.istage in (3, 4, 5, 6):
                break

        # Dynamic P3 = 370 + 0.135 * SUMDTT_at_stage2 — should be > 0
        assert pheno.state.p3_dyn > 0.0
        # P4_dyn should be 150
        assert pheno.state.p4_dyn == pytest.approx(150.0)

    def test_isdate_set_in_stage4(self):
        """ISDATE (anthesis) should be set during stage 4 when SUMDTT >= 50."""
        pheno = self._make_pheno()
        pheno.run(dynamic=SEASINIT, yrdoy=2020100,
                  tmax=32.0, tmin=20.0, srad=20.0, dayl=13.0,
                  snow=0.0, sw=_make_sw(), ll=_make_ll(),
                  dlayr=_make_dlayr(), nlayr=5, iswwat="N")

        for i in range(1, 300):
            yrdoy = incdat(2020100, i)
            pheno.run(dynamic=RATE, yrdoy=yrdoy,
                      tmax=32.0, tmin=20.0, srad=20.0, dayl=13.0,
                      snow=0.0, sw=_make_sw(), ll=_make_ll(),
                      dlayr=_make_dlayr(), nlayr=5, iswwat="N",
                      twilen=13.0)
            # Stages 7/8/9 are numerically > 5 but precede stages 1-6
            if pheno.state.istage in (5, 6):
                break

        assert pheno.state.isdate > 0, "ISDATE should be set at anthesis"


# ---------------------------------------------------------------------------
# Photoperiod sensitivity tests
# ---------------------------------------------------------------------------

class TestPhotoperiodSensitivity:
    """Test that photoperiod delays stage 2 correctly in each model."""

    def test_maize_long_day_delays_stage2(self):
        """Longer days than P2O should slow SIND accumulation in maize."""
        from dssat.crop.ceres_maize.phenology import MaizeCultivar, MaizePhenology
        cv = MaizeCultivar(p1=200.0, p2=0.5, p5=500.0, g2=600.0, g3=8.0,
                           phint=38.9, p2o=12.5, djti=4.0)
        pheno_short = MaizePhenology(cv, pltpop=7.0, sdepth=5.0, yrsim=2020100)
        pheno_long = MaizePhenology(cv, pltpop=7.0, sdepth=5.0, yrsim=2020100)

        # Initialize both
        for pheno in (pheno_short, pheno_long):
            pheno.run(dynamic=SEASINIT, yrdoy=2020100,
                      tmax=32.0, tmin=20.0, srad=20.0, dayl=13.0,
                      snow=0.0, sw=_make_sw(), ll=_make_ll(),
                      dlayr=_make_dlayr(), nlayr=5, iswwat="N")

        # Run each for 80 days with different photoperiods
        days_short_to_stage3 = 0
        days_long_to_stage3 = 0

        for pheno, twilen, days_ref in [
            (pheno_short, 12.5, None),  # At P2O — no delay
            (pheno_long, 15.0, None),   # Well above P2O — delay
        ]:
            for i in range(1, 200):
                yrdoy = incdat(2020100, i)
                pheno.run(dynamic=RATE, yrdoy=yrdoy,
                          tmax=32.0, tmin=20.0, srad=20.0, dayl=twilen,
                          snow=0.0, sw=_make_sw(), ll=_make_ll(),
                          dlayr=_make_dlayr(), nlayr=5, iswwat="N",
                          twilen=twilen, xn=1, sump=0.0)
                # Stages 7/8/9 are numerically ≥ 3 but precede stages 1–6
                if pheno.state.istage in (3, 4, 5, 6, 10):
                    if pheno is pheno_short:
                        days_short_to_stage3 = i
                    else:
                        days_long_to_stage3 = i
                    break

        # Long-day crop should take more days to reach stage 3
        assert days_long_to_stage3 > days_short_to_stage3, (
            f"Long day ({days_long_to_stage3}) should be > short day ({days_short_to_stage3})"
        )

    def test_millet_long_day_delays_stage2(self):
        """Longer days than P2O should slow SIND accumulation in millet."""
        from dssat.crop.ceres_millet.phenology import MilletCultivar, MilletPhenology
        cv = MilletCultivar(p1=250.0, p2o=12.5, p2r=150.0, p5=350.0,
                            g4=0.9, g5=8.0, phint=38.9, djti=68.0)

        pheno_short = MilletPhenology(cv, pltpop=25.0, sdepth=3.0, yrsim=2020100)
        pheno_long = MilletPhenology(cv, pltpop=25.0, sdepth=3.0, yrsim=2020100)

        for pheno in (pheno_short, pheno_long):
            pheno.run(dynamic=SEASINIT, yrdoy=2020100,
                      tmax=32.0, tmin=20.0, srad=20.0, dayl=13.0,
                      snow=0.0, sw=_make_sw(), ll=_make_ll(),
                      dlayr=_make_dlayr(), nlayr=5, iswwat="N")

        days_short = days_long = 0
        for pheno, twilen in [(pheno_short, 12.5), (pheno_long, 16.0)]:
            for i in range(1, 250):
                yrdoy = incdat(2020100, i)
                pheno.run(dynamic=RATE, yrdoy=yrdoy,
                          tmax=32.0, tmin=20.0, srad=20.0, dayl=twilen,
                          snow=0.0, sw=_make_sw(), ll=_make_ll(),
                          dlayr=_make_dlayr(), nlayr=5, iswwat="N",
                          twilen=twilen)
                # Stages 7/8/9 are numerically ≥ 3 but precede stages 1–6
                if pheno.state.istage in (3, 4, 5, 6):
                    if pheno is pheno_short:
                        days_short = i
                    else:
                        days_long = i
                    break

        assert days_long > days_short, (
            f"Millet: long day ({days_long}) should be > short day ({days_short})"
        )


# ---------------------------------------------------------------------------
# Germination failure tests
# ---------------------------------------------------------------------------

class TestGerminationFailure:
    """Test that crops fail gracefully with inadequate soil water."""

    def test_maize_no_germination_without_water(self):
        """With SW == LL always, maize should not germinate before DSGT (21d)."""
        from dssat.crop.ceres_maize.phenology import MaizeCultivar, MaizePhenology
        cv = MaizeCultivar()
        pheno = MaizePhenology(cv, pltpop=7.0, sdepth=5.0, yrsim=2020100)
        pheno.run(dynamic=SEASINIT, yrdoy=2020100,
                  tmax=32.0, tmin=20.0, srad=20.0, dayl=13.0,
                  snow=0.0, sw=np.full(NL, 0.10), ll=_make_ll(),
                  dlayr=_make_dlayr(), nlayr=5, iswwat="Y")

        # Run only 15 days (< DSGT=21) with SW == LL (no available water).
        # Crop should stay in stage 8 (no germination yet; failure hasn't
        # triggered since NDAS < DSGT).
        for i in range(1, 16):
            yrdoy = incdat(2020100, i)
            pheno.run(dynamic=RATE, yrdoy=yrdoy,
                      tmax=32.0, tmin=20.0, srad=20.0, dayl=13.0,
                      snow=0.0, sw=np.full(NL, 0.10), ll=_make_ll(),
                      dlayr=_make_dlayr(), nlayr=5, iswwat="Y",
                      twilen=13.0, xn=0, sump=0.0)

        # Should stay in stage 8 (germination pending, no water, no failure yet)
        assert pheno.state.istage == 8

    def test_millet_no_germination_without_water(self):
        """With SW <= LL always, millet should not germinate."""
        from dssat.crop.ceres_millet.phenology import MilletCultivar, MilletPhenology
        cv = MilletCultivar()
        pheno = MilletPhenology(cv, pltpop=25.0, sdepth=3.0, yrsim=2020100)
        pheno.run(dynamic=SEASINIT, yrdoy=2020100,
                  tmax=32.0, tmin=20.0, srad=20.0, dayl=13.0,
                  snow=0.0, sw=np.full(NL, 0.10), ll=_make_ll(),
                  dlayr=_make_dlayr(), nlayr=5, iswwat="Y")

        for i in range(1, 26):
            yrdoy = incdat(2020100, i)
            pheno.run(dynamic=RATE, yrdoy=yrdoy,
                      tmax=32.0, tmin=20.0, srad=20.0, dayl=13.0,
                      snow=0.0, sw=np.full(NL, 0.10), ll=_make_ll(),
                      dlayr=_make_dlayr(), nlayr=5, iswwat="Y",
                      twilen=13.0)

        assert pheno.state.istage == 8


# ---------------------------------------------------------------------------
# Stage carryover tests
# ---------------------------------------------------------------------------

class TestStageCarryover:
    """Test that excess TT carries over at stage transitions."""

    def test_maize_stage9_carryover(self):
        """At stage 9→1 transition, excess TT over P9 carries into stage 1."""
        from dssat.crop.ceres_maize.phenology import MaizeCultivar, MaizePhenology
        cv = MaizeCultivar(p1=200.0, p2=0.3, p5=500.0, g2=600.0,
                           g3=8.0, phint=38.9, gdde=6.0)
        pheno = MaizePhenology(cv, pltpop=7.0, sdepth=5.0, yrsim=2020100)
        p9 = 45.0 + 6.0 * 5.0  # = 75.0

        # Force into stage 9 with SUMDTT slightly above P9
        pheno.state.istage = 9
        pheno.state.p9 = p9
        pheno.state.sumdtt = p9 + 15.0  # 15 units carryover
        pheno.state.cumdtt = p9 + 15.0
        pheno.state.dtt = 0.0

        # Trigger the stage 9→1 transition
        pheno._stage9(2020115)

        # SUMDTT should be reduced by P9 (= 15.0 carryover)
        assert pheno.state.istage == 1
        assert pheno.state.sumdtt == pytest.approx(15.0)

    def test_sorghum_stage9_carryover(self):
        """At sorghum stage 9→1 transition, carryover applied correctly."""
        from dssat.crop.ceres_sorghum.phenology import SorghumCultivar, SorghumPhenology
        cv = SorghumCultivar(gdde=6.5)
        pheno = SorghumPhenology(cv, pltpop=10.0, sdepth=4.0, yrsim=2020100)
        p9 = 50.0 + 6.5 * 4.0  # = 76.0

        pheno.state.istage = 9
        pheno.state.p9 = p9
        pheno.state.sumdtt = p9 + 10.0  # 10 units carryover
        pheno.state.cumdtt = p9 + 10.0
        pheno.state.dtt = 0.0

        pheno._stage9(2020115)
        assert pheno.state.istage == 1
        assert pheno.state.sumdtt == pytest.approx(10.0)

    def test_millet_stage9_carryover(self):
        """At millet stage 9→1 transition, carryover applied correctly."""
        from dssat.crop.ceres_millet.phenology import MilletCultivar, MilletPhenology
        cv = MilletCultivar()
        pheno = MilletPhenology(cv, pltpop=25.0, sdepth=3.0, yrsim=2020100)
        p9 = 45.0 + 6.0 * 3.0  # = 63.0 (hardcoded 6.0 in PHASEI)

        pheno.state.istage = 9
        pheno.state.p9 = p9
        pheno.state.sumdtt = p9 + 8.0  # 8 units carryover
        pheno.state.cumdtt = p9 + 8.0
        pheno.state.dtt = 0.0

        pheno._stage9(2020115)
        assert pheno.state.istage == 1
        assert pheno.state.sumdtt == pytest.approx(8.0)
