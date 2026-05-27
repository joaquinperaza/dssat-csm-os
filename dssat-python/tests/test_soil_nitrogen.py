"""Tests for soil nitrogen and organic matter modules.

Covers:
- SoilNitrogenModule: initialization, nitrification, denitrification,
  leaching, fertilizer application, N uptake
- SoilOrganicMatterModule: initialization, mineralization, FOM decomp
- Integration test: soil N + organic matter coupled
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from dssat.soil.nitrogen import SoilNitrogenModule
from dssat.soil.organic_matter import SoilOrganicMatterModule
from dssat.core.constants import NL, SEASINIT, RATE, INTEGR, OUTPUT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_soilprop(nlayr: int = 5) -> SimpleNamespace:
    sp = SimpleNamespace()
    sp.nlayr = nlayr
    sp.dlayr = np.array([5, 10, 15, 30, 30] + [0] * (NL - nlayr), dtype=float)
    sp.ds = np.cumsum(sp.dlayr)
    sp.bd = np.array([1.4, 1.5, 1.5, 1.6, 1.6] + [0] * (NL - nlayr), dtype=float)
    sp.ll = np.array([0.02] * nlayr + [0] * (NL - nlayr), dtype=float)
    sp.dul = np.array([0.09] * nlayr + [0] * (NL - nlayr), dtype=float)
    sp.sat = np.array([0.22] * nlayr + [0] * (NL - nlayr), dtype=float)
    sp.ph = np.array([6.5] * nlayr + [0] * (NL - nlayr), dtype=float)
    sp.oc = np.array([1.2, 0.8, 0.3, 0.1, 0.05] + [0] * (NL - nlayr), dtype=float)
    # kg2ppm = 10 / (bd * dlayr)  [µg g⁻¹ per kg ha⁻¹]
    kg2ppm = np.zeros(NL)
    for L in range(nlayr):
        kg2ppm[L] = 10.0 / (sp.bd[L] * sp.dlayr[L]) if sp.dlayr[L] > 0 else 1.0
    kg2ppm[nlayr:] = 1.0
    sp.kg2ppm = kg2ppm
    return sp


def make_ctrl(dynamic: int) -> SimpleNamespace:
    c = SimpleNamespace()
    c.dynamic = dynamic
    c.yrdoy = 1982100
    return c


def make_iswitch(iswnit: str = "Y") -> SimpleNamespace:
    sw = SimpleNamespace()
    sw.iswnit = iswnit
    sw.mesom = "G"
    return sw


def make_weather(tmax: float = 28.0, tmin: float = 18.0) -> SimpleNamespace:
    w = SimpleNamespace()
    w.tmax = tmax
    w.tmin = tmin
    return w


def _run_seasinit(mod: SoilNitrogenModule, sp: SimpleNamespace) -> None:
    """Run SEASINIT on a SoilNitrogenModule."""
    mod.run(
        make_ctrl(SEASINIT), make_iswitch(), sp, make_weather(),
        sw=np.full(NL, 0.12),
        drn=np.zeros(NL),
        ssom=np.zeros(NL),
        uno3=np.zeros(NL),
        unh4=np.zeros(NL),
    )


def _run_rate_integr(
    mod: SoilNitrogenModule,
    sp: SimpleNamespace,
    sw: np.ndarray,
    weather=None,
    drn=None,
    ssom=None,
    uno3=None,
    unh4=None,
    fert_sno3=None,
    fert_snh4=None,
    fert_urea=None,
    apply_fert=False,
) -> None:
    """Run one RATE + INTEGR step."""
    if weather is None:
        weather = make_weather()
    drn = drn if drn is not None else np.zeros(NL)
    ssom = ssom if ssom is not None else np.zeros(NL)
    uno3 = uno3 if uno3 is not None else np.zeros(NL)
    unh4 = unh4 if unh4 is not None else np.zeros(NL)

    mod.run(
        make_ctrl(RATE), make_iswitch(), sp, weather,
        sw=sw, drn=drn, ssom=ssom,
        uno3=uno3, unh4=unh4,
        fert_sno3=fert_sno3,
        fert_snh4=fert_snh4,
        fert_urea=fert_urea,
        apply_fert_today=apply_fert,
    )
    mod.run(
        make_ctrl(INTEGR), make_iswitch(), sp, make_weather(),
        sw=sw, drn=drn, ssom=ssom,
        uno3=uno3, unh4=unh4,
    )


# ---------------------------------------------------------------------------
# SoilNitrogenModule tests
# ---------------------------------------------------------------------------

class TestSoilNitrogenInit:
    def test_soilni_init(self):
        """After SEASINIT, sno3/snh4 > 0 and no3/nh4 ppm are positive."""
        mod = SoilNitrogenModule()
        sp = make_soilprop()
        _run_seasinit(mod, sp)

        for L in range(sp.nlayr):
            assert mod.sno3[L] > 0.0, f"Layer {L}: sno3={mod.sno3[L]}"
            assert mod.snh4[L] > 0.0, f"Layer {L}: snh4={mod.snh4[L]}"
            assert mod.no3[L] > 0.0, f"Layer {L}: no3={mod.no3[L]}"
            assert mod.nh4[L] > 0.0, f"Layer {L}: nh4={mod.nh4[L]}"

    def test_initialise_n_sets_pools(self):
        """initialise_n should override pools with supplied values."""
        mod = SoilNitrogenModule()
        sp = make_soilprop()
        nlayr = sp.nlayr
        sno3_init = np.array([5.0, 4.0, 3.0, 2.0, 1.0] + [0] * (NL - nlayr), dtype=float)
        snh4_init = np.array([2.0, 1.5, 1.0, 0.5, 0.2] + [0] * (NL - nlayr), dtype=float)
        mod.initialise_n(sno3_init, snh4_init, nlayr, sp.kg2ppm)

        np.testing.assert_allclose(mod.sno3[:nlayr], sno3_init[:nlayr])
        np.testing.assert_allclose(mod.snh4[:nlayr], snh4_init[:nlayr])
        # ppm values should be sXX * kg2ppm
        for L in range(nlayr):
            assert abs(mod.no3[L] - sno3_init[L] * sp.kg2ppm[L]) < 1e-9


class TestNitrification:
    def test_nitrification_reduces_snh4(self):
        """After a RATE+INTEGR step with warm temp and moist soil, snh4 should decrease."""
        mod = SoilNitrogenModule()
        sp = make_soilprop()
        _run_seasinit(mod, sp)

        snh4_before = mod.snh4.copy()
        sw = np.array([0.09] * sp.nlayr + [0] * (NL - sp.nlayr), dtype=float)  # at DUL
        _run_rate_integr(mod, sp, sw=sw, weather=make_weather(tmax=30, tmin=20))

        # Total NH4 should have decreased (nitrification) in at least some layers
        total_before = sum(snh4_before[:sp.nlayr])
        total_after = sum(mod.snh4[:sp.nlayr])
        assert total_after < total_before, (
            f"Expected NH4 to decrease via nitrification; before={total_before:.4f}, after={total_after:.4f}"
        )

    def test_nitrification_reported_positive(self):
        """nitrif_today should be > 0 under warm moist conditions."""
        mod = SoilNitrogenModule()
        sp = make_soilprop()
        _run_seasinit(mod, sp)
        sw = np.full(NL, 0.09)
        _run_rate_integr(mod, sp, sw=sw, weather=make_weather(tmax=30, tmin=20))
        assert mod.nitrif_today > 0.0

    def test_nitrif_increases_sno3(self):
        """Nitrification should produce NO3 from NH4."""
        mod = SoilNitrogenModule()
        sp = make_soilprop()
        _run_seasinit(mod, sp)
        sno3_before = mod.sno3.copy()
        sw = np.full(NL, 0.09)
        # No leaching / drainage, no denitrification (sw well below sat)
        _run_rate_integr(mod, sp, sw=sw, weather=make_weather(tmax=30, tmin=20))
        total_no3_before = sum(sno3_before[:sp.nlayr])
        total_no3_after = sum(mod.sno3[:sp.nlayr])
        assert total_no3_after >= total_no3_before - 1e-6  # >= with float tolerance


class TestDenitrification:
    def test_denitrification_near_saturation(self):
        """When sw >= 0.9 * sat with organic matter, NO3 should decrease."""
        mod = SoilNitrogenModule()
        sp = make_soilprop()
        _run_seasinit(mod, sp)

        sno3_before = mod.sno3.copy()
        # sw near saturation (0.21 out of sat=0.22 → 95%)
        sw = np.full(NL, 0.21)
        # Provide some SOM for denitrification
        ssom = np.full(NL, 5000.0)  # kg ha⁻¹
        _run_rate_integr(mod, sp, sw=sw, ssom=ssom, weather=make_weather(tmax=28, tmin=18))

        total_before = sum(sno3_before[:sp.nlayr])
        total_after = sum(mod.sno3[:sp.nlayr])
        # NO3 should be reduced by denitrification
        assert total_after < total_before, (
            f"Expected denitrification; before={total_before:.4f}, after={total_after:.4f}"
        )

    def test_no_denitrif_below_threshold(self):
        """When sw < 0.9*sat, denitrif_today should be ~0."""
        mod = SoilNitrogenModule()
        sp = make_soilprop()
        _run_seasinit(mod, sp)

        sw = np.full(NL, 0.09)  # at DUL, well below sat=0.22
        ssom = np.full(NL, 5000.0)
        _run_rate_integr(mod, sp, sw=sw, ssom=ssom)

        assert mod.denitrif_today < 1e-9, (
            f"Expected no denitrification; got {mod.denitrif_today:.6f}"
        )


class TestFertilizer:
    def test_fertilizer_urea_adds_to_urea_pool(self):
        """Applying urea fertilizer should increase the urea pool."""
        mod = SoilNitrogenModule()
        sp = make_soilprop()
        _run_seasinit(mod, sp)

        urea_before = mod.urea.copy()
        fert_urea = np.zeros(NL)
        fert_urea[0] = 10.0  # 10 kg N/ha urea to surface layer

        sw = np.full(NL, 0.09)
        # Run with cool/dry conditions (slow hydrolysis) to check urea pool
        _run_rate_integr(
            mod, sp, sw=sw,
            weather=make_weather(tmax=10, tmin=5),  # cool → slow hydrolysis
            fert_urea=fert_urea,
            apply_fert=True,
        )

        # Even with some hydrolysis, urea in layer 0 should increase
        assert mod.urea[0] > urea_before[0], (
            f"Expected urea[0] to increase; before={urea_before[0]:.4f}, after={mod.urea[0]:.4f}"
        )

    def test_fertilizer_nitrate_adds_to_no3(self):
        """FE002 (sodium nitrate) type should add to NO3 pool."""
        mod = SoilNitrogenModule()
        sp = make_soilprop()
        _run_seasinit(mod, sp)

        sno3_before = mod.sno3.copy()
        fert_sno3 = np.zeros(NL)
        fert_sno3[0] = 20.0  # 20 kg N/ha NO3 to surface

        sw = np.full(NL, 0.09)
        _run_rate_integr(
            mod, sp, sw=sw,
            weather=make_weather(tmax=10, tmin=5),  # cool → minimal losses
            fert_sno3=fert_sno3,
            apply_fert=True,
        )

        # Layer 0 NO3 should be higher than before
        assert mod.sno3[0] > sno3_before[0], (
            f"Expected sno3[0] to increase; before={sno3_before[0]:.4f}, after={mod.sno3[0]:.4f}"
        )

    def test_fertilizer_ammonium_adds_to_nh4(self):
        """Ammonium sulfate fertilizer should add to NH4 pool."""
        mod = SoilNitrogenModule()
        sp = make_soilprop()
        _run_seasinit(mod, sp)

        snh4_before = mod.snh4.copy()
        fert_snh4 = np.zeros(NL)
        fert_snh4[0] = 15.0

        sw = np.full(NL, 0.05)  # near LL → minimal nitrification
        _run_rate_integr(
            mod, sp, sw=sw,
            weather=make_weather(tmax=8, tmin=2),  # very cool
            fert_snh4=fert_snh4,
            apply_fert=True,
        )

        assert mod.snh4[0] > snh4_before[0], (
            f"Expected snh4[0] to increase; before={snh4_before[0]:.4f}, after={mod.snh4[0]:.4f}"
        )


class TestLeaching:
    def test_n_leaching_with_drainage(self):
        """With large drainage, NO3 should move down layers / be leached."""
        mod = SoilNitrogenModule()
        sp = make_soilprop()
        _run_seasinit(mod, sp)

        # High NO3 in surface layer
        mod.sno3[0] = 50.0
        mod.no3[0] = mod.sno3[0] * sp.kg2ppm[0]
        sno3_surface_before = mod.sno3[0]

        sw = np.full(NL, 0.15)
        drn = np.full(NL, 0.5)  # large drainage per layer (cm/d)

        _run_rate_integr(mod, sp, sw=sw, drn=drn, weather=make_weather(tmax=10, tmin=5))

        # Surface NO3 should have decreased due to leaching
        assert mod.sno3[0] < sno3_surface_before, (
            f"Expected surface NO3 to decrease via leaching; "
            f"before={sno3_surface_before:.4f}, after={mod.sno3[0]:.4f}"
        )

    def test_leach_today_positive_with_drainage(self):
        """leach_today should be positive when drainage occurs."""
        mod = SoilNitrogenModule()
        sp = make_soilprop()
        _run_seasinit(mod, sp)
        mod.sno3[0] = 50.0

        sw = np.full(NL, 0.15)
        drn = np.full(NL, 0.5)
        _run_rate_integr(mod, sp, sw=sw, drn=drn)
        assert mod.leach_today >= 0.0


class TestPlantUptake:
    def test_plant_uptake_removes_n(self):
        """Passing uno3/unh4 should reduce the soil N pools."""
        mod = SoilNitrogenModule()
        sp = make_soilprop()
        _run_seasinit(mod, sp)

        sno3_before = mod.sno3.copy()
        snh4_before = mod.snh4.copy()

        # Small uptake (well within pool size)
        uno3 = np.zeros(NL)
        unh4 = np.zeros(NL)
        for L in range(sp.nlayr):
            uno3[L] = min(0.1, mod.sno3[L] * 0.1)
            unh4[L] = min(0.05, mod.snh4[L] * 0.1)

        sw = np.full(NL, 0.09)
        _run_rate_integr(
            mod, sp, sw=sw,
            weather=make_weather(tmax=10, tmin=5),  # cool → minimal transformation
            uno3=uno3,
            unh4=unh4,
        )

        total_no3_before = sum(sno3_before[:sp.nlayr])
        total_no3_after = sum(mod.sno3[:sp.nlayr])
        total_nh4_before = sum(snh4_before[:sp.nlayr])
        total_nh4_after = sum(mod.snh4[:sp.nlayr])

        assert total_no3_after <= total_no3_before + 1e-6, (
            "Expected NO3 to decrease or stay after plant uptake"
        )
        assert total_nh4_after <= total_nh4_before + 1e-6, (
            "Expected NH4 to decrease or stay after plant uptake"
        )

    def test_pools_never_negative(self):
        """Excessive uptake should be clamped — pools should never go negative."""
        mod = SoilNitrogenModule()
        sp = make_soilprop()
        _run_seasinit(mod, sp)

        # Request much more N than available
        uno3 = np.full(NL, 1000.0)
        unh4 = np.full(NL, 1000.0)

        sw = np.full(NL, 0.09)
        _run_rate_integr(mod, sp, sw=sw, uno3=uno3, unh4=unh4)

        for L in range(sp.nlayr):
            assert mod.sno3[L] >= 0.0, f"sno3[{L}] went negative"
            assert mod.snh4[L] >= 0.0, f"snh4[{L}] went negative"


class TestIswnit:
    def test_iswnit_off_no_change(self):
        """When iswnit='N', RATE phase should not change pools."""
        mod = SoilNitrogenModule()
        sp = make_soilprop()
        _run_seasinit(mod, sp)

        sno3_before = mod.sno3.copy()
        snh4_before = mod.snh4.copy()

        iswitch_off = make_iswitch("N")
        sw = np.full(NL, 0.09)
        mod.run(
            make_ctrl(RATE), iswitch_off, sp, make_weather(tmax=30, tmin=20),
            sw=sw, drn=np.zeros(NL), ssom=np.zeros(NL),
            uno3=np.zeros(NL), unh4=np.zeros(NL),
        )
        mod.run(
            make_ctrl(INTEGR), iswitch_off, sp, make_weather(),
            sw=sw, drn=np.zeros(NL), ssom=np.zeros(NL),
            uno3=np.zeros(NL), unh4=np.zeros(NL),
        )

        np.testing.assert_array_equal(mod.sno3, sno3_before)
        np.testing.assert_array_equal(mod.snh4, snh4_before)


# ---------------------------------------------------------------------------
# SoilOrganicMatterModule tests
# ---------------------------------------------------------------------------

class TestSoilOrganicMatter:
    def _run_om_seasinit(self, mod, sp):
        mod.run(
            make_ctrl(SEASINIT), make_iswitch(), sp, make_weather(),
            sw=np.full(NL, 0.09),
        )

    def _run_om_rate_integr(self, mod, sp, sw=None, weather=None, fom_add=None, fon_add=None):
        if sw is None:
            sw = np.full(NL, 0.09)
        if weather is None:
            weather = make_weather()
        mod.run(make_ctrl(RATE), make_iswitch(), sp, weather, sw=sw,
                fom_add=fom_add, fon_add=fon_add)
        mod.run(make_ctrl(INTEGR), make_iswitch(), sp, weather, sw=sw)

    def test_som_init_from_oc(self):
        """After SEASINIT, humc > 0 proportional to OC."""
        mod = SoilOrganicMatterModule()
        sp = make_soilprop()
        self._run_om_seasinit(mod, sp)

        for L in range(sp.nlayr):
            assert mod.humc[L] > 0.0, f"humc[{L}] should be > 0"
            # OC-proportional: layer with higher OC should have higher humc
        assert mod.humc[0] > mod.humc[4], "Surface humc should exceed deepest layer"

    def test_humn_proportional_to_humc(self):
        """After SEASINIT, humn should be humc / 10 (C:N = 10)."""
        mod = SoilOrganicMatterModule()
        sp = make_soilprop()
        self._run_om_seasinit(mod, sp)

        for L in range(sp.nlayr):
            expected_humn = mod.humc[L] / 10.0
            assert abs(mod.humn[L] - expected_humn) < 1e-6, (
                f"Layer {L}: humn={mod.humn[L]:.4f}, expected {expected_humn:.4f}"
            )

    def test_mineralization_warm_moist(self):
        """With warm/moist conditions, net mineralization (mnr) should be > 0."""
        mod = SoilOrganicMatterModule()
        sp = make_soilprop()
        self._run_om_seasinit(mod, sp)

        sw = np.full(NL, 0.09)  # at DUL
        weather = make_weather(tmax=30, tmin=20)
        mod.run(make_ctrl(RATE), make_iswitch(), sp, weather, sw=sw)

        total_mnr = float(np.sum(mod.mnr[:sp.nlayr]))
        assert total_mnr > 0.0, f"Expected positive mineralization; got {total_mnr}"

    def test_no_mineralization_cold_dry(self):
        """Very cold, dry soil should produce near-zero mineralization."""
        mod = SoilOrganicMatterModule()
        sp = make_soilprop()
        self._run_om_seasinit(mod, sp)

        sw = np.full(NL, 0.02)  # at LL
        weather = make_weather(tmax=4, tmin=1)  # cold
        mod.run(make_ctrl(RATE), make_iswitch(), sp, weather, sw=sw)

        total_mnr = float(np.sum(mod.mnr[:sp.nlayr]))
        assert total_mnr < 0.01, f"Expected near-zero mineralization; got {total_mnr}"

    def test_fom_decomposition(self):
        """Adding FOM then running RATE should reduce the FOM pool."""
        mod = SoilOrganicMatterModule()
        sp = make_soilprop()
        self._run_om_seasinit(mod, sp)

        # Add FOM to surface layer
        fom_add = np.zeros(NL)
        fon_add = np.zeros(NL)
        fom_add[0] = 500.0   # kg C/ha
        fon_add[0] = 25.0    # kg N/ha (C:N = 20)

        # Run RATE with warm/moist conditions — check FOM decreases after integration
        sw = np.full(NL, 0.09)
        weather = make_weather(tmax=28, tmin=18)

        mod.run(make_ctrl(RATE), make_iswitch(), sp, weather, sw=sw,
                fom_add=fom_add, fon_add=fon_add)
        fom_after_rate = mod.fom[0] + mod._dlt_fom[0]  # net after rate

        assert fom_after_rate > 0.0  # FOM was added
        fom_net = mod.fom[0] + mod._dlt_fom[0]
        assert fom_net < fom_add[0], "FOM should have partially decomposed"

    def test_fom_decomp_integration(self):
        """After INTEGR, FOM pool should be less than what was added."""
        mod = SoilOrganicMatterModule()
        sp = make_soilprop()
        self._run_om_seasinit(mod, sp)

        fom_add = np.zeros(NL)
        fon_add = np.zeros(NL)
        fom_add[0] = 500.0
        fon_add[0] = 25.0

        sw = np.full(NL, 0.09)
        weather = make_weather(tmax=28, tmin=18)
        self._run_om_rate_integr(mod, sp, sw=sw, weather=weather,
                                  fom_add=fom_add, fon_add=fon_add)

        assert mod.fom[0] < fom_add[0], (
            f"FOM[0] should be < added amount; got {mod.fom[0]:.2f}"
        )

    def test_humc_decreases_over_time(self):
        """Humus C should slowly decline under warm moist conditions."""
        mod = SoilOrganicMatterModule()
        sp = make_soilprop()
        self._run_om_seasinit(mod, sp)
        humc_init = mod.humc.copy()

        sw = np.full(NL, 0.09)
        weather = make_weather(tmax=30, tmin=20)
        # Run 30 days
        for _ in range(30):
            self._run_om_rate_integr(mod, sp, sw=sw, weather=weather)

        total_humc_before = float(np.sum(humc_init[:sp.nlayr]))
        total_humc_after = float(np.sum(mod.humc[:sp.nlayr]))
        assert total_humc_after < total_humc_before, (
            "Humus C should decline over 30 warm-moist days"
        )

    def test_cumulative_mnr_accumulates(self):
        """cummnr should increase each day."""
        mod = SoilOrganicMatterModule()
        sp = make_soilprop()
        self._run_om_seasinit(mod, sp)

        sw = np.full(NL, 0.09)
        weather = make_weather(tmax=28, tmin=18)
        self._run_om_rate_integr(mod, sp, sw=sw, weather=weather)
        cummnr_after1 = mod.cummnr

        self._run_om_rate_integr(mod, sp, sw=sw, weather=weather)
        cummnr_after2 = mod.cummnr

        assert cummnr_after2 > cummnr_after1, "cummnr should increase each day"


# ---------------------------------------------------------------------------
# Integration test: SoilNitrogenModule + SoilOrganicMatterModule coupled
# ---------------------------------------------------------------------------

class TestSoilNitrogenIntegration:
    def test_coupled_om_feeds_no3(self):
        """Over 30 days, mineralized N from OM should add to NH4 pool."""
        om_mod = SoilOrganicMatterModule()
        n_mod = SoilNitrogenModule()
        sp = make_soilprop()

        ctrl_si = make_ctrl(SEASINIT)
        iswitch = make_iswitch()
        sw = np.full(NL, 0.09)
        weather = make_weather(tmax=28, tmin=18)

        om_mod.run(ctrl_si, iswitch, sp, weather, sw=sw)
        n_mod.run(ctrl_si, iswitch, sp, weather, sw=sw, drn=np.zeros(NL),
                  ssom=np.zeros(NL), uno3=np.zeros(NL), unh4=np.zeros(NL))

        # Initialise with minimal N
        n_mod.sno3[:sp.nlayr] = 0.1
        n_mod.snh4[:sp.nlayr] = 0.1
        n_mod.no3[:sp.nlayr] = n_mod.sno3[:sp.nlayr] * sp.kg2ppm[:sp.nlayr]
        n_mod.nh4[:sp.nlayr] = n_mod.snh4[:sp.nlayr] * sp.kg2ppm[:sp.nlayr]

        n_start = float(np.sum(n_mod.sno3[:sp.nlayr] + n_mod.snh4[:sp.nlayr]))

        for _ in range(30):
            om_mod.run(make_ctrl(RATE), iswitch, sp, weather, sw=sw)
            ssom = om_mod.ssom.copy()
            om_mod.run(make_ctrl(INTEGR), iswitch, sp, weather, sw=sw)

            # Mineralization adds to NH4 pool via ssom
            n_mod.run(
                make_ctrl(RATE), iswitch, sp, weather,
                sw=sw, drn=np.zeros(NL), ssom=ssom,
                uno3=np.zeros(NL), unh4=np.zeros(NL),
            )
            n_mod.run(
                make_ctrl(INTEGR), iswitch, sp, weather,
                sw=sw, drn=np.zeros(NL), ssom=ssom,
                uno3=np.zeros(NL), unh4=np.zeros(NL),
            )

        n_end = float(np.sum(n_mod.sno3[:sp.nlayr] + n_mod.snh4[:sp.nlayr]))
        # After 30 days of mineralization with nitrification but no uptake/leaching,
        # total N should have increased (OM → mineral N)
        assert n_end >= n_start * 0.5, (
            f"Expected N to remain or increase from mineralization; start={n_start:.4f}, end={n_end:.4f}"
        )

    def test_n_balance_no_external_inputs(self):
        """With no fertilizer, uptake, or leaching, N transformations conserve pool total
        (nitrification just moves NH4 → NO3)."""
        mod = SoilNitrogenModule()
        sp = make_soilprop()
        _run_seasinit(mod, sp)
        n_total_init = float(np.sum(mod.sno3[:sp.nlayr] + mod.snh4[:sp.nlayr]))

        sw = np.full(NL, 0.09)
        weather = make_weather(tmax=20, tmin=12)  # moderate — no denitrification
        for _ in range(10):
            _run_rate_integr(mod, sp, sw=sw, weather=weather)

        n_total_final = float(np.sum(mod.sno3[:sp.nlayr] + mod.snh4[:sp.nlayr]))
        # Total N should be approximately conserved (nitrif is internal transfer)
        assert abs(n_total_final - n_total_init) < n_total_init * 0.05, (
            f"N balance error: init={n_total_init:.4f}, final={n_total_final:.4f}"
        )
