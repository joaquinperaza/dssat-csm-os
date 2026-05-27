"""Soil inorganic nitrogen transformations module.

Translated from ``SOILNI.for`` and ``NFLUX.for``.

Processes modelled:
- Urea hydrolysis (urea → NH₄)
- Nitrification   (NH₄ → NO₃, Gilmour method)
- Denitrification (NO₃ → N₂/N₂O, CERES method)
- N leaching with downward water flow
- Fertiliser N addition
- Plant N uptake subtraction

References:
    Godwin, D.C. & Jones, C.A. (1991) Nitrogen dynamics in soil–plant
    systems. In: *Modeling Plant and Soil Systems*, ASA Monograph 31.
"""

from __future__ import annotations

import numpy as np

from dssat.core.constants import (
    NL, RUNINIT, SEASINIT, RATE, INTEGR, OUTPUT, SEASEND, ENDRUN,
)
from dssat.core.types import (
    ControlType, SwitchType, SoilType, WeatherType,
)


class SoilNitrogenModule:
    """Daily inorganic soil nitrogen transformations.

    Manages three N pools per layer — NO₃, NH₄, and urea — and applies
    urea hydrolysis, nitrification, denitrification, leaching, fertiliser
    addition, and plant uptake each day.

    Attributes:
        sno3: Soil NO₃ by layer (kg N ha⁻¹), shape ``(NL,)``.
        snh4: Soil NH₄ by layer (kg N ha⁻¹), shape ``(NL,)``.
        urea: Soil urea by layer (kg N ha⁻¹), shape ``(NL,)``.
        no3: NO₃ concentration by layer (µg g⁻¹ = ppm), shape ``(NL,)``.
        nh4: NH₄ concentration by layer (µg g⁻¹ = ppm), shape ``(NL,)``.
        total_no3: Profile-total NO₃ (kg N ha⁻¹).
        total_nh4: Profile-total NH₄ (kg N ha⁻¹).
        nitrif_today: Daily nitrification (kg N ha⁻¹ d⁻¹).
        denitrif_today: Daily denitrification (kg N ha⁻¹ d⁻¹).
        leach_today: Daily NO₃ leached below profile (kg N ha⁻¹ d⁻¹).
        cumnitrif: Cumulative seasonal nitrification (kg N ha⁻¹).
        cumdenitrif: Cumulative seasonal denitrification (kg N ha⁻¹).
        cumleach: Cumulative seasonal leaching (kg N ha⁻¹).
    """

    def __init__(self) -> None:
        self.sno3: np.ndarray = np.zeros(NL)
        self.snh4: np.ndarray = np.zeros(NL)
        self.urea: np.ndarray = np.zeros(NL)
        self.no3: np.ndarray = np.zeros(NL)
        self.nh4: np.ndarray = np.zeros(NL)

        # Daily output totals
        self.total_no3: float = 0.0
        self.total_nh4: float = 0.0
        self.nitrif_today: float = 0.0
        self.denitrif_today: float = 0.0
        self.leach_today: float = 0.0

        # Cumulative season totals
        self.cumnitrif: float = 0.0
        self.cumdenitrif: float = 0.0
        self.cumleach: float = 0.0

        # Internal rate arrays (reset each RATE phase)
        self._dltsno3: np.ndarray = np.zeros(NL)
        self._dltsnh4: np.ndarray = np.zeros(NL)
        self._dlturea: np.ndarray = np.zeros(NL)

    # ------------------------------------------------------------------
    def run(
        self,
        control: ControlType,
        iswitch: SwitchType,
        soilprop: SoilType,
        weather: WeatherType,
        sw: np.ndarray,
        drn: np.ndarray,
        ssom: np.ndarray,
        uno3: np.ndarray,
        unh4: np.ndarray,
        fert_sno3: np.ndarray | None = None,
        fert_snh4: np.ndarray | None = None,
        fert_urea: np.ndarray | None = None,
        apply_fert_today: bool = False,
    ) -> None:
        """Advance the soil N module one simulation step.

        Args:
            control: Simulation control block (provides ``dynamic`` phase).
            iswitch: Simulation switches (``iswnit`` controls N modelling).
            soilprop: Current soil profile properties.
            weather: Today's weather (``tmax``, ``tmin``).
            sw: Volumetric soil water content by layer (cm³ cm⁻³), shape
                ``(NL,)`` — typically from the water balance module.
            drn: Drainage flux by layer (cm d⁻¹), shape ``(NL,)``.
            ssom: Total soil organic matter by layer (kg ha⁻¹), shape
                ``(NL,)`` — used for denitrification C source.
            uno3: Plant NO₃ uptake by layer (kg N ha⁻¹ d⁻¹), shape ``(NL,)``.
            unh4: Plant NH₄ uptake by layer (kg N ha⁻¹ d⁻¹), shape ``(NL,)``.
            fert_sno3: Fertiliser NO₃ additions by layer (kg N ha⁻¹),
                shape ``(NL,)``.  Only applied when *apply_fert_today* is
                ``True``.
            fert_snh4: Fertiliser NH₄ additions by layer (kg N ha⁻¹),
                shape ``(NL,)``.
            fert_urea: Fertiliser urea additions by layer (kg N ha⁻¹),
                shape ``(NL,)``.
            apply_fert_today: Set to ``True`` on fertiliser application
                days so that fert_* arrays are added to the pools.
        """
        dynamic = control.dynamic

        if dynamic == RUNINIT:
            pass  # nothing to do at run init

        elif dynamic == SEASINIT:
            self._season_init(soilprop)

        elif dynamic == RATE:
            if iswitch.iswnit == "Y":
                self._daily_rate(
                    soilprop, weather, sw, drn, ssom, uno3, unh4,
                    fert_sno3, fert_snh4, fert_urea, apply_fert_today,
                )

        elif dynamic == INTEGR:
            if iswitch.iswnit == "Y":
                self._integrate(soilprop)

        elif dynamic == OUTPUT:
            self._update_output_totals(soilprop)

        elif dynamic in (SEASEND, ENDRUN):
            pass  # seasonal summaries already accumulated

    # ------------------------------------------------------------------
    def initialise_n(
        self,
        sno3_init: np.ndarray,
        snh4_init: np.ndarray,
        nlayr: int,
        kg2ppm: np.ndarray,
    ) -> None:
        """Set initial soil N pools from experiment specification.

        Args:
            sno3_init: Initial NO₃ by layer (kg N ha⁻¹), shape ``(NL,)``.
            snh4_init: Initial NH₄ by layer (kg N ha⁻¹), shape ``(NL,)``.
            nlayr: Number of active soil layers.
            kg2ppm: Conversion factor kg ha⁻¹ → µg g⁻¹ by layer.
        """
        for L in range(nlayr):
            self.sno3[L] = max(0.0, sno3_init[L])
            self.snh4[L] = max(0.0, snh4_init[L])
            self.no3[L] = self.sno3[L] * kg2ppm[L]
            self.nh4[L] = self.snh4[L] * kg2ppm[L]

    # ------------------------------------------------------------------
    def _season_init(self, soilprop: SoilType) -> None:
        """Seasonal initialisation — reset cumulative counters."""
        self.cumnitrif = 0.0
        self.cumdenitrif = 0.0
        self.cumleach = 0.0
        self.nitrif_today = 0.0
        self.denitrif_today = 0.0
        self.leach_today = 0.0
        self._dltsno3[:] = 0.0
        self._dltsnh4[:] = 0.0
        self._dlturea[:] = 0.0

        # Default initialisation from soil OC if pools not yet set
        nlayr = soilprop.nlayr
        for L in range(nlayr):
            if self.sno3[L] <= 0.0:
                # Background NO3: ~5 ppm at surface, declining with depth
                bg_ppm = max(1.0, 5.0 * (1.0 - soilprop.ds[L] / 200.0))
                k2p = soilprop.kg2ppm[L]
                self.sno3[L] = bg_ppm / k2p if k2p > 0.0 else 0.0
            if self.snh4[L] <= 0.0:
                bg_ppm = 1.0
                k2p = soilprop.kg2ppm[L]
                self.snh4[L] = bg_ppm / k2p if k2p > 0.0 else 0.0
            self.no3[L] = self.sno3[L] * soilprop.kg2ppm[L]
            self.nh4[L] = self.snh4[L] * soilprop.kg2ppm[L]

        self.total_no3 = float(np.sum(self.sno3[:nlayr]))
        self.total_nh4 = float(np.sum(self.snh4[:nlayr]))

    # ------------------------------------------------------------------
    def _daily_rate(
        self,
        soilprop: SoilType,
        weather: WeatherType,
        sw: np.ndarray,
        drn: np.ndarray,
        ssom: np.ndarray,
        uno3: np.ndarray,
        unh4: np.ndarray,
        fert_sno3: np.ndarray | None,
        fert_snh4: np.ndarray | None,
        fert_urea: np.ndarray | None,
        apply_fert_today: bool,
    ) -> None:
        """Compute daily N transformation rates."""
        nlayr = soilprop.nlayr
        dl = self._dltsno3
        dl[:] = 0.0
        self._dltsnh4[:] = 0.0
        self._dlturea[:] = 0.0

        # ---- Soil temperature estimate
        tsoil = (weather.tmax + weather.tmin) / 2.0 + 2.0

        # ---- Temperature factor (0–1)
        tf = max(0.0, min(1.0, (tsoil - 5.0) / 25.0))

        # Per-layer accumulators
        nitrif_arr = np.zeros(NL)
        denitrif_arr = np.zeros(NL)

        for L in range(nlayr):
            ll_l = soilprop.ll[L]
            dul_l = soilprop.dul[L]
            sat_l = soilprop.sat[L]
            bd_l = soilprop.bd[L]
            ph_l = soilprop.ph[L]
            k2p_l = soilprop.kg2ppm[L]
            sw_l = sw[L]

            # Water factor for mineralization / nitrification (0–1)
            denom = dul_l - ll_l
            wf = (sw_l - ll_l) / denom if denom > 0.0 else 0.0
            wf = max(0.0, min(1.0, wf))

            # ---- 1. Urea hydrolysis
            hydrol = self.urea[L] * 0.0833 * tf * wf
            hydrol = min(hydrol, self.urea[L])
            self._dlturea[L] -= hydrol
            self._dltsnh4[L] += hydrol

            # ---- 2. Nitrification (Gilmour method)
            ph_fac = max(0.0, min(1.0, (ph_l - 4.0) / 2.5))
            snh4_avail = max(
                0.0,
                self.snh4[L] - (0.5 / k2p_l if k2p_l > 0.0 else 0.0),
            )
            nitrif_l = 0.20 * snh4_avail * tf * wf * ph_fac
            nitrif_l = min(nitrif_l, max(0.0, self.snh4[L] + self._dltsnh4[L]))
            nitrif_arr[L] = nitrif_l
            self._dltsnh4[L] -= nitrif_l
            self._dltsno3[L] += nitrif_l

            # ---- 3. Denitrification (CERES method, near-saturation only)
            thr = 0.9 * sat_l
            sat_gap = sat_l - thr
            wfdenit = (sw_l - thr) / sat_gap if sat_gap > 0.0 else 0.0
            wfdenit = max(0.0, min(1.0, wfdenit))
            c_avail = ssom[L] * 0.58  # 58 % C in SOM
            denitrif_l = min(
                max(0.0, self.sno3[L] + self._dltsno3[L]),
                6.0 * c_avail / 1000.0 * wfdenit * tf,
            )
            denitrif_arr[L] = denitrif_l
            self._dltsno3[L] -= denitrif_l

        # ---- 4. N leaching (NFLUX approach)
        leach_no3 = _nflux(
            nlayr=nlayr,
            sno3=self.sno3,
            dltsno3=self._dltsno3,
            sw=sw,
            drn=drn,
            bd=soilprop.bd,
            dul=soilprop.dul,
            dlayr=soilprop.dlayr,
        )

        # ---- 5. Fertiliser addition
        if apply_fert_today:
            if fert_sno3 is not None:
                self._dltsno3[:nlayr] += fert_sno3[:nlayr]
            if fert_snh4 is not None:
                self._dltsnh4[:nlayr] += fert_snh4[:nlayr]
            if fert_urea is not None:
                self._dlturea[:nlayr] += fert_urea[:nlayr]

        # ---- 6. Plant uptake
        self._dltsno3[:nlayr] -= uno3[:nlayr]
        self._dltsnh4[:nlayr] -= unh4[:nlayr]

        # ---- Daily summaries (before integration)
        self.nitrif_today = float(np.sum(nitrif_arr[:nlayr]))
        self.denitrif_today = float(np.sum(denitrif_arr[:nlayr]))
        self.leach_today = leach_no3
        self.cumnitrif += self.nitrif_today
        self.cumdenitrif += self.denitrif_today
        self.cumleach += self.leach_today

    # ------------------------------------------------------------------
    def _integrate(self, soilprop: SoilType) -> None:
        """Apply daily rate changes to soil N state variables."""
        nlayr = soilprop.nlayr
        for L in range(nlayr):
            self.sno3[L] = max(0.0, self.sno3[L] + self._dltsno3[L])
            self.snh4[L] = max(0.0, self.snh4[L] + self._dltsnh4[L])
            self.urea[L] = max(0.0, self.urea[L] + self._dlturea[L])
            k2p = soilprop.kg2ppm[L]
            self.no3[L] = self.sno3[L] * k2p
            self.nh4[L] = self.snh4[L] * k2p

    # ------------------------------------------------------------------
    def _update_output_totals(self, soilprop: SoilType) -> None:
        """Recompute profile totals for output."""
        nlayr = soilprop.nlayr
        self.total_no3 = float(np.sum(self.sno3[:nlayr]))
        self.total_nh4 = float(np.sum(self.snh4[:nlayr]))


# ---------------------------------------------------------------------------
# Free functions
# ---------------------------------------------------------------------------

def _nflux(
    nlayr: int,
    sno3: np.ndarray,
    dltsno3: np.ndarray,
    sw: np.ndarray,
    drn: np.ndarray,
    bd: np.ndarray,
    dul: np.ndarray,
    dlayr: np.ndarray,
    adcoef: float = 0.0,
) -> float:
    """NFLUX — downward NO₃ transport with drainage.

    Solute is partitioned between soil solution and adsorbed phase using a
    linear adsorption coefficient (*adcoef*; default 0 = no adsorption).
    Drainage from each layer carries a fraction of the dissolved NO₃
    proportional to the drainage volume relative to total pore water.

    Args:
        nlayr: Number of active soil layers.
        sno3: Current soil NO₃ per layer (kg N ha⁻¹), shape ``(NL,)``.
        dltsno3: NO₃ rate-change array to update in-place (kg N ha⁻¹),
            shape ``(NL,)``.  Leaching losses are subtracted and downward
            inputs to the next layer are added here.
        sw: Volumetric water content (cm³ cm⁻³), shape ``(NL,)``.
        drn: Drainage flux leaving base of each layer (cm d⁻¹), shape
            ``(NL,)``.
        bd: Bulk density (g cm⁻³), shape ``(NL,)``.
        dul: Drained upper limit (cm³ cm⁻³), shape ``(NL,)``.
        dlayr: Layer thickness (cm), shape ``(NL,)``.
        adcoef: Linear adsorption coefficient (cm³ g⁻¹) — default 0.

    Returns:
        Total NO₃ leached below the bottom active layer (kg N ha⁻¹ d⁻¹).
    """
    flow_in = 0.0  # NO3 entering current layer from layer above

    for L in range(nlayr):
        no3_avail = max(0.0, sno3[L] + dltsno3[L] + flow_in)

        # Fraction of NO3 in solution (vs adsorbed)
        frac_soln = 1.0 / (1.0 + bd[L] * adcoef / max(dul[L], 0.01))

        # Water volume in layer (cm³ cm⁻²) × factor → use sw × dlayr
        water_vol = sw[L] * dlayr[L]

        # NO3 carried downward with drainage
        denom = water_vol + drn[L] + 1.0e-6
        ndown = no3_avail * frac_soln * drn[L] / denom
        ndown = max(0.0, min(ndown, no3_avail))

        dltsno3[L] += flow_in - ndown
        flow_in = ndown  # becomes input to next layer

    # flow_in after last layer = leached below profile
    return flow_in
