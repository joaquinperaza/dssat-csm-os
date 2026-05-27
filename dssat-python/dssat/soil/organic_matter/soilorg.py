"""CERES-based soil organic matter and nitrogen mineralization module.

Translated from ``SOILORG.for`` (CERES method, ``mesom = "G"``).

Processes modelled:
- Humus C and N decomposition / net mineralization
- Fresh organic matter (FOM / residue) decomposition
- Immobilisation when soil C:N ratio is high

The module exposes ``mnr`` (mineralization) and ``imm`` (immobilisation)
arrays that the :class:`~dssat.soil.nitrogen.soilni.SoilNitrogenModule`
uses to update the NH₄ pool.

References:
    Godwin, D.C. & Jones, C.A. (1991) Nitrogen dynamics in soil–plant
    systems.  In: *Modeling Plant and Soil Systems*, ASA Monograph 31.
    Parton, W.J. et al. (1988) Dynamics of C, N, P, and S in grassland
    soils: a model.  *Biogeochemistry* 5, 109–131.
"""

from __future__ import annotations

import numpy as np

from dssat.core.constants import (
    NL, RUNINIT, SEASINIT, RATE, INTEGR, OUTPUT, SEASEND, ENDRUN,
)
from dssat.core.types import (
    ControlType, SwitchType, SoilType, WeatherType,
)


class SoilOrganicMatterModule:
    """CERES soil organic matter decomposition and N mineralization.

    Maintains humus and fresh organic matter (FOM) pools per layer and
    computes daily mineralization/immobilisation rates.

    Attributes:
        humc: Humus carbon by layer (kg C ha⁻¹), shape ``(NL,)``.
        humn: Humus nitrogen by layer (kg N ha⁻¹), shape ``(NL,)``.
        fom: Fresh organic matter carbon by layer (kg C ha⁻¹), shape
            ``(NL,)``.
        fon: Fresh organic nitrogen by layer (kg N ha⁻¹), shape ``(NL,)``.
        ssom: Total soil organic matter by layer (kg ha⁻¹), shape
            ``(NL,)`` — approximated as ``humc / 0.58``.
        mnr: Daily net N mineralization by layer (kg N ha⁻¹ d⁻¹), shape
            ``(NL,)``.  Positive = mineral N released to NH₄ pool.
        imm: Daily N immobilisation by layer (kg N ha⁻¹ d⁻¹), shape
            ``(NL,)``.  Positive = mineral N removed from NH₄ pool.
        cummnr: Cumulative season mineralization (kg N ha⁻¹).
        cumimm: Cumulative season immobilisation (kg N ha⁻¹).
        cumfom_dec: Cumulative FOM-C decomposed (kg C ha⁻¹).
    """

    def __init__(self) -> None:
        self.humc: np.ndarray = np.zeros(NL)
        self.humn: np.ndarray = np.zeros(NL)
        self.fom: np.ndarray = np.zeros(NL)
        self.fon: np.ndarray = np.zeros(NL)
        self.ssom: np.ndarray = np.zeros(NL)

        # Daily rate outputs (exposed for coupling to SoilNitrogenModule)
        self.mnr: np.ndarray = np.zeros(NL)
        self.imm: np.ndarray = np.zeros(NL)

        # Cumulative season totals
        self.cummnr: float = 0.0
        self.cumimm: float = 0.0
        self.cumfom_dec: float = 0.0

        # Internal delta arrays
        self._dlt_humc: np.ndarray = np.zeros(NL)
        self._dlt_humn: np.ndarray = np.zeros(NL)
        self._dlt_fom: np.ndarray = np.zeros(NL)
        self._dlt_fon: np.ndarray = np.zeros(NL)

    # ------------------------------------------------------------------
    def run(
        self,
        control: ControlType,
        iswitch: SwitchType,
        soilprop: SoilType,
        weather: WeatherType,
        sw: np.ndarray,
        fom_add: np.ndarray | None = None,
        fon_add: np.ndarray | None = None,
    ) -> None:
        """Advance the organic matter module one simulation step.

        Args:
            control: Simulation control block (provides ``dynamic`` phase).
            iswitch: Simulation switches (``mesom`` selects SOM method;
                ``iswnit`` controls N cycling).
            soilprop: Current soil profile properties (``oc``, ``bd``,
                ``dlayr``, ``ll``, ``dul`` etc.).
            weather: Today's weather (``tmax``, ``tmin``).
            sw: Volumetric soil water content by layer (cm³ cm⁻³), shape
                ``(NL,)`` — from the water balance module.
            fom_add: Fresh organic matter C added today by layer
                (kg C ha⁻¹), shape ``(NL,)``.  Pass ``None`` or zeros when
                no residue is applied.
            fon_add: Fresh organic N added today by layer (kg N ha⁻¹),
                shape ``(NL,)``.  Pass ``None`` or zeros when no residue
                is applied.
        """
        dynamic = control.dynamic

        if dynamic == RUNINIT:
            pass

        elif dynamic == SEASINIT:
            self._season_init(soilprop)

        elif dynamic == RATE:
            if iswitch.iswnit == "Y":
                self._daily_rate(soilprop, weather, sw, fom_add, fon_add)

        elif dynamic == INTEGR:
            if iswitch.iswnit == "Y":
                self._integrate(soilprop)

        elif dynamic == OUTPUT:
            self._update_ssom(soilprop)

        elif dynamic in (SEASEND, ENDRUN):
            pass

    # ------------------------------------------------------------------
    def add_residue(
        self,
        nlayr: int,
        fom_by_layer: np.ndarray,
        fon_by_layer: np.ndarray,
    ) -> None:
        """Add fresh organic matter to the FOM pools immediately.

        Typically called when a tillage/residue incorporation event occurs
        outside the normal RATE phase.

        Args:
            nlayr: Number of active soil layers.
            fom_by_layer: FOM carbon to add by layer (kg C ha⁻¹).
            fon_by_layer: FOM nitrogen to add by layer (kg N ha⁻¹).
        """
        for L in range(nlayr):
            self.fom[L] = max(0.0, self.fom[L] + fom_by_layer[L])
            self.fon[L] = max(0.0, self.fon[L] + fon_by_layer[L])

    # ------------------------------------------------------------------
    def _season_init(self, soilprop: SoilType) -> None:
        """Seasonal initialisation — build pools from OC profile."""
        nlayr = soilprop.nlayr

        # Reset cumulative counters
        self.cummnr = 0.0
        self.cumimm = 0.0
        self.cumfom_dec = 0.0

        for L in range(nlayr):
            bd_l = soilprop.bd[L]
            dl_l = soilprop.dlayr[L]
            oc_l = soilprop.oc[L]  # g per 100 g = %

            # kg C ha⁻¹ = (oc/100) * bd [g/cm³] * dlayr [cm] * 1e7 [cm²/ha]
            # × 1e-3 [g→kg]  ≡  oc * bd * dlayr * 1000 * 10
            self.humc[L] = oc_l * bd_l * dl_l * 1000.0 * 10.0
            self.humn[L] = self.humc[L] / 10.0  # C:N ≈ 10

            # FOM starts empty; residues are added separately
            self.fom[L] = 0.0
            self.fon[L] = 0.0

        # Layers beyond nlayr stay at zero
        self.humc[nlayr:] = 0.0
        self.humn[nlayr:] = 0.0
        self.fom[nlayr:] = 0.0
        self.fon[nlayr:] = 0.0

        self._update_ssom(soilprop)

    # ------------------------------------------------------------------
    def _daily_rate(
        self,
        soilprop: SoilType,
        weather: WeatherType,
        sw: np.ndarray,
        fom_add: np.ndarray | None,
        fon_add: np.ndarray | None,
    ) -> None:
        """Compute daily SOM decomposition and mineralization rates."""
        nlayr = soilprop.nlayr

        # Reset deltas
        self._dlt_humc[:] = 0.0
        self._dlt_humn[:] = 0.0
        self._dlt_fom[:] = 0.0
        self._dlt_fon[:] = 0.0
        self.mnr[:] = 0.0
        self.imm[:] = 0.0

        # ---- Soil temperature estimate
        tsom = (weather.tmax + weather.tmin) / 2.0 + 2.0
        tf = max(0.0, min(1.0, (tsom - 5.0) / 25.0))

        # ---- Add incoming FOM if any
        if fom_add is not None:
            for L in range(nlayr):
                self._dlt_fom[L] += fom_add[L]
        if fon_add is not None:
            for L in range(nlayr):
                self._dlt_fon[L] += fon_add[L]

        for L in range(nlayr):
            ll_l = soilprop.ll[L]
            dul_l = soilprop.dul[L]
            sw_l = sw[L]

            # ---- Water factor (0–1)
            denom = dul_l - ll_l
            wfsom = (sw_l - ll_l) / denom if denom > 0.0 else 0.0
            wfsom = max(0.0, min(1.0, wfsom))

            # ---- Humus decomposition
            hrate = 0.000083 * tf * wfsom
            humc_dec = self.humc[L] * hrate
            humn_dec = self.humn[L] * hrate

            # Limit decomposition to available pool
            humc_dec = min(humc_dec, self.humc[L])
            humn_dec = min(humn_dec, self.humn[L])

            # ---- Mineralization vs immobilisation
            cn_ratio = self.humc[L] / (self.humn[L] + 1.0e-6)
            if cn_ratio > 10.0:
                # C:N too high → immobilise some inorganic N
                imm_l = humn_dec * 0.3
                mnr_l = 0.0
            else:
                # Mineralise N to NH₄ pool
                mnr_l = humn_dec
                imm_l = 0.0

            self.mnr[L] = mnr_l
            self.imm[L] = imm_l

            # Humus pool losses (C and N both decline as SOM decomposes)
            self._dlt_humc[L] -= humc_dec
            self._dlt_humn[L] -= humn_dec

            # ---- FOM decomposition
            frate = 0.05 * tf * wfsom
            fom_avail = max(0.0, self.fom[L] + self._dlt_fom[L])
            fon_avail = max(0.0, self.fon[L] + self._dlt_fon[L])
            fom_dec = fom_avail * frate
            fon_dec = fon_avail * frate

            # FOM decomposition feeds into humus (60 %) + respiration (40 %)
            self._dlt_fom[L] -= fom_dec
            self._dlt_fon[L] -= fon_dec
            self._dlt_humc[L] += fom_dec * 0.6
            self._dlt_humn[L] += fon_dec * 0.6

            # Net FOM mineralization added to the mnr output
            self.mnr[L] += fon_dec * 0.4
            self.cumfom_dec += fom_dec

        # ---- Season cumulative totals
        self.cummnr += float(np.sum(self.mnr[:nlayr]))
        self.cumimm += float(np.sum(self.imm[:nlayr]))

    # ------------------------------------------------------------------
    def _integrate(self, soilprop: SoilType) -> None:
        """Apply rate deltas to SOM state variables."""
        nlayr = soilprop.nlayr
        for L in range(nlayr):
            self.humc[L] = max(0.0, self.humc[L] + self._dlt_humc[L])
            self.humn[L] = max(0.0, self.humn[L] + self._dlt_humn[L])
            self.fom[L] = max(0.0, self.fom[L] + self._dlt_fom[L])
            self.fon[L] = max(0.0, self.fon[L] + self._dlt_fon[L])

        self._update_ssom(soilprop)

    # ------------------------------------------------------------------
    def _update_ssom(self, soilprop: SoilType) -> None:
        """Recompute total SOM from humus C pool.

        SOM ≈ humus C / 0.58 (Van Bemmelen factor).
        """
        nlayr = soilprop.nlayr
        for L in range(nlayr):
            self.ssom[L] = self.humc[L] / 0.58
        self.ssom[nlayr:] = 0.0
