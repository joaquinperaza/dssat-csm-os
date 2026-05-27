"""CROPGRO nitrogen fixation sub-model (legumes only).

Translated from ``NFIX.for`` in the DSSAT-CSM CROPGRO module.

Biological nitrogen fixation (BNF) is modelled as a daily nodule-growth
process limited by carbon supply, and the actual fixation rate is modified
by temperature, soil water, and nodule age factors.

This module is activated only for soybean (and other legumes).  For canola
and other non-legumes, N fixation is zero.

References:
    Boote, K.J., Mínguez, M.I., & Sau, F. (2002).
    Adapting the CROPGRO legume model to simulate growth of faba bean.
    Agron. J. 94, 743–756.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from dssat.core.constants import RUNINIT, SEASINIT, RATE, INTEGR


# ---------------------------------------------------------------------------
# N-fixation state
# ---------------------------------------------------------------------------

@dataclass
class NFIXState:
    """State for the N fixation sub-model.

    Attributes:
        dwnod:     Nodule dry weight (g plant^-1).
        wtnfx:     Cumulative N fixed (g N plant^-1).
        nfixn:     Daily N fixation rate (g N plant^-1 d^-1).
        cnod:      Nodule C demand coefficient (g N g^-1 nodule d^-1).
        rfixn:     Relative N fixation rate (0–1).
        fnfxt:     Temperature factor for N fixation (0–1).
        fnfxw:     Soil water factor for N fixation (0–1).
        cnfact:    Nodule age / C supply factor (0–1).
    """

    dwnod: float = 0.0
    wtnfx: float = 0.0
    nfixn: float = 0.0
    cnod: float = 0.0
    rfixn: float = 0.0
    fnfxt: float = 1.0
    fnfxw: float = 1.0
    cnfact: float = 1.0


# ---------------------------------------------------------------------------
# Utility: temperature response curves for N fixation
# ---------------------------------------------------------------------------

def _fnfxt(tmean: float) -> float:
    """Temperature factor for N fixation (legacy helper, not used in main code).

    Piecewise linear: 0 below 7°C, rises to 1.0 at 25–33°C,
    falls back to 0 at 42°C.

    Args:
        tmean: Daily mean temperature (°C).

    Returns:
        Temperature factor (0–1).
    """
    if tmean <= 7.0 or tmean >= 42.0:
        return 0.0
    if tmean <= 25.0:
        return (tmean - 7.0) / (25.0 - 7.0)
    if tmean >= 33.0:
        return (42.0 - tmean) / (42.0 - 33.0)
    return 1.0


def _curv_lin_temp(params: tuple, x: float) -> float:
    """Piecewise linear response for temperature: CURV('LIN', tb, to1, to2, tm, x).

    Maps to Fortran CURV function with type 'LIN':
        0 at x <= tb
        linearly rising to 1 at to1
        1 between to1 and to2
        linearly declining to 0 at tm
        0 above tm

    This is the CROPGRO temperature response used in NFIX.for for nodule
    temperature effects.

    Args:
        params: 4-tuple (tb, to1, to2, tm) — base, lower-optimum, upper-optimum, max.
        x:      Temperature (°C).

    Returns:
        Response factor (0–1).
    """
    tb, to1, to2, tm = params
    if x <= tb or x >= tm:
        return 0.0
    if x >= to1 and x <= to2:
        return 1.0
    if x < to1:
        return (x - tb) / max(to1 - tb, 1e-9)
    # x > to2
    return (tm - x) / max(tm - to2, 1e-9)


# ---------------------------------------------------------------------------
# N-fixation sub-model
# ---------------------------------------------------------------------------

class CropGRONFix:
    """CROPGRO nitrogen fixation sub-model.

    Nodule growth and N fixation for legumes.  For non-legumes (``fix_n=False``)
    all methods are no-ops and ``state.nfixn`` remains 0.

    Args:
        fix_n:   If True, enables N fixation (soybean).
        pltpop:  Plant population (plants m^-2).
    """

    # Species-level constants (from SBGRO048.SPE !*NITROGEN FIXATION PARAMETERS)
    # Matches NFIX.for variable names:
    # SNACTM  = max specific nodule activity (g N g^-1 nodule d^-1); SPE default 0.045
    # NODRGM  = max nodule growth rate relative to root (g nod g^-1 root d^-1); SPE default 0.17
    # DWNODI  = initial nodule dry weight per plant (g plant^-1); SPE default 0.014
    # NDTHMX  = max nodule death rate (fraction d^-1); SPE default 0.07
    # CNODCR  = C requirement for nodule respiration (g C g^-1 nodule d^-1); SPE default 0.05
    _SNACTM: float = 0.045   # max specific nodule activity (g N g^-1 nodule d^-1)
    _NODRGM: float = 0.170   # max nodule growth rate (g nod g^-1 root d^-1)
    _DWNODI: float = 0.014   # initial nodule weight per plant (g plant^-1)
    _NDTHMX: float = 0.070   # max nodule death rate (fraction d^-1)
    _CNODCR: float = 0.050   # C requirement for nodule respiration (g C g^-1 nod d^-1)
    # Temperature curve for N fixation: FNFXT = (tb, to1, to2, tm) from SPE
    # NFIX.for: CURV(TYPFXT, FNFXT(1..4), ST(I)) — uses SOIL TEMPERATURE
    # SPE default (SBGRO048.SPE): 4.0  20.0  35.0  44.0  LIN
    _FNFXT: tuple = (4.0, 20.0, 35.0, 44.0)
    # Dry stress function: FNFXD = (0.0, 0.85, 1.0, 10.0) applied to TURFAC
    # NFIX.for line 288: SWFACT = CURV(TYPFXD, FNFXD(1..4), TURFAC)
    _FNFXD: tuple = (0.0, 0.85, 1.0, 10.0)
    # Nodule age/DXR57 function: FNFXA applied to DXR57
    # NFIX.for line 289: NFXAGE = CURV(TYPFXA, FNFXA(1..4), DXR57)
    # SPE default: 0.0  0.10  1.0  0.0  INL (inverse linear: max at DXR57 near 0.10)
    _FNFXA: tuple = (0.0, 0.10, 1.0, 0.0)

    def __init__(self, fix_n: bool = True, pltpop: float = 30.0) -> None:
        self.fix_n = fix_n
        self.pltpop = pltpop
        self.state = NFIXState()

    # ------------------------------------------------------------------
    def run(
        self,
        dynamic: int,
        pheno: Any,
        tmax: float,
        tmin: float,
        sw: np.ndarray,
        ll: np.ndarray,
        dul: np.ndarray,
        dlayr: np.ndarray,
        nlayr: int,
        wrt: float,
        growth_avail: float,
        iswwat: str,
        st: np.ndarray | None = None,
        turfac: float = 1.0,
        ctonod: float = 0.0,
    ) -> None:
        """Run one time step.

        Mirrors NFIX.for INTEGR block.

        Args:
            dynamic:       Phase code.
            pheno:         Phenology state (provides rstage, dxr57, emerged).
            tmax:          Daily max temperature (°C).
            tmin:          Daily min temperature (°C).
            sw:            Soil water content by layer (cm^3 cm^-3).
            ll:            Lower limit by layer.
            dul:           Drained upper limit by layer.
            dlayr:         Layer thickness (cm).
            nlayr:         Number of active layers.
            wrt:           Root dry weight (g plant^-1) — for nodule:root ratio.
            growth_avail:  Available CH2O (g m^-2 d^-1) to nodule zone (CTONOD).
            iswwat:        Water switch.
            st:            Soil temperature by layer (°C).  If None, uses
                           mean air temperature as a fallback.
            turfac:        Turgor stress factor (0–1); 0 = full stress.
                           Used for SWFACT = CURV(FNFXD, TURFAC) per NFIX.for line 288.
            ctonod:        Carbon allocated to nodule zone (g CH2O m^-2 d^-1).
                           NFIX.for uses this as CTONOD; if 0 use growth_avail proxy.
        """
        if not self.fix_n:
            return

        if dynamic in (RUNINIT, SEASINIT):
            self._init()
            return

        if not pheno.emerged:
            return

        if pheno.mdate > 0:
            return

        s = self.state

        # Fallback soil temperature to air mean if not provided
        if st is None:
            tmean = (tmax + tmin) * 0.5
            st_arr = np.full(max(nlayr, 1), tmean)
        else:
            st_arr = st

        # ---- Temperature factor for N fixation (uses soil temperature per layer)
        # NFIX.for lines 252-274: weighted average over top DNOD (50 cm)
        # ACSTF = sum(DLAYR(I)*FLAYR * CURV(TYPFXT, FNFXT, ST(I))) / DNOD
        # Python: CURV('LIN', tb, to1, to2, tm, x) = curv_tp(tb, to1, to2, tm, x)
        # FNFXT from SPE: (4.0, 20.0, 35.0, 44.0) LIN → piecewise linear
        dnod = 50.0
        acstf = 0.0
        dsw_accum = 0.0
        for i in range(nlayr):
            flayr = 1.0
            dsw_accum += dlayr[i]
            if dsw_accum > dnod:
                flayr = (dnod - (dsw_accum - dlayr[i])) / dlayr[i]
            ti = st_arr[i] if i < len(st_arr) else st_arr[-1]
            acstf += dlayr[i] * flayr * _curv_lin_temp(self._FNFXT, ti)
            if flayr < 1.0:
                break
        tnfix = acstf / dnod  # TNFIX: soil T effect on N fixation (0-1)
        s.fnfxt = max(0.0, min(1.0, tnfix))

        # ---- Soil water drought factor on N fixation
        # NFIX.for line 288: SWFACT = CURV(TYPFXD, FNFXD(1..4), TURFAC)
        # TURFAC = plant turgor stress factor (0=no water, 1=no stress)
        # FNFXD = (0.0, 0.85, 1.0, 10.0) LIN: 0 when TURFAC<0, 1 when TURFAC>=0.85
        # curv_lin(0, 0.0, 0.85, 1.0, turfac) interpreted as LIN:
        #   below 0: 0; between 0 and 0.85: interpolated; above 0.85: 1.0
        d1, d2, d3, d4 = self._FNFXD
        if turfac <= d1:
            swfact = 0.0
        elif turfac >= d2:
            swfact = 1.0
        else:
            swfact = (turfac - d1) / max(d2 - d1, 1e-9)
        s.fnfxw = max(0.0, min(1.0, swfact))

        # ---- Nodule age / DXR57 effect on N fixation
        # NFIX.for line 289: NFXAGE = CURV(TYPFXA, FNFXA(1..4), DXR57)
        # FNFXA from SPE: (0.0, 0.10, 1.0, 0.0) INL (inverse linear)
        # INL: max (=1.0) at DXR57 <= fnfxa[1], declines to 0 at DXR57 >= fnfxa[2]
        # Python mapping: curv_lin(1.0, fnfxa[1], fnfxa[2], 0.0, DXR57)
        a1, a2, a3, a4 = self._FNFXA
        dxr57 = pheno.dxr57
        if dxr57 <= a2:
            nfxage = 1.0
        elif dxr57 >= a3:
            nfxage = a4
        else:
            nfxage = 1.0 + (a4 - 1.0) * (dxr57 - a2) / max(a3 - a2, 1e-9)
        s.cnfact = max(0.0, min(1.0, nfxage))

        # ---- C carbon factor (CNFACT from C allocation)
        # NFIX.for lines 241-245:
        #   CNFACT = 1.0 (default)
        #   IF (DWNOD > 1e-4): FRCNM = CTONOD/DWNOD; IF FRCNM < CNODCR: CNFACT = FRCNM/CNODCR
        # We use growth_avail * pltpop as a proxy for CTONOD (g CH2O m^-2 d^-1)
        ctonod_val = ctonod if ctonod > 0.0 else growth_avail * self.pltpop
        cnfact_c = 1.0
        dwnod_m2 = s.dwnod * self.pltpop  # convert to g m^-2
        if dwnod_m2 > 1e-4:
            frcnm = ctonod_val / dwnod_m2
            if frcnm < self._CNODCR:
                cnfact_c = frcnm / self._CNODCR
        # Combine age factor and C factor (use minimum as most limiting)
        s.cnfact = max(0.0, min(s.cnfact, cnfact_c))

        # ---- Daily N fixation (NFIX.for lines 327-333)
        # PNFIXN = MIN((CLEFT*0.16/RFIXN), (DWNOD*SNACT)) * TNFIX
        # NFIXN = PNFIXN * MIN(SWFACT, SWMEM8, FLDACT)
        # Simplified (no 8-day memory or flooding): NFIXN = DWNOD * SNACTM * TNFIX * SWFACT * CNFACT
        # Units: g N plant^-1 d^-1 (DWNOD in g plant^-1, SNACTM in g N g^-1 nodule d^-1)
        s.rfixn = self._SNACTM * s.fnfxt * s.fnfxw * s.cnfact
        s.nfixn = max(0.0, s.dwnod * s.rfixn)  # g N plant^-1 d^-1

        # ---- Integration: nodule growth
        # NFIX.for lines 351-360:
        # NODGR = MIN(CLEFT/AGRNOD, DWNOD*NODRGR) * TNGRO * MIN(SWFACT, FLDACT) * NFXAGE
        # DWNOD is per plant; NODRGM is per root weight → DWNOD * NODRGR = DWNOD * (NODRGM * root_ratio)
        # Simplified: nodule grows as fraction of root weight
        if dynamic == INTEGR:
            # Nodule growth rate relative to root weight (NODRGM = 0.17 g nod g^-1 root d^-1)
            # DWNOD is per plant; wrt is per plant
            max_nodule_grwth = wrt * self._NODRGM * s.fnfxt * s.fnfxw
            s.dwnod = max(self._DWNODI, s.dwnod + max_nodule_grwth * 0.5)  # 0.5 = growth efficiency
            # Clamp relative to root weight (physical maximum)
            if wrt > 0.0:
                s.dwnod = min(s.dwnod, wrt * 0.20)
            s.wtnfx += s.nfixn

    # ------------------------------------------------------------------
    def _init(self) -> None:
        """Reset N fixation state for a new season."""
        s = self.state
        s.dwnod = self._DWNODI
        s.wtnfx = 0.0
        s.nfixn = 0.0
        s.cnod = self._CNODCR  # use CNODCR as the reference C demand
        s.rfixn = 0.0
        s.fnfxt = 1.0
        s.fnfxw = 1.0
        s.cnfact = 1.0
