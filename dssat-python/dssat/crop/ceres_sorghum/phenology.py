"""CERES-Sorghum phenology — growth stage determination and thermal time.

Translated from ``SG_PHENOL.for`` + ``SG_PHASEI.for`` (DSSAT CERES-Sorghum v4.8).

Growth stages (ISTAGE):

| Code | Description                                          |
|------|------------------------------------------------------|
| 7    | Sowing                                                |
| 8    | Germination                                           |
| 9    | Emergence                                             |
| 1    | End of juvenile phase                                 |
| 2    | Panicle initiation (photoperiod-sensitive)            |
| 3    | End of flag leaf expansion                            |
| 4    | Beginning of grain filling (anthesis sub-phase)       |
| 5    | End of grain filling                                  |
| 6    | Physiological maturity                                |

References:
    Ritchie, J.T. et al. (1998) DSSAT v3: Sorghum simulation.
    Revision history in SG_PHENOL.for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from dssat.core.constants import NL, RUNINIT, SEASINIT, RATE, INTEGR


@dataclass
class SorghumCultivar:
    """CERES-Sorghum cultivar and ecotype coefficients.

    Parameters come from ``SGCER048.CUL``, ``SGCER048.ECO``, and
    ``SGCER048.SPE``.  Use :meth:`from_file` to load directly from disk.

    Attributes:
        varno:  Cultivar identifier (6-char).
        vrname: Cultivar name.
        econo:  Ecotype identifier (6-char).
        p1:    Thermal time from emergence to end of juvenile phase (°C d).
        p2:    Base thermal time for photoperiod-sensitive stage (°C d).
               ``RATEIN = P2 + P2R*(TWILEN-P2O)`` — SG_PHENOL.for:562.
        p2o:   Critical photoperiod (h) — no delay below this.
        p2r:   Photoperiod sensitivity (°C d h⁻¹ above P2O).
        p3:    Thermal time: panicle initiation → anthesis (fraction of PANTH) (°C d).
        p4:    Thermal time: flag leaf → begin grain fill lag (°C d).
        p5:    Thermal time for grain filling (°C d).
        g1:    Grain number scalar (grains g⁻¹ stover).
        g2:    Individual kernel weight (mg).
        phint: Phyllochron interval (°C d leaf⁻¹).
        panth: Total thermal time: panicle initiation → anthesis (°C d).
        tbase: Base temperature for development (°C).  From ECO.
        topt:  Optimum vegetative temperature (°C).  From ECO.
        ropt:  Optimum reproductive temperature (°C).  From ECO.
        gdde:  GDD per cm seed depth for emergence (°C d cm⁻¹).  From ECO.
        rue:   Radiation use efficiency (g MJ⁻¹ PAR).  From ECO.
        kcan:  Canopy PAR extinction coefficient.  From ECO.
    """

    varno: str = "IS3301"
    vrname: str = "IS3301 Sorghum"
    econo: str = "DFAULT"

    # Cultivar coefficients (CUL file)
    p1:    float = 450.0
    p2:    float = 100.0
    p2o:   float = 12.5
    p2r:   float = 180.0
    p3:    float = 100.0
    p4:    float = 50.0
    p5:    float = 480.0
    g1:    float = 6.5
    g2:    float = 27.0
    phint: float = 38.9
    panth: float = 550.0

    # Ecotype coefficients (ECO file)
    tbase: float = 7.0
    topt:  float = 30.0
    ropt:  float = 26.0
    gdde:  float = 6.5
    rue:   float = 3.5
    kcan:  float = 0.85

    @classmethod
    def from_file(
        cls,
        varno: str,
        *,
        model_code: str = "SGCER048",
        genotype_dir=None,
    ) -> "SorghumCultivar":
        """Load cultivar from ``SGCER048.CUL / .ECO / .SPE`` files."""
        from dssat.crop.ceres_sorghum.fileio import load_ceres_sorghum_params
        p = load_ceres_sorghum_params(varno, model_code=model_code,
                                      genotype_dir=genotype_dir)
        cv = cls(
            varno=p.cul.varno,   vrname=p.cul.vrname,  econo=p.cul.econo,
            p1=p.cul.p1,         p2=p.cul.p2,           p2o=p.cul.p2o,
            p2r=p.cul.p2r,       p3=p.cul.p3,           p4=p.cul.p4,
            p5=p.cul.p5,         g1=p.cul.g1,           g2=p.cul.g2,
            phint=p.cul.phint,   panth=p.cul.panth,
            tbase=p.eco.tbase,   topt=p.eco.topt,       ropt=p.eco.ropt,
            gdde=p.eco.gdde,     rue=p.eco.rue,         kcan=p.eco.kcan,
        )
        cv._params = p
        return cv


@dataclass
class PhenologyState:
    """Mutable state for the CERES-Sorghum phenology sub-model.

    Attributes:
        istage:    Current growth stage code (7, 8, 9, 1–6).
        xstage:    Continuous growth-stage indicator.
        cumdtt:    Cumulative thermal time from germination (°C d).
        sumdtt:    Cumulative TT within current stage (°C d).
        dtt:       Daily thermal time (°C d d⁻¹).
        stgdoy:    YYYYDDD date of each stage (0-indexed array of size 20).
        mdate:     Maturity date (YYYYDDD); -99 until reached.
        isdate:    Anthesis date (YYYYDDD).
        yremrg:    Emergence date (YYYYDDD).
        gpp:       Grain number per plant.
        cumph:     Cumulative phyllochron units.
        leafno:    Number of fully expanded leaves (proxy for XN in Fortran).
        tlno:      Total leaf number at flag leaf.
        kep:       Light extinction coefficient for even-aged canopy.
        crop_status: Status flag (0=running, 1=matured, 12/13=failure).
        l0:        Soil layer containing the seed.
        ndas:      Days after sowing.
        swsd:      Modified soil water content for germination check.
        sind:      Photoperiod induction index (SIND in SG_PHENOL.for).
        cump2:     Cumulative thermal time in stage 2 (CUMP2).
        sumdtt_2:  SUMDTT at end of stage 2 (for VegFrac).
        sumdtt_3:  SUMDTT at end of stage 3 (for VegFrac).
        pflowr:    Photoperiod-adjusted flower threshold (°C d).
        idur1:     Calendar days within the current vegetative stage.
        p9:        Computed TT threshold for emergence (50 + GDDE*SDEPTH).
        vegfrac:   Vegetative fraction complete (P model).
        seedfrac:  Seed-fill fraction complete (P model).
    """

    istage: int = 7
    xstage: float = 0.1
    cumdtt: float = 0.0
    sumdtt: float = 0.0
    dtt: float = 0.0
    stgdoy: np.ndarray = field(
        default_factory=lambda: np.full(20, 9999999, dtype=int)
    )
    mdate: int = -99
    isdate: int = 0
    yremrg: int = -99
    gpp: float = 0.0
    cumph: float = 0.0
    leafno: int = 0
    tlno: float = 0.0
    kep: float = 0.7
    crop_status: int = 0
    l0: int = 0
    ndas: int = 0
    swsd: float = 0.0
    # Stage-2 base thermal time threshold (mirrors cv.p2; accessible by tests)
    p2: float = 0.0
    # Stage-2 photoperiod state (SG_PHENOL.for: SIND, CUMP2)
    sind: float = 0.0
    cump2: float = 0.0
    # Captured SUMDTT values at stage transitions (for VegFrac / TLNO)
    sumdtt_2: float = 0.0
    sumdtt_3: float = 0.0
    # Photoperiod-adjusted threshold for flower (SG_PHASEI.for: PFLOWR)
    pflowr: float = 0.0
    # Calendar days in vegetative stage (SG_PHENOL.for: IDUR1)
    idur1: int = 0
    # Computed P9 value
    p9: float = 0.0
    # P-model fractions
    vegfrac: float = 0.0
    seedfrac: float = 0.0


def compute_dtt(
    tmax: float,
    tmin: float,
    srad: float,
    dayl: float,
    snow: float,
    leafno: int,
    istage: int,
    tbase: float,
    topt: float,
    ropt: float,
) -> float:
    """Compute daily thermal time (°C d).

    Matches ``SG_PHENOL.for`` DTT algorithm (identical to maize except
    for the CTYPE=1 branch which uses ``LEAFNO <= 10``).

    Args:
        tmax:   Daily maximum temperature (°C).
        tmin:   Daily minimum temperature (°C).
        srad:   Solar radiation (MJ m⁻² d⁻¹).
        dayl:   Day length (h).
        snow:   Snow depth (mm).
        leafno: Number of fully expanded leaves.
        istage: Current CERES growth stage code.
        tbase:  Base temperature (°C).
        topt:   Vegetative optimum temperature (°C).
        ropt:   Reproductive optimum temperature (°C).

    Returns:
        Daily thermal time (°C d), ≥ 0.
    """
    # SG_PHENOL.for:287-290
    dopt = ropt if (3 < istage <= 6) else topt

    xs = min(snow, 15.0)
    tempcn = tmin
    tempcx = tmax
    if tmin < 0.0:
        tempcn = 2.0 + tmin * (0.4 + 0.0018 * (xs - 15.0) ** 2)
    if tmax < 0.0:
        tempcx = 2.0 + tmax * (0.4 + 0.0018 * (xs - 15.0) ** 2)

    if tmax < tbase:
        return 0.0
    if tmin > dopt:
        return dopt - tbase

    # CTYPE=1 (corn/sorghum): soil temp method when LEAFNO <= 10
    # SG_PHENOL.for:308
    if leafno <= 10:
        if xs > 0.0:
            dtt = (tempcn + tempcx) / 2.0 - tbase
        else:
            acoef  = 0.01061 * srad + 0.5902
            tdsoil = acoef * tmax + (1.0 - acoef) * tmin
            tnsoil = 0.36354 * tmax + 0.63646 * tmin
            if tdsoil < tbase:
                return 0.0
            tnsoil = max(tnsoil, tbase)
            tdsoil = min(tdsoil, dopt)
            tmsoil = tdsoil * (dayl / 24.0) + tnsoil * ((24.0 - dayl) / 24.0)
            if tmsoil < tbase:
                dtt = (tbase + tdsoil) / 2.0 - tbase
            else:
                dtt = (tnsoil + tdsoil) / 2.0 - tbase
            dtt = min(dtt, dopt - tbase)
    elif tmin < tbase or tmax > dopt:
        dtt = 0.0
        for i in range(1, 25):
            th = (tmax + tmin) / 2.0 + (tmax - tmin) / 2.0 * math.sin(
                math.pi / 12.0 * i
            )
            th = min(max(th, tbase), dopt)
            dtt += (th - tbase) / 24.0
    else:
        dtt = (tmax + tmin) / 2.0 - tbase

    return max(dtt, 0.0)


class SorghumPhenology:
    """CERES-Sorghum phenology sub-model.

    Implements the stage-transition logic of ``SG_PHENOL.for`` and
    ``SG_PHASEI.for`` in Python.

    Args:
        cultivar: Cultivar and ecotype coefficients.
        pltpop:   Plant population (plants m⁻²).
        sdepth:   Sowing depth (cm).
        yrsim:    Simulation start date (YYYYDDD).
    """

    def __init__(
        self,
        cultivar: SorghumCultivar,
        pltpop: float,
        sdepth: float,
        yrsim: int,
    ) -> None:
        self.cultivar = cultivar
        self.pltpop = pltpop
        self.sdepth = sdepth
        self.state = PhenologyState()
        self.state.stgdoy[13] = yrsim  # STGDOY(14)=YRSIM

        # SPE seed parameters
        spe = getattr(getattr(cultivar, "_params", None), "spe", None)
        self._dsgt: float = getattr(spe, "dsgt", 21.0)
        self._dget: float = getattr(spe, "dget", 150.0)
        self._swcg: float = getattr(spe, "swcg", 0.02)

    # ------------------------------------------------------------------
    def run(
        self,
        dynamic: int,
        yrdoy: int,
        tmax: float,
        tmin: float,
        srad: float,
        dayl: float,
        snow: float,
        sw: np.ndarray,
        ll: np.ndarray,
        dlayr: np.ndarray,
        nlayr: int,
        iswwat: str,
        twilen: float = -1.0,
    ) -> None:
        """Run one phenology time step.

        Args:
            dynamic:  Simulation phase (RUNINIT, SEASINIT, RATE, INTEGR).
            yrdoy:    Current date (YYYYDDD).
            tmax:     Max temperature (°C).
            tmin:     Min temperature (°C).
            srad:     Solar radiation (MJ m⁻² d⁻¹).
            dayl:     Day length (h).
            snow:     Snow depth (mm).
            sw:       Soil water by layer (cm³ cm⁻³).
            ll:       Lower limit by layer (cm³ cm⁻³).
            dlayr:    Layer thickness (cm).
            nlayr:    Number of active layers.
            iswwat:   Water switch (``"Y"``/``"N"``).
            twilen:   Twilight daylength (h); defaults to *dayl* if ≤ 0.
        """
        if twilen <= 0.0:
            twilen = dayl
        if dynamic in (RUNINIT, SEASINIT):
            self._init(yrdoy)
        else:
            self._rate_integr(
                yrdoy, tmax, tmin, srad, dayl, snow,
                sw, ll, dlayr, nlayr, iswwat, twilen
            )

    # ------------------------------------------------------------------
    def _init(self, yrsim: int) -> None:
        """Reset state for a new season."""
        s = self.state
        cv = self.cultivar
        s.istage     = 7
        s.xstage     = 0.1
        s.cumdtt     = 0.0
        s.sumdtt     = 0.0
        s.dtt        = 0.0
        s.stgdoy[:]  = 9999999
        s.stgdoy[13] = yrsim
        s.mdate      = -99
        s.isdate     = 0
        s.yremrg     = -99
        s.gpp        = 0.0
        s.cumph      = 0.0
        s.leafno     = 0
        s.tlno       = 0.0
        s.kep        = cv.kcan / (1.0 - 0.07) * (1.0 - 0.25)
        s.crop_status = 0
        s.l0         = 0
        s.ndas       = 0
        s.swsd       = 0.0
        s.p2         = cv.p2
        s.sind       = 0.0
        s.cump2      = 0.0
        s.sumdtt_2   = 0.0
        s.sumdtt_3   = 0.0
        s.pflowr     = 0.0
        s.idur1      = 0
        s.p9         = 0.0
        s.vegfrac    = 0.0
        s.seedfrac   = 0.0

    # ------------------------------------------------------------------
    def _rate_integr(
        self,
        yrdoy: int,
        tmax: float,
        tmin: float,
        srad: float,
        dayl: float,
        snow: float,
        sw: np.ndarray,
        ll: np.ndarray,
        dlayr: np.ndarray,
        nlayr: int,
        iswwat: str,
        twilen: float,
    ) -> None:
        """Daily rate + integration (RATE/INTEGR phases)."""
        s = self.state
        cv = self.cultivar

        # Compute thermal time (SG_PHENOL.for:368-369)
        s.dtt = compute_dtt(
            tmax, tmin, srad, dayl, snow,
            s.leafno, s.istage,
            cv.tbase, cv.topt, cv.ropt,
        )
        s.sumdtt += s.dtt
        s.cumdtt  += s.dtt

        # DOPT for XSTAGE denominators
        dopt = cv.ropt if (3 < s.istage <= 6) else cv.topt

        # Dispatch to stage-specific logic
        if s.istage == 7:
            self._stage7(yrdoy, dlayr, nlayr, iswwat)
        elif s.istage == 8:
            self._stage8(yrdoy, sw, ll, nlayr, iswwat)
        elif s.istage == 9:
            self._stage9(yrdoy)
        elif s.istage == 1:
            self._stage1(yrdoy, dopt)
        elif s.istage == 2:
            self._stage2(yrdoy, dopt, twilen)
        elif s.istage == 3:
            self._stage3(yrdoy, dopt, twilen)
        elif s.istage == 4:
            self._stage4(yrdoy, twilen)
        elif s.istage == 5:
            self._stage5(yrdoy)
        elif s.istage == 6:
            self._stage6(yrdoy)

    # ------------------------------------------------------------------
    # Stage 7: Sowing (SG_PHENOL.for:393-423, SG_PHASEI.for:270-274)
    # ------------------------------------------------------------------
    def _stage7(
        self,
        yrdoy: int,
        dlayr: np.ndarray,
        nlayr: int,
        iswwat: str,
    ) -> None:
        s = self.state
        cv = self.cultivar
        s.stgdoy[7 - 1] = yrdoy
        # SG_PHASEI: ISTAGE=8, RTDEP=SDEPTH, SUMDTT=0
        s.istage = 8
        s.sumdtt = 0.0
        if iswwat == "N":
            return
        cumdep = 0.0
        for L in range(nlayr):
            cumdep += dlayr[L]
            if self.sdepth < cumdep:
                s.l0 = L
                break

    # ------------------------------------------------------------------
    # Stage 8: Germination (SG_PHENOL.for:428-459, SG_PHASEI.for:279-285)
    # ------------------------------------------------------------------
    def _stage8(
        self,
        yrdoy: int,
        sw: np.ndarray,
        ll: np.ndarray,
        nlayr: int,
        iswwat: str,
    ) -> None:
        s = self.state
        cv = self.cultivar
        if iswwat != "N":
            l0 = s.l0
            # SG_PHENOL.for:430: only check water when SW <= LL
            if sw[l0] <= ll[l0]:
                l1 = min(l0 + 1, nlayr - 1)
                s.swsd = (
                    (sw[l0] - ll[l0]) * 0.65
                    + (sw[l1] - ll[l1]) * 0.35
                )
                if s.swsd < self._swcg:
                    return
        # Germination — SG_PHASEI: ISTAGE=9, CUMDTT=0, SUMDTT=0
        s.stgdoy[8 - 1] = yrdoy
        s.istage = 9
        s.cumdtt = 0.0
        s.sumdtt = 0.0
        # P9 = 50 + GDDE*SDEPTH  (SG_PHENOL.for:458)
        s.p9 = 50.0 + cv.gdde * self.sdepth

    # ------------------------------------------------------------------
    # Stage 9: Emergence (SG_PHENOL.for:464-502, SG_PHASEI.for:290-338)
    # ------------------------------------------------------------------
    def _stage9(self, yrdoy: int) -> None:
        s = self.state
        cv = self.cultivar
        s.idur1 += 1
        if s.sumdtt < s.p9:            # SG_PHENOL.for:467
            return
        # Failure check (SG_PHENOL.for:468-483)
        if s.p9 > self._dget:
            s.istage = 6
            self.pltpop = 0.0
            s.gpp = 1.0
            s.mdate = yrdoy
            s.crop_status = 12
            return
        s.stgdoy[9 - 1] = yrdoy
        # SG_PHASEI:292 — ISTAGE=1, SUMDTT=SUMDTT-P9 (carryover)
        s.istage  = 1
        s.sumdtt  = s.sumdtt - s.p9
        s.cumdtt  = max(0.0, s.cumdtt - s.p9)
        s.tlno    = 30.0
        s.leafno  = 1
        s.cumph   = s.sumdtt / cv.phint if cv.phint > 0.0 else 0.0
        s.yremrg  = yrdoy

    # ------------------------------------------------------------------
    # Stage 1: Emergence → End of Juvenile (SG_PHENOL.for:507-525)
    # ------------------------------------------------------------------
    def _stage1(self, yrdoy: int, dopt: float) -> None:
        s = self.state
        cv = self.cultivar
        s.idur1 += 1
        # XSTAGE = 2.0*SUMDTT/P1  (SG_PHENOL.for:508)
        s.xstage = 2.0 * s.sumdtt / cv.p1 if cv.p1 > 0.0 else 0.0
        # VegFrac (SG_PHENOL.for:522)
        denom_vf = cv.p1 + cv.p2r + cv.panth + cv.p4
        s.vegfrac = s.sumdtt / denom_vf if denom_vf > 0.0 else 0.0

        if s.sumdtt < cv.p1:           # SG_PHENOL.for:524
            return
        # Transition — SG_PHASEI:183-190: ISTAGE=2, SIND=0, CUMP2=0
        s.stgdoy[1 - 1] = yrdoy
        s.istage = 2
        s.sind   = 0.0
        s.cump2  = 0.0
        # NOTE: SUMDTT is NOT reset at stage 1→2 (SG_PHASEI does NOT reset it)

    # ------------------------------------------------------------------
    # Stage 2: End Juvenile → Panicle Initiation (SG_PHENOL.for:531-603)
    # ------------------------------------------------------------------
    def _stage2(self, yrdoy: int, dopt: float, twilen: float) -> None:
        s = self.state
        cv = self.cultivar
        s.idur1 += 1
        # XSTAGE = 2.0 + SIND  (SG_PHENOL.for:532)
        s.xstage = 2.0 + s.sind
        # VegFrac (SG_PHENOL.for:591)
        denom_vf = cv.p1 + cv.p2r + cv.panth + cv.p4
        s.vegfrac = s.sumdtt / denom_vf if denom_vf > 0.0 else 0.0

        # Accumulate CUMP2 (SG_PHENOL.for:568)
        s.cump2 += s.dtt
        # RATEIN = P2 + P2R*(TWILEN-P2O) or P2 (SG_PHENOL.for:560-566)
        if twilen > cv.p2o:
            ratein = cv.p2 + cv.p2r * (twilen - cv.p2o)
        else:
            ratein = cv.p2
        # SIND = max(SIND, CUMP2/RATEIN)  (SG_PHENOL.for:569)
        if ratein > 0.0:
            s.sind = max(s.sind, s.cump2 / ratein)

        if s.cump2 < ratein:           # SG_PHENOL.for:594
            return

        # Transition — SG_PHASEI:194-207: ISTAGE=3, TLNO=(CUMDTT/35+6), SUMDTT=0
        s.stgdoy[2 - 1] = yrdoy
        s.sumdtt_2 = s.sumdtt            # SG_PHENOL.for:601
        s.vegfrac  = max(
            s.vegfrac,
            s.sumdtt_2 / (s.sumdtt_2 + cv.panth + cv.p4)
            if (s.sumdtt_2 + cv.panth + cv.p4) > 0.0 else 0.0,
        )
        s.istage   = 3
        s.tlno     = s.cumdtt / 35.0 + 6.0   # SG_PHASEI.for:196
        s.sumdtt   = 0.0

    # ------------------------------------------------------------------
    # Stage 3: Panicle Init → End Flag Leaf (SG_PHENOL.for:615-634)
    # ------------------------------------------------------------------
    def _stage3(self, yrdoy: int, dopt: float, twilen: float) -> None:
        s = self.state
        cv = self.cultivar
        s.idur1 += 1
        # XSTAGE = 3.0 + 2.0*SUMDTT/(PANTH-P3)  (SG_PHENOL.for:618)
        dur3 = cv.panth - cv.p3
        s.xstage = 3.0 + 2.0 * s.sumdtt / dur3 if dur3 > 0.0 else 3.0
        # VegFrac (SG_PHENOL.for:620-621)
        denom_vf = s.sumdtt_2 + cv.panth + cv.p4
        s.vegfrac = max(
            s.vegfrac,
            (s.sumdtt_2 + s.sumdtt) / denom_vf if denom_vf > 0.0 else 0.0,
        )

        if s.sumdtt < dur3:            # SG_PHENOL.for:625
            return

        # Transition — SG_PHASEI:213-226: ISTAGE=4, SUMDTT=SUMDTT-(PANTH-P3), PFLOWR set
        s.stgdoy[3 - 1] = yrdoy
        s.sumdtt_3 = s.sumdtt_2 + s.sumdtt   # SG_PHASEI.for (captured P1+P2+P3)
        s.vegfrac  = max(
            s.vegfrac,
            s.sumdtt_3 / (s.sumdtt_3 + cv.p3 + cv.p4)
            if (s.sumdtt_3 + cv.p3 + cv.p4) > 0.0 else 0.0,
        )
        s.istage   = 4
        # SG_PHASEI.for:219 — carryover
        s.sumdtt   = s.sumdtt - dur3
        # PFLOWR = P3 + P2R*(TWILEN-P2O) or P3  (SG_PHASEI.for:220-224)
        if twilen > cv.p2o:
            s.pflowr = cv.p3 + cv.p2r * (twilen - cv.p2o)
        else:
            s.pflowr = cv.p3

    # ------------------------------------------------------------------
    # Stage 4: Anthesis sub-phase (SG_PHENOL.for:641-685)
    # ------------------------------------------------------------------
    def _stage4(self, yrdoy: int, twilen: float) -> None:
        s = self.state
        cv = self.cultivar
        # XSTAGE = 5.0 + SUMDTT/PFLOWR  (SG_PHENOL.for:653)
        s.xstage = 5.0 + s.sumdtt / s.pflowr if s.pflowr > 0.0 else 5.0
        # VegFrac (SG_PHENOL.for:671)
        denom_vf = s.sumdtt_3 + s.pflowr + cv.p4
        s.vegfrac = max(
            s.vegfrac,
            (s.sumdtt + s.sumdtt_3) / denom_vf if denom_vf > 0.0 else 0.0,
        )

        # ISDATE set when SUMDTT >= PFLOWR (anthesis)  (SG_PHENOL.for:658-665)
        if s.isdate == 0 and s.sumdtt >= s.pflowr:
            s.isdate = yrdoy

        # Transition when SUMDTT >= PFLOWR + P4  (SG_PHENOL.for:675)
        if s.sumdtt < s.pflowr + cv.p4:
            return

        # SG_PHASEI:230-250: ISTAGE=5, SUMDTT=SUMDTT-(PFLOWR+P4) (carryover)
        s.stgdoy[4 - 1] = yrdoy
        s.vegfrac  = 1.0
        if s.gpp <= 0.0:
            s.gpp = 1.0
        s.istage   = 5
        s.sumdtt   = s.sumdtt - (s.pflowr + cv.p4)

    # ------------------------------------------------------------------
    # Stage 5: Grain filling (SG_PHENOL.for:689-696, SG_PHASEI.for:255-257)
    # ------------------------------------------------------------------
    def _stage5(self, yrdoy: int) -> None:
        s = self.state
        cv = self.cultivar
        # XSTAGE = 6.5 + 2.5*SUMDTT/P5  (SG_PHENOL.for:691)
        s.xstage = 6.5 + 2.5 * s.sumdtt / cv.p5 if cv.p5 > 0.0 else 6.5
        s.seedfrac = s.sumdtt / cv.p5 if cv.p5 > 0.0 else 0.0

        if s.sumdtt < cv.p5:           # SG_PHENOL.for:695
            return
        s.seedfrac = 1.0
        s.stgdoy[5 - 1] = yrdoy
        # SG_PHASEI:256: ISTAGE=6
        s.istage = 6

    # ------------------------------------------------------------------
    # Stage 6: Physiological maturity (SG_PHENOL.for:701-739)
    # ------------------------------------------------------------------
    def _stage6(self, yrdoy: int) -> None:
        s = self.state
        # SG_PHENOL.for:702-703: wait for DTT>=2 and SUMDTT>=2
        if s.dtt < 2.0 or s.sumdtt < 2.0:
            return
        s.stgdoy[6 - 1] = yrdoy
        s.mdate      = yrdoy
        s.crop_status = 1
