"""CERES-Millet phenology — growth stage determination and thermal time.

Translated from ``ML_PHENOL.for`` + ``ML_PHASEI.for``.

Growth stages follow the CERES convention:

| Code | Description                                    |
|------|------------------------------------------------|
| 7    | Sowing                                          |
| 8    | Germination                                     |
| 9    | Emergence                                       |
| 1    | End of juvenile phase                           |
| 2    | Panicle initiation (photoperiod-sensitive)      |
| 3    | End of leaf growth                              |
| 4    | End of panicle growth                           |
| 5    | End grain fill                                  |
| 6    | Maturity                                        |

References:
    Ritchie, J.T. et al. (1998) DSSAT v3: Millet simulation.
    Boote, K.J. et al. (2015) Major revisions to CERES-Millet model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from dssat.core.constants import NL, RUNINIT, SEASINIT, RATE, INTEGR


@dataclass
class MilletCultivar:
    """CERES-Millet cultivar and ecotype coefficients.

    Attributes:
        varno: Cultivar identifier (6-char).
        vrname: Cultivar name.
        econo: Ecotype identifier (6-char).
        p1: Thermal time from emergence to end of juvenile phase (°C d).
        p2o: Critical photoperiod (h) — below this, no photoperiod delay.
        p2r: Photoperiod sensitivity (°C d per h above p2o) — ecotype.
        p5: Thermal time for grain filling (°C d).
        g4: Partitioning coefficient for grain fill (scalar for PANWT).
        g5: Individual kernel size (mg) — used in GPP calculation.
        phint: Phyllochron interval (°C d per leaf).
        tbase: Base temperature for development (°C).
        topt: Optimum temperature for vegetative development (°C).
        ropt: Optimum temperature for reproductive development (°C).
        djti: Thermal time base for juvenile period (°C d) — ecotype.
        gdde: Growing degree-days per cm emergence depth (°C d cm⁻¹) — ecotype.
        dget: Maximum GDD allowed for emergence (°C d) — species.
        swcg: Minimum soil water for germination (cm³ cm⁻³) — species.
        rue: Radiation use efficiency (g MJ⁻¹ PAR) — ecotype.
        kcan: Canopy light extinction coefficient for PAR — ecotype.
    """

    varno: str = "LCSA94"
    vrname: str = "LCSA9401 Millet"
    econo: str = "DFAULT"

    # Cultivar coefficients
    p1: float = 350.0
    p2o: float = 12.5
    p2r: float = 100.0
    p5: float = 400.0
    g4: float = 0.9       # G4 in MLCER048.CUL
    g5: float = 8.0       # G5 in MLCER048.CUL — kernel size (mg)
    phint: float = 38.9

    # Ecotype / species coefficients
    tbase: float = 8.0
    topt: float = 33.0
    ropt: float = 28.0
    djti: float = 68.0    # Base thermal time for photoperiod stage (°C d)
    gdde: float = 6.5     # Not used in P9 calc (6.0 is hardcoded in PHASEI)
    dget: float = 150.0   # Max GDD for emergence failure check
    swcg: float = 0.02    # Min soil water for germination
    rue: float = 3.2
    kcan: float = 0.85

    @classmethod
    def from_file(
        cls,
        varno: str,
        *,
        model_code: str = "MLCER048",
        genotype_dir=None,
    ) -> "MilletCultivar":
        """Load cultivar from MLCER048.CUL / .ECO / .SPE files.

        Parameters
        ----------
        varno:
            Cultivar identifier, e.g. ``"990001"``.
        model_code:
            DSSAT model prefix (default ``"MLCER048"``).
        genotype_dir:
            Path to directory with genotype files; auto-detected if None.

        Returns
        -------
        MilletCultivar
            Populated from file data.
        """
        from dssat.crop.ceres_millet.fileio import load_ceres_millet_params
        p = load_ceres_millet_params(varno, model_code=model_code,
                                     genotype_dir=genotype_dir)
        cv = cls(
            varno=p.cul.varno,
            vrname=p.cul.vrname,
            econo=p.cul.econo,
            p1=p.cul.p1,
            p2o=p.cul.p2o,
            p2r=p.cul.p2r,
            p5=p.cul.p5,
            g4=p.cul.g4,
            g5=p.cul.g5,
            phint=p.cul.phint,
            tbase=p.eco.tbase,
            topt=p.eco.topt,
            ropt=p.eco.ropt,
            djti=p.eco.djti,
            gdde=p.eco.gdde,
            dget=getattr(p.spe, "dget", 150.0),
            swcg=getattr(p.spe, "swcg", 0.02),
            rue=p.eco.rue,
            kcan=p.eco.kcan,
        )
        cv._params = p
        return cv


@dataclass
class PhenologyState:
    """Mutable state for the CERES-Millet phenology sub-model.

    Attributes:
        istage: Current growth stage code (7, 8, 9, 1–6).
        xstage: Continuous growth stage (decimal).
        cumdtt: Cumulative thermal time since germination (°C d).
        sumdtt: Cumulative thermal time within current stage (°C d).
        dtt: Daily thermal time (°C d d⁻¹).
        stgdoy: Day-of-year at which each stage was reached, indexed
            0–19 (YYYYDDD).
        mdate: Maturity date (YYYYDDD); -99 until reached.
        isdate: Anthesis date (YYYYDDD).
        yremrg: Emergence date (YYYYDDD).
        gpp: Grain-bearing panicles per plant.
        cumph: Cumulative phyllochron units (leaf tip appearances).
        leafno: Number of fully expanded leaves.
        tlno: Total leaf number.
        kep: Light extinction coefficient.
        crop_status: Crop status flag (0=growing, 1=mature, 12=failed).
        l0: Soil layer index containing the seed.
        ndas: Days since sowing (integer counter).
        swsd: Soil water sum at seed layer.
        sind: Photoperiod-sensitive stage accumulator (dimensionless).
        p9: Emergence thermal time threshold (°C d).
        p3_dyn: Dynamically computed P3 (end leaf growth TT) (°C d).
        p4_dyn: Dynamically set P4 = 150 (end panicle growth TT) (°C d).
        idur1: Day counter used for GPP calculation.
        ism: Flag for maturity check in stage 5.
        emat: Counter for maturity confirmation in stage 5.
        xnti: Leaf number at panicle initiation.
        vegfrac: Vegetative N fraction.
        seedfrac: Seed N fraction.
        iprint: Print flag (used to gate ISDATE setting in stage 4).
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
    tlno: float = 30.0
    kep: float = 0.7
    crop_status: int = 0
    l0: int = 0
    ndas: int = 0
    swsd: float = 0.0
    # Stage-2 base TT threshold (= cv.djti; accessible by tests as state.p2)
    p2: float = 0.0
    # Stage-2 SIND accumulator
    sind: float = 0.0
    # Emergence TT threshold (computed at stage 8→9 transition)
    p9: float = 0.0
    # Dynamically computed stage durations (set at 2→3 transition)
    p3_dyn: float = 0.0   # P3 = 370 + 0.135 * SUMDTT at transition
    p4_dyn: float = 150.0  # P4 = 150.0 (hardcoded in PHASEI)
    # Counters
    idur1: int = 0
    ism: int = 0           # Stage 5 maturity flag
    emat: float = 0.0      # Stage 5 maturity counter
    xnti: float = 0.0      # Leaf number at panicle initiation
    vegfrac: float = 0.0
    seedfrac: float = 0.0
    iprint: int = 0        # Gate for ISDATE setting in stage 4


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

    Matches the CTYPE=1 (C4 cereals) algorithm in ML_PHENOL.for.

    Args:
        tmax: Daily maximum temperature (°C).
        tmin: Daily minimum temperature (°C).
        srad: Solar radiation (MJ m⁻² d⁻¹).
        dayl: Day length (h).
        snow: Snow depth (mm water equivalent).
        leafno: Current number of expanded leaves.
        istage: Current CERES growth stage code.
        tbase: Base temperature (°C).
        topt: Optimum temperature for vegetative development (°C).
        ropt: Optimum temperature for reproductive development (°C).

    Returns:
        Daily thermal time (°C d).
    """
    # DOPT switches to ROPT after anthesis (stages 4–6), matching ML_PHENOL:269
    dopt = ropt if (3 < istage <= 6) else topt

    xs = min(snow, 15.0)

    # Crown temperature adjustments for frozen conditions (ML_PHENOL:257-261)
    tempcn = tmin
    tempcx = tmax
    if tmin < 0.0:
        tempcn = 2.0 + tmin * (0.4 + 0.0018 * (xs - 15.0) ** 2)
    if tmax < 0.0:
        tempcx = 2.0 + tmax * (0.4 + 0.0018 * (xs - 15.0) ** 2)

    # Early exit if TMAX below base (ML_PHENOL:276)
    if tmax < tbase:
        return 0.0

    # All temps above DOPT: cap at DOPT - TBASE (ML_PHENOL:279)
    if tmin > dopt:
        return dopt - tbase

    # C4 cereal soil temperature method (CTYPE=1, LEAFNO <= 10) (ML_PHENOL:288)
    if leafno <= 10:
        if xs > 0.0:
            # Snow on ground: simple average (ML_PHENOL:298)
            dtt = (tempcn + tempcx) / 2.0 - tbase
        else:
            # No snow: compute soil temperature (ML_PHENOL:303-326)
            acoef = 0.01061 * srad + 0.5902
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
        # Hourly integration when Tmin or Tmax out of range (ML_PHENOL:332-343)
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


class MilletPhenology:
    """CERES-Millet phenology sub-model.

    Determines the current growth stage and accumulates thermal time.
    Called once per day with the current weather and soil state.

    Implements both ML_PHENOL.for and ML_PHASEI.for logic.

    Args:
        cultivar: Cultivar and ecotype coefficients.
        pltpop: Plant population (plants m⁻²).
        sdepth: Sowing depth (cm).
        yrsim: Simulation start date (YYYYDDD).
    """

    def __init__(
        self,
        cultivar: MilletCultivar,
        pltpop: float,
        sdepth: float,
        yrsim: int,
    ) -> None:
        self.cultivar = cultivar
        self.pltpop = pltpop
        self.sdepth = sdepth
        self.state = PhenologyState()
        self.state.stgdoy[13] = yrsim  # STGDOY(14) in Fortran (1-indexed)

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
            dynamic: DSSAT dynamic switch (RUNINIT, SEASINIT, RATE, INTEGR).
            yrdoy: Current date (YYYYDDD).
            tmax: Daily maximum temperature (°C).
            tmin: Daily minimum temperature (°C).
            srad: Solar radiation (MJ m⁻² d⁻¹).
            dayl: Day length (h).
            snow: Snow depth (mm water equivalent).
            sw: Soil water content by layer (cm³ cm⁻³).
            ll: Lower limit of plant-available water by layer (cm³ cm⁻³).
            dlayr: Soil layer thickness by layer (cm).
            nlayr: Number of active soil layers.
            iswwat: Water simulation switch ('Y' or 'N').
            twilen: Twilight day length (h); defaults to dayl if not given.
        """
        if dynamic in (RUNINIT, SEASINIT):
            self._init(yrdoy)
        else:
            if twilen < 0.0:
                twilen = dayl
            self._rate_integr(
                yrdoy, tmax, tmin, srad, dayl, snow, sw, ll, dlayr, nlayr,
                iswwat, twilen,
            )

    # ------------------------------------------------------------------
    def _init(self, yrsim: int) -> None:
        """Reset state for a new season."""
        s = self.state
        cv = self.cultivar
        s.istage = 7
        s.xstage = 0.1
        s.cumdtt = 0.0
        s.sumdtt = 0.0
        s.dtt = 0.0
        s.stgdoy[:] = 9999999
        s.stgdoy[13] = yrsim
        s.mdate = -99
        s.isdate = 0
        s.yremrg = -99
        s.gpp = 0.0
        s.cumph = 0.0
        s.leafno = 0
        s.tlno = 30.0
        s.kep = cv.kcan
        s.crop_status = 0
        s.l0 = 0
        s.ndas = 0
        s.swsd = 0.0
        s.p2 = cv.djti    # Base TT for photoperiod-sensitive stage (ECO DJTI)
        s.sind = 0.0
        s.p9 = 0.0
        s.p3_dyn = 0.0
        s.p4_dyn = 150.0
        s.idur1 = 0
        s.ism = 0
        s.emat = 0.0
        s.xnti = 0.0
        s.vegfrac = 0.0
        s.seedfrac = 0.0
        s.iprint = 0

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
        """Daily rate and integration for phenology (ML_PHENOL main body)."""
        s = self.state
        cv = self.cultivar

        if s.crop_status == 1 and s.istage not in (6, 7):
            return

        # Compute DTT (ML_PHENOL:246-346)
        s.dtt = compute_dtt(
            tmax, tmin, srad, dayl, snow,
            s.leafno, s.istage,
            cv.tbase, cv.topt, cv.ropt,
        )

        # Accumulate thermal time SUMDTT and CUMDTT (ML_PHENOL:348-349)
        s.sumdtt += s.dtt
        s.cumdtt += s.dtt

        # Run stage-specific logic (mirrors the IF/ELSEIF chain in ML_PHENOL)
        if s.istage == 7:
            self._stage7(yrdoy, dlayr, nlayr, iswwat)
        elif s.istage == 8:
            self._stage8(yrdoy, sw, ll, iswwat)
        elif s.istage == 9:
            self._stage9(yrdoy)
        elif s.istage == 1:
            self._stage1(yrdoy)
        elif s.istage == 2:
            self._stage2(yrdoy, twilen)
        elif s.istage == 3:
            self._stage3(yrdoy)
        elif s.istage == 4:
            self._stage4(yrdoy)
        elif s.istage == 5:
            self._stage5(yrdoy)
        elif s.istage == 6:
            self._stage6(yrdoy)

    # ==================== Stage handlers ==================================

    def _stage7(
        self,
        yrdoy: int,
        dlayr: np.ndarray,
        nlayr: int,
        iswwat: str,
    ) -> None:
        """ISTAGE = 7: Sowing date (ML_PHENOL:377-410 + PHASEI:325-329)."""
        s = self.state
        # Record sowing day (1-indexed: STGDOY(7))
        s.stgdoy[6] = yrdoy
        s.ndas = 0

        # ML_PHASEI ISTAGE=7: advance to 8, set rtdep, reset SUMDTT
        # (PHASEI:325-329)
        s.istage = 8
        s.sumdtt = 0.0
        # rtdep handled by root module — initialised to sdepth by coordinator

        if iswwat == "N":
            return

        # Find seed layer (ML_PHENOL:401-410)
        cumdep = 0.0
        for L in range(nlayr):
            cumdep += dlayr[L]
            if self.sdepth < cumdep:
                s.l0 = L
                break

    def _stage8(
        self,
        yrdoy: int,
        sw: np.ndarray,
        ll: np.ndarray,
        iswwat: str,
    ) -> None:
        """ISTAGE = 8: Germination (ML_PHENOL:415-444 + PHASEI:334-341)."""
        s = self.state
        cv = self.cultivar

        # Check soil water (ML_PHENOL:416-423)
        if iswwat != "N":
            l0 = s.l0
            # Only check / increment NDAS when SW <= LL (ML_PHENOL:417)
            if sw[l0] <= ll[l0]:
                l1 = min(l0 + 1, len(sw) - 1)
                s.swsd = (sw[l0] - ll[l0]) * 0.65 + (sw[l1] - ll[l1]) * 0.35
                s.ndas += 1
                if s.swsd < cv.swcg:
                    return  # Not enough water yet

        # Germinated — record day (ML_PHENOL:425)
        s.stgdoy[7] = yrdoy  # STGDOY(8) 1-indexed

        # ML_PHASEI ISTAGE=8 → 9 (PHASEI:334-341)
        s.istage = 9
        s.cumdtt = 0.0
        s.sumdtt = 0.0
        # P9 = 45.0 + 6.0 * SDEPTH (hardcoded coefficient in PHASEI:340)
        s.p9 = 45.0 + 6.0 * self.sdepth

    def _stage9(self, yrdoy: int) -> None:
        """ISTAGE = 9: Emergence (ML_PHENOL:450-491 + PHASEI:346-402)."""
        s = self.state
        cv = self.cultivar

        # Increment day counter (ML_PHENOL:451)
        s.ndas += 1
        s.idur1 += 1

        if s.sumdtt < s.p9:
            return  # Not yet emerged

        # Failure check: P9 > DGET means too deep to emerge (ML_PHENOL:456-472)
        if s.p9 > cv.dget:
            s.istage = 6
            self.pltpop = 0.0
            s.gpp = 1.0
            s.mdate = yrdoy
            s.crop_status = 12
            return

        # Record emergence (ML_PHENOL:474)
        s.stgdoy[8] = yrdoy  # STGDOY(9)
        s.yremrg = yrdoy

        # ML_PHASEI ISTAGE=9 → 1 (PHASEI:346-402)
        s.istage = 1
        # Carryover: excess TT over P9 carries into stage 1 (PHASEI:348)
        s.sumdtt = s.sumdtt - s.p9
        s.cumdtt = s.cumdtt - s.p9
        s.tlno = 30.0
        s.leafno = 1
        # Initialise CUMPH from carryover TT (PHASEI:392)
        s.cumph = s.sumdtt / max(cv.phint, 0.001)

    def _stage1(self, yrdoy: int) -> None:
        """ISTAGE = 1: End of juvenile phase (ML_PHENOL:496-501 + PHASEI:193-200)."""
        s = self.state
        cv = self.cultivar

        s.ndas += 1
        s.idur1 += 1
        # Continuous XSTAGE (ML_PHENOL:498)
        s.xstage = 2.0 * s.sumdtt / cv.p1

        if s.sumdtt < cv.p1:
            return  # Still in juvenile phase

        s.stgdoy[0] = yrdoy  # STGDOY(1)

        # ML_PHASEI ISTAGE=1 → 2 (PHASEI:193-200)
        # SUMDTT is NOT reset here (unlike maize); only SIND is zeroed
        s.istage = 2
        s.sind = 0.0
        # (BIOMS1, BIOMS2 etc. are growth state — not tracked in phenology)

    def _stage2(self, yrdoy: int, twilen: float) -> None:
        """ISTAGE = 2: Panicle initiation, photoperiod-sensitive
        (ML_PHENOL:507-532 + PHASEI:204-213)."""
        s = self.state
        cv = self.cultivar

        s.ndas += 1
        s.idur1 += 1

        # XSTAGE (ML_PHENOL:509)
        s.xstage = 2.0 + s.sind

        # PDTT: on first day use carryover from P1 transition (ML_PHENOL:518-519)
        # ICSDUR==1 in Fortran corresponds to the day SUMDTT still holds P1 carryover
        # We track this by checking if SIND is still 0.0 (first call in stage 2)
        pdtt = s.dtt
        if s.sind == 0.0:
            # First entry into stage 2: PDTT = carryover from P1 (ML_PHENOL:518)
            pdtt = max(s.sumdtt - cv.p1, 0.0)

        # Photoperiod sensitivity (ML_PHENOL:522-528)
        if twilen > cv.p2o:
            ratein = 1.0 / (cv.djti + cv.p2r * (twilen - cv.p2o))
        else:
            ratein = 1.0 / cv.djti

        # Accumulate SIND (ML_PHENOL:530)
        s.sind += ratein * pdtt

        if s.sind < 1.0:
            return  # Not yet at panicle initiation

        s.stgdoy[1] = yrdoy  # STGDOY(2)

        # ML_PHASEI ISTAGE=2 → 3 (PHASEI:204-213)
        s.istage = 3
        s.tlno = s.cumdtt / 35.0 + 6.0
        s.xnti = s.sumdtt / 43.0
        # Dynamic P3 and P4 computed from current accumulated TT (PHASEI:209-210)
        s.p3_dyn = 370.0 + 0.135 * s.sumdtt
        s.p4_dyn = 150.0
        # SUMDTT = DTT (not 0!), carries partial day into stage 3 (PHASEI:211)
        s.sumdtt = s.dtt

    def _stage3(self, yrdoy: int) -> None:
        """ISTAGE = 3: End of leaf growth (ML_PHENOL:537-553 + PHASEI:219-233)."""
        s = self.state

        s.ndas += 1
        s.idur1 += 1

        p3 = s.p3_dyn  # Dynamic P3 set at 2→3 transition
        s.xstage = 3.0 + 2.0 * s.sumdtt / max(p3, 0.001)

        if s.sumdtt < p3:
            return

        s.stgdoy[2] = yrdoy  # STGDOY(3)

        # ML_PHASEI ISTAGE=3 → 4 (PHASEI:219-233)
        s.istage = 4
        # Carryover (PHASEI:231)
        s.sumdtt = s.sumdtt - p3

    def _stage4(self, yrdoy: int) -> None:
        """ISTAGE = 4: End of panicle growth (ML_PHENOL:555-598 + PHASEI:236-305).

        PFLOWR = 50.0 is hardcoded — anthesis occurs when SUMDTT >= 50.
        P4 = 150.0 was set dynamically at 2→3 transition.
        """
        s = self.state
        cv = self.cultivar

        s.ndas += 1

        pflowr = 50.0  # Hardcoded in ML_PHENOL:557

        s.xstage = 5.0 + s.sumdtt / pflowr

        if s.sumdtt <= pflowr:
            s.idur1 += 1

        # ISDATE: set once when SUMDTT crosses PFLOWR (ML_PHENOL:583-589)
        if s.sumdtt >= pflowr and s.iprint == 0:
            s.stgdoy[3] = yrdoy   # STGDOY(4)
            s.isdate = yrdoy
            s.iprint = 1

        # After P4 passed, update XSTAGE and transition (ML_PHENOL:591-598)
        if s.sumdtt < s.p4_dyn:
            return

        # STGDOY(16) = YRDOY — note: 1-indexed 16 → 0-indexed 15
        s.stgdoy[15] = yrdoy
        s.xstage = 6.0 + 0.5 * (s.sumdtt - 50.0) / max(s.p4_dyn - 50.0, 0.001)

        if s.gpp <= 0.0:
            s.gpp = 1.0

        # ML_PHASEI ISTAGE=4 → 5 (PHASEI:236-305)
        # GPP is computed here using growth-state biomass — simplified version
        # (full formula requires LFWT, STMWT, BIOMS2/IDUR1 from growth module)
        s.istage = 5
        # Carryover (PHASEI:303)
        s.sumdtt = s.sumdtt - s.p4_dyn
        s.emat = 0.0
        s.ism = 0

    def _stage5(self, yrdoy: int) -> None:
        """ISTAGE = 5: Grain fill (ML_PHENOL:605-623 + PHASEI:310-312)."""
        s = self.state
        cv = self.cultivar

        s.ndas += 1

        # Continuous XSTAGE (ML_PHENOL:611)
        s.xstage = 6.5 + 2.5 * s.sumdtt / max(cv.p5, 0.001)

        if s.sumdtt < cv.p5:
            return

        # ISM maturity mechanism (ML_PHENOL:613-624)
        if s.ism > 0:
            # Already triggered — wait for P5 + 90 and EMAT >= 2
            if s.sumdtt < cv.p5 + 90.0:
                return
            s.emat += 1
            if s.emat < 2:
                return
            # Confirmed maturity
            s.stgdoy[4] = yrdoy  # STGDOY(5)
        else:
            # First time SUMDTT >= P5: set ISM flag (ML_PHENOL:620-623)
            s.ism = 1
            s.stgdoy[16] = yrdoy  # STGDOY(17)
            return

        # ML_PHASEI ISTAGE=5 → 6 (PHASEI:310-312)
        s.istage = 6

    def _stage6(self, yrdoy: int) -> None:
        """ISTAGE = 6: Physiological maturity (ML_PHENOL:629-668)."""
        s = self.state

        # Requires DTT >= 2 and SUMDTT >= 2 (ML_PHENOL:630-631)
        if s.dtt < 2.0:
            return
        if s.sumdtt < 2.0:
            return

        s.stgdoy[5] = yrdoy  # STGDOY(6)
        s.mdate = yrdoy
        s.crop_status = 1
