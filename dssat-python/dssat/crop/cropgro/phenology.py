"""CROPGRO phenology — R stages, V stages, and photoperiod.

Translated from ``PHENOL.for``, ``RSTAGES.for``, and ``VSTAGES.for`` in the
DSSAT-CSM CROPGRO module.

CROPGRO uses 13 phenological phases accumulated as photothermal days:

    phzacc[j] += ft[j] * fuday[j] * fsw[j] * fnstr[j]

where:
- ``ft[j]``    = piecewise-linear temperature response (0–1)
- ``fuday[j]`` = photoperiod factor (short-day or long-day)
- ``fsw[j]``   = soil-water modulation factor
- ``fnstr[j]`` = N-stress modulation factor

Key R-stage thresholds are derived from the cultivar coefficients:
- R1 (flowering): phzacc >= phthrs[3]  (phases 2+3 = EM_FL equivalent)
- R3 (pod set):   phzacc >= phthrs[5]
- R5 (seed fill): phzacc >= phthrs[7]
- R7 (maturity):  phzacc >= phthrs[9]

References:
    Boote, K.J., Jones, J.W., Hoogenboom, G., & Pickering, N.B. (1998).
    The CROPGRO model for grain legumes. Understanding Options for Agricultural
    Production, 99–128.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dssat.core.constants import NL, RUNINIT, SEASINIT, RATE, INTEGR


# ---------------------------------------------------------------------------
# Utility curve functions
# ---------------------------------------------------------------------------

def curv_tp(tb: float, to1: float, to2: float, tm: float, x: float) -> float:
    """Piecewise-linear temperature response, returns 0–1.

    Implements ``CURV('TP', tb, to1, to2, tm, x)`` from the Fortran CURV
    function.

    Args:
        tb:  Base temperature (°C) — response = 0 below this.
        to1: Lower optimum temperature (°C) — response reaches 1.0 here.
        to2: Upper optimum temperature (°C) — response stays 1.0 to here.
        tm:  Maximum temperature (°C) — response = 0 above this.
        x:   Query temperature (°C).

    Returns:
        Response factor (0–1).
    """
    if x <= tb or x >= tm:
        return 0.0
    if x <= to1:
        return (x - tb) / (to1 - tb)
    if x >= to2:
        return (tm - x) / (tm - to2)
    return 1.0


def curv_lin(c1: float, c2: float, c3: float, c4: float, x: float) -> float:
    """Piecewise-linear response from (c2, c1) to (c3, c4).

    Implements ``CURV('LIN', c1, c2, c3, c4, x)``.

    Args:
        c1: Y value at x <= c2.
        c2: X breakpoint where rise starts.
        c3: X breakpoint where rise ends.
        c4: Y value at x >= c3.
        x:  Query value.

    Returns:
        Interpolated response.
    """
    if x <= c2:
        return c1
    if x >= c3:
        return c4
    return c1 + (c4 - c1) * (x - c2) / (c3 - c2)


# ---------------------------------------------------------------------------
# State dataclass
# ---------------------------------------------------------------------------

@dataclass
class CropGROPhenoState:
    """Mutable phenology state for the CROPGRO model.

    Attributes:
        rstage:     Current R stage (0=pre-emergence, 1–7 = R1–R7).
        vstage:     Continuous V stage (node/leaf number).
        phzacc:     Phase accumulator array, shape (14,).  phzacc[j] counts
                    photothermal days towards the threshold for phase j.
        phthrs:     Threshold array for each phase (photothermal days).
        stgdoy:     DOY at which each stage was reached; -99 until triggered.
        mdate:      Maturity date (YYYYDDD); -99 until reached.
        r1_date:    Flowering date (YYYYDDD).
        r3_date:    Pod-set date (YYYYDDD).
        r5_date:    Seed-fill date (YYYYDDD).
        r7_date:    Physiological maturity date (YYYYDDD).
        yremrg:     Emergence date (YYYYDDD).
        dtx:        Vegetative thermal-time (°C d) today (drives VSTAGE).
        drpp:       Daily photothermal day accumulation (combined).
        dxr57:      Fraction of the R5→R7 period elapsed (0–1).
        xpod:       Pod partitioning factor.
        fracdn:     Fraction of day above critical photoperiod.
        crop_status: Status flag (0=normal, 1=mature).
    """

    rstage: int = 0
    vstage: float = 0.0
    phzacc: np.ndarray = field(default_factory=lambda: np.zeros(14))
    phthrs: np.ndarray = field(default_factory=lambda: np.zeros(14))
    stgdoy: np.ndarray = field(default_factory=lambda: np.full(14, -99, dtype=int))
    mdate: int = -99
    r1_date: int = -99
    r3_date: int = -99
    r5_date: int = -99
    r7_date: int = -99
    yremrg: int = -99
    dtx: float = 0.0
    drpp: float = 0.0
    dxr57: float = 0.0
    xpod: float = 0.0
    fracdn: float = 0.0
    crop_status: int = 0
    # Germination and emergence tracking
    ndas: float = 0.0          # Days after sowing
    germinated: bool = False
    emerged: bool = False
    emerg_tt: float = 0.0      # Thermal time accumulated for emergence
    emerg_tt_req: float = 0.0  # Required TT for emergence


# ---------------------------------------------------------------------------
# Phenology sub-model
# ---------------------------------------------------------------------------

class CropGROPhenology:
    """CROPGRO phenology sub-model.

    Drives stage transitions and computes the daily photothermal day rate
    for all 13 phases of CROPGRO development.

    Args:
        cultivar:  Cultivar dataclass (SoybeanCultivar or CanolaCultivar).
        pltpop:    Plant population (plants m^-2).
        sdepth:    Sowing depth (cm).
        yrsim:     Simulation start date (YYYYDDD).
        short_day: If True the crop is short-day (soybean); if False it is
                   long-day (canola).
    """

    # Number of phenological phases tracked
    NPHS: int = 13

    def __init__(
        self,
        cultivar: Any,
        pltpop: float,
        sdepth: float,
        yrsim: int,
        short_day: bool = True,
    ) -> None:
        self.cultivar = cultivar
        self.pltpop = pltpop
        self.sdepth = sdepth
        self.yrsim = yrsim
        self.short_day = short_day
        self.state = CropGROPhenoState()
        self._build_phthrs()

    # ------------------------------------------------------------------
    def _build_phthrs(self) -> None:
        """Build the PHTHRS threshold array from cultivar/ecotype coefficients.

        Mirrors IPPHENOL.for (Ipphenol.for lines 235-237) and the ECO file
        mapping in fileio.py CropGROParams._derive().

        Fortran PHTHRS indices (1-based) → Python (0-based):
            PHTHRS(1)  = PL_EM  : planting → emergence
            PHTHRS(2)  = EM_V1  : emergence → V1
            PHTHRS(3)  = V1_JU  : V1 → end of juvenile phase
            PHTHRS(4)  = JU_R0  : juvenile → R0 (floral induction end)
            PHTHRS(5)  = MAX(0, EM_FL - PHTHRS(3) - PHTHRS(4)) : R0 → R1
            PHTHRS(6)  = FL_SH  : R1 → R3 (first pod/shell)
            PHTHRS(7)  = FL_SH + MAX(0,(FL_SD-FL_SH)*PM06) : R3 → R3.5
            PHTHRS(8)  = FL_SD  : R1 → R5 (first seed)
            PHTHRS(9)  = MAX(0, SD_PM * PM09) : R5 → NDSET
            PHTHRS(10) = SD_PM  : R5 → R7 (physiological maturity)
            PHTHRS(11) = R7_R8  : R7 → R8 (harvest maturity)
            PHTHRS(12) = FL_VS  : R1 → end of vegetative growth
            PHTHRS(13) = FL_LF  : R1 → last leaf appearance
        """
        cv = self.cultivar
        s = self.state

        # Ecotype parameters (with safe defaults matching DSSAT soybean defaults)
        pl_em  = getattr(cv, "pl_em",  2.2)    # PL-EM from ECO
        em_v1  = getattr(cv, "em_v1",  6.0)    # EM-V1 from ECO
        v1_ju  = getattr(cv, "v1_ju",  0.0)    # V1-JU from ECO
        ju_r0  = getattr(cv, "ju_r0",  5.0)    # JU-R0 from ECO
        pm06   = getattr(cv, "pm06",   0.0)    # PM06 from ECO
        pm09   = getattr(cv, "pm09",   0.35)   # PM09 from ECO
        r7_r8  = getattr(cv, "r7_r8",  12.0)   # R7-R8 from ECO
        fl_vs  = getattr(cv, "fl_vs",  26.0)   # FL-VS from ECO

        # Cultivar parameters
        em_fl = cv.em_fl
        fl_sh = cv.fl_sh
        fl_sd = cv.fl_sd
        sd_pm = cv.sd_pm
        fl_lf = cv.fl_lf

        # Derived thresholds (Ipphenol.for lines 235-237)
        ph5 = max(0.0, em_fl - v1_ju - ju_r0)      # R0 → R1
        ph7 = fl_sh + max(0.0, (fl_sd - fl_sh) * pm06)  # R3 → R3.5
        ph9 = max(0.0, sd_pm * pm09)                # R5 → NDSET

        # Store in array (0-indexed = Fortran 1-indexed - 1)
        s.phthrs[0]  = pl_em   # PHTHRS(1): PL → emergence
        s.phthrs[1]  = em_v1   # PHTHRS(2): EM → V1
        s.phthrs[2]  = v1_ju   # PHTHRS(3): V1 → juvenile end
        s.phthrs[3]  = ju_r0   # PHTHRS(4): juvenile → R0 (floral induction)
        s.phthrs[4]  = ph5     # PHTHRS(5): R0 → R1 (flowering)
        s.phthrs[5]  = fl_sh   # PHTHRS(6): R1 → R3 (first pod)
        s.phthrs[6]  = ph7     # PHTHRS(7): R3 → intermediate
        s.phthrs[7]  = fl_sd   # PHTHRS(8): R1 → R5 (first seed)
        s.phthrs[8]  = ph9     # PHTHRS(9): R5 → NDSET
        s.phthrs[9]  = sd_pm   # PHTHRS(10): R5 → R7 (maturity)
        s.phthrs[10] = r7_r8   # PHTHRS(11): R7 → R8 (harvest maturity)
        s.phthrs[11] = fl_vs   # PHTHRS(12): R1 → end veg growth
        s.phthrs[12] = fl_lf   # PHTHRS(13): R1 → last leaf
        for j in range(13, 14):
            s.phthrs[j] = 999.0

    # ------------------------------------------------------------------
    def run(
        self,
        dynamic: int,
        yrdoy: int,
        tmax: float,
        tmin: float,
        dayl: float,
        sw: np.ndarray,
        ll: np.ndarray,
        dlayr: np.ndarray,
        nlayr: int,
        swfac: float,
        nstres: float,
        iswwat: str,
    ) -> None:
        """Run one phenology time step.

        Args:
            dynamic:  Simulation phase code.
            yrdoy:    Current date (YYYYDDD).
            tmax:     Daily max temperature (°C).
            tmin:     Daily min temperature (°C).
            dayl:     Day length (h).
            sw:       Soil water by layer (cm^3 cm^-3).
            ll:       Lower limit by layer.
            dlayr:    Layer thickness (cm).
            nlayr:    Number of active layers.
            swfac:    Photosynthesis water stress factor (0–1).
            nstres:   N stress factor (0–1).
            iswwat:   Water switch ("Y"/"N").
        """
        if dynamic in (RUNINIT, SEASINIT):
            self._init(yrdoy)
        elif dynamic == INTEGR:
            # Phenological accumulation runs once per day (INTEGR phase only)
            # to avoid double-counting when both RATE and INTEGR are called.
            self._rate_integr(
                yrdoy, tmax, tmin, dayl, sw, ll, dlayr, nlayr,
                swfac, nstres, iswwat,
            )
        # else RATE: stage checks happen in INTEGR; nothing to do here
        # (soybean.run() reads pheno.emerged after phenology runs, so we
        # propagate the rate-phase query by caching the value set in INTEGR)

    # ------------------------------------------------------------------
    def _init(self, yrsim: int) -> None:
        """Reset phenology state for a new season."""
        s = self.state
        cv = self.cultivar
        s.rstage = 0
        s.vstage = 0.0
        s.phzacc[:] = 0.0
        s.stgdoy[:] = -99
        s.mdate = -99
        s.r1_date = -99
        s.r3_date = -99
        s.r5_date = -99
        s.r7_date = -99
        s.yremrg = -99
        s.dtx = 0.0
        s.drpp = 0.0
        s.dxr57 = 0.0
        s.xpod = 0.0
        s.fracdn = 0.0
        s.crop_status = 0
        s.ndas = 0.0
        s.germinated = False
        s.emerged = False
        s.emerg_tt = 0.0
        # Build phthrs first so phthrs[0] = PL_EM is available
        self._build_phthrs()
        # Emergence accumulator threshold: PHTHRS(1) + SDEPTH * 0.6
        # Mirrors PHENOL/RSTAGES: PHTEM = PHTHRS(1) + SDEPTH * 0.6
        # PHTHRS(1) = PL_EM from ecotype (default 2.2 photothermal days).
        # RStages.for line 203.
        s.emerg_tt_req = s.phthrs[0] + self.sdepth * 0.6

    # ------------------------------------------------------------------
    def _rate_integr(
        self,
        yrdoy: int,
        tmax: float,
        tmin: float,
        dayl: float,
        sw: np.ndarray,
        ll: np.ndarray,
        dlayr: np.ndarray,
        nlayr: int,
        swfac: float,
        nstres: float,
        iswwat: str,
    ) -> None:
        """Daily rate and integration of phenology.

        Mirrors PHENOL.for RATE + INTEGR blocks and the accumulation
        logic in RStages.for.
        """
        s = self.state
        cv = self.cultivar

        if s.mdate > 0 and s.mdate <= yrdoy:
            return

        s.ndas += 1.0
        tempm = (tmax + tmin) * 0.5

        # -- temperature factor (vegetative, using tbase/topt1/topt2/tmax)
        # Corresponds to FT(2) = veg temperature function (TSELC=1 for phases 2-5)
        # PHENOL.for lines 233-250; uses hourly temps, but we approximate with tmean
        tb  = cv.tbase
        to1 = cv.topt1
        to2 = cv.topt2
        tm  = cv.tmax

        ft_veg = curv_tp(tb, to1, to2, tm, tempm)
        s.dtx = ft_veg  # vegetative thermal factor (drives VSTAGE); PHENOL.for line 280

        # -- germination phase (phase 1 in Fortran: sowing → emergence)
        # DLTYP(1) = 'NON' — no photoperiod effect; FUDAY(1) = 1.0
        # RStages.for lines 202-223
        if not s.germinated:
            s.phzacc[0] += ft_veg * 1.0  # FUDAY=1 for emergence phase
            if s.phzacc[0] >= s.emerg_tt_req:
                s.germinated = True
                s.emerged = True
                s.yremrg = yrdoy
                s.stgdoy[0] = yrdoy
                s.stgdoy[1] = yrdoy
                s.rstage = 0  # vegetative
            return

        # -- Minimum-temperature modifier ZMODTE for pre-R1 phases
        # Applies to phases 4 and 5 (V1→end-juvenile and juv→R0→R1)
        # PHENOL.for lines 266-271:
        #   IF (TMIN < OPTBI): ZMODTE = MAX(0, MIN(1, 1 - SLOBI*(OPTBI-TMIN)))
        optbi = getattr(cv, "optbi", 18.0)
        slobi = getattr(cv, "slobi", 0.028)
        zmodte = 1.0
        if tmin < optbi:
            zmodte = max(0.0, min(1.0, 1.0 - slobi * (optbi - tmin)))

        # -- photoperiod factor
        # Pre-R1: use CSDVAR/CLDVAR; post-R1: use CSDVRR/CLDVRR
        # PHENOL.for lines 243-245
        after_r1 = (s.rstage >= 1)
        fuday = self._photoperiod_factor(dayl, after_r1=after_r1)

        # -- soil water modulation
        # WSENP values from SPE: most phases 0, so FSW=1 by default
        # Simplified: phases 3-8 wsenp=-0.4 → FSW = 1+(1-swfac)*(-0.4)
        # but for simplicity we pass no SW effect here (safe default)
        fsw = 1.0
        fnstr = 1.0

        # PHTHRS index mapping (0-based = Fortran 1-based − 1):
        # [0]=PL_EM  [1]=EM_V1  [2]=V1_JU  [3]=JU_R0  [4]=R0→R1
        # [5]=FL_SH  [6]=FL_SH+  [7]=FL_SD  [8]=R5→NDSET  [9]=SD_PM
        # [10]=R7_R8  [11]=FL_VS  [12]=FL_LF

        # -- phases 2-4: EM → V1 → JuvenileEnd → R0 (pre-flowering accumulators)
        # DLTYP='NON' → FUDAY=1.0 for phases 2 & 3; DLTYP='INL' for phases 4 & 5
        # RStages.for lines 233-296
        if s.rstage == 0:
            # Phase 2: EM → V1 (NON photoperiod)
            if s.phzacc[1] < s.phthrs[1]:
                s.phzacc[1] += ft_veg * 1.0
                if s.phzacc[1] >= s.phthrs[1]:
                    s.stgdoy[1] = yrdoy
            # Phase 3: V1 → JuvenileEnd (NON photoperiod; PHTHRS[2] can be 0)
            if s.phzacc[2] < s.phthrs[2] + 1e-6:
                s.phzacc[2] += ft_veg * 1.0
                if s.phzacc[2] >= s.phthrs[2]:
                    s.stgdoy[2] = yrdoy
            # Phase 4: JuvenileEnd → R0 (INL photoperiod, ZMODTE applied)
            # RStages.for lines 280-296
            if s.phzacc[3] < s.phthrs[3]:
                s.phzacc[3] += ft_veg * fuday * fsw * fnstr * zmodte

        # -- Phase 5: R0 → R1 (flowering); INL photoperiod, ZMODTE
        # RStages.for lines 298-318; triggers when PHZACC(4)>=PHTHRS(4) AND PHZACC(5)>=PHTHRS(5)
        if s.rstage < 1:
            # Only start phase 5 after phase 4 is done
            if s.phzacc[3] >= s.phthrs[3]:
                s.phzacc[4] += ft_veg * fuday * fsw * fnstr * zmodte
                if s.phzacc[4] >= s.phthrs[4]:
                    s.rstage = 1
                    s.r1_date = yrdoy
                    s.stgdoy[3] = yrdoy  # keep stgdoy[3] as R1 date for compatibility

        # -- Phase 6: R1 → R3 (pod set) — use post-R1 photoperiod (CSDVRR/CLDVRR)
        # RStages.for lines 320-366; DRPP = FUDAY(6) — PHENOL.for line 289
        elif s.rstage == 1:
            fuday_r = self._photoperiod_factor(dayl, after_r1=True)
            s.phzacc[5] += ft_veg * fuday_r * fsw * fnstr
            s.vstage += ft_veg * fuday_r  # leafing continues after R1
            if s.phzacc[5] >= s.phthrs[5]:
                s.rstage = 3
                s.r3_date = yrdoy
                s.stgdoy[4] = yrdoy

        # -- Phase 8: R1 → R5 via R3 (seed fill start)
        # PHTHRS(8) = FL_SD = R1 → R5; accumulated in phzacc[7]
        # RStages.for lines 368-392
        elif s.rstage == 3:
            fuday_r = self._photoperiod_factor(dayl, after_r1=True)
            s.phzacc[7] += ft_veg * fuday_r * fsw * fnstr
            if s.phzacc[7] >= s.phthrs[7]:
                s.rstage = 5
                s.r5_date = yrdoy
                s.stgdoy[5] = yrdoy

        # -- Phase 10: R5 → R7 (physiological maturity)
        # PHTHRS(10) = SD_PM; DXR57 = PHZACC(10)/PHTHRS(10)
        # PHENOL.for lines 335-340; RStages.for lines 416-439
        elif s.rstage == 5:
            s.phzacc[9] += ft_veg * fsw * fnstr
            s.dxr57 = min(s.phzacc[9] / max(s.phthrs[9], 1e-6), 1.0)
            if s.phzacc[9] >= s.phthrs[9]:
                s.rstage = 7
                s.r7_date = yrdoy
                s.mdate = yrdoy
                s.stgdoy[6] = yrdoy
                s.crop_status = 1

        # -- update VSTAGE (continuous leaf number, driven by ft_veg)
        # VSTAGES.for: VSTAGE += DTX * TRIFOL * TURFAC * (1-XPOD)
        # We use a simplified rate here; trifol ~0.32 leaves/thermal day
        trifol = getattr(cv, "trifol", 0.32)
        if s.emerged and s.rstage < 7:
            if s.rstage < 1:
                # Pre-R1: V-stage grows proportional to DTX * TRIFOL
                # VSTAGES.for line 469
                s.vstage += ft_veg * trifol * (1.0 - s.xpod)
        s.vstage = max(s.vstage, 0.0)

        # -- DRPP (daily photothermal day): FUDAY(6) in Fortran = FUDAY for phase 6
        # Phase 6 = R1→R3 phase. PHENOL.for line 289: DRPP = FUDAY(6)
        # After R1 use post-R1 photoperiod parameters (CSDVRR/CLDVRR).
        s.drpp = self._photoperiod_factor(dayl, after_r1=(s.rstage >= 1))

        # -- XPOD: pod partitioning factor (0 before R1, ramps to 1 by R3)
        # PHENOL.for / PODS.for: xpod ramps from 0→1 during R1→R3 phase (phzacc[5]/phthrs[5])
        if s.rstage >= 3:
            s.xpod = 1.0
        elif s.rstage == 1 and s.phthrs[5] > 0:
            s.xpod = min(s.phzacc[5] / s.phthrs[5], 1.0)
        else:
            s.xpod = 0.0

    # ------------------------------------------------------------------
    def _photoperiod_factor(self, dayl: float, after_r1: bool = False) -> float:
        """Compute the photoperiod factor (fuday) using the INL curve.

        Mirrors ``CURV('INL', 1.0, CSDVAR, CLDVAR, THVAR, DAYL)`` from
        PHENOL.for line 243-245.  After R1 (NR1), CSDVRR/CLDVRR are used
        instead of CSDVAR/CLDVAR (PHENOL.for line 245).

        For a short-day plant (soybean):
            DLTYP = 'INL' → inverse linear:
            fuday = 1.0  when dayl <= csdvar
            fuday = THVAR when dayl >= cldvar
            linearly interpolated between

        For a long-day plant (canola), the same INL curve is used but with
        csdvar < cldvar such that short days give the minimum response.

        Args:
            dayl:      Current daylength (h).
            after_r1:  If True, use post-R1 parameters (CSDVRR/CLDVRR).
                       Corresponds to PHENOL.for line 245.

        Returns:
            Photoperiod factor (0–1 range, minimum = thvar).
        """
        cv = self.cultivar
        thvar = getattr(cv, "thvar", 0.0)  # minimum rate at long day (default 0)

        if after_r1:
            # CSDVRR = CSDVAR - R1PPO,  CLDVRR = CLDVAR - R1PPO  (Ipphenol.for l.245-246)
            csdvar = getattr(cv, "csdvrr", cv.csdvar)
            cldvar = getattr(cv, "cldvrr", cv.cldvar)
        else:
            csdvar = cv.csdvar
            cldvar = cv.cldvar

        if self.short_day:
            # INL (inverse linear): max at short days, min (THVAR) at long days
            # CURV('INL', 1.0, CSDVAR, CLDVAR, THVAR, DAYL) — PHENOL.for line 243
            fuday = curv_lin(1.0, csdvar, cldvar, thvar, dayl)
        else:
            # Long-day: INL effectively means max at long days.
            # For canola, DLTYP phases 4-9 are 'INL' and the curve gives low
            # values at short days and 1.0 at long days.  We keep the same INL
            # form but swap the endpoints.
            fuday = curv_lin(thvar, csdvar, cldvar, 1.0, dayl)

        return max(0.0, min(1.0, fuday))
