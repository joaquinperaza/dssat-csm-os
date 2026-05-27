"""CERES-Maize phenology — growth stage determination and thermal time.

Translated from ``MZ_PHENOL.for`` (DSSAT CERES-Maize v4.8).

Growth stages (ISTAGE):

| Code | Description                                        |
|------|----------------------------------------------------|
| 7    | Sowing                                              |
| 8    | Germination                                         |
| 9    | Emergence                                           |
| 1    | End of juvenile phase (floral induction)            |
| 2    | Tassel initiation (end photoperiod-sensitive phase) |
| 3    | End of leaf growth / silking                        |
| 4    | End of panicle growth / effective grain fill start  |
| 5    | Effective grain filling → 95 % of P5               |
| 6    | End of effective GF → physiological maturity        |
| 10   | Post-maturity (Fortran sets ISTAGE=10 after PM)     |

References:
    Jones, C.A. & Kiniry, J.R. (1986) CERES-Maize. Texas A&M Press.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np

from dssat.core.constants import NL, RUNINIT, SEASINIT, RATE, INTEGR
from dssat.core.date_utils import yr_doy

if TYPE_CHECKING:
    from dssat.crop.ceres_maize.fileio import CeresMaizeParams


@dataclass
class MaizeCultivar:
    """CERES-Maize cultivar and ecotype coefficients.

    Parameters come from ``MZCER048.CUL`` (cultivar), ``MZCER048.ECO``
    (ecotype) and ``MZCER048.SPE`` (species).  Use :meth:`from_file` to
    load directly from disk.

    Attributes:
        varno:  Cultivar identifier (6-char).
        vrname: Cultivar name.
        econo:  Ecotype identifier (6-char).
        p1:    Thermal time from emergence to end of juvenile phase (°C d).
        p2:    Photoperiod sensitivity (d h⁻¹ above p2o). [MZ_PHENOL RATEIN]
        p5:    Thermal time from silking to physiological maturity (°C d).
        g2:    Maximum possible number of kernels per plant.
        g3:    Kernel filling rate (mg d⁻¹ kernel⁻¹) under optimum conditions.
        phint: Phyllochron interval (°C d leaf⁻¹).
        tbase: Base temperature for development (°C).  From ECO.
        topt:  Optimum temperature for vegetative development (°C).  From ECO.
        ropt:  Optimum temperature for reproductive development (°C).  From ECO.
        p2o:   Critical photoperiod (h).  From ECO.
        djti:  Minimum days from end-juvenile to tassel initiation.  From ECO.
        gdde:  GDD per cm seed depth for emergence (°C d cm⁻¹).  From ECO.
        dsgft: GDD from silking to effective grain fill (°C d).  From ECO.
        rue:   Radiation use efficiency (g MJ⁻¹ PAR).  From ECO.
        kcan:  PAR extinction coefficient.  From ECO.
        tsen:  Critical temperature for cold leaf damage (°C).  From ECO.
        cday:  Consecutive cold days for early maturity.  From ECO.
    """

    varno: str = "GENERIC"
    vrname: str = "Generic Maize"
    econo: str = "DFAULT"

    # Cultivar coefficients (from CUL file)
    p1:    float = 300.0
    p2:    float = 0.52
    p5:    float = 800.0
    g2:    float = 800.0
    g3:    float = 8.5
    phint: float = 38.9

    # Ecotype coefficients (from ECO file)
    tbase: float = 8.0
    topt:  float = 34.0
    ropt:  float = 34.0
    p2o:   float = 12.5
    djti:  float = 4.0
    gdde:  float = 6.0
    dsgft: float = 170.0
    rue:   float = 4.2
    kcan:  float = 0.85
    tsen:  float = 6.0
    cday:  int   = 15

    @classmethod
    def from_file(
        cls,
        varno: str,
        *,
        model_code: str = "MZCER048",
        genotype_dir: Optional[Path | str] = None,
    ) -> "MaizeCultivar":
        """Load a cultivar from MZCER048.CUL + .ECO + .SPE.

        Parameters
        ----------
        varno:
            Cultivar identifier, e.g. ``"IB0001"``.
        model_code:
            DSSAT model prefix, default ``"MZCER048"``.
        genotype_dir:
            Path to the directory with the genotype files.  Auto-detected
            if omitted.

        Returns
        -------
        MaizeCultivar
            Fully populated cultivar object with ``._params`` attribute
            giving access to the raw :class:`~dssat.crop.ceres_maize.fileio.CeresMaizeParams`.
        """
        from dssat.crop.ceres_maize.fileio import load_ceres_maize_params
        p = load_ceres_maize_params(varno, model_code=model_code,
                                    genotype_dir=genotype_dir)
        cv = cls(
            varno  = p.cul.varno,
            vrname = p.cul.vrname,
            econo  = p.cul.econo,
            p1     = p.cul.p1,
            p2     = p.cul.p2,
            p5     = p.cul.p5,
            g2     = p.cul.g2,
            g3     = p.cul.g3,
            phint  = p.cul.phint,
            tbase  = p.eco.tbase,
            topt   = p.eco.topt,
            ropt   = p.eco.ropt,
            p2o    = p.eco.p2o,
            djti   = p.eco.djti,
            gdde   = p.eco.gdde,
            dsgft  = p.eco.dsgft,
            rue    = p.eco.rue,
            kcan   = p.eco.kcan,
            tsen   = p.eco.tsen,
            cday   = p.eco.cday,
        )
        # Store full params so growth module can access SPE data
        cv._params = p
        return cv

    @classmethod
    def from_dict(cls, d: dict) -> "MaizeCultivar":
        """Construct from a coefficient dict (backward compatibility)."""
        mapping = {k.lower(): k for k in cls.__dataclass_fields__}
        kwargs: dict = {}
        for k, v in d.items():
            canonical = mapping.get(k.lower())
            if canonical:
                kwargs[canonical] = v
        return cls(**kwargs)


@dataclass
class PhenologyState:
    """Mutable state for the CERES-Maize phenology sub-model.

    Attributes:
        istage:    Current growth stage (7, 8, 9, 1–6, 10).
        xstage:    Continuous growth stage indicator.
        cumdtt:    Cumulative thermal time from germination (°C d).
        sumdtt:    Cumulative TT within the current stage (°C d).
        dtt:       Daily thermal time (°C d d⁻¹).
        stgdoy:    YYYYDDD date when each stage was reached (1-indexed in
                   Fortran; stored 0-indexed here, size 20).
        mdate:     Maturity date (YYYYDDD); -99 until reached.
        isdate:    Silking / anthesis date (YYYYDDD).
        yremrg:    Emergence date (YYYYDDD); -99 until reached.
        gpp:       Grain number per plant (grains plant⁻¹).
        ears:      Ears per unit area (m⁻²).
        tlno:      Total leaf number at final stage (leaves plant⁻¹).
        xnti:      Leaf number at tassel initiation.
        p3:        Computed TT for stage 3 (tassel init → silking) (°C d).
        kep:       Light extinction coefficient for even-aged canopy.
        crop_status: Crop status flag (0=running, 1=matured normally, 12/13=failure).
        l0:        Soil layer containing the seed.
        ndas:      Days after sowing.
        swsd:      Modified soil water for germination check.
        sind:      Cumulative photoperiod induction rate (stage 2; MZ_PHENOL).
        sumdtt_2:  SUMDTT at start of stage 3 (used for P3/VegFrac calculation).
        idurp:     Calendar days elapsed in stage 4 (IDURP in Fortran).
        dummy:     First-day flag for stage 2 (DUMMY in MZ_PHENOL.for).
        p9:        Computed TT threshold for emergence (45 + GDDE*SDEPTH) (°C d).
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
    ears: float = 0.0
    tlno: float = 0.0
    xnti: float = 0.0
    p3: float = 0.0
    kep: float = 0.7
    crop_status: int = 0
    l0: int = 0
    ndas: int = 0
    swsd: float = 0.0
    # Stage-2 photoperiod induction (MZ_PHENOL.for: SIND)
    sind: float = 0.0
    # SUMDTT captured at end of stage 2 (used to compute P3)
    sumdtt_2: float = 0.0
    # Calendar days in stage 4 (MZ_PHENOL.for: IDURP)
    idurp: int = 0
    # First-day flag for stage 2 (MZ_PHENOL.for: DUMMY)
    dummy: int = 0
    # Computed P9 value (MZ_PHENOL.for: P9 = 45 + GDDE*SDEPTH)
    p9: float = 0.0
    # P-model fractions
    vegfrac: float = 0.0
    seedfrac: float = 0.0
    # Cumulative growth during stage 4 (g plant⁻¹); fed from growth module
    sump: float = 0.0


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

    Implements the crown-temperature and photoperiod-aware DTT algorithm
    from ``MZ_PHENOL.for`` (JTR/CIMMYT method, 1998).

    Args:
        tmax:   Daily maximum temperature (°C).
        tmin:   Daily minimum temperature (°C).
        srad:   Solar radiation (MJ m⁻² d⁻¹).
        dayl:   Day length (h).
        snow:   Snow depth (mm).
        leafno: Current number of fully expanded leaves.
        istage: Current CERES growth stage code.
        tbase:  Base temperature (°C).
        topt:   Optimum temperature for vegetative development (°C).
        ropt:   Optimum temperature for reproductive development (°C).

    Returns:
        Daily thermal time (°C d), ≥ 0.
    """
    # MZ_PHENOL.for:430 — DOPT switches to ROPT after anthesis
    dopt = ropt if (3 < istage <= 6) else topt

    # Crown temperature adjustment under snow (MZ_PHENOL.for:402-416)
    xs = min(snow, 15.0)
    tempcn = tmin
    tempcx = tmax
    if tmin < 0.0:
        tempcn = 2.0 + tmin * (0.4 + 0.0018 * (xs - 15.0) ** 2)
    if tmax < 0.0:
        tempcx = 2.0 + tmax * (0.4 + 0.0018 * (xs - 15.0) ** 2)

    # MZ_PHENOL.for:437-506
    if tmax < tbase:
        dtt = 0.0
    elif tmin > dopt:
        dtt = dopt - tbase
    elif leafno <= 10:
        # Seedling stage: soil temperature method
        if xs > 0.0:
            dtt = (tempcn + tempcx) / 2.0 - tbase
        else:
            acoef  = 0.01061 * srad + 0.5902
            tdsoil = acoef * tmax + (1.0 - acoef) * tmin
            tnsoil = 0.36354 * tmax + 0.63646 * tmin
            if tdsoil < tbase:
                dtt = 0.0
            else:
                tnsoil = max(tnsoil, tbase)
                tdsoil = min(tdsoil, dopt)
                tmsoil = tdsoil * (dayl / 24.0) + tnsoil * ((24.0 - dayl) / 24.0)
                if tmsoil < tbase:
                    dtt = (tbase + tdsoil) / 2.0 - tbase
                else:
                    dtt = (tnsoil + tdsoil) / 2.0 - tbase
                dtt = min(dtt, dopt - tbase)
    elif tmin < tbase or tmax > dopt:
        # Hourly integration when Tmax or Tmin out of range (MZ_PHENOL.for:492-503)
        dtt = 0.0
        for i in range(1, 25):
            th = (tmax + tmin) / 2.0 + (tmax - tmin) / 2.0 * math.sin(
                math.pi / 12.0 * i
            )
            th = min(max(th, tbase), dopt)
            dtt += (th - tbase) / 24.0
    else:
        dtt = (tmax + tmin) / 2.0 - tbase

    return max(dtt, 0.0)  # MZ_PHENOL.for:508


class MaizePhenology:
    """CERES-Maize phenology sub-model.

    Determines growth stage and accumulates thermal time, matching
    ``MZ_PHENOL.for`` logic exactly.

    Args:
        cultivar: Cultivar and ecotype coefficients.
        pltpop:   Plant population (plants m⁻²).
        sdepth:   Sowing depth (cm).
        yrsim:    Simulation start date (YYYYDDD).
    """

    def __init__(
        self,
        cultivar: MaizeCultivar,
        pltpop: float,
        sdepth: float,
        yrsim: int,
    ) -> None:
        self.cultivar = cultivar
        self.pltpop = pltpop
        self.sdepth = sdepth
        self.state = PhenologyState()
        self.state.stgdoy[13] = yrsim  # STGDOY(14) = YRSIM (MZ_PHENOL.for:343)

        # SPE parameters — from MZCER048.SPE *SEED section (MZ_PHENOL.for:249-258)
        spe = getattr(getattr(cultivar, "_params", None), "spe", None)
        self._dsgt: float = getattr(spe, "dsgt", 21.0)   # max days germ→failure
        self._dget: float = getattr(spe, "dget", 150.0)  # max GDD germ→emergence
        self._swcg: float = getattr(spe, "swcg", 0.02)   # min soil water for germination

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
        xn: int = 0,
        sump: float = 0.0,
    ) -> None:
        """Run one phenology time step.

        Args:
            dynamic:  Simulation phase (RUNINIT, SEASINIT, RATE, INTEGR).
            yrdoy:    Current date (YYYYDDD).
            tmax:     Max temperature (°C).
            tmin:     Min temperature (°C).
            srad:     Solar radiation (MJ m⁻² d⁻¹).
            dayl:     Astronomical day length (h).
            snow:     Snow depth (mm).
            sw:       Soil water by layer (cm³ cm⁻³).
            ll:       Lower limit by layer (cm³ cm⁻³).
            dlayr:    Layer thickness (cm).
            nlayr:    Number of active layers.
            iswwat:   Water switch (``"Y"``/``"N"``).
            twilen:   Twilight daylength (h); defaults to *dayl* if ≤ 0.
            xn:       Leaf number from growth module (passed as XN from grosub).
            sump:     Cumulative plant growth during stage 4 (g plant⁻¹); for GPP.
        """
        # Use astronomical dayl as twilight approximation when not provided
        if twilen <= 0.0:
            twilen = dayl

        if dynamic in (RUNINIT, SEASINIT):
            self._init(yrdoy)
        elif dynamic == RATE:
            # Only run the daily rate/integrate logic once per day (RATE phase).
            # The INTEGR phase is handled by root growth in the crop model;
            # running phenology again in INTEGR would double-count thermal time.
            self._rate_integr(
                yrdoy, tmax, tmin, srad, dayl, snow,
                sw, ll, dlayr, nlayr, iswwat, twilen, xn, sump
            )

    # ------------------------------------------------------------------
    def _init(self, yrsim: int) -> None:
        """Reset state for a new season (MZ_PHENOL.for:340-383)."""
        s = self.state
        cv = self.cultivar

        s.istage    = 7
        s.xstage    = 0.1
        s.cumdtt    = 0.0
        s.sumdtt    = 0.0
        s.dtt       = 0.0
        s.stgdoy[:] = 9999999
        s.stgdoy[13] = yrsim              # STGDOY(14) = YRSIM
        s.mdate     = -99
        s.isdate    = 0
        s.yremrg    = -99                 # MZ_PHENOL.for:345
        s.gpp       = 0.0
        s.ears      = 0.0
        s.tlno      = 0.0
        s.xnti      = 0.0
        s.p3        = 0.0
        # KEP = KCAN/(1-0.07)*(1-0.25)  (MZ_PHENOL.for:338)
        s.kep       = cv.kcan / (1.0 - 0.07) * (1.0 - 0.25)
        s.crop_status = 0
        s.l0        = 0
        s.ndas      = 0
        s.swsd      = 0.0
        s.sind      = 0.0
        s.sumdtt_2  = 0.0
        s.idurp     = 0
        s.dummy     = 0
        s.p9        = 0.0
        s.vegfrac   = 0.0
        s.seedfrac  = 0.0
        s.sump      = 0.0

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
        xn: int,
        sump: float,
    ) -> None:
        """Daily rate + integration (RATE/INTEGR dynamic phases)."""
        s = self.state
        cv = self.cultivar

        # Update cumulative sump from growth module (stage 4 accumulation)
        if s.istage == 4:
            s.sump = sump

        # -- Compute thermal time (MZ_PHENOL.for:509-510)
        s.dtt = compute_dtt(
            tmax, tmin, srad, dayl, snow,
            int(s.tlno), s.istage,         # LEAFNO proxy
            cv.tbase, cv.topt, cv.ropt,
        )
        s.sumdtt += s.dtt
        s.cumdtt += s.dtt

        # DOPT for VegFrac / XSTAGE denominators
        dopt = cv.ropt if (3 < s.istage <= 6) else cv.topt

        # -- Stage transitions (MZ_PHENOL.for:531-912)
        if s.istage == 7:
            self._stage7(yrdoy, dlayr, nlayr, iswwat)

        elif s.istage == 8:
            self._stage8(yrdoy, sw, ll, nlayr, iswwat)

        elif s.istage == 9:
            self._stage9(yrdoy)

        elif s.istage == 1:
            self._stage1(yrdoy, dopt)

        elif s.istage == 2:
            self._stage2(yrdoy, dopt, twilen, xn)

        elif s.istage == 3:
            self._stage3(yrdoy, dopt)

        elif s.istage == 4:
            self._stage4(yrdoy, dopt, sump)

        elif s.istage == 5:
            self._stage5(yrdoy)

        elif s.istage == 6:
            self._stage6(yrdoy)

    # ------------------------------------------------------------------
    # Stage 7: Sowing (MZ_PHENOL.for:531-549)
    # ------------------------------------------------------------------
    def _stage7(
        self,
        yrdoy: int,
        dlayr: np.ndarray,
        nlayr: int,
        iswwat: str,
    ) -> None:
        s = self.state
        s.stgdoy[7 - 1] = yrdoy       # STGDOY(ISTAGE=7) = YRDOY
        s.ndas = 0
        s.istage = 8
        s.sumdtt = 0.0
        if iswwat == "N":
            return
        # Find seed layer (MZ_PHENOL.for:541-547)
        cumdep = 0.0
        for L in range(nlayr):
            cumdep += dlayr[L]
            if self.sdepth < cumdep:
                s.l0 = L
                break

    # ------------------------------------------------------------------
    # Stage 8: Germination (MZ_PHENOL.for:555-592)
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
            # Only count dry days and check soil water when below LL
            # MZ_PHENOL.for:557: IF (SW(L0) .LE. LL(L0)) THEN
            if sw[l0] <= ll[l0]:
                l1 = min(l0 + 1, nlayr - 1)
                s.swsd = (
                    (sw[l0] - ll[l0]) * 0.65
                    + (sw[l1] - ll[l1]) * 0.35
                )
                s.ndas += 1
                if s.ndas >= self._dsgt:          # MZ_PHENOL.for:562: NDAS >= DSGT
                    s.istage = 6
                    self.pltpop = 0.0
                    s.gpp = 1.0
                    s.mdate = yrdoy
                    s.crop_status = 12
                    return
                if s.swsd < self._swcg:           # MZ_PHENOL.for:580
                    return

        # Germination occurs — move to stage 9
        s.stgdoy[8 - 1] = yrdoy
        s.istage = 9
        s.cumdtt = 0.0
        s.sumdtt = 0.0
        # P9 = 45 + GDDE*SDEPTH  (MZ_PHENOL.for:591)
        s.p9 = 45.0 + cv.gdde * self.sdepth

    # ------------------------------------------------------------------
    # Stage 9: Emergence (MZ_PHENOL.for:598-636)
    # ------------------------------------------------------------------
    def _stage9(self, yrdoy: int) -> None:
        s = self.state
        s.ndas += 1
        if s.sumdtt < s.p9:                       # MZ_PHENOL.for:603
            return
        # Emergence failure check (MZ_PHENOL.for:609-626)
        if s.p9 > self._dget:
            s.istage = 6
            self.pltpop = 0.0
            s.gpp = 1.0
            s.mdate = yrdoy
            s.crop_status = 13
            return
        s.stgdoy[9 - 1] = yrdoy
        s.istage = 1
        s.sumdtt = s.sumdtt - s.p9               # MZ_PHENOL.for:633 — carryover!
        s.tlno = 30.0                             # MZ_PHENOL.for:634
        s.yremrg = s.stgdoy[9 - 1]              # MZ_PHENOL.for:635

    # ------------------------------------------------------------------
    # Stage 1: Emergence → End of Juvenile (MZ_PHENOL.for:641-677)
    # ------------------------------------------------------------------
    def _stage1(self, yrdoy: int, dopt: float) -> None:
        s = self.state
        cv = self.cultivar
        s.ndas += 1
        # XSTAGE = SUMDTT/P1  (MZ_PHENOL.for:643)
        s.xstage = s.sumdtt / cv.p1 if cv.p1 > 0.0 else 0.0
        # VegFrac (MZ_PHENOL.for:668)
        s.vegfrac = s.sumdtt / (cv.p1 + 25.0 * (dopt - cv.tbase))
        if s.sumdtt < cv.p1:                      # MZ_PHENOL.for:670
            return
        # Transition to stage 2
        s.stgdoy[1 - 1] = yrdoy
        s.istage = 2
        s.sind = 0.0                              # MZ_PHENOL.for:677 — reset SIND

    # ------------------------------------------------------------------
    # Stage 2: End Juvenile → Tassel Initiation (MZ_PHENOL.for:683-742)
    # ------------------------------------------------------------------
    def _stage2(self, yrdoy: int, dopt: float, twilen: float, xn: int) -> None:
        s = self.state
        cv = self.cultivar
        s.ndas += 1
        # XSTAGE = 1.0 + 0.5*SIND  (MZ_PHENOL.for:687)
        s.xstage = 1.0 + 0.5 * s.sind
        # VegFrac (MZ_PHENOL.for:709)
        s.vegfrac = max(
            s.vegfrac,
            s.sumdtt / (cv.p1 + 25.0 * (dopt - cv.tbase)),
        )

        # Photoperiod-sensitive rate (MZ_PHENOL.for:716-722)
        # PDTT = 1.0 (MZ_PHENOL.for:721 sets PDTT=1.0 unconditionally)
        if twilen > cv.p2o:
            ratein = 1.0 / (cv.djti + cv.p2 * (twilen - cv.p2o))
        else:
            ratein = 1.0 / cv.djti
        s.sind += ratein * 1.0                    # MZ_PHENOL.for:722: SIND += RATEIN*PDTT

        if s.sind < 1.0:                          # MZ_PHENOL.for:724
            return

        # Transition to stage 3 (MZ_PHENOL.for:730-742)
        s.stgdoy[2 - 1] = yrdoy
        s.istage = 3
        # TLNO = SUMDTT/(PHINT*0.5) + 5.0  (MZ_PHENOL.for:734)
        s.tlno = s.sumdtt / (cv.phint * 0.5) + 5.0
        # P3 = ((TLNO + 0.5) * PHINT) - SUMDTT  (MZ_PHENOL.for:735)
        s.p3 = (s.tlno + 0.5) * cv.phint - s.sumdtt
        # XNTI = XN (leaf number from growth module)  (MZ_PHENOL.for:736)
        s.xnti = float(xn)
        # SUMDTT_2 = SUMDTT  (MZ_PHENOL.for:739)
        s.sumdtt_2 = s.sumdtt
        # VegFrac update (MZ_PHENOL.for:740)
        s.vegfrac = max(
            s.vegfrac,
            s.sumdtt_2 / (s.sumdtt_2 + s.p3 + cv.dsgft),
        )
        s.sumdtt = 0.0                            # MZ_PHENOL.for:742

    # ------------------------------------------------------------------
    # Stage 3: Tassel Init → Silking (MZ_PHENOL.for:750-780)
    # ------------------------------------------------------------------
    def _stage3(self, yrdoy: int, dopt: float) -> None:
        s = self.state
        cv = self.cultivar
        s.ndas += 1
        # XSTAGE = 1.5 + 3.0*SUMDTT/P3  (MZ_PHENOL.for:755)
        s.xstage = 1.5 + 3.0 * s.sumdtt / s.p3 if s.p3 > 0.0 else 1.5
        # VegFrac (MZ_PHENOL.for:766)
        s.vegfrac = max(
            s.vegfrac,
            (s.sumdtt + s.sumdtt_2) / (s.sumdtt_2 + s.p3)
            if (s.sumdtt_2 + s.p3) > 0.0 else 0.0,
        )
        if s.sumdtt < s.p3:                       # MZ_PHENOL.for:768
            return
        # Transition to stage 4 (MZ_PHENOL.for:773-780)
        s.stgdoy[3 - 1] = yrdoy
        s.isdate = yrdoy                          # MZ_PHENOL.for:774: ISDATE=YRDOY
        s.istage = 4
        s.sumdtt = s.sumdtt - s.p3               # MZ_PHENOL.for:776 — carryover
        s.idurp  = 0                             # MZ_PHENOL.for:777
        s.vegfrac = 1.0                           # MZ_PHENOL.for:780

    # ------------------------------------------------------------------
    # Stage 4: Silking → Effective GF Start (MZ_PHENOL.for:785-848)
    # ------------------------------------------------------------------
    def _stage4(self, yrdoy: int, dopt: float, sump: float) -> None:
        s = self.state
        cv = self.cultivar
        s.ndas += 1
        s.idurp += 1
        # XSTAGE = 4.5 + 5.5*SUMDTT/(P5*0.95)  (MZ_PHENOL.for:790)
        denom = cv.p5 * 0.95
        s.xstage = 4.5 + 5.5 * s.sumdtt / denom if denom > 0.0 else 4.5
        # SeedFrac (MZ_PHENOL.for:798)
        s.seedfrac = s.sumdtt / cv.p5 if cv.p5 > 0.0 else 0.0

        if s.sumdtt < cv.dsgft:                   # MZ_PHENOL.for:800
            return

        # Transition to stage 5 — compute GPP (MZ_PHENOL.for:810-848)
        # PSKER = SUMP*1000.0/IDURP*3.4/5.0  (MZ_PHENOL.for:810)
        psker = sump * 1000.0 / max(s.idurp, 1) * 3.4 / 5.0
        gpp = cv.g2 * psker / 7200.0 + 50.0      # MZ_PHENOL.for:811
        gpp = min(gpp, cv.g2)
        gpp = max(gpp, 0.0)
        gpp = max(gpp, 51.0)                      # MZ_PHENOL.for:817

        # Barrenness (MZ_PHENOL.for:823-844)
        ears = self.pltpop
        if gpp < cv.g2 * 0.15:
            ears = self.pltpop * (gpp / (cv.g2 * 0.15)) ** 0.33
        elif self.pltpop > 12.0:
            if gpp < cv.g2 * 0.5:
                barfac = 0.0085 * (1.0 - gpp / cv.g2) * self.pltpop ** 1.5
                ears = self.pltpop * (gpp / (cv.g2 * 0.50)) ** barfac

        s.ears = max(ears, 0.0)
        s.gpp  = gpp
        s.stgdoy[4 - 1] = yrdoy
        s.istage = 5

    # ------------------------------------------------------------------
    # Stage 5: Effective GF → 95 % of P5 (MZ_PHENOL.for:858-875)
    # ------------------------------------------------------------------
    def _stage5(self, yrdoy: int) -> None:
        s = self.state
        cv = self.cultivar
        s.ndas += 1
        # XSTAGE = 4.5 + 5.5*SUMDTT/P5  (MZ_PHENOL.for:860)
        s.xstage = 4.5 + 5.5 * s.sumdtt / cv.p5 if cv.p5 > 0.0 else 4.5
        # SeedFrac (MZ_PHENOL.for:867)
        s.seedfrac = s.sumdtt / cv.p5 if cv.p5 > 0.0 else 0.0

        if s.sumdtt < cv.p5 * 0.95:              # MZ_PHENOL.for:870
            return
        s.stgdoy[5 - 1] = yrdoy
        s.istage = 6

    # ------------------------------------------------------------------
    # Stage 6: End EFG → Physiological Maturity (MZ_PHENOL.for:880-911)
    # ------------------------------------------------------------------
    def _stage6(self, yrdoy: int) -> None:
        s = self.state
        cv = self.cultivar
        # Low-DTT protection (MZ_PHENOL.for:881)
        if s.dtt < 2.0:
            s.sumdtt = cv.p5
        # SeedFrac (MZ_PHENOL.for:889)
        s.seedfrac = s.sumdtt / cv.p5 if cv.p5 > 0.0 else 0.0

        if s.sumdtt < cv.p5:                      # MZ_PHENOL.for:891
            return
        # Maturity reached (MZ_PHENOL.for:895-910)
        s.stgdoy[6 - 1] = yrdoy
        s.mdate      = yrdoy
        s.crop_status = 1                         # matured normally
        s.istage     = 10                         # MZ_PHENOL.for:899: ISTAGE=10
        s.cumdtt     = 0.0
        s.dtt        = 0.0
        s.seedfrac   = 1.0
        if self.pltpop != 0.0 and s.gpp <= 0.0:
            s.gpp = 1.0
