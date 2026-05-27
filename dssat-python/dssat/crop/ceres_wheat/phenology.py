"""CERES-Wheat/Barley phenology — growth stage determination and thermal time.

Translated from ``CER_Integrate.for``, ``CER_Growth.for``, and
``CER_Init.for``.

Architecture (Cropsim-CERES):
  - TFAC4 piecewise-linear 4-cardinal temperature response function.
  - Germination / emergence tracked via CUMGEU / GESTAGE (Gstages).
  - Post-emergence development tracked via CUMDU / RSTAGE (Rstages),
    where DU = TT * VF * DF * LIF2.
  - ISTAGE / XSTAGE derived from GESTAGE and RSTAGE.
  - Vernalization factor VF based on cumulative vernalization units CUMVD.
  - Photoperiod factor DF based on daylength vs threshold P1DT.

Growth stages follow the CERES convention:

| Code | Description                                       |
|------|---------------------------------------------------|
| 7    | Pre-sowing / initial state                        |
| 8    | Sowing (seed placed)                              |
| 9    | Germination (radicle emergence)                   |
| 1    | Emergence                                         |
| 2    | End spikelet production (terminal spikelet)       |
| 3    | End leaf growth                                   |
| 4    | End spike growth                                  |
| 5    | End lag phase of grain growth                     |
| 6    | End grain fill (physiological maturity)           |

References:
    Ritchie, J.T. & Otter, S. (1985) Description and performance of
    CERES-Wheat: A user-oriented wheat yield model. USDA-ARS ARS-38.

    Hunt, L.A. & Pararajasingham, S. (1994) CROPSIM-CERES wheat.
    Can. J. Plant Sci. 74, 113–125.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, TYPE_CHECKING

import numpy as np

from dssat.core.constants import NL, RUNINIT, SEASINIT, RATE, INTEGR

if TYPE_CHECKING:
    from dssat.crop.ceres_wheat.fileio import CeresWheatParams

# Standard day thermal time (TT in a 20 °C day, from CER_Init.for line 103)
STDAY = 20.0


# ---------------------------------------------------------------------------
# TFAC4 — 4-cardinal piecewise-linear temperature response
# ---------------------------------------------------------------------------

def _tfac4(tcard: List[float], tmean: float):
    """4-cardinal piecewise-linear temperature response.

    Matches ``TFAC4`` / ``CURV('LIN', ...)`` in ``CSUTS.for``.

    Args:
        tcard: [T_base, T_opt_low, T_opt_high, T_ceiling].
        tmean: Mean temperature (°C).

    Returns:
        Tuple (factor, tunit) where factor is in [0, 1] and
        tunit = factor * (T_opt_low - T_base) is the thermal time
        per day.
    """
    t0, t1, t2, t3 = tcard
    if tmean <= t0:
        f = 0.0
    elif tmean <= t1:
        f = (tmean - t0) / (t1 - t0) if t1 > t0 else 1.0
    elif tmean <= t2:
        f = 1.0
    elif tmean <= t3:
        f = 1.0 - (tmean - t2) / (t3 - t2) if t3 > t2 else 0.0
    else:
        f = 0.0
    f = max(0.0, min(1.0, f))
    tunit = f * (t1 - t0)
    return f, tunit


def compute_wheat_tt(trdv1: List[float], tmean: float) -> float:
    """Compute daily thermal time for wheat using TFAC4 with TRDV1 cardinals.

    Args:
        trdv1: 4-cardinal temperature array [Tbase, Topt, Topt2, Tmax].
        tmean: Mean temperature (°C).

    Returns:
        Daily thermal time (°Cd).
    """
    _, tt = _tfac4(trdv1, tmean)
    return tt


def compute_wheat_ttgem(trgem: List[float], tmean: float) -> float:
    """Compute germination thermal time using TFAC4 with TRGEM cardinals.

    Args:
        trgem: 4-cardinal temperature array for germination.
        tmean: Mean temperature (°C).

    Returns:
        Daily germination thermal time (°Cd).
    """
    _, ttgem = _tfac4(trgem, tmean)
    return ttgem


def compute_tfv(trvrn: List[float], tmean: float) -> float:
    """Compute daily vernalization temperature factor (TFV).

    TFV is the amount of vernalization (in °Cd units of TRVRN optimum range)
    accumulated per day.

    Args:
        trvrn: 4-cardinal temperature array for vernalization.
        tmean: Mean temperature (°C).

    Returns:
        Vernalization units per day.
    """
    f, _ = _tfac4(trvrn, tmean)
    # TFV = factor * (T1 - T0) following TFAC4 convention
    return f * (trvrn[1] - trvrn[0])


# ---------------------------------------------------------------------------
# Backward-compatible compute_wheat_dtt for external callers
# ---------------------------------------------------------------------------

def compute_wheat_dtt(
    tmax: float,
    tmin: float,
    tbase: float = 0.0,
    topt: float = 26.0,
) -> float:
    """Backward-compatible daily thermal time function.

    Constructs a synthetic TRDV1 = [tbase, topt, topt+24, topt+34] and
    calls TFAC4.  This keeps old callers working while the main phenology
    now uses the full 4-cardinal trdv1 array.

    Args:
        tmax: Daily maximum temperature (°C).
        tmin: Daily minimum temperature (°C).
        tbase: Base temperature (°C).
        topt: Optimum temperature (°C).

    Returns:
        Daily thermal time (°Cd), >= 0.
    """
    tmean = (tmax + tmin) * 0.5
    trdv1 = [tbase, topt, topt + 24.0, topt + 34.0]
    return compute_wheat_tt(trdv1, tmean)


# ---------------------------------------------------------------------------
# WheatCultivar
# ---------------------------------------------------------------------------

@dataclass
class WheatCultivar:
    """CERES-Wheat cultivar and ecotype coefficients.

    All parameters originate from the three DSSAT genotype files:
    ``WHCER048.CUL`` (cultivar), ``WHCER048.ECO`` (ecotype), and
    ``WHCER048.SPE`` (species).  Use :meth:`from_file` to load directly
    from disk rather than constructing with hardcoded values.

    Attributes (CUL file):
        id (varno): Cultivar identifier (6-char).
        name (vrname): Cultivar name.
        ecotype (econo): Ecotype identifier.
        p1v: Vernalization requirement (P1V — effective °Cd for full vern).
        p1d: Photoperiod sensitivity (P1D — % reduction per 10 h drop).
        p5: Grain fill thermal time (°C d) — mapped to PD(5).
        g1: Kernel number per unit stem+spike weight at anthesis (#/g).
        g2: Standard kernel weight (mg kernel⁻¹).
        g3: Tiller death coefficient.
        phint: Phyllochron interval (°C d leaf⁻¹).

    Attributes (ECO file — resolved after CUL overrides):
        p1:    Phase 1 duration end-juvenile → TS (PD(1)).
        p2fr1: Fraction of P2 to jointing.
        p2:    Phase 2 duration TS → end leaf growth (PD(2)).
        p3:    Phase 3 duration end leaf → end spike (PD(3)).
        p4fr1: Fraction of P4 end spike → anthesis start.
        p4fr2: Fraction of P4 = anthesis duration.
        p4:    Phase 4 duration end spike → end grain fill lag (PD(4)).
        veff:  Vernalization effect factor (0–1).
        parue: Radiation use efficiency before last leaf (g MJ⁻¹).
        paru2: RUE after last leaf (g MJ⁻¹).
        kcan:  PAR extinction coefficient.
        la1s:  Area of standard first leaf (cm²).
        slas:  Specific leaf area standard (cm² g⁻¹).
        lafv, lafr: Leaf area increase per leaf, veg/repro phases.
        rdgs:  Root depth growth rate (cm d⁻¹).
        gn_s:  Grain N standard (%).
        gn_mn: Grain N minimum (%).

    Attributes (SPE file defaults):
        trdv1:  4-cardinal temperature array for development [Tb,To,To2,Tm].
        trgem:  4-cardinal temperature array for germination.
        trvrn:  4-cardinal temperature array for vernalization.
        pgerm:  Germination phase duration (hydrothermal units).
        pemrg:  Emergence rate (thermal units per cm depth).
        p1dt:   Photoperiod threshold (h). Default 20.0 h.
        ppend:  Photoperiod sensitivity end stage. Default 2.0.
        ppfpe:  Photoperiod factor pre-emergence. Default 1.0.
        wfgeu:  Water factor for seed germination upper limit. Default 0.5.

    Aliases (backward compatibility):
        tbase: trdv1[0].
        topt:  trdv1[1].
    """

    # CUL fields
    id:      str   = "IB1500"
    name:    str   = "Generic Spring Wheat"
    ecotype: str   = "USWH01"
    p1v:     float = 5.0
    p1d:     float = 75.0
    p5:      float = 450.0
    g1:      float = 30.0
    g2:      float = 35.0
    g3:      float = 1.0
    phint:   float = 60.0

    # Resolved ECO values (set by from_file or manually)
    p1:    float = 200.0
    p2fr1: float = 0.25
    p2:    float = 200.0
    p3:    float = 200.0
    p4fr1: float = 0.25
    p4fr2: float = 0.10
    p4:    float = 200.0
    veff:  float = 0.6
    parue: float = 2.7
    paru2: float = 2.7
    kcan:  float = 0.85
    la1s:  float = 5.0
    slas:  float = 400.0
    lafv:  float = 0.10
    lafr:  float = 0.50
    rdgs:  float = 3.0
    gn_s:  float = 3.0
    gn_mn: float = 0.0

    # tbase / topt: backward-compat aliases for trdv1 first two cardinals.
    # When constructing manually, set these and trdv1 will be derived.
    tbase: float = 0.0
    topt:  float = 26.0

    # SPE temperature response arrays (4-element each, from WHCER048.SPE)
    # Defaults match the WHCER048.SPE values.
    trdv1: List[float] = field(default_factory=lambda: [1.0, 26.0, 50.0, 60.0])
    trgem: List[float] = field(default_factory=lambda: [1.0, 26.0, 50.0, 60.0])
    trvrn: List[float] = field(default_factory=lambda: [-5.0, 0.0, 7.0, 15.0])

    # SPE scalar defaults
    pgerm:  float = 10.0   # PGERM — germination phase hydrothermal units
    pemrg:  float = 8.0    # PEMRG — emergence rate (TU/cm)
    p1dt:   float = 20.0   # PPTHR — photoperiod threshold (h)
    ppend:  float = 2.0    # PPEND — photoperiod sensitivity end stage
    ppfpe:  float = 1.0    # PPFPE — photoperiod factor pre-emergence
    wfgeu:  float = 0.5    # WFGEU — water factor upper limit for germination

    # ------------------------------------------------------------------ #
    def __post_init__(self):
        """Ensure trdv1 is consistent with tbase/topt when set explicitly."""
        # If trdv1 is still the default but tbase/topt differ from defaults,
        # reconstruct trdv1 from tbase/topt.
        default_trdv1 = [1.0, 26.0, 50.0, 60.0]
        if self.trdv1 == default_trdv1:
            if self.tbase != 0.0 or self.topt != 26.0:
                self.trdv1 = [
                    self.tbase,
                    self.topt,
                    self.topt + 24.0,
                    self.topt + 34.0,
                ]
                # Also update trgem to match if still at default
                default_trgem = [1.0, 26.0, 50.0, 60.0]
                if self.trgem == default_trgem:
                    self.trgem = list(self.trdv1)
        # Keep tbase/topt aliases consistent
        self.tbase = self.trdv1[0]
        self.topt = self.trdv1[1]

    # ------------------------------------------------------------------ #
    @property
    def varno(self) -> str:
        return self.id

    @property
    def vrname(self) -> str:
        return self.name

    @property
    def econo(self) -> str:
        return self.ecotype

    # ------------------------------------------------------------------ #
    @classmethod
    def from_file(
        cls,
        varno: str,
        *,
        model_code: str = "WHCER048",
        genotype_dir: Optional[Path | str] = None,
    ) -> "WheatCultivar":
        """Load a cultivar from WHCER048.CUL + .ECO + .SPE.

        Parameters
        ----------
        varno:
            Cultivar identifier, e.g. ``"IB1500"``.
        model_code:
            DSSAT model prefix, default ``"WHCER048"``.
        genotype_dir:
            Path to the directory with the genotype files.  Auto-detected if
            omitted.

        Returns
        -------
        WheatCultivar
            Fully populated cultivar.  The full
            :class:`~dssat.crop.ceres_wheat.fileio.CeresWheatParams`
            (including SPE) is stored as ``cultivar._params``.
        """
        from dssat.crop.ceres_wheat.fileio import load_ceres_wheat_params
        p = load_ceres_wheat_params(varno, model_code=model_code,
                                    genotype_dir=genotype_dir)
        # Build trdv1 / trgem / trvrn from SPE arrays
        spe = p.spe
        trdv1 = list(spe.trdv1) if spe.trdv1 else [1.0, 26.0, 50.0, 60.0]
        trgem = list(spe.trgem) if spe.trgem else list(trdv1)
        trvrn = list(spe.trvrn) if spe.trvrn else [-5.0, 0.0, 7.0, 15.0]

        cv = cls(
            id      = p.cul.varno,
            name    = p.cul.vrname,
            ecotype = p.cul.econo,
            p1v     = p.cul.p1v,
            p1d     = p.cul.p1d,
            p5      = p.cul.p5,
            g1      = p.cul.g1,
            g2      = p.cul.g2,
            g3      = p.cul.g3,
            phint   = p.cul.phint,
            # Resolved ECO values
            p1      = p.p1,
            p2fr1   = p.p2fr1,
            p2      = p.p2,
            p3      = p.p3,
            p4fr1   = p.p4fr1,
            p4fr2   = p.p4fr2,
            p4      = p.p4,
            veff    = p.veff,
            parue   = p.parue,
            paru2   = p.paru2,
            kcan    = p.kcan,
            la1s    = p.la1s,
            slas    = p.slas,
            lafv    = p.lafv,
            lafr    = p.lafr,
            rdgs    = p.rdgs,
            gn_s    = p.gn_s,
            gn_mn   = p.gn_mn,
            # tbase / topt aliases from SPE
            tbase   = trdv1[0],
            topt    = trdv1[1],
            # Full 4-cardinal arrays
            trdv1   = trdv1,
            trgem   = trgem,
            trvrn   = trvrn,
        )
        cv._params = p
        return cv

    @classmethod
    def from_dict(cls, d: dict) -> "WheatCultivar":
        """Construct from a coefficient dict (backward compatibility)."""
        mapping = {k.lower(): k for k in cls.__dataclass_fields__}
        kwargs: dict = {}
        for k, v in d.items():
            canonical = mapping.get(k.lower())
            if canonical:
                kwargs[canonical] = v
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# WheatPhenologyState
# ---------------------------------------------------------------------------

@dataclass
class WheatPhenologyState:
    """Mutable state for the CERES-Wheat phenology sub-model.

    Implements the Cropsim-CERES dual-stage system:
      - Gstages (0–1): germination/emergence via cumgeu / gestage.
      - Rstages (0–6.9): post-emergence development via cumdu / rstage.

    Attributes:
        istage: Discrete growth stage (7, 8, 9, 1–6).
        xstage: Continuous growth stage (decimal).
        cumdu: Cumulative developmental units since emergence (°Cd modified).
        rstage: Continuous reproductive stage (0–6.9).
        cumgeu: Cumulative germination/emergence units.
        gestage: Germination–emergence stage (0→1).
        cumvd: Cumulative vernalization units.
        vrnstage: Vernalization completion (0→1).
        vf: Vernalization factor (0–1).
        df: Photoperiod factor (0–1).
        dtt: Daily thermal time (°Cd) from TFAC4(trdv1).
        stgdoy: Day-of-year at which each stage was reached (YYYYDDD).
        mdate: Maturity date (YYYYDDD); -99 until reached.
        isdate: Anthesis date (YYYYDDD).
        yremrg: Emergence date (YYYYDDD).
        gpp: Grain number per plant (kernels plant⁻¹).
        crop_status: Crop status flag.
        l0: Soil layer containing the seed.
        leaf_num: Cumulative leaf number (simplified tracking).
        lnph: Target leaf number at terminal spikelet.

        # Backward-compat aliases (kept so old code reading these doesn't break)
        sumdtt: Alias for within-stage thermal time accumulation (approx).
        cumdtt: Alias for cumdu.
        sumdtt_vp: Set to 0 (no longer used internally).
        ppfac: Alias for df.
        ndas: Days since sowing (germination failure check).
    """

    istage: int = 7
    xstage: float = 0.1
    # New Cropsim-CERES state variables
    cumdu: float = 0.0
    rstage: float = 0.0
    cumgeu: float = 0.0
    gestage: float = 0.0
    cumvd: float = 0.0
    vrnstage: float = 0.0
    vf: float = 1.0
    df: float = 1.0
    dtt: float = 0.0
    stgdoy: np.ndarray = field(
        default_factory=lambda: np.full(20, 9999999, dtype=int)
    )
    mdate: int = -99
    isdate: int = 0
    yremrg: int = -99
    gpp: float = 0.0
    crop_status: int = 0
    l0: int = 0
    leaf_num: float = 0.0
    lnph: float = 7.0
    ndas: float = 0.0

    # Backward-compat aliases
    @property
    def cumdtt(self) -> float:
        """Backward-compat alias for cumdu."""
        return self.cumdu

    @property
    def sumdtt(self) -> float:
        """Approximate within-stage progress for old growth-module callers."""
        return self.cumdu

    @property
    def sumdtt_vp(self) -> float:
        """No longer used; returns 0."""
        return 0.0

    @property
    def ppfac(self) -> float:
        """Backward-compat alias for df (photoperiod factor)."""
        return self.df


# ---------------------------------------------------------------------------
# WheatPhenology
# ---------------------------------------------------------------------------

class WheatPhenology:
    """CERES-Wheat phenology sub-model (Cropsim-CERES architecture).

    Tracks germination/emergence (Gstages) and post-emergence
    development (Rstages) separately, then derives ISTAGE/XSTAGE.

    Args:
        cultivar: Cultivar and ecotype coefficients.
        pltpop: Plant population (plants m⁻²).
        sdepth: Sowing depth (cm).
        yrsim: Simulation start date (YYYYDDD).
    """

    def __init__(
        self,
        cultivar: WheatCultivar,
        pltpop: float,
        sdepth: float,
        yrsim: int,
    ) -> None:
        self.cultivar = cultivar
        self.pltpop = pltpop
        self.sdepth = sdepth
        self.state = WheatPhenologyState()
        self.state.stgdoy[13] = yrsim

        # Build PTH array from phase durations PD(0..10)
        # PD(0) = 0 (no pre-emergence DU accumulation),
        # PD(1..5) from cultivar, PD(6..10) default 0
        self._pd = self._build_pd(cultivar)
        self._pth = self._build_pth(self._pd)

    # ------------------------------------------------------------------
    def _build_pd(self, cv: WheatCultivar) -> List[float]:
        """Build PD array [0..10] from cultivar phase durations.

        PD(0) = 0 (pre-emergence phase, handled by GESTAGE).
        PD(1) = cv.p1 (emergence → terminal spikelet).
        PD(2) = cv.p2 (TS → end leaf growth).
        PD(3) = cv.p3 (end leaf → end spike).
        PD(4) = cv.p4 (end spike → end lag).
        PD(5) = cv.p5 (grain filling, from SPE default P6=200 but cultivar p5).
        PD(6..10) = 0 (not used for wheat).
        """
        pd = [0.0] * 11
        pd[0] = 0.0       # P0 — pre-emergence (handled by Gstage system)
        pd[1] = cv.p1     # emergence → TS
        pd[2] = cv.p2     # TS → end leaf growth
        pd[3] = cv.p3     # end leaf → end spike
        pd[4] = cv.p4     # end spike → end grain fill lag
        pd[5] = cv.p5     # grain filling
        pd[6] = 200.0     # P6 from SPE default (post-maturity, not critical)
        # 7–10 unused
        return pd

    def _build_pth(self, pd: List[float]) -> List[float]:
        """Build PTH cumulative phase threshold array.

        PTH(0) = PD(0)
        PTH(L) = PTH(L-1) + max(0, PD(L))  for L = 1..10
        """
        pth = [0.0] * 11
        pth[0] = pd[0]
        for L in range(1, 11):
            pth[L] = pth[L - 1] + max(0.0, pd[L])
        return pth

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
        iswwat: str,
    ) -> None:
        """Run one phenology time step.

        Args:
            dynamic: Simulation phase (RUNINIT, SEASINIT, RATE/INTEGR).
            yrdoy: Current date (YYYYDDD).
            tmax: Max temperature (°C).
            tmin: Min temperature (°C).
            dayl: Day length (h).
            sw: Soil water content by layer (cm³ cm⁻³).
            ll: Lower limit by layer (cm³ cm⁻³).
            dlayr: Layer thickness (cm).
            nlayr: Number of active layers.
            iswwat: Water switch (``"Y"``/``"N"``).
        """
        if dynamic in (RUNINIT, SEASINIT):
            self._init(yrdoy)
        else:
            self._rate_integr(
                yrdoy, tmax, tmin, dayl, sw, ll, dlayr, nlayr, iswwat
            )

    # ------------------------------------------------------------------
    def _init(self, yrsim: int) -> None:
        """Reset state for a new season."""
        s = self.state
        cv = self.cultivar

        s.istage = 7
        s.xstage = 0.1
        s.cumdu = 0.0
        s.rstage = 0.0
        s.cumgeu = 0.0
        s.gestage = 0.0
        s.cumvd = 0.0
        s.vrnstage = 0.0
        # VF initial: spring types (P1V=0) → VF=1.0; winter types → 1-veff
        if cv.p1v <= 0.0:
            s.vf = 1.0
        else:
            s.vf = max(0.0, 1.0 - cv.veff)
        s.df = cv.ppfpe   # pre-emergence DF = PPFPE
        s.dtt = 0.0
        s.stgdoy[:] = 9999999
        s.stgdoy[13] = yrsim
        s.mdate = -99
        s.isdate = 0
        s.yremrg = -99
        s.gpp = 0.0
        s.crop_status = 0
        s.l0 = 0
        s.leaf_num = 0.0
        s.lnph = max(4.0, 7.0 - cv.p1v * 0.5)
        s.ndas = 0.0

        # Rebuild PTH in case cultivar changed
        self._pd = self._build_pd(cv)
        self._pth = self._build_pth(self._pd)

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
        iswwat: str,
    ) -> None:
        """Daily rate and integration for phenology (Cropsim-CERES)."""
        s = self.state
        cv = self.cultivar

        # ---- Stage 7: Sowing day — advance to stage 8 immediately
        if s.istage == 7:
            s.stgdoy[7 - 1] = yrdoy
            s.ndas = 0.0
            s.istage = 8
            s.xstage = 8.0
            # Find soil layer containing seed
            if iswwat != "N":
                cumdep = 0.0
                for L in range(nlayr):
                    cumdep += dlayr[L]
                    if self.sdepth < cumdep:
                        s.l0 = L
                        break
            return

        # Skip computation after maturity
        if s.mdate > 0 and s.mdate < yrdoy:
            return

        tmean = (tmax + tmin) * 0.5

        # ---- Thermal time: TT using TRDV1 cardinals
        s.dtt = compute_wheat_tt(cv.trdv1, tmean)

        # ---- Germination thermal time: TTGEM using TRGEM cardinals
        ttgem = compute_wheat_ttgem(cv.trgem, tmean)

        # ---- Devernalization (before CUMVD update)
        # VDLOST = 0.5*(TMAX-30) when TMAX>30 AND CUMVD<10
        vdlost = 0.0
        if tmax > 30.0 and s.cumvd < 10.0:
            vdlost = 0.5 * (tmax - 30.0)

        # ---- Vernalization update (active during pre-anthesis: ISTAGE>=7 or <=1)
        if s.istage >= 7 or s.istage <= 1:
            tfv = compute_tfv(cv.trvrn, tmean)
            s.cumvd = max(0.0, s.cumvd + tfv - vdlost)
            if cv.p1v > 0.0:
                s.vrnstage = max(0.0, min(1.0, s.cumvd / cv.p1v))
                s.vf = max(0.0, (1.0 - cv.veff) + cv.veff * s.vrnstage)
            else:
                s.vf = 1.0
                s.vrnstage = 1.0
        elif 1 < s.istage < 7:
            s.vf = 1.0

        # ---- Photoperiod factor DF
        if s.istage >= 7:
            s.df = cv.ppfpe
        elif s.istage >= 1:
            if cv.p1d >= 0.0:
                # Long-day plant
                s.df = max(0.0, min(1.0,
                    1.0 - (cv.p1d / 10000.0) * (cv.p1dt - dayl) ** 2
                ))
            else:
                # Short-day plant
                s.df = max(0.0, min(1.0,
                    1.0 - (abs(cv.p1d) / 1000.0) * (dayl - cv.p1dt)
                ))

        # PPEND: once RSTAGE > ppend, DF is fixed at 1
        if s.rstage > cv.ppend and s.istage < 7:
            s.df = 1.0

        # ---- Developmental units (LIF2 = 1 for wheat)
        du = s.dtt * s.vf * s.df
        du = max(0.0, du)

        # ---- Water factor for germination
        wfge = 1.0
        if iswwat != "N" and s.istage > 7:
            l0 = s.l0
            swp = np.zeros(nlayr + 1)
            cumdep = 0.0
            for L in range(nlayr):
                cumdep += dlayr[L]
                sw_frac = max(0.0, min(1.0,
                    (sw[L] - ll[L]) / max(0.3 - ll[L], 0.01)
                ))
                swp[L + 1] = sw_frac
            # SWP(0) at soil surface (extrapolate)
            if nlayr >= 2:
                swp[0] = min(1.0, max(0.0, swp[1] - 0.5 * (swp[2] - swp[1])))
            elif nlayr >= 1:
                swp[0] = swp[1]
            if l0 > 0:
                swpsd = swp[l0]
            else:
                if nlayr >= 2:
                    swpsd = swp[0] + (self.sdepth / max(dlayr[0], 0.01)) * (swp[2] - swp[0])
                else:
                    swpsd = swp[0]
            wfge = max(0.0, min(1.0, swpsd / max(cv.wfgeu, 0.01)))

        # ---- Germination units
        geu = ttgem * wfge

        # ---- Integrate GEU (only when ISTAGE > 7, i.e., seed is in ground)
        if s.istage > 7:
            s.cumgeu += geu

        # ---- Update GESTAGE
        pegd = cv.pgerm   # PEGD = PGERM when no dormancy (PLMAGE >= 0)
        pemrg = cv.pemrg
        if s.cumgeu < pegd:
            s.gestage = min(1.0, s.cumgeu / max(pegd, 1e-6) * 0.5)
        else:
            s.gestage = min(1.0,
                0.5 + 0.5 * (s.cumgeu - pegd) / max(pemrg * self.sdepth, 1e-6)
            )

        # ---- Integrate DU → CUMDU → RSTAGE (only after emergence, ISTAGE 1-6)
        if 1 <= s.istage <= 6:
            s.cumdu += du

            # Compute RSTAGE from PTH array
            pth = self._pth
            pd = self._pd
            if pd[0] > 0.0 and s.cumdu < pth[0]:
                s.rstage = s.cumdu / pd[0]
            else:
                for L in range(6, 0, -1):
                    if s.cumdu >= pth[L - 1]:
                        denom = pd[L] if pd[L] > 0.0 else 1.0
                        s.rstage = float(L) + (s.cumdu - pth[L - 1]) / denom
                        s.rstage = min(6.9, s.rstage)
                        break

        # ---- ISTAGE / XSTAGE transitions (from CER_Integrate.for lines 495-513)
        if s.istage == 8:
            s.xstage = 8.0 + s.gestage * 2.0
            if s.gestage >= 0.5:
                s.istage = 9
                s.xstage = 9.0 + (s.gestage - 0.5) * 2.0
        elif s.istage == 9:
            s.xstage = 9.0 + (s.gestage - 0.5) * 2.0
            if s.gestage >= 1.0:
                s.istage = 1
                s.xstage = 1.0
                s.stgdoy[9 - 1] = yrdoy
                s.yremrg = yrdoy
        elif 1 <= s.istage <= 6:
            prev_istage = s.istage
            s.istage = int(s.rstage)
            s.xstage = min(6.9, s.rstage)
            # Record transition dates
            for stage in range(max(prev_istage, 1), s.istage + 1):
                if stage in range(1, 7):
                    idx = stage - 1
                    if s.stgdoy[idx] == 9999999:
                        s.stgdoy[idx] = yrdoy
            # Maturity check
            if s.istage >= 6:
                s.istage = 6
                s.xstage = min(6.9, s.rstage)
                if s.mdate < 0:
                    s.mdate = yrdoy
                    s.crop_status = 1

        # ---- Leaf number accumulation (simplified, for growth module)
        if 1 <= s.istage <= 2:
            s.leaf_num += s.dtt / max(cv.phint, 1.0)

        # ---- ndas tracking (for germination failure check)
        if s.istage == 8:
            s.ndas += 1.0
            if s.ndas >= 40.0 and iswwat != "N":
                # Germination failure
                s.istage = 6
                s.mdate = yrdoy
                s.crop_status = 12

        # ---- Record specific stage dates
        if s.istage >= 3 and s.isdate == 0:
            # isdate is set when RSTAGE passes stage 4 (spike growth end)
            # Use stage 4 as a proxy for anthesis
            if s.rstage >= 4.0:
                s.isdate = yrdoy
