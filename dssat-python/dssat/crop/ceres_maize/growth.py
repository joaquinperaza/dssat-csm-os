"""CERES-Maize biomass growth, leaf area, and organ partitioning.

Translated from ``MZ_GROSUB.for``.

The photosynthesis–partitioning model:

1. Daily gross photosynthesis is computed from intercepted PAR and RUE,
   reduced by temperature, water, N, P and K stress factors.
2. Carbon is partitioned to leaves, stems, ears/grain and roots according
   to the current growth stage.
3. Grain filling (stage 5) uses the G3 kernel-filling rate with a
   temperature-response function.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from dssat.core.constants import NL, RUNINIT, SEASINIT, INTEGR, RATE
from dssat.crop.ceres_maize.phenology import MaizeCultivar, PhenologyState

# Default CO2 and temperature response tables.
# These are ONLY used when the cultivar was NOT loaded from the SPE file via
# MaizeCultivar.from_file().  Values here are placeholders — actual values
# MUST come from MZCER048.SPE (MaizeSpeParams).
_CO2X_DEFAULT = np.array([0.0, 220.0, 280.0, 330.0, 400.0,
                           490.0, 570.0, 750.0, 990.0, 9999.0])
_CO2Y_DEFAULT = np.array([0.00, 0.85, 0.95, 1.00, 1.02,
                           1.04, 1.05, 1.06, 1.07, 1.08])
_PRFTC_DEFAULT  = (6.2, 16.5, 33.0, 44.0)   # from MZCER048.SPE PRFTC
_RGFILC_DEFAULT = (5.5, 16.0, 27.0, 35.0)   # from MZCER048.SPE RGFIL


def _tabex(xarr: np.ndarray, yarr: np.ndarray, x: float) -> float:
    """1-D linear interpolation from a look-up table.

    Args:
        xarr: X values (must be monotone increasing).
        yarr: Y values corresponding to *xarr*.
        x: Query value.

    Returns:
        Interpolated Y value.
    """
    if x <= xarr[0]:
        return yarr[0]
    if x >= xarr[-1]:
        return yarr[-1]
    idx = int(np.searchsorted(xarr, x)) - 1
    frac = (x - xarr[idx]) / (xarr[idx + 1] - xarr[idx])
    return yarr[idx] + frac * (yarr[idx + 1] - yarr[idx])


def _curv_lin(t1: float, t2: float, t3: float, t4: float, x: float) -> float:
    """CERES trapezoidal temperature response (Fortran ``CURV('LIN', ...)``.

    Matches the 4-point trapezoidal shape used throughout the Fortran CERES
    source (MZ_GROSUB.for, SG_GROSUB.for, etc.).

    Args:
        t1: Base temperature — response is 0 below this.
        t2: Lower optimum — response reaches 1.0 here (linear rise from t1).
        t3: Upper optimum — response stays 1.0 until here.
        t4: Ceiling temperature — response drops back to 0 here (linear fall).
        x:  Query temperature.

    Returns:
        Response factor in [0, 1].
    """
    if x <= t1:
        return 0.0
    if x <= t2:
        return (x - t1) / (t2 - t1)
    if x <= t3:
        return 1.0
    if x <= t4:
        return 1.0 - (x - t3) / (t4 - t3)
    return 0.0


@dataclass
class GrowthState:
    """Mutable state for the CERES-Maize growth sub-model.

    All biomass variables are in g plant⁻¹ unless noted.

    Attributes:
        lfwt: Leaf dry weight (g plant⁻¹).
        stmwt: Stem dry weight (g plant⁻¹).
        rtwt: Root dry weight (g plant⁻¹).
        earwt: Ear (cob + grain) dry weight (g plant⁻¹).
        grnwt: Grain dry weight (g plant⁻¹).
        seedrv: Seed carbohydrate reserve available for seedling (g plant⁻¹).
        pla: Total plant leaf area (cm² plant⁻¹).
        lai: Leaf area index (m² m⁻²).
        xlai: Same as *lai* (exported name matching Fortran).
        xhlai: Half-LAI for light interception calculation.
        carbo: Daily assimilate produced (g plant⁻¹ d⁻¹).
        grolf: Leaf growth rate (g plant⁻¹ d⁻¹).
        grostm: Stem growth rate (g plant⁻¹ d⁻¹).
        grort: Root growth rate (g plant⁻¹ d⁻¹).
        grogrn: Grain growth rate (g plant⁻¹ d⁻¹).
        seedno: Grain number per plant.
        gpsm: Grain yield (g m⁻²).
        yield_: Grain yield (kg ha⁻¹).
        swfac: Photosynthesis water stress factor (0–1).
        turfac: Turgor (expansion) water stress factor (0–1).
        nstres: N stress factor for photosynthesis (0–1).
        agefac: Leaf age (N) stress factor (0–1).
        satfac: Waterlogging (oxygen) stress factor (0–1).
        canht: Canopy height (m).
        tanc: Total above-ground N concentration (g N g⁻¹ DM).
        rootn: Root N (g N plant⁻¹).
        stovn: Stover N (g N plant⁻¹).
        wtnup: Cumulative N uptake (g N plant⁻¹).
        topwt: Total above-ground dry weight (g m⁻²).
        stovwt: Stover (leaf+stem) dry weight (g plant⁻¹).
        biomas: Total above-ground biomass (g m⁻²).
        cumph: Cumulative phyllochron units.
        vstage: Vegetative stage (leaf number).
        rstage: Reproductive stage.
        senesce_wt: Senesced tissue weight (kg ha⁻¹ d⁻¹).
        slan: Leaf area senescence per plant (cm² plant⁻¹).
    """

    lfwt: float = 0.0
    stmwt: float = 0.0
    rtwt: float = 0.0
    earwt: float = 0.0
    grnwt: float = 0.0
    seedrv: float = 0.0
    pla: float = 0.0
    lai: float = 0.0
    xlai: float = 0.0
    xhlai: float = 0.0
    carbo: float = 0.0
    grolf: float = 0.0
    grostm: float = 0.0
    grort: float = 0.0
    grogrn: float = 0.0
    seedno: float = 0.0
    gpsm: float = 0.0
    yield_: float = 0.0
    swfac: float = 1.0
    turfac: float = 1.0
    nstres: float = 1.0
    agefac: float = 1.0
    satfac: float = 0.0
    canht: float = 0.0
    tanc: float = 0.0
    tmnc: float = 0.0
    ranc: float = 0.0
    rmnc: float = 0.0
    rootn: float = 0.0
    stovn: float = 0.0
    wtnup: float = 0.0
    topwt: float = 0.0
    stovwt: float = 0.0
    biomas: float = 0.0
    cumph: float = 0.0
    vstage: float = 0.0
    rstage: int = 0
    senesce_wt: float = 0.0
    slan: float = 0.0
    cumdtteg: float = 0.0
    stg2cls: float = 0.0
    sump: float = 0.0
    swmin: float = 0.0
    swmax: float = 0.0
    prft: float = 1.0
    pcnl: float = 0.0
    pcnst: float = 0.0
    pcnrt: float = 0.0
    pcngrn: float = 0.0
    wtnlf: float = 0.0
    wtnst: float = 0.0
    wtnsd: float = 0.0
    wtncan: float = 0.0
    tss: np.ndarray = field(default_factory=lambda: np.zeros(NL))


class MaizeGrowth:
    """CERES-Maize biomass growth and partitioning.

    Args:
        cultivar: Cultivar and ecotype coefficients.
        pltpop: Plant population (plants m⁻²).
        rowspc: Row spacing (cm).
        slpf: Soil fertility factor (0–1).
    """

    def __init__(
        self,
        cultivar: MaizeCultivar,
        pltpop: float,
        rowspc: float = 75.0,
        slpf: float = 1.0,
    ) -> None:
        self.cultivar = cultivar
        self.pltpop = pltpop
        self.rowspc = rowspc
        self.slpf = slpf
        self.state = GrowthState()

        # Load SPE species constants from file when available (via from_file),
        # otherwise fall back to the SPE defaults declared at module level.
        spe = getattr(getattr(cultivar, "_params", None), "spe", None)

        self._parsr   = spe.parsr   if spe else 0.50
        self._pormin  = spe.porm    if spe else 0.05
        self._rlwr    = spe.rlwr    if spe else 0.98
        self._rwumx   = spe.rwumx   if spe else 0.03
        self._rwuep1  = spe.rwuep1  if spe else 1.50
        self._sdsz    = spe.sdsz    if spe else 0.275
        self._rsgr    = spe.rsgr    if spe else 0.10
        self._rsgrt   = spe.rsgrt   if spe else 5.0
        self._tance   = spe.tance   if spe else 0.0440
        self._rance   = spe.rance   if spe else 0.0220
        self._tmnco   = spe.tmnc    if spe else 0.0045
        self._rmnco   = spe.rcnp    if spe else 0.0106
        self._fslfw   = spe.fslfw   if spe else 0.050
        self._fslfn   = spe.fslfn   if spe else 0.050

        # Temperature response tuples (tb, opt_low, opt_high, tm)
        self._prftc  = tuple(spe.prftc)  if spe else _PRFTC_DEFAULT
        self._rgfilc = tuple(spe.rgfil)  if spe else _RGFILC_DEFAULT

        # CO2 response table
        if spe and len(spe.co2x) >= 2:
            self._co2x = np.array(spe.co2x, dtype=float)
            self._co2y = np.array(spe.co2y, dtype=float)
        else:
            self._co2x = _CO2X_DEFAULT.copy()
            self._co2y = _CO2Y_DEFAULT.copy()

        self._rowspc = rowspc
        self._lifac  = 0.0  # light extinction factor (computed at run-time)

    # ------------------------------------------------------------------
    def run(
        self,
        dynamic: int,
        pheno: PhenologyState,
        yrdoy: int,
        tmax: float,
        tmin: float,
        srad: float,
        co2: float,
        sw: np.ndarray,
        dul: np.ndarray,
        sat: np.ndarray,
        ll: np.ndarray,
        dlayr: np.ndarray,
        nlayr: int,
        nh4: np.ndarray,
        no3: np.ndarray,
        kg2ppm: np.ndarray,
        shf: np.ndarray,
        rlv: np.ndarray,
        rtdep: float,
        trwup: float,
        eop: float,
        iswwat: str,
        iswnit: str,
    ) -> None:
        """Run one growth time step.

        Args:
            dynamic: Simulation phase.
            pheno: Current phenology state (read-only reference).
            yrdoy: Current date (YYYYDDD).
            tmax: Max temperature (°C).
            tmin: Min temperature (°C).
            srad: Solar radiation (MJ m⁻² d⁻¹).
            co2: CO₂ concentration (ppm).
            sw: Soil water content by layer.
            dul: Drained upper limit by layer.
            sat: Saturation by layer.
            ll: Lower limit by layer.
            dlayr: Layer thickness (cm).
            nlayr: Number of active layers.
            nh4: Ammonium by layer (µg g⁻¹).
            no3: Nitrate by layer (µg g⁻¹).
            kg2ppm: kg ha⁻¹ to ppm conversion by layer.
            shf: Soil hospitality factor by layer.
            rlv: Root length density by layer (cm root cm⁻³ soil).
            rtdep: Rooting depth (cm).
            trwup: Potential root water uptake (cm d⁻¹).
            eop: Potential transpiration (mm d⁻¹).
            iswwat: Water switch.
            iswnit: Nitrogen switch.
        """
        if dynamic in (RUNINIT, SEASINIT):
            self._init(pheno)
            return

        if dynamic == INTEGR:
            # Growth state is fully updated in the RATE phase.  The INTEGR
            # phase only handles root growth (in the crop model coordinator).
            return

        if pheno.mdate == yrdoy:
            return

        if pheno.istage in (7, 8):
            return

        g = self.state
        cv = self.cultivar

        tempm = (tmax + tmin) * 0.5

        # ---- critical N concentrations
        if pheno.xstage < 4.0:
            tmnc = (1.25 - 0.200 * pheno.xstage) / 100.0
        else:
            tmnc = self._tmnco
        g.tmnc = tmnc
        tcnp = math.exp(1.52 - 0.160 * pheno.xstage) / 100.0

        # ---- N stress factors
        if iswnit != "N" and pheno.istage < 7:
            g.agefac, _, g.nstres = _n_stress_factors(g.tanc, tcnp, tmnc)
        else:
            g.agefac = 1.0
            g.nstres = 1.0

        # ---- water stress factors
        g.swfac = 1.0
        g.turfac = 1.0
        if iswwat != "N":
            if eop > 0.0:
                ep1 = eop * 0.1
                if trwup / ep1 < self._rwuep1:
                    g.turfac = (1.0 / self._rwuep1) * trwup / ep1
                if ep1 >= trwup:
                    g.swfac = trwup / ep1
        # Truncate turfac to 3 decimal places (matching Fortran INT(TURFAC*1000)/1000)
        g.turfac = int(g.turfac * 1000) / 1000.0

        # ---- waterlogging stress
        g.satfac = _saturation_factor(
            sw, sat, dul, dlayr, nlayr, rlv, self._pormin, g.tss
        )

        # ---- photosynthesis
        lifac = 1.5 - 0.768 * ((self._rowspc * 0.01) ** 2 * self.pltpop) ** 0.1
        pco2 = _tabex(self._co2x, self._co2y, co2)
        par = srad * self._parsr
        if self.pltpop > 0.0:
            ipar = par / self.pltpop * (1.0 - math.exp(-lifac * g.lai))
        else:
            ipar = 0.0
        pcarb = ipar * cv.rue * pco2

        # temperature response on photosynthesis
        tavgd = 0.25 * tmin + 0.75 * tmax
        prft = _curv_lin(*self._prftc, tavgd)
        prft = min(max(prft, 0.0), 1.0)
        g.prft = prft

        # gross assimilation (g plant⁻¹ d⁻¹)
        g.carbo = (
            pcarb
            * min(prft, g.swfac, g.nstres)
            * self.slpf
        )
        g.carbo = max(g.carbo, 0.0)

        # ---- organ growth by stage
        self._partition(pheno, tempm)

        # ---- update totals
        g.stovwt = g.lfwt + g.stmwt
        g.topwt = (g.stovwt + g.earwt) * self.pltpop  # g plant⁻¹ × plant m⁻² = g m⁻²
        g.biomas = g.topwt
        g.lai = self.pltpop * g.pla * 1.0e-4
        g.xlai = g.lai
        g.xhlai = g.lai / 2.0

        # ---- yield
        g.gpsm = g.grnwt * self.pltpop  # g m⁻²
        g.yield_ = g.gpsm * 10.0  # g m⁻² × (10000 m²/ha ÷ 1000 g/kg) = kg ha⁻¹

        # ---- N accumulation in organs
        g.pcnl = g.tanc
        g.pcnst = g.tanc * 0.8
        g.pcnrt = g.ranc
        if g.grnwt > 0.0:
            g.pcngrn = g.wtnsd / g.grnwt / self.pltpop / 10.0 if self.pltpop > 0 else 0.0
        g.wtnlf = g.lfwt * g.pcnl * self.pltpop * 10.0
        g.wtnst = g.stmwt * g.pcnst * self.pltpop * 10.0
        g.wtncan = g.wtnlf + g.wtnst + g.wtnsd
        g.wtnup = g.wtncan

    # ------------------------------------------------------------------
    def _init(self, pheno: PhenologyState) -> None:
        """Initialise growth state for a new season."""
        g = self.state
        cv = self.cultivar
        g.lfwt = self._sdsz * 0.20
        g.stmwt = 0.0
        g.rtwt = self._sdsz * 0.20
        g.earwt = 0.0
        g.grnwt = 0.0
        g.seedrv = self._sdsz * 0.60
        g.pla = g.lfwt ** 0.8 * 267.0
        g.lai = 0.0
        g.xlai = 0.0
        g.xhlai = 0.0
        g.carbo = 0.0
        g.seedno = 0.0
        g.gpsm = 0.0
        g.yield_ = 0.0
        g.swfac = 1.0
        g.turfac = 1.0
        g.nstres = 1.0
        g.agefac = 1.0
        g.satfac = 0.0
        g.canht = 0.0
        g.tanc = self._tance
        g.ranc = self._rance
        g.tmnc = self._tmnco
        g.rmnc = self._rmnco
        g.rootn = g.ranc * g.rtwt
        g.stovn = g.lfwt * g.tanc
        g.wtnup = 0.0
        g.slan = 0.0
        g.cumdtteg = 0.0
        g.stg2cls = 0.0
        g.sump = 0.0
        g.tss[:] = 0.0
        g.cumph = 0.514

    # ------------------------------------------------------------------
    def _partition(self, pheno: PhenologyState, tempm: float) -> None:
        """Partition daily assimilate to organs by growth stage."""
        g = self.state
        cv = self.cultivar
        dtt = pheno.dtt
        sumdtt = pheno.sumdtt
        istage = pheno.istage
        xstage = pheno.xstage
        cumph = g.cumph  # Accumulated in growth state (MZ_GROSUB)
        xnti = pheno.xnti
        tlno = pheno.tlno

        if self.pltpop <= 0.01:
            return

        # Phyllochron counter (TI) — matches MZ_GROSUB.for TI/CUMPH logic
        if 1 <= istage <= 3:
            pc = 1.0
            if cumph < 5.0:
                pc = 0.66 + 0.068 * cumph
            ti = dtt / (cv.phint * pc)
            xn = cumph + 1.0
            # Accumulate phyllochron units (MZ_GROSUB.for: CUMPH = CUMPH + TI)
            g.cumph += ti
        else:
            ti = 0.0
            xn = cumph + 1.0

        # ---- Stage 1: Emergence → End juvenile
        if istage == 1:
            if xn < 4.0:
                plag = 4.0 * xn * ti * min(g.turfac, 1.0 - g.satfac)
            else:
                plag = 3.0 * xn * xn * ti * min(g.turfac, 1.0 - g.satfac)
            g.pla += plag
            xlfwt = (g.pla / 250.0) ** 1.25
            xlfwt = max(xlfwt, g.lfwt)
            grolf = xlfwt - g.lfwt
            grort = g.carbo - grolf
            if grort <= 0.25 * g.carbo:
                grort = g.carbo * 0.25
                g.seedrv += g.carbo - grolf - grort
                if g.seedrv <= 0.0:
                    g.seedrv = 0.0
                    grolf = g.carbo * 0.75
                    g.pla = (g.lfwt + grolf) ** 0.8 * 267.0
            g.lfwt += grolf
            g.slan = sumdtt * g.pla / 10000.0
            g.lfwt -= g.slan / 600.0
            g.grolf = grolf
            g.grort = grort
            g.grostm = 0.0

        # ---- Stage 2: End juvenile → Tassel initiation
        elif istage == 2:
            plag = 3.5 * xn * xn * ti * min(g.agefac, g.turfac, 1.0 - g.satfac)
            g.pla += plag
            xlfwt = (g.pla / 267.0) ** 1.25
            grolf = xlfwt - g.lfwt
            if grolf >= g.carbo * 0.75:
                grolf = g.carbo * 0.75
                g.pla = (g.lfwt + grolf) ** 0.8 * 267.0
            grort = g.carbo - grolf
            g.lfwt += grolf
            g.slan = sumdtt * g.pla / 10000.0
            g.lfwt -= g.slan / 600.0
            g.stg2cls = g.slan / 600.0 * self.pltpop * 10.0
            g.grolf = grolf
            g.grort = grort
            g.grostm = 0.0

        # ---- Stage 3: Tassel init → Silking
        elif istage == 3:
            # Ear growth starts at P3 - BSGDD
            bsgdd = 250.0
            asgdd = 100.0
            p3 = pheno.p3
            if sumdtt >= p3 - bsgdd:
                g.cumdtteg = sumdtt - (p3 - bsgdd)
                groear = (0.81 / (1.0 + math.exp(-0.02 * (g.cumdtteg - 210.0)))) * g.carbo
            else:
                g.cumdtteg = 0.0
                groear = 0.0

            if xn < 12.0:
                plag = 3.5 * xn * xn * ti * min(g.agefac, g.turfac, 1.0 - g.satfac)
                grolf = 0.00116 * plag * g.pla ** 0.25
                grostm = grolf * 0.0182 * (xn - xnti) ** 2
            elif xn < tlno - 2.9999:
                plag = 3.5 * 170.0 * ti * min(g.agefac, g.turfac, 1.0 - g.satfac)
                grolf = 0.00116 * plag * g.pla ** 0.25
                grostm = grolf * 0.0182 * (xn - xnti) ** 2
            else:
                plag = (
                    170.0 * 3.5 / (xn + 3.0 - tlno) ** 0.5
                    * ti
                    * min(g.agefac, g.turfac, 1.0 - g.satfac)
                )
                grolf = 0.00116 * plag * g.pla ** 0.25
                grostm = 3.000 * 3.1 * ti * min(g.agefac, g.turfac, 1.0 - g.satfac)

            if grostm > groear:
                grostm -= groear
            else:
                groear = grostm * 0.5
                grostm *= 0.5

            grort = g.carbo - grolf - grostm - groear
            if grort <= 0.10 * g.carbo and g.turfac > 0.0:
                if grolf > 0.0 or grostm > 0.0:
                    grf = g.carbo * 0.90 / (grostm + grolf + groear)
                    grort = g.carbo * 0.10
                else:
                    grf = 1.0
                grolf *= grf
                grostm *= grf
                groear *= grf

            g.pla = (g.lfwt + grolf) ** 0.8 * 267.0
            g.lfwt += grolf
            g.slan = g.pla / 1000.0
            g.lfwt -= g.slan / 600.0
            g.stmwt += grostm
            g.earwt += groear
            g.grolf = grolf
            g.grort = grort
            g.grostm = grostm

        # ---- Stage 4: Silking → Start effective grain fill
        elif istage == 4:
            g.cumdtteg += dtt
            groear = (
                0.81 / (1.0 + math.exp(-0.02 * (g.cumdtteg - 210.0)))
                * g.carbo
                * min(g.agefac, g.turfac, 1.0 - g.satfac)
            )
            grort = g.carbo * 0.08
            if g.carbo > groear + grort:
                grostm = g.carbo - groear - grort
            else:
                grort = (g.carbo - groear) * 0.5
                grostm = grort
            g.slan = g.pla * (0.05 + sumdtt / 200.0 * 0.05)
            g.lfwt -= g.slan / 600.0
            g.earwt += groear
            g.stmwt += grostm
            g.sump += g.carbo
            g.grolf = 0.0
            g.grort = grort
            g.grostm = grostm

        # ---- Stage 5: Effective grain filling
        elif istage == 5:
            g.slan = g.pla * (0.1 + 0.60 * (sumdtt / cv.p5) ** 3)
            # Temperature response for grain fill
            rgfill = _curv_lin(*self._rgfilc, tempm)
            rgfill = min(max(rgfill, 0.0), 1.0)
            if g.seedno <= 0.0:
                g.seedno = pheno.gpp
            g.grogrn = rgfill * pheno.gpp * cv.g3 * 0.001 * (0.45 + 0.55 * g.swfac)
            g.grnwt += g.grogrn
            g.grolf = 0.0
            g.grort = 0.0
            g.grostm = 0.0
            g.senesce_wt = g.slan / 600.0 * self.pltpop * 10.0

        # ---- Stage 6: Post-maturity (no more growth)
        elif istage == 6:
            g.carbo = 0.0
            g.grolf = 0.0
            g.grort = 0.0
            g.grostm = 0.0
            g.grogrn = 0.0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _n_stress_factors(
    tanc: float, tcnp: float, tmnc: float
) -> tuple[float, float, float]:
    """Compute N stress factors for leaf age, N demand, and photosynthesis.

    Implements ``MZ_NFACTO`` logic.

    Args:
        tanc: Current above-ground N concentration (g N g⁻¹ DM).
        tcnp: Critical N concentration (g N g⁻¹ DM).
        tmnc: Minimum N concentration (g N g⁻¹ DM).

    Returns:
        Tuple ``(agefac, ndef3, nstres)``:

        - *agefac* — N age factor for leaf area expansion (0–1).
        - *ndef3* — N stress factor for expansion growth (0–1).
        - *nstres* — N stress factor for photosynthesis (0–1).
    """
    if tcnp <= tmnc:
        return 1.0, 1.0, 1.0
    ratio = min(max((tanc - tmnc) / (tcnp - tmnc), 0.0), 1.0)
    agefac = ratio
    ndef3 = ratio
    nstres = min(ratio, 1.0)
    return agefac, ndef3, nstres


def _saturation_factor(
    sw: np.ndarray,
    sat: np.ndarray,
    dul: np.ndarray,
    dlayr: np.ndarray,
    nlayr: int,
    rlv: np.ndarray,
    pormin: float,
    tss: np.ndarray,
) -> float:
    """Compute root oxygen-deficiency stress factor.

    Args:
        sw: Soil water content by layer.
        sat: Saturation by layer.
        dul: Drained upper limit by layer.
        dlayr: Layer thickness (cm).
        nlayr: Number of active layers.
        rlv: Root length density by layer.
        pormin: Minimum air porosity for aerobic roots.
        tss: Days each layer has been saturated above *pormin*.

    Returns:
        Saturation stress factor (0 = none, 1 = maximum stress).
    """
    sumex = 0.0
    sumrl = 0.0
    for L in range(nlayr):
        if (sat[L] - sw[L]) >= pormin:
            tss[L] = 0.0
        else:
            tss[L] += 1.0
        if tss[L] > 2.0:
            swexf = (sat[L] - sw[L]) / pormin
            swexf = min(max(swexf, 0.0), 1.0)
        else:
            swexf = 1.0
        sumex += dlayr[L] * rlv[L] * (1.0 - swexf)
        sumrl += dlayr[L] * rlv[L]
    return min(max(sumex / sumrl, 0.0), 1.0) if sumrl > 0 else 0.0
