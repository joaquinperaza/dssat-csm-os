"""CERES-Wheat biomass growth, leaf area, and organ partitioning.

Adapted from ``CER_Growth.for`` / ``CSCER.for``.

The photosynthesis–partitioning framework is the same RUE-based approach
used in CERES-Maize, with wheat-specific parameters:

- Higher specific leaf area (SLA) than maize
- Grain number set from stem+leaf weight at anthesis using ``g1``
- Grain filling driven by ``g2`` (standard kernel weight) and ``p5`` duration

All biomass state variables are in g plant⁻¹ unless otherwise noted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from dssat.core.constants import NL, RUNINIT, SEASINIT, INTEGR, RATE
from dssat.crop.ceres_wheat.phenology import WheatCultivar, WheatPhenologyState

# Default CO2 and temperature response tables — used ONLY when cultivar
# was not loaded via WheatCultivar.from_file().  Values here are from
# WHCER048.SPE (CO2RF/CO2F table, TRPHS array for photosynthesis,
# TRGFW array for grain fill).
_CO2X_DEFAULT = np.array([0.0, 220.0, 330.0, 440.0, 550.0,
                           660.0, 770.0, 880.0, 990.0, 9999.0])
_CO2Y_DEFAULT = np.array([0.00, 0.71, 1.00, 1.08, 1.17,
                           1.25, 1.32, 1.38, 1.43, 1.50])
_PRFTC_DEFAULT  = (0.0,  5.0, 25.0, 35.0)   # WHCER048.SPE TRPHS
_RGFILC_DEFAULT = (0.0, 16.0, 35.0, 45.0)   # WHCER048.SPE TRGFW


def _tabex(xarr: np.ndarray, yarr: np.ndarray, x: float) -> float:
    """1-D linear interpolation from a look-up table."""
    if x <= xarr[0]:
        return yarr[0]
    if x >= xarr[-1]:
        return yarr[-1]
    idx = int(np.searchsorted(xarr, x)) - 1
    frac = (x - xarr[idx]) / (xarr[idx + 1] - xarr[idx])
    return yarr[idx] + frac * (yarr[idx + 1] - yarr[idx])


def _curv_lin(t1: float, t2: float, t3: float, t4: float, x: float) -> float:
    """CERES trapezoidal temperature response (Fortran ``CURV('LIN', ...)``.

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
class WheatGrowthState:
    """Mutable state for the CERES-Wheat growth sub-model.

    All biomass variables are in g plant⁻¹ unless noted.

    Attributes:
        lfwt: Leaf dry weight (g plant⁻¹).
        stmwt: Stem + spike dry weight (g plant⁻¹).
        rtwt: Root dry weight (g plant⁻¹).
        grnwt: Grain dry weight (g plant⁻¹).
        seedrv: Seed carbohydrate reserve for seedling (g plant⁻¹).
        pla: Total plant leaf area (cm² plant⁻¹).
        lai: Leaf area index (m² m⁻²).
        xlai: Same as lai (Fortran alias).
        xhlai: Half-LAI for light interception.
        carbo: Daily assimilate (g plant⁻¹ d⁻¹).
        grolf: Leaf growth rate (g plant⁻¹ d⁻¹).
        grostm: Stem growth rate (g plant⁻¹ d⁻¹).
        grort: Root growth rate (g plant⁻¹ d⁻¹).
        grogrn: Grain growth rate (g plant⁻¹ d⁻¹).
        seedno: Grain number per plant.
        gpsm: Grain yield (g m⁻²).
        yield_: Grain yield (kg ha⁻¹).
        swfac: Photosynthesis water stress factor (0–1).
        turfac: Turgor water stress factor (0–1).
        nstres: N stress for photosynthesis (0–1).
        agefac: Leaf age (N) stress factor (0–1).
        satfac: Waterlogging stress factor (0–1).
        canht: Canopy height (m).
        tanc: Total above-ground N concentration (g N g⁻¹ DM).
        ranc: Root N concentration (g N g⁻¹ DM).
        rootn: Root N content (g N plant⁻¹).
        stovn: Stover N content (g N plant⁻¹).
        wtnup: Cumulative N uptake (g N plant⁻¹).
        topwt: Total above-ground dry weight (g m⁻²).
        stovwt: Stover (leaf + stem) dry weight (g plant⁻¹).
        biomas: Total above-ground biomass (g m⁻²).
        senesce_wt: Senesced tissue (kg ha⁻¹ d⁻¹).
        slan: Leaf area senescence per plant (cm² plant⁻¹).
        prft: Photosynthesis temperature factor.
    """

    lfwt: float = 0.0
    stmwt: float = 0.0
    rtwt: float = 0.0
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
    senesce_wt: float = 0.0
    slan: float = 0.0
    prft: float = 1.0
    tss: np.ndarray = field(default_factory=lambda: np.zeros(NL))
    # N accounting
    pcnl: float = 0.0
    pcnst: float = 0.0
    pcnrt: float = 0.0
    pcngrn: float = 0.0
    wtnlf: float = 0.0
    wtnst: float = 0.0
    wtnsd: float = 0.0
    wtncan: float = 0.0


class WheatGrowth:
    """CERES-Wheat biomass growth and partitioning.

    Args:
        cultivar: Cultivar and ecotype coefficients.
        pltpop: Plant population (plants m⁻²).
        rowspc: Row spacing (cm).
        slpf: Soil fertility factor (0–1).
    """

    def __init__(
        self,
        cultivar: WheatCultivar,
        pltpop: float,
        rowspc: float = 20.0,
        slpf: float = 1.0,
    ) -> None:
        self.cultivar = cultivar
        self.pltpop = pltpop
        self.rowspc = rowspc
        self.slpf = slpf
        self.state = WheatGrowthState()

        # Load SPE species constants from file when available (via from_file),
        # otherwise fall back to the SPE defaults declared at module level.
        spe = getattr(getattr(cultivar, "_params", None), "spe", None)

        self._parsr  = spe.tpar    if spe else 0.07   # WHCER048.SPE TPAR
        self._rue    = cultivar.parue                 # RUE from ECO (via cultivar)
        self._kcan   = cultivar.kcan                  # KCAN from ECO (via cultivar)
        self._pormin = spe.rwupm   if spe else 0.02
        self._rlwr   = spe.rlwr    if spe else 0.98
        self._rwumx  = spe.rwumx   if spe else 0.03
        self._rwuep1 = 1.5                            # not in SPE; standard value
        self._sdsz   = spe.sdwt    if spe else 0.0284

        # Temperature response tuples (tb, opt1, opt2, tm)
        # TRPHS for photosynthesis, TRGFW for grain fill DW
        self._prftc  = tuple(spe.trphs) if spe else _PRFTC_DEFAULT
        self._rgfilc = tuple(spe.trgfw) if spe else _RGFILC_DEFAULT

        # CO2 response table
        if spe and len(spe.co2rf) >= 2:
            self._co2x = np.array(spe.co2rf, dtype=float)
            self._co2y = np.array(spe.co2f,  dtype=float)
        else:
            self._co2x = _CO2X_DEFAULT.copy()
            self._co2y = _CO2Y_DEFAULT.copy()

        # N concentration constants (initial values at emergence).
        # TANCE = above-ground N at emergence ≈ SDN% (seed N) from SPE.
        # RANCE = root N at emergence.
        # TMNCO = minimum plant N concentration (overridden per-timestep by
        #         TMNC = (2.97 - 0.455 * XSTAGE) / 100  from CER_Integrate.for).
        # RMNCO = minimum root N concentration.
        sdn_pct = spe.sdn_pct if spe else 1.9
        self._tance = sdn_pct / 100.0    # e.g. 0.019 for wheat (1.9% from SPE SDN%)
        self._rance = 0.020              # root N at emergence (~2%)
        self._tmnco = 0.0045             # minimum plant N (g N g⁻¹ DM)
        self._rmnco = 0.0045             # minimum root N (g N g⁻¹ DM)

    # ------------------------------------------------------------------
    def run(
        self,
        dynamic: int,
        pheno: WheatPhenologyState,
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

        if pheno.mdate == yrdoy:
            return

        if pheno.istage in (7, 8):
            return

        g = self.state
        cv = self.cultivar

        tempm = (tmax + tmin) * 0.5

        # ---- Critical N concentrations (decline with stage)
        if pheno.xstage < 4.0:
            tmnc = max((1.10 - 0.15 * pheno.xstage) / 100.0, self._tmnco)
        else:
            tmnc = self._tmnco
        g.tmnc = tmnc
        tcnp = math.exp(1.40 - 0.14 * pheno.xstage) / 100.0

        # ---- N stress factors
        if iswnit != "N" and pheno.istage < 7:
            g.agefac, _, g.nstres = _n_stress_factors(g.tanc, tcnp, tmnc)
        else:
            g.agefac = 1.0
            g.nstres = 1.0

        # ---- Water stress factors
        g.swfac = 1.0
        g.turfac = 1.0
        if iswwat != "N":
            if eop > 0.0:
                ep1 = eop * 0.1
                if trwup / ep1 < self._rwuep1:
                    g.turfac = (1.0 / self._rwuep1) * trwup / ep1
                if ep1 >= trwup:
                    g.swfac = trwup / ep1
        g.turfac = int(g.turfac * 1000) / 1000.0

        # ---- Waterlogging stress
        g.satfac = _saturation_factor(
            sw, sat, dul, dlayr, nlayr, rlv, self._pormin, g.tss
        )

        # ---- Photosynthesis
        lifac = 1.5 - 0.768 * ((self.rowspc * 0.01) ** 2 * self.pltpop) ** 0.1
        pco2 = _tabex(self._co2x, self._co2y, co2)
        par = srad * self._parsr
        if self.pltpop > 0.0:
            ipar = par / self.pltpop * (1.0 - math.exp(-lifac * g.lai))
        else:
            ipar = 0.0
        pcarb = ipar * self._rue * pco2

        # Temperature response on photosynthesis
        tavgd = 0.25 * tmin + 0.75 * tmax
        prft = _curv_lin(*self._prftc, tavgd)
        prft = min(max(prft, 0.0), 1.0)
        g.prft = prft

        # Gross assimilation (g plant⁻¹ d⁻¹)
        g.carbo = (
            pcarb
            * min(prft, g.swfac, g.nstres)
            * self.slpf
        )
        g.carbo = max(g.carbo, 0.0)

        # ---- Organ growth by stage
        self._partition(pheno, tempm)

        # ---- Update totals
        g.stovwt = g.lfwt + g.stmwt
        g.topwt = (g.stovwt + g.grnwt) * self.pltpop
        g.biomas = g.topwt
        g.lai = self.pltpop * g.pla * 1.0e-4
        g.xlai = g.lai
        g.xhlai = g.lai / 2.0

        # ---- Yield
        g.gpsm = g.grnwt * self.pltpop
        g.yield_ = g.gpsm * 10.0

        # ---- N accounting
        g.pcnl = g.tanc
        g.pcnst = g.tanc * 0.8
        g.pcnrt = g.ranc
        g.wtnlf = g.lfwt * g.pcnl * self.pltpop * 10.0
        g.wtnst = g.stmwt * g.pcnst * self.pltpop * 10.0
        g.wtncan = g.wtnlf + g.wtnst + g.wtnsd
        g.wtnup = g.wtncan

    # ------------------------------------------------------------------
    def _init(self, pheno: WheatPhenologyState) -> None:
        """Initialise growth state for a new season."""
        g = self.state
        g.lfwt = self._sdsz * 0.20
        g.stmwt = 0.0
        g.rtwt = self._sdsz * 0.20
        g.grnwt = 0.0
        g.seedrv = self._sdsz * 0.60
        g.pla = g.lfwt ** 0.8 * 300.0  # slightly larger initial SLA for wheat
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
        g.tss[:] = 0.0

    # ------------------------------------------------------------------
    def _partition(self, pheno: WheatPhenologyState, tempm: float) -> None:
        """Partition daily assimilate to organs by growth stage."""
        g = self.state
        cv = self.cultivar
        dtt = pheno.dtt
        sumdtt = pheno.sumdtt
        istage = pheno.istage
        xstage = pheno.xstage

        if self.pltpop <= 0.01:
            return

        # Specific leaf area for wheat: higher than maize, decreases with stage
        sla = max(100.0, 400.0 - 30.0 * xstage)  # cm² g⁻¹

        # ---- Stage 9: Emergence (seedling using reserves)
        if istage == 9:
            grolf = min(g.carbo * 0.8, g.seedrv * 0.1)
            grolf = max(grolf, 0.0)
            g.seedrv = max(0.0, g.seedrv - grolf)
            g.lfwt += grolf
            g.pla = g.lfwt * sla
            g.slan = 0.0
            g.grolf = grolf
            g.grort = g.carbo - grolf
            g.grostm = 0.0

        # ---- Stage 1: Emergence → Terminal spikelet
        elif istage == 1:
            # Leaf-dominant vegetative growth
            grolf = g.carbo * 0.75 * min(g.turfac, g.agefac, 1.0 - g.satfac)
            grolf = max(grolf, 0.0)
            grort = g.carbo - grolf
            g.lfwt += grolf
            g.pla = g.lfwt * sla
            # Slight senescence of oldest leaves
            g.slan = max(0.0, sumdtt * g.pla / 20000.0)
            g.lfwt -= g.slan / sla if sla > 0.0 else 0.0
            g.lfwt = max(g.lfwt, 0.001)
            g.grolf = grolf
            g.grort = grort
            g.grostm = 0.0

        # ---- Stage 2: Terminal spikelet → Heading
        elif istage == 2:
            # Leaf and stem growing together (stem elongation)
            grolf = g.carbo * 0.45 * min(g.turfac, g.agefac, 1.0 - g.satfac)
            grostm = g.carbo * 0.35 * min(g.turfac, g.agefac, 1.0 - g.satfac)
            grort = g.carbo - grolf - grostm
            g.lfwt += grolf
            g.stmwt += grostm
            g.pla = g.lfwt * sla
            g.slan = max(0.0, sumdtt * g.pla / 15000.0)
            g.lfwt -= g.slan / sla if sla > 0.0 else 0.0
            g.lfwt = max(g.lfwt, 0.001)
            g.grolf = grolf
            g.grort = grort
            g.grostm = grostm

        # ---- Stage 3: Heading → Anthesis
        elif istage == 3:
            # Stem (spike) fills, leaf growth minimal
            grostm = g.carbo * 0.70 * min(g.turfac, 1.0 - g.satfac)
            grolf = g.carbo * 0.10 * min(g.turfac, 1.0 - g.satfac)
            grort = g.carbo - grostm - grolf
            g.lfwt += grolf
            g.stmwt += grostm
            g.pla = g.lfwt * sla
            g.slan = g.pla / 2000.0
            g.lfwt -= g.slan / sla if sla > 0.0 else 0.0
            g.lfwt = max(g.lfwt, 0.001)
            g.grolf = grolf
            g.grort = grort
            g.grostm = grostm

        # ---- Stage 4: Anthesis → Beginning effective grain fill
        elif istage == 4:
            # Set grain number if not done yet (uses stem+leaf weight at anthesis)
            if g.seedno <= 0.0:
                gpp_est = cv.g1 * (g.stmwt + g.lfwt) * self.pltpop * 0.001
                g.seedno = max(50.0, min(500.0, gpp_est))
                # Pass to phenology state for use in grain fill
                pheno.gpp = g.seedno
            # Continue some stem filling, roots stop
            grostm = g.carbo * 0.50
            grort = g.carbo * 0.10
            g.stmwt += grostm
            g.grolf = 0.0
            g.grort = grort
            g.grostm = grostm
            g.grogrn = 0.0

        # ---- Stage 5: Effective grain filling
        elif istage == 5:
            # Temperature response for grain filling
            rgfill = _curv_lin(*self._rgfilc, tempm)
            rgfill = min(max(rgfill, 0.0), 1.0)
            if g.seedno <= 0.0:
                g.seedno = max(50.0, pheno.gpp)
            # Daily grain growth (g plant⁻¹ d⁻¹)
            # Based on g2 (mg/kernel), seedno, and p5 duration
            grort_fill = (
                cv.g2 * g.seedno * 0.001
                / max(cv.p5, 1.0)
                * dtt
                * rgfill
                * (0.45 + 0.55 * g.swfac)
            )
            g.grogrn = grort_fill
            g.grnwt += g.grogrn
            # Remobilize stem carbohydrates to grain
            remob = min(g.stmwt * 0.01, g.grogrn * 0.3)
            g.stmwt = max(0.0, g.stmwt - remob)
            # Leaf senescence accelerates in grain fill
            sla_s5 = max(50.0, sla - 50.0)
            g.slan = g.pla * (0.1 + 0.60 * (sumdtt / max(cv.p5, 1.0)) ** 3)
            g.lfwt -= g.slan / sla_s5 if sla_s5 > 0.0 else 0.0
            g.lfwt = max(g.lfwt, 0.0001)
            g.pla = g.lfwt * sla_s5
            g.grolf = 0.0
            g.grort = 0.0
            g.grostm = 0.0
            g.senesce_wt = g.slan / sla_s5 * self.pltpop * 10.0 if sla_s5 > 0.0 else 0.0

        # ---- Stage 6: Post-maturity
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
    """Compute N stress factors.

    Args:
        tanc: Current above-ground N concentration (g N g⁻¹ DM).
        tcnp: Critical N concentration (g N g⁻¹ DM).
        tmnc: Minimum N concentration (g N g⁻¹ DM).

    Returns:
        Tuple ``(agefac, ndef3, nstres)``.
    """
    if tcnp <= tmnc:
        return 1.0, 1.0, 1.0
    ratio = min(max((tanc - tmnc) / (tcnp - tmnc), 0.0), 1.0)
    return ratio, ratio, min(ratio, 1.0)


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
        tss: Days each layer has been saturated above pormin.

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
