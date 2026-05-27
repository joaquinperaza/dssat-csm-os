"""CERES-Millet biomass growth, leaf area, and organ partitioning.

Translated from ``ML_GROSUB.for``.

Uses the same RUE-based approach as CERES-Sorghum with millet-specific
parameters (smaller seeds, higher tbase, different SLA).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from dssat.core.constants import NL, RUNINIT, SEASINIT, INTEGR, RATE
from dssat.crop.ceres_millet.phenology import MilletCultivar, PhenologyState

# Default CO2 response table from MLCER048.SPE (CO2X / CO2Y)
_CO2X_DEFAULT = np.array([0.0, 220.0, 280.0, 330.0, 400.0, 490.0, 570.0, 750.0, 990.0, 9999.0])
_CO2Y_DEFAULT = np.array([0.00, 0.85,  0.95,  1.00,  1.02,  1.04,  1.05,  1.06,  1.07,  1.08])

# Default temperature responses from MLCER048.SPE (PRFTC / RGFIL)
_PRFTC_DEFAULT  = (11.0, 22.0, 35.0, 48.0)
_RGFILC_DEFAULT = (7.0,  22.0, 27.0, 60.0)


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
class MilletGrowthState:
    """Mutable state for the CERES-Millet growth sub-model.

    All biomass variables are in g plant⁻¹ unless noted.
    """

    lfwt: float = 0.0
    stmwt: float = 0.0
    rtwt: float = 0.0
    panwt: float = 0.0
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


# Alias for backward compatibility
GrowthState = MilletGrowthState


class MilletGrowth:
    """CERES-Millet biomass growth and partitioning.

    Args:
        cultivar: Cultivar and ecotype coefficients.
        pltpop: Plant population (plants m⁻²).
        rowspc: Row spacing (cm).
        slpf: Soil fertility factor (0–1).
    """

    def __init__(
        self,
        cultivar: MilletCultivar,
        pltpop: float,
        rowspc: float = 75.0,
        slpf: float = 1.0,
    ) -> None:
        self.cultivar = cultivar
        self.pltpop = pltpop
        self.rowspc = rowspc
        self.slpf = slpf
        self.state = MilletGrowthState()
        # Load SPE params if cultivar has file-loaded params attached
        spe = getattr(getattr(cultivar, "_params", None), "spe", None)
        eco = getattr(getattr(cultivar, "_params", None), "eco", None)
        # Species constants — from MLCER048.SPE when available, else defaults
        self._parsr  = spe.parsr  if spe else 0.5
        self._rowspc = rowspc
        self._tance  = 0.030
        self._rance  = 0.020
        self._tmnco  = 0.0045
        self._rmnco  = 0.0045
        self._lifac  = 0.0
        self._pormin = spe.porm   if spe else 0.02
        self._rlwr   = spe.rlwr   if spe else 0.98
        self._rwumx  = spe.rwumx  if spe else 0.03
        self._rwuep1 = spe.rwuep1 if spe else 1.5
        self._sdsz   = 0.008      # millet seed ~8 mg (not in SPE)
        self._prftc  = tuple(spe.prftc)  if spe else _PRFTC_DEFAULT
        self._rgfilc = tuple(spe.rgfil)  if spe else _RGFILC_DEFAULT
        self._co2x   = np.array(spe.co2x) if spe else _CO2X_DEFAULT
        self._co2y   = np.array(spe.co2y) if spe else _CO2Y_DEFAULT
        # Partitioning from ECO when available
        self._stpc = eco.stpc if (eco and hasattr(eco, "stpc")) else 0.10
        self._rtpc = eco.rtpc if (eco and hasattr(eco, "rtpc")) else 0.25

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
        """Run one growth time step."""
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

        tavgd = 0.25 * tmin + 0.75 * tmax
        prft = _curv_lin(*self._prftc, tavgd)
        prft = min(max(prft, 0.0), 1.0)
        g.prft = prft

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
        g.topwt = (g.stovwt + g.panwt) * self.pltpop
        g.biomas = g.topwt
        g.lai = self.pltpop * g.pla * 1.0e-4
        g.xlai = g.lai
        g.xhlai = g.lai / 2.0

        # ---- yield
        g.gpsm = g.grnwt * self.pltpop
        g.yield_ = g.gpsm * 10.0

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
        g.panwt = 0.0
        g.grnwt = 0.0
        g.seedrv = self._sdsz * 0.60
        # Millet: initial leaf area based on seed size
        g.pla = max(1.0, 6000.0 * self._sdsz)
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

        if self.pltpop <= 0.01:
            return

        # SLA for millet: similar to sorghum
        sla = max(200.0, 300.0 - 30.0 * xstage)

        # ---- Stage 1: Emergence → End juvenile
        if istage == 1:
            grolf = g.carbo * (1.0 - self._rtpc)
            grort = g.carbo * self._rtpc
            g.lfwt += grolf
            # Update leaf area from SLA
            g.pla = g.lfwt * sla
            g.slan = sumdtt * g.pla / 10000.0
            g.lfwt -= g.slan / 600.0
            g.grolf = grolf
            g.grort = grort
            g.grostm = 0.0

        # ---- Stage 2: End juvenile → Panicle initiation
        elif istage == 2:
            grolf = g.carbo * (1.0 - self._stpc - self._rtpc)
            grostm = g.carbo * self._stpc
            grort = g.carbo * self._rtpc
            g.lfwt += grolf
            g.stmwt += grostm
            g.pla = g.lfwt * sla
            g.slan = sumdtt * g.pla / 10000.0
            g.lfwt -= g.slan / 600.0
            g.grolf = grolf
            g.grort = grort
            g.grostm = grostm

        # ---- Stage 3: Panicle init → End flag leaf
        elif istage == 3:
            grolf = g.carbo * (1.0 - self._stpc - self._rtpc) * 0.5
            grostm = g.carbo * self._stpc
            grort = g.carbo * self._rtpc
            gropan = g.carbo * (1.0 - self._stpc - self._rtpc) * 0.5
            g.lfwt += grolf
            g.stmwt += grostm
            g.panwt += gropan
            g.pla = g.lfwt * sla
            g.slan = g.pla / 1000.0
            g.lfwt -= g.slan / 600.0
            g.grolf = grolf
            g.grort = grort
            g.grostm = grostm

        # ---- Stage 4: End flag leaf → Anthesis
        elif istage == 4:
            grort = g.carbo * self._rtpc
            grostm = g.carbo * self._stpc
            gropan = g.carbo * (1.0 - self._stpc - self._rtpc)
            g.panwt += gropan
            g.stmwt += grostm
            g.slan = g.pla * (0.05 + sumdtt / 200.0 * 0.05)
            g.lfwt -= g.slan / 600.0
            g.grolf = 0.0
            g.grort = grort
            g.grostm = grostm

        # ---- Stage 5: Effective grain filling
        elif istage == 5:
            g.slan = g.pla * (0.1 + 0.60 * (sumdtt / cv.p5) ** 3)
            rgfill = _curv_lin(*self._rgfilc, tempm)
            rgfill = min(max(rgfill, 0.0), 1.0)
            if g.seedno <= 0.0:
                g.seedno = pheno.gpp
            # Millet grain fill: g5 (kernel size, mg) * gpp * 0.001 / p5 * dtt
            g.grogrn = rgfill * pheno.gpp * cv.g5 * 0.001 / max(cv.p5, 1.0) * dtt
            g.grogrn *= (0.45 + 0.55 * g.swfac)
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
    """Compute N stress factors for leaf age, N demand, and photosynthesis."""
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
    """Compute root oxygen-deficiency stress factor."""
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
