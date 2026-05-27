"""CROPGRO growth — photosynthesis, C partitioning, organ growth, senescence.

Translated from ``PHOTO.for``, ``GROW.for``, and ``VEGGR.for`` in the
DSSAT-CSM CROPGRO module.

The core daily cycle is:

1. Compute daily gross photosynthesis (``_compute_pg``).
2. Partition assimilate to leaf, stem, root, shell and seed organs according
   to the current R stage (``_partition``).
3. Update LAI, biomass totals, and senescence (``_update_lai``).

All biomass variables are g plant^-1 unless noted.

References:
    Boote, K.J., Jones, J.W., & Pickering, N.B. (1996). Potential uses and
    limitations of crop models. Agron. J. 88, 704–716.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dssat.core.constants import NL, RUNINIT, SEASINIT, RATE, INTEGR
from dssat.crop.cropgro.phenology import CropGROPhenoState, curv_tp, curv_lin


# ---------------------------------------------------------------------------
# CO2 response look-up table (matches CERES approach for consistency)
# ---------------------------------------------------------------------------
_CO2X = np.array(
    [0.0, 220.0, 330.0, 440.0, 550.0, 660.0, 770.0, 990.0, 1100.0, 1650.0]
)
_CO2Y = np.array(
    [0.0, 0.71, 1.00, 1.08, 1.17, 1.25, 1.32, 1.43, 1.50, 1.50]
)


def _tabex(xarr: np.ndarray, yarr: np.ndarray, x: float) -> float:
    """1-D piecewise-linear interpolation from a look-up table.

    Args:
        xarr: Monotone-increasing X values.
        yarr: Corresponding Y values.
        x:    Query value.

    Returns:
        Interpolated Y.
    """
    if x <= xarr[0]:
        return float(yarr[0])
    if x >= xarr[-1]:
        return float(yarr[-1])
    idx = int(np.searchsorted(xarr, x)) - 1
    frac = (x - xarr[idx]) / (xarr[idx + 1] - xarr[idx])
    return float(yarr[idx] + frac * (yarr[idx + 1] - yarr[idx]))


# ---------------------------------------------------------------------------
# Photosynthesis helper functions (shared by soybean and canola)
# ---------------------------------------------------------------------------

def compute_pg(
    srad: float,
    tday: float,
    co2: float,
    xhlai: float,
    swfac: float,
    nstres: float,
    rnitp: float,
    kcan: float,
    slpf: float,
    phtmax: float,
    parmax: float,
    pglfmx: float,
    lnref: float,
    fnpgt: tuple,
    fnpgn: tuple,
    cceff: float = 0.50,
    ccmax: float = 0.90,
    ccmp: float = 60.0,
) -> tuple[float, float]:
    """Compute daily gross photosynthesis (g CH2O m^-2 d^-1).

    Implements the canopy photosynthesis sub-model from ``PHOTO.for``.

    Args:
        srad:    Solar radiation (MJ m^-2 d^-1).
        tday:    Mean daytime temperature (°C).
        co2:     CO2 concentration (ppm).
        xhlai:   Half-LAI for light interception (m^2 m^-2).
        swfac:   Photosynthesis water stress factor (0–1).
        nstres:  N stress factor for photosynthesis (0–1).
        rnitp:   Leaf N concentration (g N g^-1 DM).
        kcan:    Light extinction coefficient.
        slpf:    Soil fertility factor (0–1).
        phtmax:  Maximum canopy photosynthesis at light saturation
                 (g CH2O m^-2 d^-1).
        parmax:  PAR at which PG = 63% of PHTMAX (MJ m^-2 d^-1).
        pglfmx:  Genetic adjustment for leaf maximum photosynthesis.
        lnref:   Reference leaf N concentration for age factor.
        fnpgt:   4-element tuple (tb, to1, to2, tm) for temperature response.
        fnpgn:   4-element tuple (xmin, xlow, xhigh, xmax) for N response
                 in a LIN curve: curv_lin(0, fnpgn[0], fnpgn[1], 1.0, rnitp).
        cceff:   CO2 carboxylation efficiency (μmol m^-2 s^-1 / ppm).
        ccmax:   Maximum CO2 assimilation rate (g CH2O m^-2 d^-1).
        ccmp:    CO2 compensation point (ppm).

    Returns:
        Tuple ``(pg, agefac)``:
        - pg     — Gross photosynthesis (g CH2O m^-2 d^-1).
        - agefac — Leaf-N age factor used for photosynthesis (0–1+).
    """
    # PAR = 0.5 * SRAD (standard CROPGRO/CERES fraction)
    par = srad * 0.5

    # Light-saturation curve (Thornley–Johnson rectangular hyperbola variant)
    ptsmax = phtmax * (1.0 - math.exp(-par / max(parmax, 0.001)))

    # Canopy light interception factor
    pgfac = 1.0 - math.exp(-kcan * max(xhlai, 0.0))

    # Temperature response on photosynthesis
    tpgfac = curv_tp(fnpgt[0], fnpgt[1], fnpgt[2], fnpgt[3], tday)
    tpgfac = max(0.0, min(1.0, tpgfac))

    # Leaf-N age factor (relative to reference N concentration)
    # Uses a LIN curve: 0 at fnpgn[0], 1.0 at fnpgn[1]
    agefac_raw = curv_lin(0.0, fnpgn[0], fnpgn[1], 1.0, rnitp)
    ageref_raw = curv_lin(0.0, fnpgn[0], fnpgn[1], 1.0, lnref)
    if ageref_raw > 0.0:
        agefac = agefac_raw / ageref_raw
    else:
        agefac = 1.0
    # Canopy scaling of agefac to mimic leaf-level N effect
    agefcc = (1.0 - math.exp(-2.0 * agefac)) / max(
        1.0 - math.exp(-2.0), 1e-9
    )

    # CO2 response — Michaelis-Menten type from PHOTO.for lines 159-161:
    #   CCK = CCEFF / CCMAX
    #   A0  = -CCMAX * (1 - EXP(-CCK * CCMP))
    #   PRATIO = A0 + CCMAX * (1 - EXP(-CCK * CO2))
    # Note: PRATIO is absolute (not relative), so we normalize to CO2=330 ppm
    # to get a dimensionless multiplier, as the Python code uses it multiplicatively.
    cck = cceff / max(ccmax, 1e-9)
    a0 = -ccmax * (1.0 - math.exp(-cck * ccmp))
    pratio_co2 = a0 + ccmax * (1.0 - math.exp(-cck * co2))
    pratio_ref = a0 + ccmax * (1.0 - math.exp(-cck * 330.0))
    pratio = pratio_co2 / max(pratio_ref, 1e-9)

    # Daily gross photosynthesis (g CH2O m^-2 d^-1)
    pg = (
        ptsmax
        * slpf
        * pgfac
        * tpgfac
        * agefcc
        * pratio
        * pglfmx
        * swfac
    )
    pg = max(pg, 0.0)
    return pg, agefac


# ---------------------------------------------------------------------------
# State dataclass
# ---------------------------------------------------------------------------

@dataclass
class CropGROGrowthState:
    """Mutable growth state for the CROPGRO model.

    All biomass fields are g plant^-1 unless noted.

    Attributes:
        wlf:     Leaf dry weight (g plant^-1).
        wst:     Stem dry weight (g plant^-1).
        wrt:     Root dry weight (g plant^-1).
        wsh:     Shell/pod-wall dry weight (g plant^-1).
        wsd:     Seed dry weight (g plant^-1).
        dwnod:   Nodule weight (g plant^-1) — soybean only.
        pla:     Plant leaf area (cm^2 plant^-1).
        lai:     Leaf area index (m^2 m^-2).
        xhlai:   Half-LAI for light interception.
        sla:     Specific leaf area (cm^2 g^-1).
        slaad:   SLA used in photosynthesis (accounts for age).
        rnitp:   Leaf N concentration driving photosynthesis (g N g^-1).
        pcnl:    Leaf N concentration (g N g^-1 DM).
        wtnup:   Cumulative N uptake (g N plant^-1).
        wtnfx:   Cumulative N fixed (g N plant^-1) — soybean only.
        seedno:  Grain/seed number per plant.
        biomas:  Total above-ground dry weight (g m^-2).
        yield_:  Grain yield (kg ha^-1).
        swfac:   Photosynthesis water stress factor (0–1).
        nstres:  N stress factor (0–1).
        agefac:  Leaf-N age factor from photosynthesis routine (0–1+).
        pg:      Daily gross photosynthesis (g CH2O m^-2 d^-1).
        growth:  Daily assimilate available for growth (g plant^-1 d^-1).
        grolf:   Leaf growth rate (g plant^-1 d^-1).
        grost:   Stem growth rate (g plant^-1 d^-1).
        grort:   Root growth rate (g plant^-1 d^-1).
        grosh:   Shell growth rate (g plant^-1 d^-1).
        grosd:   Seed growth rate (g plant^-1 d^-1).
        topwt:   Total above-ground weight (g m^-2).
        stovn:   Stover N content (g N plant^-1) — for N uptake logic.
        rootn:   Root N content (g N plant^-1).
    """

    wlf: float = 0.0
    wst: float = 0.0
    wrt: float = 0.0
    wsh: float = 0.0
    wsd: float = 0.0
    dwnod: float = 0.0
    pla: float = 0.0
    lai: float = 0.0
    xhlai: float = 0.0
    sla: float = 370.0
    slaad: float = 370.0
    rnitp: float = 0.045
    pcnl: float = 0.045
    wtnup: float = 0.0
    wtnfx: float = 0.0
    seedno: float = 0.0
    biomas: float = 0.0
    yield_: float = 0.0
    swfac: float = 1.0
    nstres: float = 1.0
    agefac: float = 1.0
    pg: float = 0.0
    growth: float = 0.0
    grolf: float = 0.0
    grost: float = 0.0
    grort: float = 0.0
    grosh: float = 0.0
    grosd: float = 0.0
    topwt: float = 0.0
    stovn: float = 0.0
    rootn: float = 0.0


# ---------------------------------------------------------------------------
# Growth sub-model
# ---------------------------------------------------------------------------

class CropGROGrowth:
    """CROPGRO biomass growth and organ partitioning.

    Args:
        cultivar:   Cultivar dataclass (SoybeanCultivar or CanolaCultivar).
        pltpop:     Plant population (plants m^-2).
        rowspc:     Row spacing (cm).
        slpf:       Soil fertility factor (0–1).
    """

    # Species-level constants (drawn from CROPGRO.SPE defaults for soybean)
    _PARSR: float = 0.50        # PAR fraction of total SRAD
    _PHTMAX: float = 70.0       # Maximum canopy PG (g CH2O m^-2 d^-1)
    _PARMAX: float = 3.50       # PAR at ~63% of PHTMAX (MJ m^-2 d^-1)
    _LNREF: float = 0.045       # Reference leaf N conc. (g N g^-1)
    _CCEFF: float = 0.50        # CO2 carboxylation efficiency
    _CCMAX: float = 0.90        # Max CO2 assimilation
    _CCMP: float = 60.0         # CO2 compensation point (ppm)
    _LMXSTD: float = 1.00       # Standard leaf max PG (mg CO2 dm^-2 s^-1)
    _PGREF: float = 1.00        # Reference leaf PG for adjusting PGLFMX
    # Temperature response curve for photosynthesis:
    # FNPGT = (tb, to1, to2, tm)
    _FNPGT: tuple = (7.0, 25.0, 35.0, 45.0)
    # Leaf-N response for photosynthesis:
    # FNPGN = (xmin, xhigh) — LIN curve: 0 at xmin, 1.0 at xhigh
    _FNPGN: tuple = (0.010, 0.045)
    # SLA table: not used directly — SLA comes from cultivar.slavr
    # Root parameters
    _RLWR: float = 0.98
    _RWUMX: float = 0.03
    _RWUEP1: float = 1.5
    # Initial seed reserve
    _SDSZ: float = 0.40         # Seed initial weight at emergence (g)
    # N concentrations
    _TANCE: float = 0.040       # Initial leaf N concentration
    _RANCE: float = 0.020       # Initial root N concentration
    _TMNCO: float = 0.012       # Minimum top N concentration
    _RMNCO: float = 0.008       # Minimum root N concentration
    # CH2O to dry matter conversion
    _CH2O_TO_DM: float = 1.43

    def __init__(
        self,
        cultivar: Any,
        pltpop: float,
        rowspc: float = 50.0,
        slpf: float = 1.0,
    ) -> None:
        self.cultivar = cultivar
        self.pltpop = pltpop
        self.rowspc = rowspc
        self.slpf = slpf
        self.state = CropGROGrowthState()
        # Allow cultivar to override species-level photosynthesis temperature
        # response curve (tb, to1, to2, tm).  CanolaCultivar supplies a cooler
        # optimum range than the soybean default.
        if hasattr(cultivar, "fnpgt") and cultivar.fnpgt is not None:
            self._FNPGT = tuple(cultivar.fnpgt)
        # Genetic photosynthesis adjustment factor
        if self._PGREF > 0.0:
            self._pglfmx = (
                (1.0 - math.exp(-1.6 * cultivar.lfmax))
                / max(1.0 - math.exp(-1.6 * self._PGREF), 1e-9)
            )
        else:
            self._pglfmx = 1.0

    # ------------------------------------------------------------------
    def run(
        self,
        dynamic: int,
        pheno: CropGROPhenoState,
        yrdoy: int,
        tmax: float,
        tmin: float,
        tday: float,
        srad: float,
        co2: float,
        sw: np.ndarray,
        dul: np.ndarray,
        ll: np.ndarray,
        dlayr: np.ndarray,
        nlayr: int,
        eop: float,
        trwup: float,
        iswwat: str,
        iswnit: str,
    ) -> None:
        """Run one growth time step.

        Args:
            dynamic:  Simulation phase code.
            pheno:    Current phenology state (read-only reference).
            yrdoy:    Current date (YYYYDDD).
            tmax:     Daily maximum temperature (°C).
            tmin:     Daily minimum temperature (°C).
            tday:     Mean daytime temperature (°C).
            srad:     Solar radiation (MJ m^-2 d^-1).
            co2:      CO2 concentration (ppm).
            sw:       Soil water by layer.
            dul:      Drained upper limit by layer.
            ll:       Lower limit by layer.
            dlayr:    Layer thickness (cm).
            nlayr:    Number of active layers.
            eop:      Potential plant transpiration (mm d^-1).
            trwup:    Actual root water uptake (cm d^-1).
            iswwat:   Water switch.
            iswnit:   Nitrogen switch.
        """
        if dynamic in (RUNINIT, SEASINIT):
            self._init()
            return

        if not pheno.emerged:
            return

        if pheno.mdate > 0 and pheno.mdate <= yrdoy:
            return

        g = self.state
        cv = self.cultivar

        # ---- Water stress factors
        # EOP (potential transpiration) is for the whole canopy, but at
        # emergence the canopy intercepts only a small fraction.
        # Scale the transpiration demand by canopy Beer's law fraction so
        # sparse seedlings are not immediately water-stressed.
        g.swfac = 1.0
        if iswwat != "N" and eop > 0.0:
            canopy_frac = 1.0 - math.exp(-cv.kcan * max(g.xhlai, 0.0))
            ep_canopy = eop * canopy_frac * 0.1  # mm → cm, scaled by canopy
            if ep_canopy > 1.0e-4:
                g.swfac = min(1.0, trwup / ep_canopy)
            # else: demand is negligible → no stress
        g.swfac = max(0.0, g.swfac)

        # ---- Compute photosynthesis
        pg, agefac = compute_pg(
            srad=srad,
            tday=tday,
            co2=co2,
            xhlai=g.xhlai,
            swfac=g.swfac,
            nstres=g.nstres,
            rnitp=g.rnitp,
            kcan=cv.kcan,
            slpf=self.slpf * cv.slpf,
            phtmax=self._PHTMAX,
            parmax=self._PARMAX,
            pglfmx=self._pglfmx,
            lnref=self._LNREF,
            fnpgt=self._FNPGT,
            fnpgn=self._FNPGN,
            cceff=self._CCEFF,
            ccmax=self._CCMAX,
            ccmp=self._CCMP,
        )
        g.pg = pg
        g.agefac = agefac

        # Convert CH2O to dry matter per plant (g plant^-1 d^-1)
        if self.pltpop > 0.0:
            g.growth = pg / self._CH2O_TO_DM / self.pltpop
        else:
            g.growth = 0.0

        # ---- Organ partitioning
        if dynamic == RATE:
            self._partition_rates(pheno)
        elif dynamic == INTEGR:
            self._integrate(pheno, tmax, tmin)

        # ---- Update LAI and biomass totals
        self._update_lai()
        self._update_n(iswnit)

    # ------------------------------------------------------------------
    def _init(self) -> None:
        """Initialise growth state for a new season."""
        g = self.state
        cv = self.cultivar
        # Start with small seed-derived leaf area and weight
        init_lf = self._SDSZ * 0.15
        init_rt = self._SDSZ * 0.10
        g.wlf = init_lf
        g.wst = self._SDSZ * 0.05
        g.wrt = init_rt
        g.wsh = 0.0
        g.wsd = 0.0
        g.dwnod = 0.0
        # Initial leaf area from SLA
        g.sla = cv.slavr
        g.slaad = cv.slavr
        g.pla = init_lf * g.sla
        g.lai = g.pla * self.pltpop * 1.0e-4
        g.xhlai = g.lai / 2.0
        g.rnitp = self._TANCE
        g.pcnl = self._TANCE
        g.wtnup = 0.0
        g.wtnfx = 0.0
        g.seedno = 0.0
        g.biomas = 0.0
        g.yield_ = 0.0
        g.swfac = 1.0
        g.nstres = 1.0
        g.agefac = 1.0
        g.pg = 0.0
        g.growth = 0.0
        g.grolf = 0.0
        g.grost = 0.0
        g.grort = 0.0
        g.grosh = 0.0
        g.grosd = 0.0
        g.topwt = 0.0
        g.stovn = g.wlf * self._TANCE
        g.rootn = g.wrt * self._RANCE

    # ------------------------------------------------------------------
    def _partition_rates(self, pheno: CropGROPhenoState) -> None:
        """Compute organ growth rates (g plant^-1 d^-1) without integrating.

        Called during RATE phase.  Results are stored in g.grolf etc. and
        consumed during INTEGR phase by ``_integrate``.
        """
        g = self.state
        rstage = pheno.rstage
        growth = max(g.growth, 0.0)

        # Stage-dependent partition fractions (from GROW.for logic)
        frlf, frstm, frrt, frshel, frsd = self._partition_fractions(
            rstage, pheno
        )

        g.grolf = growth * frlf
        g.grost = growth * frstm
        g.grort = growth * frrt
        g.grosh = growth * frshel
        g.grosd = growth * frsd

    # ------------------------------------------------------------------
    def _integrate(
        self, pheno: CropGROPhenoState, tmax: float, tmin: float
    ) -> None:
        """Integrate daily growth rates into organ weights.

        Called during INTEGR phase.
        """
        g = self.state
        cv = self.cultivar
        rstage = pheno.rstage

        # Accumulate organ weights
        g.wlf = max(0.0, g.wlf + g.grolf)
        g.wst = max(0.0, g.wst + g.grost)
        g.wrt = max(0.0, g.wrt + g.grort)
        g.wsh = max(0.0, g.wsh + g.grosh)
        g.wsd = max(0.0, g.wsd + g.grosd)

        # Leaf area from SLA (SLA declines slightly with age past R5)
        if rstage >= 5:
            # SLA decreases as canopy ages during seed fill
            sla_decline = g.sla * 0.005  # 0.5% per day
            g.sla = max(cv.slavr * 0.5, g.sla - sla_decline)
        else:
            g.sla = cv.slavr

        g.slaad = g.sla

        # Leaf area: accumulate via SLA from new leaf growth, senescence after R5
        new_leaf_area = g.grolf * g.sla
        g.pla = max(0.0, g.pla + new_leaf_area)

        # Leaf senescence — begins after R5 (seed fill)
        if rstage >= 5 and pheno.phthrs[6] > 0:
            # Senescence rate proportional to DXR57 (fraction of R5→R7 elapsed)
            senes_rate = pheno.dxr57 * g.pla * 0.03  # up to 3% per day at maturity
            g.pla = max(0.0, g.pla - senes_rate)
            # Leaf weight decreases accordingly
            leaf_loss = senes_rate / max(g.sla, 1.0)
            g.wlf = max(0.0, g.wlf - leaf_loss)

        # Seed number: initialised at R3 (pod set)
        if rstage >= 3 and g.seedno <= 0.0:
            g.seedno = cv.sdpdv * cv.wtpsd  # placeholder; will be refined
            g.seedno = max(1.0, g.seedno)

        # Grain yield
        g.yield_ = g.wsd * self.pltpop * 10.0  # g plant^-1 × plants m^-2 × 10 = kg ha^-1

    # ------------------------------------------------------------------
    def _partition_fractions(
        self, rstage: int, pheno: CropGROPhenoState
    ) -> tuple[float, float, float, float, float]:
        """Return (frlf, frstm, frrt, frshel, frsd) for the current stage.

        Based on the stage-dependent lookup in ``GROW.for``.

        Args:
            rstage:  Current integer R stage (0, 1, 3, 5, 7).
            pheno:   Phenology state (provides xpod for smooth transition).

        Returns:
            Five-tuple of partition fractions that sum to ≤ 1.0.
        """
        xpod = pheno.xpod  # 0 before R1, ramps to 1 by R3

        if rstage < 1:
            # Vegetative: leaves dominate (GROW.for: FRLF~0.55-0.65 early)
            frlf = 0.55
            frstm = 0.25
            frrt = 0.20
            frshel = 0.0
            frsd = 0.0
        elif rstage == 1:
            # Early reproductive: leaves decline, pod wall starts filling
            # Smooth ramp from vegetative to pod-fill partition
            frlf  = 0.40 * (1.0 - xpod) + 0.02 * xpod
            frstm = 0.25 * (1.0 - xpod) + 0.05 * xpod
            frrt  = 0.10 * (1.0 - xpod) + 0.03 * xpod
            frshel = 0.50 * xpod
            frsd = 0.0
        elif rstage == 3:
            # Pod fill: shell and seed fill together (PODS.for logic)
            frlf = 0.02
            frstm = 0.05
            frrt = 0.03
            frshel = 0.50
            frsd = 0.40
        elif rstage >= 5:
            # Rapid seed fill: nearly all assimilate to seed
            frlf = 0.0
            frstm = 0.0
            frrt = 0.0
            frshel = 0.15
            frsd = 0.85
        else:
            frlf = 0.55
            frstm = 0.25
            frrt = 0.20
            frshel = 0.0
            frsd = 0.0

        return frlf, frstm, frrt, frshel, frsd

    # ------------------------------------------------------------------
    def _update_lai(self) -> None:
        """Update LAI and related canopy variables from current leaf area."""
        g = self.state
        g.lai = g.pla * self.pltpop * 1.0e-4  # cm^2 plant^-1 * plants m^-2 → m^2 m^-2
        g.lai = max(0.0, g.lai)
        g.xhlai = g.lai / 2.0
        # Total above-ground biomass (g m^-2)
        g.topwt = (g.wlf + g.wst + g.wsh + g.wsd) * self.pltpop
        g.biomas = g.topwt

    # ------------------------------------------------------------------
    def _update_n(self, iswnit: str) -> None:
        """Update N-related state variables.

        Args:
            iswnit: Nitrogen switch ("Y"/"N").
        """
        g = self.state
        if iswnit == "N":
            g.rnitp = self._LNREF
            g.pcnl = self._TANCE
            return

        # Keep leaf N at a reasonable concentration
        # (full N model would call NUPTAK; here we track simply)
        total_wt = g.wlf + g.wst + g.wsh + g.wsd
        if total_wt > 0.0:
            g.pcnl = g.stovn / max(g.wlf, 0.001)
            g.pcnl = max(self._TMNCO, min(self._TANCE, g.pcnl))
        else:
            g.pcnl = self._TANCE
        g.rnitp = g.pcnl
