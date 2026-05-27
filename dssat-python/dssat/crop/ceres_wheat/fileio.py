"""CERES-Wheat parameter file readers (WHCER048.CUL / .ECO / .SPE).

Reads the three DSSAT-CERES-Wheat genotype files exactly as the Fortran
routines in ``CER_Init.for`` / ``CSCER.for`` do via the CROPSIM
``CUREADR`` / ``ECREADR`` / ``SPREADR`` subroutines.

* ``WHCER048.CUL`` — cultivar coefficients (P1V, P1D, P5, G1, G2, G3, PHINT)
* ``WHCER048.ECO`` — ecotype coefficients (P1, P2FR1, P2, P3, P4FR1, P4FR2,
  P4, VEFF, PARUE, PARU2, PHL2, PHF3, LA1S, LAFV, LAFR, SLAS, LSPHS, LSPHE,
  TIL#S, TIPHE, TIFAC, TDPHS, TDPHE, TDFAC, RDGS, HTSTD, AWNS, KCAN, RS%S,
  GN%S, GN%MN, TKFH)
* ``WHCER048.SPE`` — species constants in ``@``-header + data rows format
  (PGERM, PEMRG, P0, P6, PPFPE, PPTHR, PPEND, root/leaf/stem/grain
  properties, CO2 table, temperature response arrays, water/N stress factors)

Parameter priority (as in Fortran ``CER_Init.for``):
  CUL value overrides ECO value when CUL value is non-zero / non-negative.
  If CUL value is zero/missing, the ECO value is used.

No values are invented here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dssat.io.converters.genotype_io import (
    read_genotype_row,
    parse_spe_csreads,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WheatCulParams:
    """Parameters from one row of ``WHCER048.CUL``.

    CUL format: ``A6 VAR#, 1X, A16 VAR-NAME, 1X, A5 EXP#, 1X, A6 ECO#,
    7F7.0`` (P1V, P1D, P5, G1, G2, G3, PHINT).

    The file may also contain override columns for ECO parameters
    (P1, P2, P3, P4, P2FR1, P4FR1, P4FR2, VEFF, PARUE, PARU2, LA1S, SLAS,
    LAFV, LAFR, RDGS, GN%S, GN%MN, NFPU, NFPL, NFGU, NFGL, RTNO3, RTNH4).
    When non-zero these override the matching ECO values (``CER_Init.for``
    lines 592–615 pattern: ``IF(PD(1).LE.0) CALL ECREADR(…)``).
    """

    varno:  str   = "DFAULT"
    vrname: str   = "Default Wheat"
    econo:  str   = "DFAULT"
    p1v:    float = 5.0      # Vernalization requirement (days at opt T)
    p1d:    float = 75.0     # Photoperiod sensitivity (% per 10 h drop)
    p5:     float = 450.0    # Grain fill thermal time (°C d)
    g1:     float = 30.0     # Kernel number per unit stem weight (#/g)
    g2:     float = 35.0     # Standard kernel weight (mg)
    g3:     float = 1.0      # Tiller death coefficient
    phint:  float = 60.0     # Phyllochron interval (°C d leaf⁻¹)

    # Optional CUL-level overrides for ECO parameters (0 = use ECO value)
    p1:     float = 0.0      # Phase 1 duration (PVTU)
    p2:     float = 0.0      # Phase 2 duration (TU)
    p2fr1:  float = 0.0      # Phase 2 fraction 1
    p3:     float = 0.0      # Phase 3 duration (TU)
    p4:     float = 0.0      # Phase 4 duration (TU)
    p4fr1:  float = 0.0
    p4fr2:  float = 0.0
    veff:   float = -99.0    # Vernalization effect (negative → use ECO)
    parue:  float = 0.0      # RUE before last leaf
    paru2:  float = -99.0    # RUE after last leaf (negative → use ECO)
    la1s:   float = 0.0      # Area of standard first leaf
    slas:   float = 0.0      # Specific leaf area
    lafv:   float = 0.0
    lafr:   float = 0.0
    rdgs:   float = -99.0
    gn_s:   float = -99.0    # Grain N standard (%)
    gn_mn:  float = -99.0    # Grain N minimum (%)


@dataclass
class WheatEcoParams:
    """Parameters from one row of ``WHCER048.ECO``.

    ECO columns (32 numeric fields after ECO#):
    P1, P2FR1, P2, P3, P4FR1, P4FR2, P4, VEFF, PARUE, PARU2,
    PHL2, PHF3, LA1S, LAFV, LAFR, SLAS, LSPHS, LSPHE, TIL#S, TIPHE,
    TIFAC, TDPHS, TDPHE, TDFAC, RDGS, HTSTD, AWNS, KCAN, RS%S, GN%S,
    GN%MN, TKFH.
    """

    econo:  str   = "DFAULT"
    p1:     float = 200.0    # Phase 1: TS → end leaf growth (PVTU)
    p2fr1:  float = 0.25     # Fraction of P2 to jointing
    p2:     float = 200.0    # Phase 2: TS → end leaf growth (TU)
    p3:     float = 200.0    # Phase 3: end leaf growth → end spike (TU)
    p4fr1:  float = 0.25     # Fraction of P4 to anthesis start
    p4fr2:  float = 0.10     # Fraction of P4 = anthesis duration
    p4:     float = 200.0    # Phase 4: end spike → end grain fill lag (TU)
    veff:   float = 0.6      # Vernalization effect (1=full, 0=none)
    parue:  float = 2.7      # RUE before last leaf (g MJ⁻¹ PAR)
    paru2:  float = 2.7      # RUE after last leaf (g MJ⁻¹ PAR)
    phl2:   float = 15.0     # Leaf # produced during phyllochron phase 2
    phf3:   float = 1.3      # PHINT factor for phase 3
    la1s:   float = 5.0      # Area of standard first leaf (cm²)
    lafv:   float = 0.10     # Leaf area increase per leaf, veg phase
    lafr:   float = 0.50     # Leaf area increase per leaf, repro phase
    slas:   float = 400.0    # Specific leaf area standard (cm² g⁻¹)
    lsphs:  float = 5.5      # Leaf senescence start stage
    lsphe:  float = 6.3      # Leaf senescence end stage
    til_s:  float = 3.5      # Tiller production starts (leaf #)
    tiphe:  float = 2.5      # Tillering phase end stage
    tifac:  float = 1.0      # Tiller initiation rate factor
    tdphs:  float = 2.5      # Tiller death start stage
    tdphe:  float = 6.0      # Tiller death end stage
    tdfac:  float = 4.0      # Tiller death factor
    rdgs:   float = 3.0      # Root depth growth rate (cm d⁻¹)
    htstd:  float = 100.0    # Standard canopy height (cm)
    awns:   float = 0.0      # Awn score (0–10)
    kcan:   float = 0.85     # Canopy PAR extinction coefficient
    rs_s:   float = 30.0     # Reserves to stem (%)
    gn_s:   float = 3.0      # Grain N standard (%)
    gn_mn:  float = 0.0      # Grain N minimum (%)
    tkfh:   float = -10.0    # Temperature at which killed when hardened (°C)


@dataclass
class WheatSpeParams:
    """Species-level constants from ``WHCER048.SPE`` (CSREADS @-header format).

    All values are read from the file.  Defaults listed here match the
    values in the actual ``WHCER048.SPE`` file.
    """

    # Phase durations
    pgerm:  float = 10.0    # Germination (hydro-thermal units)
    pemrg:  float = 8.0     # Emergence per cm seed depth (TU cm⁻¹)
    p0:     float = 0.0     # Juvenile phase (°C d)
    p6:     float = 200.0   # Post-maturity phase (°C d)

    # Photoperiod
    ppfpe:  float = 1.0     # Photoperiod factor pre-emergence
    ppthr:  float = 20.0    # Photoperiod threshold (h, above = no effect)
    ppend:  float = 2.0     # Photoperiod sensitivity end stage

    # Roots
    rlig_pct: float = 10.0  # Root lignin (%)
    rlwr:     float = 0.98  # Root length/weight ratio (cm g⁻¹ × 1e4)
    rsen:     float = 0.10  # Root senescence (%/d)
    rresp:    float = 0.40  # Root respiration fraction
    rldgr:    float = 500.0 # Root length/depth growth ratio

    # Leaves
    llig_pct: float = 10.0  # Leaf lignin (%)
    laxs:     float = 900.0 # Max leaf area on main stem (cm²)
    lshfr:    float = 0.33  # Leaf sheath fraction
    lshaw:    float = 70.0  # Leaf sheath area/weight ratio (cm² g⁻¹)
    phl1:     float = 2.0   # Leaf # during phyllochron phase 1
    phf1:     float = 0.5   # PHINT factor for phase 1
    slamn:    float = 0.5   # Min SLA fraction of non-stressed
    slacf:    float = 0.02  # SLA change with leaf position (per leaf)
    llife:    float = 4.0   # Leaf life (phyllochrons)
    lwlos:    float = 0.50  # Leaf DM loss at normal senescence (fraction)
    lrphs:    float = 3.0   # Stage after which dead leaves retained

    # Canopy
    tpar:   float = 0.07    # PAR transmission fraction
    tsrad:  float = 0.25    # Solar radiation transmission fraction

    # Tillers
    tgr02:  float = 0.80    # Growth rate of tiller 2 (rel. to main shoot)

    # Reserves
    rs_x:   float = 80.0    # Max reserve concentration (%)
    rsuse:  float = 0.10    # Reserve utilisation fraction per day

    # Stems
    slig_pct: float = 10.0  # Stem lignin (%)
    saws:     float = 25.0  # Stem area/weight ratio (cm² g⁻¹)
    sgphe:    float = 4.45  # Stem growth phase end stage
    ssphs:    float = 5.8   # Stem senescence phase start stage
    ssen_pct: float = 0.53  # Stem senescence (%/d)

    # Chaff
    chfr:   float = 0.65    # Fraction of assimilates to chaff
    chstg:  float = 3.8     # Chaff growth start stage

    # Grain
    glig_pct: float = 10.0  # Grain lignin (%)

    # Seed
    sdwt:   float = 0.0284  # Seed dry weight (g seed⁻¹)
    sdafr:  float = 0.50    # Seed reserves availability fraction per day

    # CO2 response table (ppm and relative response)
    co2rf:  Tuple[float, ...] = (0, 220, 330, 440, 550, 660, 770, 880, 990, 9999)
    co2f:   Tuple[float, ...] = (0.00, 0.71, 1.00, 1.08, 1.17, 1.25, 1.32, 1.38, 1.43, 1.50)

    # CH2O partitioning
    ptfmx:  float = 0.98    # Max partition fraction to tops

    # Temperature responses (4 values each: base, opt1, opt2, max °C)
    trgem:  Tuple[float, ...] = (1.0, 26.0, 50.0, 60.0)   # germination
    trdv1:  Tuple[float, ...] = (0.0, 26.0, 50.0, 60.0)   # development 1
    trdv2:  Tuple[float, ...] = (0.0, 30.0, 50.0, 60.0)   # development 2
    trlfg:  Tuple[float, ...] = (0.0, 10.0, 20.0, 35.0)   # leaf growth
    trphs:  Tuple[float, ...] = (0.0,  5.0, 25.0, 35.0)   # photosynthesis
    trvrn:  Tuple[float, ...] = (-5.0, 0.0,  7.0, 15.0)   # vernalization
    trhar:  Tuple[float, ...] = (-5.0, 0.0,  5.0, 10.0)   # hardening
    trgfw:  Tuple[float, ...] = (0.0, 16.0, 35.0, 45.0)   # grain fill DW
    trgfn:  Tuple[float, ...] = (0.0, 16.0, 35.0, 45.0)   # grain fill N

    # Water factors
    rwupm:  float = 0.02    # Soil saturation limit for root growth
    rwumx:  float = 0.03    # Max water uptake (cm³ cm⁻¹ root d⁻¹)
    wfpu:   float = 1.0     # Water stress, photosynthesis upper
    wfpgf:  float = 1.0     # Water stress, photosynthesis during grain fill
    wfgu:   float = 1.3     # Water stress, growth upper
    wftu:   float = 1.0     # Water stress, tillering upper
    wftl:   float = 0.5     # Water stress, tillering lower
    wfsu:   float = 0.6     # Water stress, senescence threshold
    wfgeu:  float = 0.50    # Germination upper water threshold
    wfrgu:  float = 0.25    # Root growth upper water threshold
    llosw:  float = 0.02    # Leaf area loss from water stress (fraction d⁻¹)

    # N uptake
    nh4mn:  float = 0.0     # Min NH4 for uptake (mg kg⁻¹)
    no3mn:  float = 0.0     # Min NO3 for uptake (mg kg⁻¹)
    rtno3:  float = 0.006   # NO3 uptake per root length (mg cm⁻¹ d⁻¹)
    rtnh4:  float = 0.006   # NH4 uptake per root length (mg cm⁻¹ d⁻¹)
    ntupf:  float = 0.05    # N top-up fraction per day

    # N concentrations (grain, leaf/stem/root tables)
    gn_mx:  float = 3.2     # Max grain N (%)
    sdn_pct:float = 1.9     # Seed N standard (%)

    # N stress factors
    nfpu:   float = 1.00    # N stress, photosynthesis upper
    nfpl:   float = 0.00    # N stress, photosynthesis lower
    nfgu:   float = 1.00    # N stress, growth upper
    nfgl:   float = 0.00    # N stress, growth lower
    nftu:   float = 1.00    # N stress, tillering upper
    nftl:   float = 0.00    # N stress, tillering lower
    nfsu:   float = 0.40    # N stress, senescence threshold
    nfsf:   float = 0.10    # N factor final senescence trigger
    llosn:  float = 0.02    # Leaf area loss from N stress (fraction d⁻¹)
    ncrg:   float = 30.0    # N concentration for max root growth (ppm)
    nlab_pct: float = 20.0  # Labile N during grain filling (%)

    # Cold hardiness
    tkuh:   float = -6.0    # Temp at which 50% kill, unhardened seedling (°C)
    hdur:   float = 10.0    # Days for complete cold hardening
    tklf:   float = -10.0   # Temp at which leaves start to be killed (°C)


@dataclass
class CeresWheatParams:
    """Combined parameters from WHCER048.SPE + .ECO + .CUL, with CUL overrides.

    Call :meth:`_apply_overrides` after construction (done automatically by
    :func:`load_ceres_wheat_params`) to apply the CUL-over-ECO priority logic
    from ``CER_Init.for``.
    """

    spe: WheatSpeParams = field(default_factory=WheatSpeParams)
    eco: WheatEcoParams = field(default_factory=WheatEcoParams)
    cul: WheatCulParams = field(default_factory=WheatCulParams)

    # Resolved final values (after CUL overrides ECO)
    p1:    float = 0.0
    p2fr1: float = 0.0
    p2:    float = 0.0
    p3:    float = 0.0
    p4fr1: float = 0.0
    p4fr2: float = 0.0
    p4:    float = 0.0
    veff:  float = 0.0
    parue: float = 0.0
    paru2: float = 0.0
    la1s:  float = 0.0
    slas:  float = 0.0
    lafv:  float = 0.0
    lafr:  float = 0.0
    rdgs:  float = 0.0
    gn_s:  float = 0.0
    gn_mn: float = 0.0
    kcan:  float = 0.0

    def _apply_overrides(self) -> None:
        """Apply CUL-over-ECO priority exactly as Fortran CER_Init.for does.

        Pattern: use CUL value if non-zero (or non-negative for signed
        fields); otherwise fall back to ECO value.
        """
        cul, eco = self.cul, self.eco

        def _pick(cul_val: float, eco_val: float, neg_ok: bool = False) -> float:
            """Return cul_val if it is considered 'set', else eco_val."""
            if neg_ok:
                # For fields that may be intentionally negative (e.g. VEFF),
                # CUL value of -99 means "not set".
                return cul_val if cul_val > -90.0 else eco_val
            return cul_val if cul_val > 0.0 else eco_val

        self.p1    = _pick(cul.p1,    eco.p1)
        self.p2fr1 = _pick(cul.p2fr1, eco.p2fr1)
        self.p2    = _pick(cul.p2,    eco.p2)
        self.p3    = _pick(cul.p3,    eco.p3)
        self.p4fr1 = _pick(cul.p4fr1, eco.p4fr1)
        self.p4fr2 = _pick(cul.p4fr2, eco.p4fr2)
        self.p4    = _pick(cul.p4,    eco.p4)
        self.veff  = _pick(cul.veff,  eco.veff,  neg_ok=True)
        self.parue = _pick(cul.parue, eco.parue)
        self.paru2 = _pick(cul.paru2, eco.paru2, neg_ok=True)
        self.la1s  = _pick(cul.la1s,  eco.la1s)
        self.slas  = _pick(cul.slas,  eco.slas)
        self.lafv  = _pick(cul.lafv,  eco.lafv)
        self.lafr  = _pick(cul.lafr,  eco.lafr)
        self.rdgs  = _pick(cul.rdgs,  eco.rdgs,  neg_ok=True)
        self.gn_s  = _pick(cul.gn_s,  eco.gn_s,  neg_ok=True)
        self.gn_mn = _pick(cul.gn_mn, eco.gn_mn, neg_ok=True)
        self.kcan  = eco.kcan   # KCAN comes only from ECO in wheat


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _parse_wh_cul(path: Path, varno: str) -> WheatCulParams:
    """Parse one cultivar row from ``WHCER048.CUL``."""
    row = read_genotype_row(
        path, varno,
        id_col="VAR#",
        fallback_id="DFAULT",
        vrname_col="VAR-NAME",
        econame_col=None,
    )

    cul = WheatCulParams()
    cul.varno  = str(row.get("VAR#", varno))
    cul.vrname = str(row.get("VAR-NAME", row.get("VRNAME", "")))

    # ECO# is the column after VRNAME (token after name field)
    cul.econo = str(row.get("ECO#", row.get("ECONO", "DFAULT")) or "DFAULT")

    def _f(key: str, default: float) -> float:
        v = row.get(key)
        return float(v) if v is not None else default

    cul.p1v   = _f("P1V",   cul.p1v)
    cul.p1d   = _f("P1D",   cul.p1d)
    cul.p5    = _f("P5",    cul.p5)
    cul.g1    = _f("G1",    cul.g1)
    cul.g2    = _f("G2",    cul.g2)
    cul.g3    = _f("G3",    cul.g3)
    cul.phint = _f("PHINT", cul.phint)

    # Optional override fields (zero / -99 = not set)
    cul.p1    = _f("P1",    0.0)
    cul.p2    = _f("P2",    0.0)
    cul.p2fr1 = _f("P2FR1", 0.0)
    cul.p3    = _f("P3",    0.0)
    cul.p4    = _f("P4",    0.0)
    cul.p4fr1 = _f("P4FR1", 0.0)
    cul.p4fr2 = _f("P4FR2", 0.0)
    cul.veff  = _f("VEFF",  -99.0)
    cul.parue = _f("PARUE", 0.0)
    cul.paru2 = _f("PARU2", -99.0)
    cul.la1s  = _f("LA1S",  0.0)
    cul.slas  = _f("SLAS",  0.0)
    cul.lafv  = _f("LAFV",  0.0)
    cul.lafr  = _f("LAFR",  0.0)
    cul.rdgs  = _f("RDGS",  -99.0)
    cul.gn_s  = _f("GN%S",  -99.0)
    cul.gn_mn = _f("GN%MN", -99.0)

    return cul


def _parse_wh_eco(path: Path, econo: str) -> WheatEcoParams:
    """Parse one ecotype row from ``WHCER048.ECO``."""
    row = read_genotype_row(
        path, econo,
        id_col="ECO#",
        fallback_id="DFAULT",
        vrname_col=None,
        econame_col=None,
    )

    eco = WheatEcoParams()
    eco.econo = str(row.get("ECO#", econo))

    def _f(key: str, default: float) -> float:
        v = row.get(key)
        return float(v) if v is not None else default

    eco.p1     = _f("P1",     eco.p1)
    eco.p2fr1  = _f("P2FR1",  eco.p2fr1)
    eco.p2     = _f("P2",     eco.p2)
    eco.p3     = _f("P3",     eco.p3)
    eco.p4fr1  = _f("P4FR1",  eco.p4fr1)
    eco.p4fr2  = _f("P4FR2",  eco.p4fr2)
    eco.p4     = _f("P4",     eco.p4)
    eco.veff   = _f("VEFF",   eco.veff)
    eco.parue  = _f("PARUE",  eco.parue)
    eco.paru2  = _f("PARU2",  eco.paru2)
    eco.phl2   = _f("PHL2",   eco.phl2)
    eco.phf3   = _f("PHF3",   eco.phf3)
    eco.la1s   = _f("LA1S",   eco.la1s)
    eco.lafv   = _f("LAFV",   eco.lafv)
    eco.lafr   = _f("LAFR",   eco.lafr)
    eco.slas   = _f("SLAS",   eco.slas)
    eco.lsphs  = _f("LSPHS",  eco.lsphs)
    eco.lsphe  = _f("LSPHE",  eco.lsphe)
    eco.til_s  = _f("TIL#S",  eco.til_s)
    eco.tiphe  = _f("TIPHE",  eco.tiphe)
    eco.tifac  = _f("TIFAC",  eco.tifac)
    eco.tdphs  = _f("TDPHS",  eco.tdphs)
    eco.tdphe  = _f("TDPHE",  eco.tdphe)
    eco.tdfac  = _f("TDFAC",  eco.tdfac)
    eco.rdgs   = _f("RDGS",   eco.rdgs)
    eco.htstd  = _f("HTSTD",  eco.htstd)
    eco.awns   = _f("AWNS",   eco.awns)
    eco.kcan   = _f("KCAN",   eco.kcan)
    eco.rs_s   = _f("RS%S",   eco.rs_s)
    eco.gn_s   = _f("GN%S",   eco.gn_s)
    eco.gn_mn  = _f("GN%MN",  eco.gn_mn)
    eco.tkfh   = _f("TKFH",   eco.tkfh)

    return eco


def _parse_wh_spe(path: Path) -> WheatSpeParams:
    """Parse ``WHCER048.SPE`` (@-header + data-rows CSREADS format)."""
    kv = parse_spe_csreads(path)

    spe = WheatSpeParams()

    def _s(key: str, default: float) -> float:
        v = kv.get(key)
        if v is None:
            return default
        return float(v) if not isinstance(v, list) else float(v[0])

    def _a(key: str, n: int, default: Tuple[float, ...]) -> Tuple[float, ...]:
        v = kv.get(key)
        if v is None:
            return default
        if isinstance(v, list):
            vals = [float(x) for x in v[:n]]
        else:
            vals = [float(v)]
        while len(vals) < n:
            vals.append(0.0)
        return tuple(vals[:n])

    spe.pgerm     = _s("PGERM",  spe.pgerm)
    spe.pemrg     = _s("PEMRG",  spe.pemrg)
    spe.p0        = _s("P0",     spe.p0)
    spe.p6        = _s("P6",     spe.p6)
    spe.ppfpe     = _s("PPFPE",  spe.ppfpe)
    spe.ppthr     = _s("PPTHR",  spe.ppthr)
    spe.ppend     = _s("PPEND",  spe.ppend)
    spe.rlig_pct  = _s("RLIG%",  spe.rlig_pct)
    spe.rlwr      = _s("RLWR",   spe.rlwr)
    spe.rsen      = _s("RSEN",   spe.rsen)
    spe.rresp     = _s("RRESP",  spe.rresp)
    spe.rldgr     = _s("RLDGR",  spe.rldgr)
    spe.llig_pct  = _s("LLIG%",  spe.llig_pct)
    spe.laxs      = _s("LAXS",   spe.laxs)
    spe.lshfr     = _s("LSHFR",  spe.lshfr)
    spe.lshaw     = _s("LSHAW",  spe.lshaw)
    spe.phl1      = _s("PHL1",   spe.phl1)
    spe.phf1      = _s("PHF1",   spe.phf1)
    spe.slamn     = _s("SLAMN",  spe.slamn)
    spe.slacf     = _s("SLACF",  spe.slacf)
    spe.llife     = _s("LLIFE",  spe.llife)
    spe.lwlos     = _s("LWLOS",  spe.lwlos)
    spe.lrphs     = _s("LRPHS",  spe.lrphs)
    spe.tpar      = _s("TPAR",   spe.tpar)
    spe.tsrad     = _s("TSRAD",  spe.tsrad)
    spe.tgr02     = _s("TGR02",  spe.tgr02)
    spe.rs_x      = _s("RS%X",   spe.rs_x)
    spe.rsuse     = _s("RSUSE",  spe.rsuse)
    spe.slig_pct  = _s("SLIG%",  spe.slig_pct)
    spe.saws      = _s("SAWS",   spe.saws)
    spe.sgphe     = _s("SGPHE",  spe.sgphe)
    spe.ssphs     = _s("SSPHS",  spe.ssphs)
    spe.ssen_pct  = _s("SSEN%",  spe.ssen_pct)
    spe.chfr      = _s("CHFR",   spe.chfr)
    spe.chstg     = _s("CHSTG",  spe.chstg)
    spe.glig_pct  = _s("GLIG%",  spe.glig_pct)
    spe.sdwt      = _s("SDWT",   spe.sdwt)
    spe.sdafr     = _s("SDAFR",  spe.sdafr)
    spe.ptfmx     = _s("PTFMX",  spe.ptfmx)
    spe.rwupm     = _s("RWUPM",  spe.rwupm)
    spe.rwumx     = _s("RWUMX",  spe.rwumx)
    spe.wfpu      = _s("WFPU",   spe.wfpu)
    spe.wfpgf     = _s("WFPGF",  spe.wfpgf)
    spe.wfgu      = _s("WFGU",   spe.wfgu)
    spe.wftu      = _s("WFTU",   spe.wftu)
    spe.wftl      = _s("WFTL",   spe.wftl)
    spe.wfsu      = _s("WFSU",   spe.wfsu)
    spe.wfgeu     = _s("WFGEU",  spe.wfgeu)
    spe.wfrgu     = _s("WFRGU",  spe.wfrgu)
    spe.llosw     = _s("LLOSW",  spe.llosw)
    spe.nh4mn     = _s("NH4MN",  spe.nh4mn)
    spe.no3mn     = _s("NO3MN",  spe.no3mn)
    spe.rtno3     = _s("RTNO3",  spe.rtno3)
    spe.rtnh4     = _s("RTNH4",  spe.rtnh4)
    spe.ntupf     = _s("NTUPF",  spe.ntupf)
    spe.gn_mx     = _s("GN%MX",  spe.gn_mx)
    spe.sdn_pct   = _s("SDN%",   spe.sdn_pct)
    spe.nfpu      = _s("NFPU",   spe.nfpu)
    spe.nfpl      = _s("NFPL",   spe.nfpl)
    spe.nfgu      = _s("NFGU",   spe.nfgu)
    spe.nfgl      = _s("NFGL",   spe.nfgl)
    spe.nftu      = _s("NFTU",   spe.nftu)
    spe.nftl      = _s("NFTL",   spe.nftl)
    spe.nfsu      = _s("NFSU",   spe.nfsu)
    spe.nfsf      = _s("NFSF",   spe.nfsf)
    spe.llosn     = _s("LLOSN",  spe.llosn)
    spe.ncrg      = _s("NCRG",   spe.ncrg)
    spe.nlab_pct  = _s("NLAB%",  spe.nlab_pct)
    spe.tkuh      = _s("TKUH",   spe.tkuh)
    spe.hdur      = _s("HDUR",   spe.hdur)
    spe.tklf      = _s("TKLF",   spe.tklf)

    # CO2 table (CO2RF → list of concentrations, CO2F → list of response)
    co2rf_v = kv.get("CO2RF")
    co2f_v  = kv.get("CO2F")
    if isinstance(co2rf_v, list) and len(co2rf_v) > 0:
        spe.co2rf = tuple(float(x) for x in co2rf_v)
    if isinstance(co2f_v, list) and len(co2f_v) > 0:
        spe.co2f  = tuple(float(x) for x in co2f_v)

    # Temperature response arrays — TRGEM…TRGFN share one @-header row;
    # parse_spe_csreads stores each column name separately as a 4-element list.
    spe.trgem = _a("TRGEM", 4, spe.trgem)
    spe.trdv1 = _a("TRDV1", 4, spe.trdv1)
    spe.trdv2 = _a("TRDV2", 4, spe.trdv2)
    spe.trlfg = _a("TRLFG", 4, spe.trlfg)
    spe.trphs = _a("TRPHS", 4, spe.trphs)
    spe.trvrn = _a("TRVRN", 4, spe.trvrn)
    spe.trhar = _a("TRHAR", 4, spe.trhar)
    spe.trgfw = _a("TRGFW", 4, spe.trgfw)
    spe.trgfn = _a("TRGFN", 4, spe.trgfn)

    return spe


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _find_genotype_dir() -> Path:
    env = os.environ.get("DSSAT_GENOTYPE_DIR")
    if env:
        return Path(env)
    p = Path(__file__).resolve().parent
    for _ in range(12):
        candidate = p / "Data" / "Genotype"
        if candidate.is_dir():
            return candidate
        p = p.parent
    raise FileNotFoundError(
        "Cannot locate Data/Genotype directory. "
        "Set the DSSAT_GENOTYPE_DIR environment variable."
    )


def load_ceres_wheat_params(
    varno: str,
    *,
    model_code: str = "WHCER048",
    genotype_dir: Optional[Path | str] = None,
) -> CeresWheatParams:
    """Load CERES-Wheat parameters from CUL + ECO + SPE files.

    Parameters
    ----------
    varno:
        Cultivar identifier, e.g. ``"IB1500"``.
    model_code:
        DSSAT model prefix, default ``"WHCER048"``.
    genotype_dir:
        Path to the directory containing the genotype files.

    Returns
    -------
    CeresWheatParams
        Combined cultivar + ecotype + species parameters, with CUL overrides
        already applied (via :meth:`CeresWheatParams._apply_overrides`).
    """
    gdir = Path(genotype_dir) if genotype_dir else _find_genotype_dir()
    prefix = model_code[:8].upper()

    cul = _parse_wh_cul(gdir / f"{prefix}.CUL", varno)
    eco = _parse_wh_eco(gdir / f"{prefix}.ECO", cul.econo)
    spe = _parse_wh_spe(gdir / f"{prefix}.SPE")

    params = CeresWheatParams(spe=spe, eco=eco, cul=cul)
    params._apply_overrides()
    return params
