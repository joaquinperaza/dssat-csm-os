"""CROPGRO parameter file readers for .SPE, .ECO, and .CUL files.

Parses the three DSSAT-CROPGRO genotype files exactly as the Fortran routines
``PHOTIP``, ``IPPHENOL``, and the growth/respiration readers do:

* ``SBGRO048.SPE`` / ``CNGRO048.SPE`` — species-level constants
* ``SBGRO048.ECO`` / ``CNGRO048.ECO`` — ecotype-level constants
* ``SBGRO048.CUL`` / ``CNGRO048.CUL`` — cultivar-level constants

The public entry point is :func:`load_cropgro_params`, which combines all
three files into a single :class:`CropGROParams` object and computes the
derived phenology thresholds (``PHTHRS`` array, ``CLDVAR``) exactly as
``IPPHENOL`` does.

No values are invented here.  Every field is read directly from the files.

Unit notes
----------
* ``RNITP`` / ``PCNL`` in Fortran GROW.for:
  ``PCNL = WTNLF / WLFI * 100.`` and ``RNITP = PCNL``
  — leaf N concentrations are stored as **percent** (g N / 100 g DM).
* Photosynthesis curves ``FNPGN`` and reference value ``LNREF`` are also
  in percent (consistent with ``RNITP``).
* All cultivar / ecotype duration fields (EM-FL, FL-SH, etc.) are in
  **photothermal days** exactly as in the Fortran PHTHRS array.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _free_floats(text: str, n: int) -> List[float]:
    """Extract up to *n* floats from *text* by whitespace splitting.

    Non-numeric tokens stop extraction silently.
    """
    result: List[float] = []
    for t in text.split():
        if len(result) >= n:
            break
        try:
            result.append(float(t))
        except ValueError:
            pass  # skip non-numeric tokens (type codes etc.)
    while len(result) < n:
        result.append(0.0)
    return result


def _word3_after_floats(line: str, n_floats: int) -> str:
    """Return the first 3-char token that appears after *n_floats* numbers."""
    tokens = line.split()
    count = 0
    for t in tokens:
        if count >= n_floats:
            return t.upper()[:3]
        try:
            float(t)
            count += 1
        except ValueError:
            if count >= n_floats:
                return t.upper()[:3]
    return "LIN"


def _spe_sections(path: Path) -> dict[str, List[str]]:
    """Parse a .SPE file into sections.

    Section headers start with ``!*`` (a comment-marker followed by ``*``).
    We use the full stripped header as the dictionary key so that sections
    with similar short prefixes (e.g. ``!*SEED  COMPOSITION VALUES`` vs
    ``!*SEED AND SHELL GROWTH PARAMETERS``) do not collide.

    Data lines (non-blank, not starting with ``!`` or ``*``) are appended to
    the current section's list.
    """
    sections: dict[str, List[str]] = {}
    current_key = "__preamble__"
    sections[current_key] = []

    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("!*"):
            current_key = stripped          # full header as key
            sections.setdefault(current_key, [])
            continue
        if stripped.startswith("!") or stripped.startswith("*"):
            continue
        sections.setdefault(current_key, []).append(raw)

    return sections


def _spe_section(sections: dict, *candidates: str) -> List[str]:
    """Return the data lines for the first matching section key.

    *candidates* are substrings to test against section header lines.
    E.g. ``_spe_section(secs, '!*LEAF GROWTH', '!*LEAF GRO')`` returns
    the lines for the leaf-growth section.
    """
    for key, lines in sections.items():
        for c in candidates:
            if c.upper() in key.upper():
                return lines
    return []


# ---------------------------------------------------------------------------
# Dataclasses — one per major SPE section + ECO + CUL
# ---------------------------------------------------------------------------

@dataclass
class PhotoParams:
    """From ``!*PHOTOSYNTHESIS PARAMETERS`` in .SPE."""

    parmax: float = 40.0    # PAR at 63 % of PHTMAX (mol quanta m^-2 d^-1)
    phtmax: float = 61.0    # Maximum canopy PG (g CH2O m^-2 d^-1)
    kcan: float = 0.67      # Light extinction coefficient
    kc_slope: float = 0.10  # Kcan slope with row/plant spacing ratio
    ccmp: float = 79.0      # CO2 compensation point (ppm)
    ccmax: float = 2.08     # Max CO2 assimilation relative to 330 ppm
    cceff: float = 0.0106   # CO2 carboxylation efficiency
    # FNPGN — leaf N effect on PG; values in percent (g N per 100 g DM)
    fnpgn: Tuple[float, ...] = (1.90, 5.50, 20.0, 20.0)
    typpgn: str = "QDR"
    # FNPGT — temp effect on canopy PG; (tb, to1, to2, tm) in °C
    fnpgt: Tuple[float, ...] = (6.0, 22.0, 34.0, 45.0)
    typpgt: str = "LIN"
    # LNREF — reference leaf N for photosynthesis, same units as FNPGN (%)
    lnref: float = 4.90
    pgref: float = 1.030    # Reference leaf PG (mg CO2 dm^-2 s^-1)
    xpgslw: Tuple[float, ...] = (
        0.0, 0.001, 0.002, 0.003, 0.0035, 0.004,
        0.005, 0.006, 0.008, 0.010,
    )
    ypgslw: Tuple[float, ...] = (
        0.162, 0.679, 0.867, 0.966, 1.000, 1.027,
        1.069, 1.100, 1.141, 1.167,
    )


@dataclass
class RespParams:
    """From ``!*RESPIRATION PARAMETERS`` in .SPE."""

    res30c: float = 3.5e-4   # Maintenance respiration coeff at 30°C (g CH2O g^-1 DM d^-1)
    r30c2: float = 0.0040    # Q10 exponent for maintenance respiration
    rno3c: float = 2.556     # CH2O cost per g N taken up as NO3
    rnh4c: float = 2.556     # CH2O cost per g N taken up as NH4
    rpro: float = 0.360      # CH2O cost for protein synthesis
    rfixn: float = 2.830     # CH2O cost per g N fixed biologically
    rch2o: float = 1.242     # Growth resp. factor for carbohydrate
    rlip: float = 3.106      # Growth resp. factor for lipid
    rlig: float = 2.174      # Growth resp. factor for lignin
    roa: float = 0.929       # Growth resp. factor for organic acids
    rmin: float = 0.05       # Growth resp. factor for minerals
    pch2o: float = 1.13      # Fraction of net assimilate after growth respiration


@dataclass
class NFIXParams:
    """From ``!*NITROGEN FIXATION PARAMETERS`` in .SPE."""

    snactm: float = 0.045    # Max specific nodule activity (g N g^-1 nod d^-1)
    nodrgm: float = 0.170    # Max nodule growth rate (g nod g^-1 root d^-1)
    dwnodi: float = 0.014    # Initial nodule weight (g plant^-1)
    ttfix: float = 0.0       # Thermal time delay before nodule growth
    ndthmx: float = 0.07     # Max nodule death rate (fraction d^-1)
    cnodcr: float = 0.05     # C:N ratio of nodule tissue
    fnngt: Tuple[float, ...] = (6.0, 22.0, 35.0, 44.0)   # Temp → nodule growth
    typngt: str = "LIN"
    fnfxt: Tuple[float, ...] = (4.0, 20.0, 35.0, 44.0)   # Temp → N fixation
    typfxt: str = "LIN"
    fnfxd: Tuple[float, ...] = (0.0, 0.85, 1.0, 10.0)    # Soil water dry stress
    typfxd: str = "LIN"
    fnfxw: Tuple[float, ...] = (-0.02, 0.001, 1.0, 2.0)  # Soil water wet stress
    typfxw: str = "LIN"
    fnfxa: Tuple[float, ...] = (0.0, 0.10, 1.0, 0.0)     # Nodule age effect
    typfxa: str = "INL"


@dataclass
class VegPartParams:
    """From ``!*VEGETATIVE PARTITIONING PARAMETERS`` in .SPE.

    XLEAF / YLEAF / YSTEM are look-up table entries as a function of V-stage.
    """

    xleaf: Tuple[float, ...] = (0.0, 1.5, 3.3, 5.0, 7.8, 10.5, 30.0, 40.0)
    yleaf: Tuple[float, ...] = (0.41, 0.42, 0.42, 0.41, 0.36, 0.32, 0.31, 0.31)
    ystem: Tuple[float, ...] = (0.09, 0.13, 0.21, 0.29, 0.37, 0.49, 0.49, 0.49)
    wtfsd: float = 0.55      # Weight fraction of seed at first seed fill
    porpt: float = 0.58      # Proportion of reproductive growth to pods
    frstmf: float = 0.55     # Stem fraction at end of vegetative growth
    frlff: float = 0.24      # Leaf fraction at end of vegetative growth
    atop: float = 1.00       # Above-ground weight fraction
    frcnod: float = 0.05     # Fraction of growth to nodules (legumes)
    frlfmx: float = 0.70     # Maximum leaf fraction of total growth


@dataclass
class LeafGrowthParams:
    """From ``!*LEAF GROWTH PARAMETERS`` in .SPE."""

    finref: float = 180.0    # Initial leaf area reference (cm^2 plant^-1)
    slaref: float = 350.0    # Reference SLA (cm^2 g^-1)
    sizref: float = 171.4    # Reference leaf size (cm^2)
    vssink: float = 5.0      # V-stage node-sink for leaf expansion
    evmodc: float = 0.0      # Modifier for early node appearance
    slamax: float = 950.0    # Maximum SLA (cm^2 g^-1)
    slamin: float = 250.0    # Minimum SLA (cm^2 g^-1)
    slapar: float = -0.048   # SLA response to PAR (cm^2 g^-1 per MJ PAR)
    tursla: float = 1.50     # Turgor effect on SLA
    nsla: float = 0.40       # N effect on SLA
    xvgrow: Tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
    yvref: Tuple[float, ...] = (0.0, 20.0, 55.0, 110.0, 200.0, 320.0)
    xslatm: Tuple[float, ...] = (-50.0, 0.0, 12.0, 22.0, 60.0)
    yslatm: Tuple[float, ...] = (0.25, 0.25, 0.25, 1.00, 1.00)


@dataclass
class LeafSenesParams:
    """From ``!*LEAF SENESCENCE FACTORS`` in .SPE."""

    senrte: float = 0.80     # Leaf senescence rate (fraction d^-1)
    senrt2: float = 0.20     # Secondary senescence rate
    senday: float = 0.06     # Days before hard senescence
    freez1: float = -2.22    # Frost damage threshold 1 (°C)
    freez2: float = -5.00    # Frost damage threshold 2 (°C)
    icmp: float = 0.80       # Light compensation point for shade senescence
    tcmp: float = 10.0       # Time constant for shade-induced senescence (d)
    xstage: Tuple[float, ...] = (0.0, 5.0, 14.0, 30.0)
    xsenmx: Tuple[float, ...] = (3.0, 5.0, 10.0, 30.0)
    senpor: Tuple[float, ...] = (0.0, 0.0, 0.12, 0.12)
    senmax: Tuple[float, ...] = (0.0, 0.2, 0.6, 0.6)


@dataclass
class RootParams:
    """From ``!*ROOT PARAMETERS`` in .SPE."""

    rtdepi: float = 20.0     # Initial rooting depth (cm)
    rfac1: float = 7500.0    # Root length per unit weight (cm g^-1)
    rtsen: float = 0.020     # Root senescence rate (fraction d^-1)
    rldsm: float = 0.1       # Min RLV for full uptake
    rtsdf: float = 0.015     # Root sink decline factor
    rwuep1: float = 1.50     # EOP/EP ratio for potential transpiration
    rwumx: float = 0.04      # Max water uptake rate (cm^3 cm^-1 root d^-1)
    xrtfac: Tuple[float, ...] = (0.0, 2.5, 3.0, 2.5, 6.0, 2.6, 30.0, 2.6)
    yrtfac: Tuple[float, ...] = (0.0, 2.5, 3.0, 2.5, 6.0, 2.6, 30.0, 2.6)
    rtno3: float = 0.006
    rtnh4: float = 0.006
    pormin: float = 0.02
    rtexf: float = 0.10


@dataclass
class SeedShellParams:
    """From ``!*SEED AND SHELL GROWTH PARAMETERS`` in .SPE."""

    setmax: float = 0.60
    srmax: float = 0.30
    rflwab: float = 0.0
    xmpage: float = 100.0
    dswbar: float = 15.0
    xfrmax: float = 0.0
    shlag: float = 0.0
    fnpdt: Tuple[float, ...] = (13.0, 21.0, 26.5, 40.0)
    typpdt: str = "QDR"
    fnsdt: Tuple[float, ...] = (6.0, 21.0, 23.5, 41.0)
    typsdt: str = "QDR"
    xxftem: Tuple[float, ...] = (0.0, 5.0, 20.0, 35.0, 45.0, 60.0)
    yxftem: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 0.0, 0.0)
    xswfac: Tuple[float, ...] = (0.0, 0.5, 1.0, 1.0)
    yswfac: Tuple[float, ...] = (0.0, 1.0, 1.0, 1.0)
    xswbar: Tuple[float, ...] = (0.0, 0.01, 0.25, 1.0, 1.0)
    yswbar: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)
    xtrfac: Tuple[float, ...] = (0.0, 0.5, 0.75, 1.0)
    ytrfac: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)


@dataclass
class PhenoTempParams:
    """Temperature curves and phase descriptors from ``!*PHENOLOGY PARAMETERS``.

    ``tb``, ``to1``, ``to2``, ``tm`` — three temperature functions indexed
    0 (vegetative), 1 (early reproductive), 2 (late reproductive).  Correspond
    to Fortran ``TB(1..3)`` etc.

    ``dltyp``, ``ctmp``, ``tselc``, ``nprior``, ``wsenp``, ``nsenp``,
    ``psenp`` — per-phase arrays with 13 elements (phases 1..13 → 0-indexed).
    ``tselc`` values are 0-indexed (Fortran 1-based − 1).
    ``nprior`` values are 0-indexed (Fortran 1-based − 1).
    """

    tb:  Tuple[float, ...] = (7.0, 6.0, -15.0)
    to1: Tuple[float, ...] = (28.0, 26.0, 26.0)
    to2: Tuple[float, ...] = (35.0, 30.0, 34.0)
    tm:  Tuple[float, ...] = (45.0, 45.0, 45.0)

    dltyp: Tuple[str, ...] = (
        "NON", "NON", "NON", "INL", "INL",
        "INL", "INL", "INL", "INL", "INL",
        "NON", "INL", "INL",
    )
    ctmp: Tuple[str, ...] = (
        "LIN", "LIN", "LIN", "LIN", "LIN",
        "LIN", "LIN", "LIN", "LIN", "LIN",
        "NON", "LIN", "LIN",
    )
    tselc: Tuple[int, ...] = (0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 0, 1, 1)
    nprior: Tuple[int, ...] = (0, 1, 1, 3, 4, 5, 5, 5, 8, 8, 10, 5, 5)
    wsenp: Tuple[float, ...] = (
        -0.20, -0.20, -0.40, -0.40, -0.40,
        -0.40, -0.40, -0.40, 0.70, 0.70,
        0.00, -0.60, -0.90,
    )
    nsenp: Tuple[float, ...] = (0.0,) * 13
    psenp: Tuple[float, ...] = (0.0,) * 13


@dataclass
class SpeciesParams:
    """All species-level parameters from a CROPGRO .SPE file."""

    photo: PhotoParams = field(default_factory=PhotoParams)
    resp: RespParams = field(default_factory=RespParams)
    nfix: NFIXParams = field(default_factory=NFIXParams)
    vegpart: VegPartParams = field(default_factory=VegPartParams)
    leafgrow: LeafGrowthParams = field(default_factory=LeafGrowthParams)
    leafsenes: LeafSenesParams = field(default_factory=LeafSenesParams)
    roots: RootParams = field(default_factory=RootParams)
    seedshell: SeedShellParams = field(default_factory=SeedShellParams)
    phenotemp: PhenoTempParams = field(default_factory=PhenoTempParams)


@dataclass
class EcotypeParams:
    """Parameters from one row of a CROPGRO .ECO file."""

    econo: str = "DFAULT"
    econam: str = ""
    mg: int = 0
    tm: int = 1
    thvar: float = 0.0
    pl_em: float = 2.2
    em_v1: float = 6.0
    v1_ju: float = 0.0
    ju_r0: float = 5.0
    pm06: float = 0.0
    pm09: float = 0.35
    lngsh: float = 10.0
    r7_r8: float = 12.0
    fl_vs: float = 26.0
    trifl: float = 0.32
    rwdth: float = 1.0
    rhght: float = 1.0
    r1ppo: float = 0.0
    optbi: float = 18.0
    slobi: float = 0.028


@dataclass
class CultivarParams:
    """Parameters from one row of a CROPGRO .CUL file."""

    varno: str = "990007"
    vrname: str = "M GROUP   7"
    econo: str = "SB0701"
    csdl: float = 12.33    # Critical short-day length (h)
    ppsen: float = 0.320   # Photoperiod sensitivity (1/h; + = short-day, − = long-day)
    em_fl: float = 20.8    # Photothermal days emergence → R1 (= PH2T5)
    fl_sh: float = 10.0    # Photothermal days R1 → R3
    fl_sd: float = 16.0    # Photothermal days R1 → R5
    sd_pm: float = 36.4    # Photothermal days R5 → R7
    fl_lf: float = 18.0    # Photothermal days R1 → end of leaf expansion
    lfmax: float = 1.030   # Max leaf PG (mg CO2 dm^-2 s^-1)
    slavr: float = 375.0   # Specific leaf area (cm^2 g^-1)
    sizlf: float = 180.0   # Max leaf size (cm^2)
    xfrt: float = 1.00
    wtpsd: float = 0.18    # Max seed weight (g seed^-1)
    sfdur: float = 23.0    # Seed fill duration (photothermal days)
    sdpdv: float = 2.05    # Seeds per pod
    podur: float = 10.0
    thrsh: float = 78.0    # Threshing percentage
    sdpro: float = 0.40
    sdlip: float = 0.20


@dataclass
class CropGROParams:
    """Combined parameters from .SPE + .ECO + .CUL, plus derived scalars.

    Call :meth:`_derive` after construction to fill in ``phthrs``,
    ``csdvar``, ``cldvar``, etc.
    """

    spe: SpeciesParams = field(default_factory=SpeciesParams)
    eco: EcotypeParams = field(default_factory=EcotypeParams)
    cul: CultivarParams = field(default_factory=CultivarParams)

    # Derived daylength thresholds (computed by _derive)
    csdvar: float = 12.33
    cldvar: float = 15.45
    csdvrr: float = 12.33
    cldvrr: float = 15.45
    thvar: float = 0.0

    # PHTHRS[0..19] = Fortran PHTHRS(1..20), 0-indexed
    phthrs: np.ndarray = field(default_factory=lambda: np.zeros(20))

    def _derive(self) -> None:
        """Compute CLDVAR and full PHTHRS exactly as Fortran IPPHENOL does."""
        cul = self.cul
        eco = self.eco

        self.csdvar = cul.csdl
        self.thvar  = eco.thvar

        ppsen = cul.ppsen
        if ppsen >= 0.0:
            self.cldvar = self.csdvar + (1.0 - eco.thvar) / max(ppsen, 1e-6)
        else:
            self.cldvar = self.csdvar + (1.0 - eco.thvar) / min(ppsen, -1e-6)

        self.csdvrr = self.csdvar - eco.r1ppo
        self.cldvrr = self.cldvar - eco.r1ppo

        # Build PHTHRS array (Python 0-indexed = Fortran 1-indexed)
        ph = np.zeros(20)
        ph[0] = eco.pl_em    # PHTHRS(1) = PL-EM
        ph[1] = eco.em_v1    # PHTHRS(2) = EM-V1
        ph[2] = eco.v1_ju    # PHTHRS(3) = V1-JU
        ph[3] = eco.ju_r0    # PHTHRS(4) = JU-R0
        ph[5] = cul.fl_sh    # PHTHRS(6) = FL-SH
        ph[7] = cul.fl_sd    # PHTHRS(8) = FL-SD
        ph[9] = cul.sd_pm    # PHTHRS(10) = SD-PM
        ph[12] = cul.fl_lf   # PHTHRS(13) = FL-LF
        ph[10] = eco.r7_r8   # PHTHRS(11) = R7-R8
        ph[11] = eco.fl_vs   # PHTHRS(12) = FL-VS

        # PHTHRS(5) = MAX(0, EM-FL − PHTHRS(3) − PHTHRS(4))
        ph[4] = max(0.0, cul.em_fl - ph[2] - ph[3])

        # PHTHRS(7) = PHTHRS(6) + MAX(0, (PHTHRS(8)-PHTHRS(6)) * PM06)
        ph[6] = ph[5] + max(0.0, (ph[7] - ph[5]) * eco.pm06)

        # PHTHRS(9) = MAX(0, PHTHRS(10) * PM09)
        ph[8] = max(0.0, ph[9] * eco.pm09)

        self.phthrs = ph


# ---------------------------------------------------------------------------
# SPE parser
# ---------------------------------------------------------------------------

def parse_spe(path: Path | str) -> SpeciesParams:
    """Parse a CROPGRO .SPE file into :class:`SpeciesParams`.

    Follows the same sequential read order as the Fortran routines
    PHOTIP, IPPHENOL, and the GROW / ROOTS / RESPIR parameter readers.
    """
    path = Path(path)
    secs = _spe_sections(path)

    # Helper — find section by substring match on header
    def _sec(*substrings: str) -> List[str]:
        return _spe_section(secs, *substrings)

    # ---- Photosynthesis -------------------------------------------------
    photo = PhotoParams()
    lines = _sec("!*PHOTOSYNTHESIS")
    if len(lines) >= 1:
        v = _free_floats(lines[0], 4)
        photo.parmax = v[0]; photo.phtmax = v[1]
        if v[2] > 0: photo.kcan = v[2]
        if v[3] > 0: photo.kc_slope = v[3]
    if len(lines) >= 2:
        v = _free_floats(lines[1], 3)
        photo.ccmp = v[0]; photo.ccmax = v[1]; photo.cceff = v[2]
    if len(lines) >= 3:
        v = _free_floats(lines[2], 4)
        photo.fnpgn = tuple(v[:4])
        photo.typpgn = _word3_after_floats(lines[2], 4)
    if len(lines) >= 4:
        v = _free_floats(lines[3], 4)
        photo.fnpgt = tuple(v[:4])
        photo.typpgt = _word3_after_floats(lines[3], 4)
    # lines[4] = XLMAXT, lines[5] = YLMAXT, lines[6] = FNPGL, lines[7] = PGEFF...
    # lines[8] = SLWREF,SLWSLO,NSLOPE,LNREF,PGREF
    if len(lines) >= 9:
        v = _free_floats(lines[8], 5)
        photo.lnref = v[3]; photo.pgref = v[4]
    if len(lines) >= 10:
        v = _free_floats(lines[9], 10)
        photo.xpgslw = tuple(v[:10])
    if len(lines) >= 11:
        v = _free_floats(lines[10], 10)
        photo.ypgslw = tuple(v[:10])

    # ---- Respiration ----------------------------------------------------
    resp = RespParams()
    lines = _sec("!*RESPIRATION")
    if len(lines) >= 1:
        v = _free_floats(lines[0], 2)
        resp.res30c = v[0]; resp.r30c2 = v[1]
    if len(lines) >= 2:
        v = _free_floats(lines[1], 4)
        resp.rno3c = v[0]; resp.rnh4c = v[1]; resp.rpro = v[2]; resp.rfixn = v[3]
    if len(lines) >= 3:
        v = _free_floats(lines[2], 6)
        resp.rch2o = v[0]; resp.rlip = v[1]; resp.rlig = v[2]
        resp.roa = v[3]; resp.rmin = v[4]; resp.pch2o = v[5]

    # ---- N fixation -----------------------------------------------------
    nfix = NFIXParams()
    lines = _sec("!*NITROGEN FIXATION")
    if len(lines) >= 1:
        v = _free_floats(lines[0], 6)
        nfix.snactm = v[0]; nfix.nodrgm = v[1]
        nfix.dwnodi = v[2]; nfix.ttfix = v[3]
        nfix.ndthmx = v[4]; nfix.cnodcr = v[5]
    if len(lines) >= 2:
        v = _free_floats(lines[1], 4)
        nfix.fnngt = tuple(v[:4])
        nfix.typngt = _word3_after_floats(lines[1], 4)
    if len(lines) >= 3:
        v = _free_floats(lines[2], 4)
        nfix.fnfxt = tuple(v[:4])
        nfix.typfxt = _word3_after_floats(lines[2], 4)
    if len(lines) >= 4:
        v = _free_floats(lines[3], 4)
        nfix.fnfxd = tuple(v[:4])
        nfix.typfxd = _word3_after_floats(lines[3], 4)
    if len(lines) >= 5:
        v = _free_floats(lines[4], 4)
        nfix.fnfxw = tuple(v[:4])
        nfix.typfxw = _word3_after_floats(lines[4], 4)
    if len(lines) >= 6:
        v = _free_floats(lines[5], 4)
        nfix.fnfxa = tuple(v[:4])
        nfix.typfxa = _word3_after_floats(lines[5], 4)

    # ---- Vegetative partitioning ----------------------------------------
    vegpart = VegPartParams()
    lines = _sec("!*VEGETATIVE PARTITIONING")
    if len(lines) >= 1:
        v = _free_floats(lines[0], 8); vegpart.xleaf = tuple(v[:8])
    if len(lines) >= 2:
        v = _free_floats(lines[1], 8); vegpart.yleaf = tuple(v[:8])
    if len(lines) >= 3:
        v = _free_floats(lines[2], 8); vegpart.ystem = tuple(v[:8])
    if len(lines) >= 4:
        v = _free_floats(lines[3], 6)
        vegpart.wtfsd = v[0]; vegpart.porpt = v[1]
        vegpart.frstmf = v[2]; vegpart.frlff = v[3]
        vegpart.atop = v[4]; vegpart.frcnod = v[5]
    if len(lines) >= 5:
        v = _free_floats(lines[4], 1); vegpart.frlfmx = v[0]

    # ---- Leaf growth parameters -----------------------------------------
    leafgrow = LeafGrowthParams()
    lines = _sec("!*LEAF GROWTH")
    if len(lines) >= 1:
        v = _free_floats(lines[0], 5)
        leafgrow.finref = v[0]; leafgrow.slaref = v[1]; leafgrow.sizref = v[2]
        leafgrow.vssink = v[3]; leafgrow.evmodc = v[4]
    if len(lines) >= 2:
        v = _free_floats(lines[1], 5)
        leafgrow.slamax = v[0]; leafgrow.slamin = v[1]; leafgrow.slapar = v[2]
        leafgrow.tursla = v[3]; leafgrow.nsla = v[4]
    if len(lines) >= 3:
        v = _free_floats(lines[2], 6); leafgrow.xvgrow = tuple(v[:6])
    if len(lines) >= 4:
        v = _free_floats(lines[3], 6); leafgrow.yvref = tuple(v[:6])
    if len(lines) >= 5:
        v = _free_floats(lines[4], 5); leafgrow.xslatm = tuple(v[:5])
    if len(lines) >= 6:
        v = _free_floats(lines[5], 5); leafgrow.yslatm = tuple(v[:5])

    # ---- Leaf senescence ------------------------------------------------
    leafsenes = LeafSenesParams()
    lines = _sec("!*LEAF SENESCENCE")
    if len(lines) >= 1:
        v = _free_floats(lines[0], 5)
        leafsenes.senrte = v[0]; leafsenes.senrt2 = v[1]; leafsenes.senday = v[2]
        leafsenes.freez1 = v[3]; leafsenes.freez2 = v[4]
    if len(lines) >= 2:
        v = _free_floats(lines[1], 2)
        leafsenes.icmp = v[0]; leafsenes.tcmp = v[1]
    if len(lines) >= 3:
        v = _free_floats(lines[2], 8)
        leafsenes.xstage = tuple(v[:4]); leafsenes.xsenmx = tuple(v[4:8])
    if len(lines) >= 4:
        v = _free_floats(lines[3], 8)
        leafsenes.senpor = tuple(v[:4]); leafsenes.senmax = tuple(v[4:8])

    # ---- Roots ----------------------------------------------------------
    roots = RootParams()
    lines = _sec("!*ROOT PARAMETERS")
    if len(lines) >= 1:
        v = _free_floats(lines[0], 7)
        roots.rtdepi = v[0]; roots.rfac1 = v[1]; roots.rtsen = v[2]
        roots.rldsm = v[3]; roots.rtsdf = v[4]; roots.rwuep1 = v[5]; roots.rwumx = v[6]
    if len(lines) >= 2:
        # "0.0  2.50   3.0  2.50   6.0  2.60  30.0  2.60  XRTFAC,YRTFAC"
        # Interleaved X and Y pairs: x0 y0 x1 y1 ... → unpack into XRTFAC, YRTFAC
        v = _free_floats(lines[1], 8)
        roots.xrtfac = tuple(v[0::2])  # even positions
        roots.yrtfac = tuple(v[1::2])  # odd positions
    if len(lines) >= 3:
        v = _free_floats(lines[2], 4)
        roots.rtno3 = v[0]; roots.rtnh4 = v[1]; roots.pormin = v[2]; roots.rtexf = v[3]

    # ---- Seed and shell growth ------------------------------------------
    seedshell = SeedShellParams()
    lines = _sec("!*SEED AND SHELL")
    if len(lines) >= 1:
        v = _free_floats(lines[0], 4)
        seedshell.setmax = v[0]; seedshell.srmax = v[1]
        seedshell.rflwab = v[2]; seedshell.xmpage = v[3]
    if len(lines) >= 2:
        v = _free_floats(lines[1], 3)
        seedshell.dswbar = v[0]; seedshell.xfrmax = v[1]; seedshell.shlag = v[2]
    if len(lines) >= 3:
        v = _free_floats(lines[2], 4)
        seedshell.fnpdt = tuple(v[:4])
        seedshell.typpdt = _word3_after_floats(lines[2], 4)
    if len(lines) >= 4:
        v = _free_floats(lines[3], 4)
        seedshell.fnsdt = tuple(v[:4])
        seedshell.typsdt = _word3_after_floats(lines[3], 4)
    if len(lines) >= 5:
        v = _free_floats(lines[4], 6); seedshell.xxftem = tuple(v[:6])
    if len(lines) >= 6:
        v = _free_floats(lines[5], 6); seedshell.yxftem = tuple(v[:6])
    if len(lines) >= 7:
        v = _free_floats(lines[6], 4); seedshell.xswfac = tuple(v[:4])
    if len(lines) >= 8:
        v = _free_floats(lines[7], 4); seedshell.yswfac = tuple(v[:4])
    if len(lines) >= 9:
        v = _free_floats(lines[8], 5); seedshell.xswbar = tuple(v[:5])
    if len(lines) >= 10:
        v = _free_floats(lines[9], 5); seedshell.yswbar = tuple(v[:5])
    if len(lines) >= 11:
        v = _free_floats(lines[10], 4); seedshell.xtrfac = tuple(v[:4])
    if len(lines) >= 12:
        v = _free_floats(lines[11], 4); seedshell.ytrfac = tuple(v[:4])

    # ---- Phenology temperature + phase descriptors ----------------------
    phenotemp = PhenoTempParams()
    lines = _sec("!*PHENOLOGY PARAMETERS")
    tb  = list(phenotemp.tb)
    to1 = list(phenotemp.to1)
    to2 = list(phenotemp.to2)
    tm_ = list(phenotemp.tm)
    for i in range(3):
        if len(lines) > i:
            v = _free_floats(lines[i], 4)
            tb[i] = v[0]; to1[i] = v[1]; to2[i] = v[2]; tm_[i] = v[3]
    phenotemp.tb  = tuple(tb)
    phenotemp.to1 = tuple(to1)
    phenotemp.to2 = tuple(to2)
    phenotemp.tm  = tuple(tm_)

    dltyp  = list(phenotemp.dltyp)
    ctmp   = list(phenotemp.ctmp)
    tselc  = list(phenotemp.tselc)
    nprior = list(phenotemp.nprior)
    wsenp  = list(phenotemp.wsenp)
    nsenp  = list(phenotemp.nsenp)
    psenp  = list(phenotemp.psenp)

    for line in lines[3:3 + 13]:
        tokens = line.split()
        if not tokens:
            continue
        try:
            j = int(tokens[0]) - 1   # Fortran 1-based → Python 0-based
        except ValueError:
            continue
        if not (0 <= j < 13):
            continue
        if len(tokens) > 1:
            try:
                nprior[j] = int(tokens[1]) - 1
            except ValueError:
                pass
        if len(tokens) > 2:
            dltyp[j] = tokens[2].upper()[:3]
        if len(tokens) > 3:
            ctmp[j] = tokens[3].upper()[:3]
        if len(tokens) > 4:
            try:
                tselc[j] = int(tokens[4]) - 1
            except ValueError:
                pass
        if len(tokens) > 5:
            try:
                wsenp[j] = float(tokens[5])
            except ValueError:
                pass
        if len(tokens) > 6:
            try:
                nsenp[j] = float(tokens[6])
            except ValueError:
                pass
        if len(tokens) > 7:
            try:
                psenp[j] = float(tokens[7])
            except ValueError:
                pass

    phenotemp.dltyp  = tuple(dltyp)
    phenotemp.ctmp   = tuple(ctmp)
    phenotemp.tselc  = tuple(tselc)
    phenotemp.nprior = tuple(nprior)
    phenotemp.wsenp  = tuple(wsenp)
    phenotemp.nsenp  = tuple(nsenp)
    phenotemp.psenp  = tuple(psenp)

    return SpeciesParams(
        photo=photo, resp=resp, nfix=nfix, vegpart=vegpart,
        leafgrow=leafgrow, leafsenes=leafsenes, roots=roots,
        seedshell=seedshell, phenotemp=phenotemp,
    )


# ---------------------------------------------------------------------------
# ECO parser
# ---------------------------------------------------------------------------

def parse_eco(path: Path | str, econo: str) -> EcotypeParams:
    """Parse one ecotype row from a CROPGRO .ECO file.

    Column layout from Fortran format 3100 in IPPHENOL::

        A6 ECO#, 1X, A16 ECONAME, 1X, 2(1X,A2) MG TM,
        7(1X,F5.0) THVAR PL-EM EM-V1 V1-JU JU-R0 PM06 PM09,
        6X (LNGSH),
        3(1X,F5.0) R7-R8 FL-VS TRIFL,
        2(6X) (RWDTH RHGHT — read here for rwdth/rhght),
        3(1X,F5.0) R1PPO OPTBI SLOBI

    Total chars before first numeric (THVAR): 6+1+16+1+2*(1+2) = 30.
    We start numeric extraction at column 30.
    """
    path = Path(path)
    econo_up = econo.strip().upper()
    match_row: Optional[str] = None
    default_row: Optional[str] = None

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = raw.strip()
        if not s or s.startswith("!") or s.startswith("@") or s.startswith("*"):
            continue
        row_eco = raw[:6].strip().upper()
        if row_eco in ("999991", "999992"):
            continue
        if row_eco == econo_up:
            match_row = raw
            break
        if row_eco == "DFAULT":
            default_row = raw

    row = match_row or default_row
    if row is None:
        return EcotypeParams(econo=econo)

    eco = EcotypeParams()
    eco.econo = row[:6].strip()
    eco.econam = row[7:23].strip()

    # Numeric fields start at column 30 (after ECO#+space+ECONAME+space+MG+TM)
    nums = _free_floats(row[30:], 16)
    # Order: THVAR, PL-EM, EM-V1, V1-JU, JU-R0, PM06, PM09, LNGSH,
    #        R7-R8, FL-VS, TRIFL, RWDTH, RHGHT, R1PPO, OPTBI, SLOBI
    if len(nums) > 0:  eco.thvar  = nums[0]
    if len(nums) > 1:  eco.pl_em  = nums[1]
    if len(nums) > 2:  eco.em_v1  = nums[2]
    if len(nums) > 3:  eco.v1_ju  = nums[3]
    if len(nums) > 4:  eco.ju_r0  = nums[4]
    if len(nums) > 5:  eco.pm06   = nums[5]
    if len(nums) > 6:  eco.pm09   = nums[6]
    if len(nums) > 7:  eco.lngsh  = nums[7]
    if len(nums) > 8:  eco.r7_r8  = nums[8]
    if len(nums) > 9:  eco.fl_vs  = nums[9]
    if len(nums) > 10: eco.trifl  = nums[10]
    if len(nums) > 11: eco.rwdth  = nums[11]
    if len(nums) > 12: eco.rhght  = nums[12]
    if len(nums) > 13: eco.r1ppo  = nums[13]
    if len(nums) > 14: eco.optbi  = nums[14]
    if len(nums) > 15: eco.slobi  = nums[15]

    return eco


# ---------------------------------------------------------------------------
# CUL parser
# ---------------------------------------------------------------------------

def parse_cul(path: Path | str, varno: str) -> CultivarParams:
    """Parse one cultivar row from a CROPGRO .CUL file.

    Column layout (standard CROPGRO CUL format)::

        Cols  0-5:   VAR# (6 chars)
        Col   6:     space
        Cols  7-22:  VRNAME (16 chars)
        Col   23:    space
        Cols  24:    EXPNO (1 char, often '.')
        Col   25:    space
        Cols  26-31: ECO# (6 chars)
        Col   32+:   18 numeric coefficients (free format, space-delimited)

    CSDL, PPSEN, EM-FL, FL-SH, FL-SD, SD-PM, FL-LF, LFMAX, SLAVR, SIZLF,
    XFRT, WTPSD, SFDUR, SDPDV, PODUR, THRSH, SDPRO, SDLIP
    """
    path = Path(path)
    varno_up = varno.strip().upper()
    match_row: Optional[str] = None
    fallback_row: Optional[str] = None

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = raw.strip()
        if not s or s.startswith("!") or s.startswith("@") or s.startswith("*"):
            continue
        row_var = raw[:6].strip().upper()
        if row_var in ("999991", "999992"):
            continue
        if row_var == varno_up:
            match_row = raw
            break
        if fallback_row is None:
            fallback_row = raw

    row = match_row or fallback_row
    if row is None:
        return CultivarParams(varno=varno)

    cul = CultivarParams()
    cul.varno  = row[:6].strip()
    cul.vrname = row[7:23].strip()
    # ECO# at cols 26-31 (6 chars), numeric data at col 32+
    cul.econo  = row[26:32].strip()

    nums = _free_floats(row[32:], 18)
    fields = [
        "csdl", "ppsen", "em_fl", "fl_sh", "fl_sd", "sd_pm", "fl_lf",
        "lfmax", "slavr", "sizlf", "xfrt", "wtpsd", "sfdur", "sdpdv",
        "podur", "thrsh", "sdpro", "sdlip",
    ]
    for i, name in enumerate(fields):
        if i < len(nums):
            setattr(cul, name, nums[i])

    return cul


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _find_genotype_dir() -> Path:
    """Locate the ``Data/Genotype`` directory in the DSSAT-CSM-OS repository."""
    import os
    env = os.environ.get("DSSAT_GENOTYPE_DIR")
    if env:
        return Path(env)
    # Search upward from this file
    p = Path(__file__).resolve().parent
    for _ in range(10):
        candidate = p / "Data" / "Genotype"
        if candidate.is_dir():
            return candidate
        p = p.parent
    raise FileNotFoundError(
        "Cannot locate Data/Genotype directory. "
        "Set the DSSAT_GENOTYPE_DIR environment variable."
    )


def load_cropgro_params(
    model_code: str,
    varno: str,
    *,
    genotype_dir: Optional[Path | str] = None,
) -> CropGROParams:
    """Load and combine .SPE + .ECO + .CUL for one cultivar.

    Parameters
    ----------
    model_code:
        DSSAT model code prefix, e.g. ``'SBGRO048'`` or ``'CNGRO048'``.
    varno:
        Cultivar identifier, e.g. ``'IB0001'``.
    genotype_dir:
        Path to the directory containing the .SPE / .ECO / .CUL files.
        Defaults to ``Data/Genotype`` relative to the repository root.

    Returns
    -------
    CropGROParams
        Combined parameters with PHTHRS and derived scalars ready to use.
    """
    gdir = Path(genotype_dir) if genotype_dir else _find_genotype_dir()
    prefix = model_code[:8].upper()
    spe = parse_spe(gdir / f"{prefix}.SPE")
    cul = parse_cul(gdir / f"{prefix}.CUL", varno)
    eco = parse_eco(gdir / f"{prefix}.ECO", cul.econo)

    params = CropGROParams(spe=spe, eco=eco, cul=cul)
    params._derive()
    return params
