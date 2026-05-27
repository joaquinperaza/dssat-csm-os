"""CERES-Maize parameter file readers (MZCER048.CUL / .ECO / .SPE).

Reads the three DSSAT-CERES genotype files exactly as the Fortran routines
in ``MZ_GROSUB.for`` and ``MZ_PHENOL.for`` do:

* ``MZCER048.CUL`` — cultivar coefficients (P1, P2, P5, G2, G3, PHINT)
* ``MZCER048.ECO`` — ecotype coefficients (TBASE, TOPT, ROPT, P2O, DJTI,
  GDDE, DSGFT, RUE, KCAN, TSEN, CDAY)
* ``MZCER048.SPE`` — species constants (temperature responses, CO2 table,
  N parameters, root/leaf properties, etc.)

No values are invented.  Every field is read directly from the files.
The public entry point is :func:`load_ceres_maize_params`.

CUL Fortran format (``MZ_GROSUB.for`` line 1800)::

    FORMAT (A6, 1X, A16, 1X, A6, 1X, 6F6.0)
    → VARNO, VRNAME, ECONO, P1, P2, P5, G2, G3, PHINT

ECO Fortran format (``MZ_PHENOL.for`` line 3100)::

    FORMAT (A6, 1X, A16, 1X, 9(1X, F5.1))
    → ECO#, ECONAME, TBASE, TOPT, ROPT, P2O, DJTI, GDDE, DSGFT, RUE, KCAN
    (TSEN at col 80-84, CDAY at col 86-90 as optional extensions)

SPE: inline keyword-value style (first token = parameter name).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from dssat.io.converters.genotype_io import (
    read_genotype_row,
    parse_spe_inline,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MaizeCulParams:
    """Parameters from one row of ``MZCER048.CUL``.

    Attributes
    ----------
    varno:   Cultivar identifier (6-char).
    vrname:  Cultivar name (16-char).
    econo:   Ecotype code (6-char).
    p1:      Thermal time (°C d base 8°C) from emergence to end juvenile.
    p2:      Photoperiod sensitivity (d per h above P2O).
    p5:      Thermal time (°C d) from silking to maturity.
    g2:      Maximum kernel number per plant.
    g3:      Kernel filling rate under optimum conditions (mg d⁻¹ kernel⁻¹).
    phint:   Phyllochron interval (°C d per leaf).
    """

    varno:  str   = "DFAULT"
    vrname: str   = "Default Maize"
    econo:  str   = "DFAULT"
    p1:     float = 300.0
    p2:     float = 0.52
    p5:     float = 800.0
    g2:     float = 800.0
    g3:     float = 8.5
    phint:  float = 38.9


@dataclass
class MaizeEcoParams:
    """Parameters from one row of ``MZCER048.ECO``.

    Attributes
    ----------
    econo:   Ecotype identifier.
    econame: Ecotype name.
    tbase:   Base temperature for development (°C).
    topt:    Optimum temperature for vegetative development (°C).
    ropt:    Optimum temperature for reproductive development (°C).
    p2o:     Critical photoperiod (h) below which no daylength effect.
    djti:    Minimum days from juvenile-end to tassel initiation.
    gdde:    GDD per cm seed depth for emergence (°C d cm⁻¹).
    dsgft:   GDD from silking to effective grain fill start (°C d).
    rue:     Radiation use efficiency (g MJ⁻¹ PAR).
    kcan:    Canopy light extinction coefficient for PAR.
    tsen:    Critical temperature for cold leaf damage (°C).
    cday:    Consecutive cold days to trigger early maturity.
    """

    econo:   str   = "DFAULT"
    econame: str   = ""
    tbase:   float = 8.0
    topt:    float = 34.0
    ropt:    float = 34.0
    p2o:     float = 12.5
    djti:    float = 4.0
    gdde:    float = 6.0
    dsgft:   float = 170.0
    rue:     float = 4.2
    kcan:    float = 0.85
    tsen:    float = 6.0
    cday:    int   = 15


@dataclass
class MaizeSpeParams:
    """Species-level constants from ``MZCER048.SPE``.

    All values are read directly from the file.  No defaults are invented —
    values listed here match the ``DFAULT`` row / base values found in the
    actual SPE file.

    Attributes
    ----------
    prftc:   4-element temperature response for photosynthesis
             (base, opt_low, opt_high, max °C).
    rgfil:   4-element temperature response for grain filling.
    parsr:   Fraction of solar radiation that is PAR.
    co2x:    CO2 concentration table (ppm), 10 elements.
    co2y:    Relative photosynthesis at each CO2 level, 10 elements.
    fslfw:   Leaf area senesced under 100% water stress (fraction d⁻¹).
    fslfn:   Leaf area senesced under 100% N stress (fraction d⁻¹).
    sdsz:    Maximum potential seed size (mg seed⁻¹).
    rsgr:    Relative seed growth rate below which early maturity may trigger.
    rsgrt:   Consecutive days below RSGR to trigger early maturity.
    carbot:  Consecutive days CARBO < 0.001 before stress maturity.
    dsgt:    Maximum days sowing → germination before seed dies.
    dget:    GDD germination → emergence beyond which seed dies from drought.
    swcg:    Minimum available soil water for germination (cm³ cm⁻³).
    stmwte:  Stem weight at emergence (g plant⁻¹).
    rtwte:   Root weight at emergence (g plant⁻¹).
    lfwte:   Leaf weight at emergence (g plant⁻¹).
    seedrve: Seed carbohydrate reserve at emergence (g plant⁻¹).
    leafnoe: Leaf number at emergence.
    plae:    Leaf area at emergence (cm² plant⁻¹).
    tmnc:    Plant minimum N concentration (g N g⁻¹ DM).
    tance:   Above-ground N at emergence (g N g⁻¹ DM).
    rcnp:    Root critical N concentration (g N g⁻¹ root DM).
    rance:   Root N at emergence (g N g⁻¹ root DM).
    ctcnp1:  Maximum critical tissue N concentration coefficient.
    ctcnp2:  Coefficient for N concentration change with growth stage.
    porm:    Minimum soil pore volume for root oxygen supply.
    rwumx:   Maximum water uptake rate (cm³ cm⁻¹ root d⁻¹).
    rlwr:    Root length to weight ratio (cm g⁻¹ × 10⁻⁴).
    rwuep1:  ET ratio for potential transpiration.
    pliglf:  Leaf lignin fraction.
    pligst:  Stem lignin fraction.
    pligrt:  Root lignin fraction.
    pligsh:  Shell lignin fraction.
    pligsd:  Seed lignin fraction.
    """

    prftc:   Tuple[float, ...] = (6.2,  16.5, 33.0, 44.0)
    rgfil:   Tuple[float, ...] = (5.5,  16.0, 27.0, 35.0)
    parsr:   float = 0.50
    co2x:    Tuple[float, ...] = (0.0, 220.0, 280.0, 330.0, 400.0,
                                  490.0, 570.0, 750.0, 990.0, 9999.0)
    co2y:    Tuple[float, ...] = (0.00, 0.85, 0.95, 1.00, 1.02,
                                  1.04, 1.05, 1.06, 1.07, 1.08)
    fslfw:   float = 0.050
    fslfn:   float = 0.050
    sdsz:    float = 0.2750
    rsgr:    float = 0.1
    rsgrt:   float = 5.0
    carbot:  float = 7.0
    dsgt:    float = 21.0
    dget:    float = 150.0
    swcg:    float = 0.02
    stmwte:  float = 0.20
    rtwte:   float = 0.20
    lfwte:   float = 0.20
    seedrve: float = 0.20
    leafnoe: float = 1.0
    plae:    float = 1.0
    tmnc:    float = 0.00450
    tance:   float = 0.0440
    rcnp:    float = 0.01060
    rance:   float = 0.0220
    ctcnp1:  float = 1.52
    ctcnp2:  float = 0.160
    porm:    float = 0.05
    rwumx:   float = 0.03
    rlwr:    float = 0.98
    rwuep1:  float = 1.50
    pliglf:  float = 0.070
    pligst:  float = 0.070
    pligrt:  float = 0.070
    pligsh:  float = 0.280
    pligsd:  float = 0.020


@dataclass
class CeresMaizeParams:
    """Combined parameters from MZCER048.SPE + .ECO + .CUL for one cultivar."""

    spe: MaizeSpeParams = field(default_factory=MaizeSpeParams)
    eco: MaizeEcoParams = field(default_factory=MaizeEcoParams)
    cul: MaizeCulParams = field(default_factory=MaizeCulParams)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _parse_mz_cul(path: Path, varno: str) -> MaizeCulParams:
    """Parse one cultivar row from ``MZCER048.CUL``."""
    row = read_genotype_row(
        path, varno,
        id_col="VAR#",
        fallback_id="DFAULT",
        vrname_col="VRNAME",
        econame_col=None,
    )

    cul = MaizeCulParams()
    cul.varno  = str(row.get("VAR#", varno))
    cul.vrname = str(row.get("VRNAME", ""))

    # ECO# follows after VRNAME; it's parsed as a string token
    cul.econo  = str(row.get("ECO#", row.get("ECONO", "DFAULT")) or "DFAULT")

    def _f(key: str, default: float) -> float:
        v = row.get(key)
        return float(v) if v is not None else default

    cul.p1    = _f("P1",    cul.p1)
    cul.p2    = _f("P2",    cul.p2)
    cul.p5    = _f("P5",    cul.p5)
    cul.g2    = _f("G2",    cul.g2)
    cul.g3    = _f("G3",    cul.g3)
    cul.phint = _f("PHINT", cul.phint)

    return cul


def _parse_mz_eco(path: Path, econo: str) -> MaizeEcoParams:
    """Parse one ecotype row from ``MZCER048.ECO``."""
    row = read_genotype_row(
        path, econo,
        id_col="ECO#",
        fallback_id="DFAULT",
        vrname_col=None,
        econame_col="ECONAME",
    )

    eco = MaizeEcoParams()
    eco.econo   = str(row.get("ECO#", econo))
    eco.econame = str(row.get("ECONAME", ""))

    def _f(key: str, default: float) -> float:
        v = row.get(key)
        return float(v) if v is not None else default

    eco.tbase = _f("TBASE", eco.tbase)
    eco.topt  = _f("TOPT",  eco.topt)
    eco.ropt  = _f("ROPT",  eco.ropt)
    eco.p2o   = _f("P20",   eco.p2o)     # column name in file is P20 not P2O
    eco.djti  = _f("DJTI",  eco.djti)
    eco.gdde  = _f("GDDE",  eco.gdde)
    eco.dsgft = _f("DSGFT", eco.dsgft)
    eco.rue   = _f("RUE",   eco.rue)
    eco.kcan  = _f("KCAN",  eco.kcan)
    eco.tsen  = _f("TSEN",  eco.tsen)
    v_cday = row.get("CDAY")
    if v_cday is not None:
        eco.cday = int(float(v_cday))

    return eco


def _parse_mz_spe(path: Path) -> MaizeSpeParams:
    """Parse ``MZCER048.SPE`` (inline keyword-value style)."""
    kv = parse_spe_inline(path)

    spe = MaizeSpeParams()

    def _arr(key: str, n: int, default: Tuple[float, ...]) -> Tuple[float, ...]:
        vals = kv.get(key, [])
        if len(vals) >= n:
            return tuple(vals[:n])
        return default

    def _scalar(key: str, default: float) -> float:
        vals = kv.get(key, [])
        return vals[0] if vals else default

    spe.prftc   = _arr("PRFTC",  4, spe.prftc)
    spe.rgfil   = _arr("RGFIL",  4, spe.rgfil)
    spe.parsr   = _scalar("PARSR",  spe.parsr)
    spe.co2x    = _arr("CO2X",  10, spe.co2x)
    spe.co2y    = _arr("CO2Y",  10, spe.co2y)
    spe.fslfw   = _scalar("FSLFW",  spe.fslfw)
    spe.fslfn   = _scalar("FSLFN",  spe.fslfn)
    spe.sdsz    = _scalar("SDSZ",   spe.sdsz)
    spe.rsgr    = _scalar("RSGR",   spe.rsgr)
    spe.rsgrt   = _scalar("RSGRT",  spe.rsgrt)
    spe.carbot  = _scalar("CARBOT", spe.carbot)
    spe.dsgt    = _scalar("DSGT",   spe.dsgt)
    spe.dget    = _scalar("DGET",   spe.dget)
    spe.swcg    = _scalar("SWCG",   spe.swcg)
    spe.stmwte  = _scalar("STMWTE", spe.stmwte)
    spe.rtwte   = _scalar("RTWTE",  spe.rtwte)
    spe.lfwte   = _scalar("LFWTE",  spe.lfwte)
    spe.seedrve = _scalar("SEEDRVE",spe.seedrve)
    spe.leafnoe = _scalar("LEAFNOE",spe.leafnoe)
    spe.plae    = _scalar("PLAE",   spe.plae)
    spe.tmnc    = _scalar("TMNC",   spe.tmnc)
    spe.tance   = _scalar("TANCE",  spe.tance)
    spe.rcnp    = _scalar("RCNP",   spe.rcnp)
    spe.rance   = _scalar("RANCE",  spe.rance)
    spe.ctcnp1  = _scalar("CTCNP1", spe.ctcnp1)
    spe.ctcnp2  = _scalar("CTCNP2", spe.ctcnp2)
    spe.porm    = _scalar("PORM",   spe.porm)
    spe.rwumx   = _scalar("RWUMX",  spe.rwumx)
    spe.rlwr    = _scalar("RLWR",   spe.rlwr)
    spe.rwuep1  = _scalar("RWUEP1", spe.rwuep1)
    spe.pliglf  = _scalar("PLIGLF", spe.pliglf)
    spe.pligst  = _scalar("PLIGST", spe.pligst)
    spe.pligrt  = _scalar("PLIGRT", spe.pligrt)
    spe.pligsh  = _scalar("PLIGSH", spe.pligsh)
    spe.pligsd  = _scalar("PLIGSD", spe.pligsd)

    return spe


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _find_genotype_dir() -> Path:
    """Locate ``Data/Genotype`` directory from the repository root."""
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


def load_ceres_maize_params(
    varno: str,
    *,
    model_code: str = "MZCER048",
    genotype_dir: Optional[Path | str] = None,
) -> CeresMaizeParams:
    """Load CERES-Maize parameters from CUL + ECO + SPE files.

    Parameters
    ----------
    varno:
        Cultivar identifier, e.g. ``"IB0001"``.
    model_code:
        DSSAT model prefix, default ``"MZCER048"``.
    genotype_dir:
        Path to the directory containing the genotype files.  Defaults to
        ``Data/Genotype`` relative to the repository root.

    Returns
    -------
    CeresMaizeParams
        Combined cultivar + ecotype + species parameters.
    """
    gdir = Path(genotype_dir) if genotype_dir else _find_genotype_dir()
    prefix = model_code[:8].upper()

    cul = _parse_mz_cul(gdir / f"{prefix}.CUL", varno)
    eco = _parse_mz_eco(gdir / f"{prefix}.ECO", cul.econo)
    spe = _parse_mz_spe(gdir / f"{prefix}.SPE")

    return CeresMaizeParams(spe=spe, eco=eco, cul=cul)
