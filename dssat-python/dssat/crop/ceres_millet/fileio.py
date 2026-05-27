"""CERES-Millet parameter file readers (MLCER048.CUL / .ECO / .SPE).

ECO columns: ECO#, ECONAME, TBASE, TOPT, ROPT, DJTI, GDDE, RUE, KCAN
CUL columns: VAR#, VAR-NAME, EXPNO, ECO#, P1, P2O, P2R, P5, G1, G4,
             PHINT, GT, G5
SPE: inline keyword-value format (same as MZCER048).

No values are invented; all defaults match the DFAULT/file values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from dssat.io.converters.genotype_io import read_genotype_row, parse_spe_inline


@dataclass
class MilletCulParams:
    varno:  str   = "DFAULT"
    vrname: str   = "Default Millet"
    econo:  str   = "DFAULT"
    p1:     float = 300.0   # Thermal time end-juvenile phase (°C d base 8°C)
    p2o:    float = 12.5    # Critical photoperiod (h)
    p2r:    float = 0.0     # Photoperiod sensitivity (d h⁻¹)
    p5:     float = 400.0   # Grain fill thermal time (°C d)
    g1:     float = 20.0    # Kernel number coefficient
    g4:     float = 3.5     # Kernel weight coefficient
    phint:  float = 40.0    # Phyllochron interval (°C d leaf⁻¹)
    gt:     float = 0.0     # Tiller growth coefficient
    g5:     float = 0.0     # Other growth coefficient


@dataclass
class MilletEcoParams:
    econo:   str   = "DFAULT"
    econame: str   = ""
    tbase:   float = 8.0
    topt:    float = 33.0
    ropt:    float = 33.0
    djti:    float = 4.0
    gdde:    float = 6.0
    rue:     float = 3.5
    kcan:    float = 0.85


@dataclass
class MilletSpeParams:
    prftc:   Tuple[float, ...] = (11.0, 22.0, 35.0, 48.0)
    rgfil:   Tuple[float, ...] = (7.0,  22.0, 27.0, 60.0)
    rgset:   Tuple[float, ...] = (-10., 12.0, 33.0, 39.0)
    rglai:   Tuple[float, ...] = (8.0,  23.0, 32.0, 42.0)
    parsr:   float = 0.50
    co2x:    Tuple[float, ...] = (0.0, 220.0, 280.0, 330.0, 400.0,
                                  490.0, 570.0, 750.0, 990.0, 9999.0)
    co2y:    Tuple[float, ...] = (0.00, 0.85, 0.95, 1.00, 1.02,
                                  1.04, 1.05, 1.06, 1.07, 1.08)
    fslfw:   float = 0.050
    fslfn:   float = 0.050
    dsgt:    float = 21.0
    dget:    float = 150.0
    swcg:    float = 0.02
    porm:    float = 0.05
    rwumx:   float = 0.03
    rlwr:    float = 0.98
    rwuep1:  float = 1.50


@dataclass
class CeresMilletParams:
    spe: MilletSpeParams = field(default_factory=MilletSpeParams)
    eco: MilletEcoParams = field(default_factory=MilletEcoParams)
    cul: MilletCulParams = field(default_factory=MilletCulParams)


def _parse_ml_cul(path: Path, varno: str) -> MilletCulParams:
    row = read_genotype_row(path, varno, id_col="VAR#", fallback_id="DFAULT",
                            vrname_col="VAR-NAME", econame_col=None)
    cul = MilletCulParams()
    cul.varno  = str(row.get("VAR#", varno))
    cul.vrname = str(row.get("VAR-NAME", row.get("VRNAME", "")))
    cul.econo  = str(row.get("ECO#", row.get("ECONO", "DFAULT")) or "DFAULT")

    def _f(k, d):
        v = row.get(k); return float(v) if v is not None else d

    cul.p1    = _f("P1",    cul.p1)
    cul.p2o   = _f("P2O",   cul.p2o)
    cul.p2r   = _f("P2R",   cul.p2r)
    cul.p5    = _f("P5",    cul.p5)
    cul.g1    = _f("G1",    cul.g1)
    cul.g4    = _f("G4",    cul.g4)
    cul.phint = _f("PHINT", cul.phint)
    cul.gt    = _f("GT",    cul.gt)
    cul.g5    = _f("G5",    cul.g5)
    return cul


def _parse_ml_eco(path: Path, econo: str) -> MilletEcoParams:
    row = read_genotype_row(path, econo, id_col="ECO#", fallback_id="DFAULT",
                            vrname_col=None, econame_col="ECONAME")
    eco = MilletEcoParams()
    eco.econo   = str(row.get("ECO#", econo))
    eco.econame = str(row.get("ECONAME", ""))

    def _f(k, d):
        v = row.get(k); return float(v) if v is not None else d

    eco.tbase = _f("TBASE", eco.tbase)
    eco.topt  = _f("TOPT",  eco.topt)
    eco.ropt  = _f("ROPT",  eco.ropt)
    eco.djti  = _f("DJTI",  eco.djti)
    eco.gdde  = _f("GDDE",  eco.gdde)
    eco.rue   = _f("RUE",   eco.rue)
    eco.kcan  = _f("KCAN",  eco.kcan)
    return eco


def _parse_ml_spe(path: Path) -> MilletSpeParams:
    kv = parse_spe_inline(path)
    spe = MilletSpeParams()

    def _a(k, n, d):
        v = kv.get(k, [])
        return tuple(v[:n]) if len(v) >= n else d

    def _s(k, d):
        v = kv.get(k, [])
        return v[0] if v else d

    spe.prftc  = _a("PRFTC", 4, spe.prftc)
    spe.rgfil  = _a("RGFIL", 4, spe.rgfil)
    spe.rgset  = _a("RGSET", 4, spe.rgset)
    spe.rglai  = _a("RGLAI", 4, spe.rglai)
    spe.parsr  = _s("PARSR",  spe.parsr)
    spe.co2x   = _a("CO2X", 10, spe.co2x)
    spe.co2y   = _a("CO2Y", 10, spe.co2y)
    spe.fslfw  = _s("FSLFW",  spe.fslfw)
    spe.fslfn  = _s("FSLFN",  spe.fslfn)
    spe.dsgt   = _s("DSGT",   spe.dsgt)
    spe.dget   = _s("DGET",   spe.dget)
    spe.swcg   = _s("SWCG",   spe.swcg)
    spe.porm   = _s("PORM",   spe.porm)
    spe.rwumx  = _s("RWMX",   spe.rwumx)
    spe.rlwr   = _s("RLWR",   spe.rlwr)
    spe.rwuep1 = _s("RWUEP1", spe.rwuep1)
    return spe


def _find_genotype_dir() -> Path:
    env = os.environ.get("DSSAT_GENOTYPE_DIR")
    if env:
        return Path(env)
    p = Path(__file__).resolve().parent
    for _ in range(12):
        c = p / "Data" / "Genotype"
        if c.is_dir():
            return c
        p = p.parent
    raise FileNotFoundError("Cannot locate Data/Genotype directory.")


def load_ceres_millet_params(
    varno: str,
    *,
    model_code: str = "MLCER048",
    genotype_dir: Optional[Path | str] = None,
) -> CeresMilletParams:
    """Load CERES-Millet parameters from CUL + ECO + SPE files."""
    gdir = Path(genotype_dir) if genotype_dir else _find_genotype_dir()
    prefix = model_code[:8].upper()
    cul = _parse_ml_cul(gdir / f"{prefix}.CUL", varno)
    eco = _parse_ml_eco(gdir / f"{prefix}.ECO", cul.econo)
    spe = _parse_ml_spe(gdir / f"{prefix}.SPE")
    return CeresMilletParams(spe=spe, eco=eco, cul=cul)
