"""CERES-Sorghum parameter file readers (SGCER048.CUL / .ECO / .SPE).

ECO columns: ECO#, ECONAME, TBASE, TOPT, ROPT, GDDE, RUE, KCAN,
             STPC, RTPC, TILFC, PLAM
CUL columns: VAR#, VAR-NAME, EXPNO, ECO#, P1, P2, P2O, P2R, PANTH,
             P3, P4, P5, PHINT, G1, G2, PBASE, PSAT
SPE: inline keyword-value format.

No values are invented; all defaults match the file values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from dssat.io.converters.genotype_io import read_genotype_row, parse_spe_inline


@dataclass
class SorghumCulParams:
    varno:  str   = "DFAULT"
    vrname: str   = "Default Sorghum"
    econo:  str   = "DFAULT"
    p1:     float = 300.0   # Thermal time end-juvenile (°C d base 10°C)
    p2:     float = 0.5     # Photoperiod sensitivity (d h⁻¹)
    p2o:    float = 12.5    # Critical photoperiod (h)
    p2r:    float = 0.0     # Photoperiod response slope
    panth:  float = 60.0    # Thermal time panicle initiation to anthesis (°C d)
    p3:     float = 0.0     # Phase 3 duration
    p4:     float = 0.0     # Phase 4 duration
    p5:     float = 400.0   # Grain fill thermal time (°C d)
    phint:  float = 38.9    # Phyllochron interval (°C d leaf⁻¹)
    g1:     float = 20.0    # Kernel number coefficient
    g2:     float = 8.0     # Kernel weight coefficient (mg d⁻¹ kernel⁻¹)
    pbase:  float = 10.0    # Base photoperiod (h)
    psat:   float = 14.0    # Saturating photoperiod (h)


@dataclass
class SorghumEcoParams:
    econo:   str   = "DFAULT"
    econame: str   = ""
    tbase:   float = 8.0
    topt:    float = 34.0
    ropt:    float = 34.0
    gdde:    float = 6.0
    rue:     float = 4.0
    kcan:    float = 0.85
    stpc:    float = 0.10   # Stem partitioning coefficient
    rtpc:    float = 0.25   # Root partitioning coefficient
    tilfc:   float = 0.0    # Tiller fraction coefficient
    plam:    float = 0.0    # Plant area modification factor


@dataclass
class SorghumSpeParams:
    prftc:   Tuple[float, ...] = (8.0,  20.0, 40.0, 44.0)
    rgfil:   Tuple[float, ...] = (7.0,  22.0, 27.0, 35.0)
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
    sla1:    float = 500.0  # SLA stage 1 (cm² g⁻¹)
    sla2:    float = 300.0  # SLA stage 2
    sla3:    float = 200.0  # SLA stage 3
    stpc:    float = 0.10   # Stem partitioning (default from SPE)
    rtpc:    float = 0.25   # Root partitioning


@dataclass
class CeresSorghumParams:
    spe: SorghumSpeParams = field(default_factory=SorghumSpeParams)
    eco: SorghumEcoParams = field(default_factory=SorghumEcoParams)
    cul: SorghumCulParams = field(default_factory=SorghumCulParams)


def _parse_sg_cul(path: Path, varno: str) -> SorghumCulParams:
    row = read_genotype_row(path, varno, id_col="VAR#", fallback_id="DFAULT",
                            vrname_col="VAR-NAME", econame_col=None)
    cul = SorghumCulParams()
    cul.varno  = str(row.get("VAR#", varno))
    cul.vrname = str(row.get("VAR-NAME", row.get("VRNAME", "")))
    cul.econo  = str(row.get("ECO#", row.get("ECONO", "DFAULT")) or "DFAULT")

    def _f(k, d):
        v = row.get(k); return float(v) if v is not None else d

    cul.p1    = _f("P1",    cul.p1)
    cul.p2    = _f("P2",    cul.p2)
    cul.p2o   = _f("P2O",   cul.p2o)
    cul.p2r   = _f("P2R",   cul.p2r)
    cul.panth = _f("PANTH", cul.panth)
    cul.p3    = _f("P3",    cul.p3)
    cul.p4    = _f("P4",    cul.p4)
    cul.p5    = _f("P5",    cul.p5)
    cul.phint = _f("PHINT", cul.phint)
    cul.g1    = _f("G1",    cul.g1)
    cul.g2    = _f("G2",    cul.g2)
    cul.pbase = _f("PBASE", cul.pbase)
    cul.psat  = _f("PSAT",  cul.psat)
    return cul


def _parse_sg_eco(path: Path, econo: str) -> SorghumEcoParams:
    row = read_genotype_row(path, econo, id_col="ECO#", fallback_id="DFAULT",
                            vrname_col=None, econame_col="ECONAME")
    eco = SorghumEcoParams()
    eco.econo   = str(row.get("ECO#", econo))
    eco.econame = str(row.get("ECONAME", ""))

    def _f(k, d):
        v = row.get(k); return float(v) if v is not None else d

    eco.tbase = _f("TBASE", eco.tbase)
    eco.topt  = _f("TOPT",  eco.topt)
    eco.ropt  = _f("ROPT",  eco.ropt)
    eco.gdde  = _f("GDDE",  eco.gdde)
    eco.rue   = _f("RUE",   eco.rue)
    eco.kcan  = _f("KCAN",  eco.kcan)
    eco.stpc  = _f("STPC",  eco.stpc)
    eco.rtpc  = _f("RTPC",  eco.rtpc)
    eco.tilfc = _f("TILFC", eco.tilfc)
    eco.plam  = _f("PLAM",  eco.plam)
    return eco


def _parse_sg_spe(path: Path) -> SorghumSpeParams:
    kv = parse_spe_inline(path)
    spe = SorghumSpeParams()

    def _a(k, n, d):
        v = kv.get(k, [])
        return tuple(v[:n]) if len(v) >= n else d

    def _s(k, d):
        v = kv.get(k, [])
        return v[0] if v else d

    spe.prftc  = _a("PRFTC", 4, spe.prftc)
    spe.rgfil  = _a("RGFIL", 4, spe.rgfil)
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
    spe.sla1   = _s("SLA1",   spe.sla1)
    spe.sla2   = _s("SLA2",   spe.sla2)
    spe.sla3   = _s("SLA3",   spe.sla3)
    spe.stpc   = _s("STPC",   spe.stpc)
    spe.rtpc   = _s("RTPC",   spe.rtpc)
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


def load_ceres_sorghum_params(
    varno: str,
    *,
    model_code: str = "SGCER048",
    genotype_dir: Optional[Path | str] = None,
) -> CeresSorghumParams:
    """Load CERES-Sorghum parameters from CUL + ECO + SPE files."""
    gdir = Path(genotype_dir) if genotype_dir else _find_genotype_dir()
    prefix = model_code[:8].upper()
    cul = _parse_sg_cul(gdir / f"{prefix}.CUL", varno)
    eco = _parse_sg_eco(gdir / f"{prefix}.ECO", cul.econo)
    spe = _parse_sg_spe(gdir / f"{prefix}.SPE")
    return CeresSorghumParams(spe=spe, eco=eco, cul=cul)
