"""DSSAT experiment file (``.MZX``, ``.WHX``, ``.SBX``, …) ↔ JSON converter.

DSSAT experiment files encode one or more treatments, each pointing to factor
levels in sections for cultivars, fields, management, and simulation controls.
This converter handles the most common sections:

- ``*GENERAL``
- ``*TREATMENTS``
- ``*CULTIVARS``
- ``*FIELDS``
- ``*INITIAL CONDITIONS``
- ``*PLANTING DETAILS``
- ``*FERTILIZERS (INORGANIC)``
- ``*IRRIGATION AND WATER MANAGEMENT``
- ``*TILLAGE``
- ``*HARVEST DETAILS``
- ``*SIMULATION CONTROLS``

Reading strategy
~~~~~~~~~~~~~~~~
All sections are parsed into tables keyed by their level number.  The first
treatment is then "resolved" — its factor-level pointers are used to select
the correct row from each section — producing a flat single-treatment JSON
matching the experiment schema.

Writing strategy
~~~~~~~~~~~~~~~~
A single-treatment JSON is converted into a minimal X-file containing exactly
one treatment with one level per section.

The JSON schema produced/consumed here is the same as
``dssat/io/schemas/experiment.schema.json``.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_MISSING = {"-99", "-9", ".", ""}
_SECTION_RE = re.compile(r"^\*(\S[^-]*?)(?:\s+-{3,}.*)?$")


def _v(tok: str) -> float | None:
    tok = tok.strip()
    if tok in _MISSING:
        return None
    try:
        return float(tok)
    except ValueError:
        return None


def _vi(tok: str) -> int | None:
    v = _v(tok)
    return int(v) if v is not None else None


def _s(tok: str) -> str | None:
    tok = tok.strip()
    return None if tok in _MISSING else tok


def _yyddd(raw: str) -> int | None:
    """Parse a YYDDD or YYYYDDD date token → 7-digit YYYYDDD int."""
    try:
        n = int(raw.strip())
    except ValueError:
        return None
    if n <= 0:
        return None
    if n < 100_000:
        # 5-digit YYDDD
        yy = n // 1000
        ddd = n % 1000
        yyyy = 2000 + yy if yy <= 30 else 1900 + yy
        return yyyy * 1000 + ddd
    return n  # already 7-digit


def _yyddd_out(yyyyddd: int) -> str:
    """Format a 7-digit YYYYDDD as 5-digit YYDDD for X-file output."""
    yyyy = yyyyddd // 1000
    ddd = yyyyddd % 1000
    yy = yyyy % 100
    return f"{yy:02d}{ddd:03d}"


def _toks(line: str) -> list[str]:
    return line.split()


def _section_name(line: str) -> str | None:
    m = _SECTION_RE.match(line.strip())
    return m.group(1).upper() if m else None


# ---------------------------------------------------------------------------
# reader
# ---------------------------------------------------------------------------

def read_xfile(path: str | Path) -> dict[str, Any]:
    """Parse a DSSAT experiment file into a JSON-compatible dict.

    Only the *first* treatment is resolved.  All raw section tables are also
    available under the ``"_tables"`` key for advanced use.

    Args:
        path: Path to the experiment file (``.MZX``, ``.WHX``, etc.).

    Returns:
        Experiment dict matching ``experiment.schema.json``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If required sections are missing.

    Example::

        exp = read_xfile("UFGA8201.MZX")
        print(exp["experiment_id"], exp["crop"]["cultivar"]["id"])
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"X-file not found: {path}")
    text = path.read_text(encoding="latin-1")
    tables = _parse_sections(text)
    return _resolve_treatment(tables, path)


def _parse_sections(text: str) -> dict[str, Any]:
    """Parse all sections into dict of lists (tables)."""
    tables: dict[str, Any] = defaultdict(list)

    current_section = ""
    current_header: list[str] = []
    exp_details = ""

    for raw in text.splitlines():
        line = raw.rstrip()

        if not line.strip() or line.strip().startswith("!"):
            continue

        stripped = line.strip()

        # Section header
        sec = _section_name(stripped)
        if sec is not None:
            if sec.startswith("EXP.DETAILS"):
                exp_details = stripped[len("*EXP.DETAILS:"):].strip()
            current_section = sec
            current_header = []
            continue

        # Column header within section
        if stripped.startswith("@"):
            current_header = stripped.lstrip("@").split()
            continue

        # Data line
        if current_section and current_header:
            parts = line.split()
            if parts:
                row = _zip_row(current_header, parts)
                tables[current_section].append(row)

    tables["_exp_details"] = exp_details
    return dict(tables)


def _zip_row(headers: list[str], values: list[str]) -> dict[str, str]:
    """Zip headers and values; last header captures remaining tokens."""
    row: dict[str, str] = {}
    for i, h in enumerate(headers):
        if i < len(values) - 1:
            row[h.upper()] = values[i]
        elif i == len(headers) - 1:
            # Last header gets all remaining tokens joined
            row[h.upper()] = " ".join(values[i:])
        else:
            row[h.upper()] = values[i] if i < len(values) else "-99"
    return row


def _first(table: list[dict], n_col: str, n_val: str | int) -> dict | None:
    """Find first row in a table where column n_col matches n_val."""
    target = str(n_val).strip()
    for row in table:
        if row.get(n_col.upper(), "").strip() == target:
            return row
    return None


def _resolve_treatment(tables: dict, path: Path) -> dict[str, Any]:
    exp: dict[str, Any] = {}

    # --- experiment metadata ---
    details = tables.get("_exp_details", "")
    exp["experiment_id"] = details.split()[0] if details.split() else path.stem.upper()
    exp["description"] = " ".join(details.split()[1:]) if len(details.split()) > 1 else ""

    # --- pick first treatment ---
    treatments = tables.get("TREATMENTS", [])
    if not treatments:
        raise ValueError("No *TREATMENTS section found in X-file.")
    treat = treatments[0]
    tnum = treat.get("N", "1")

    # factor levels: CU, FL, SA, IC, MP, MI, MF, MR, MC, MT, ME, MH, SM
    cu = treat.get("CU", "1")
    fl = treat.get("FL", "1")
    ic_lv = treat.get("IC", "1")
    mp = treat.get("MP", "1")
    mi = treat.get("MI", "0")
    mf = treat.get("MF", "0")
    mh = treat.get("MH", "1")
    sm = treat.get("SM", "1")

    # --- cultivar ---
    cul_row = _first(tables.get("CULTIVARS", []), "C", cu) or {}
    cr = cul_row.get("CR", "MZ")
    crop_model_map = {
        "MZ": "MZCER", "WH": "WHCER", "SB": "SBGRO", "SG": "SGCER",
        "ML": "MLCER", "BA": "BACER", "TR": "TRPNI", "PE": "PZCER",
        "PT": "PTPNI", "BN": "BNPNI", "TM": "TMGRO", "CS": "CSCAS",
        "SC": "SCCAN", "PI": "PIALO",
    }
    model = crop_model_map.get(cr.upper()[:2], f"{cr.upper()[:2]}CER")
    exp["crop"] = {
        "model": model,
        "cultivar": {
            "id": _s(cul_row.get("INGENO", "-99")) or "-99",
            "name": _s(cul_row.get("CNAME", "-99")) or "-99",
        },
    }

    # --- field / weather / soil ---
    fld_row = _first(tables.get("FIELDS", []), "L", fl) or {}
    wsta = _s(fld_row.get("WSTA....", "")) or _s(fld_row.get("WSTA", "")) or ""
    soil_id = _s(fld_row.get("ID_SOIL....", "")) or _s(fld_row.get("ID_SOIL", ""))

    exp["field"] = {
        "wsta": wsta.strip("." if wsta else ""),
        "soil_id": soil_id,
        "lat": _v(fld_row.get("YCRD", "-99")),
        "lon": _v(fld_row.get("XCRD", "-99")),
        "elev": _v(fld_row.get("ELEV", "-99")),
    }
    exp["weather"] = {"station": wsta.strip(".")}

    # --- simulation controls ---
    sim_rows = tables.get("SIMULATION CONTROLS", [])
    sim_ge = _first(sim_rows, "N", sm) or {}
    sim_op = _find_sim_option(sim_rows, sm, "OPTIONS") or {}
    sim_me = _find_sim_option(sim_rows, sm, "METHODS") or {}
    sim_ma = _find_sim_option(sim_rows, sm, "MANAGEMENT") or {}
    sim_ou = _find_sim_option(sim_rows, sm, "OUTPUTS") or {}
    sim_au = _find_sim_option(sim_rows, sm, "AUTOMATIC") or {}

    sdate = _yyddd(sim_ge.get("SDATE", sim_ge.get("YRSIM", "-99") or "-99") or "-99")
    nyrs = _vi(sim_ge.get("NYERS", "1") or "1") or 1
    co2_raw = _s(sim_me.get("CO2", "M") or "M") or "M"

    exp["simulation"] = {
        "start_date": sdate or 0,
        "nyrs": nyrs,
        "water": _s(sim_op.get("WATER", "Y") or "Y") or "Y",
        "nitrogen": _s(sim_op.get("NITRO", "Y") or "Y") or "Y",
        "phosphorus": _s(sim_op.get("PHOSP", "N") or "N") or "N",
        "potassium": _s(sim_op.get("POTAS", "N") or "N") or "N",
        "et_method": _s(sim_me.get("EVAP", "R") or "R") or "R",
        "som_method": _s(sim_me.get("SOM", "G") or "G") or "G",
    }

    # --- initial conditions ---
    ic_rows = [r for r in tables.get("INITIAL CONDITIONS", []) if r.get("C") == ic_lv]
    if ic_rows:
        ic_hdr = ic_rows[0]  # first row has header-level data
        exp["initial_conditions"] = {
            "sw_layers": [],
            "no3_layers": [],
            "nh4_layers": [],
        }
        for row in ic_rows:
            slb = row.get("SLB", "")
            if slb:
                sw = _v(row.get("SH2O", "-99") or "-99")
                no3 = _v(row.get("SNO3", "-99") or "-99")
                nh4 = _v(row.get("SNH4", "-99") or "-99")
                if sw is not None:
                    exp["initial_conditions"]["sw_layers"].append(sw)
                if no3 is not None:
                    exp["initial_conditions"]["no3_layers"].append(no3)
                if nh4 is not None:
                    exp["initial_conditions"]["nh4_layers"].append(nh4)

    # --- planting ---
    plt_rows = [r for r in tables.get("PLANTING DETAILS", []) if r.get("P") == mp]
    if plt_rows:
        pr = plt_rows[0]
        pdate = _yyddd(pr.get("PDATE", "-99") or "-99")
        exp["planting"] = {
            "date": pdate,
            "pltpop": _v(pr.get("PPOP", "-99") or "-99"),
            "sdepth": _v(pr.get("PLDP", "-99") or "-99"),
            "rowspc": _v(pr.get("PLRS", "-99") or "-99"),
        }

    # --- fertilizers ---
    fert_rows = [r for r in tables.get("FERTILIZERS (INORGANIC)", [])
                 if r.get("F") == mf]
    if fert_rows:
        ferts = []
        for fr in fert_rows:
            fdate = _yyddd(fr.get("FDATE", "-99") or "-99")
            amount = _v(fr.get("FAMN", "-99") or "-99")
            if fdate is not None and amount is not None:
                ferts.append({
                    "date": fdate,
                    "amount": amount,
                    "type": _s(fr.get("FMCD", "FE005") or "FE005") or "FE005",
                })
        if ferts:
            exp["fertilizer"] = ferts

    # --- irrigation ---
    irr_rows = [r for r in tables.get("IRRIGATION AND WATER MANAGEMENT", [])
                if r.get("I") == mi]
    if irr_rows:
        irrs = []
        for ir in irr_rows:
            idate = _yyddd(ir.get("IDATE", "-99") or "-99")
            amount = _v(ir.get("IRVAL", "-99") or "-99")
            if idate is not None and amount is not None:
                irrs.append({
                    "date": idate,
                    "amount": amount,
                    "method": _s(ir.get("IMCD", "IR001") or "IR001") or "IR001",
                })
        if irrs:
            exp["irrigation"] = irrs

    # --- harvest ---
    hrv_rows = [r for r in tables.get("HARVEST DETAILS", []) if r.get("H") == mh]
    if hrv_rows:
        hr = hrv_rows[0]
        hdate = _yyddd(hr.get("HDATE", "-99") or "-99")
        exp["harvest"] = {
            "method": _s(hr.get("HCOM", "M") or "M") or "M",
            "date": hdate,
        }

    # Store raw tables for round-trip fidelity
    exp["_tables"] = dict(tables)

    return exp


def _find_sim_option(sim_rows: list[dict], n: str, kw: str) -> dict | None:
    """Find a SIM CONTROLS row with matching N and keyword in GENERAL field."""
    for row in sim_rows:
        if row.get("N", "").strip() == n.strip():
            # The "GENERAL" column holds things like "GE", "OP", "ME", etc.
            gen = row.get("GENERAL", row.get("OPTIONS", row.get("METHODS", "")))
            if kw[:2].upper() in gen.upper():
                return row
    return None


# ---------------------------------------------------------------------------
# writer
# ---------------------------------------------------------------------------

def write_xfile(
    exp: dict[str, Any],
    path: str | Path,
    *,
    crop_code: str | None = None,
) -> None:
    """Write a single-treatment experiment JSON to a DSSAT X-file.

    Args:
        exp: Experiment dict (output of :func:`read_xfile` or hand-built JSON).
        path: Destination file path.  The extension is used as-is (e.g.
            ``.MZX``, ``.WHX``).
        crop_code: Two-letter crop code override (e.g. ``"MZ"``).  Inferred
            from ``exp["crop"]["model"]`` if not given.

    Example::

        write_xfile(exp, "UFGA8201.MZX")
    """
    path = Path(path)
    cr = _infer_crop_code(exp, crop_code)
    lines = _build_xfile(exp, cr)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _infer_crop_code(exp: dict, override: str | None) -> str:
    if override:
        return override.upper()[:2]
    model = exp.get("crop", {}).get("model", "MZCER")
    model_to_cr = {
        "MZCER": "MZ", "WHCER": "WH", "SBGRO": "SB", "SGCER": "SG",
        "MLCER": "ML", "BACER": "BA", "PZCER": "PE", "BNPNI": "BN",
        "TMGRO": "TM", "CSCAS": "CS", "SCCAN": "SC",
    }
    return model_to_cr.get(model.upper()[:5], model[:2].upper())


def _dd(v) -> str:
    """Format a date as 5-digit YYDDD for X-file output."""
    if v is None:
        return "   -99"
    return f"{_yyddd_out(int(v)):>6}"


def _fld(v, w=6) -> str:
    if v is None:
        return f"{'  -99':>{w}}"
    try:
        return f"{float(v):>{w}.1f}"
    except (TypeError, ValueError):
        return f"{'  -99':>{w}}"


def _build_xfile(exp: dict, cr: str) -> list[str]:
    lines: list[str] = []
    eid = exp.get("experiment_id", "UNKNOWN")
    desc = exp.get("description", "")
    model = exp.get("crop", {}).get("model", f"{cr}CER")

    lines.append(f"*EXP.DETAILS: {eid} {desc}")
    lines.append("")

    # GENERAL
    lines.append("*GENERAL")
    lines.append("@PEOPLE")
    lines.append(" DSSAT Python package")
    lines.append("@ADDRESS")
    lines.append(" -99")
    lines.append("@NOTES")
    lines.append(" -99")
    lines.append("")

    # TREATMENTS
    lines.append("*TREATMENTS                        "
                 "-------------FACTOR LEVELS------------")
    lines.append("@N R O C TNAME.................... "
                 "CU FL SA IC MP MI MF MR MC MT ME MH SM")
    tname = f"{eid}-1"[:25].ljust(25)
    # Determine which optional sections are present
    has_irr = 1 if exp.get("irrigation") else 0
    has_fert = 1 if exp.get("fertilizer") else 0
    has_harv = 1 if exp.get("harvest") else 0
    lines.append(
        f" 1 1 0 0 {tname}"
        f"  1  1  0  1  1  {has_irr}  {has_fert}  0  0  0  0  {has_harv}  1"
    )
    lines.append("")

    # CULTIVARS
    cul = exp.get("crop", {}).get("cultivar", {})
    ingeno = (cul.get("id") or "-99")[:6].ljust(6)
    cname = (cul.get("name") or "-99")[:20]
    lines.append("*CULTIVARS")
    lines.append("@C CR INGENO CNAME")
    lines.append(f" 1 {cr} {ingeno} {cname}")
    lines.append("")

    # FIELDS
    fld = exp.get("field", {})
    wsta = (fld.get("wsta") or exp.get("weather", {}).get("station") or "-99")[:8].ljust(8)
    soil_id = (fld.get("soil_id") or "-99")[:10].ljust(10)
    lat = fld.get("lat")
    lon = fld.get("lon")
    elev = fld.get("elev")
    lat_s = f"{lat:8.3f}" if lat is not None else "    -99."
    lon_s = f"{lon:8.3f}" if lon is not None else "    -99."
    elev_s = f"{elev:9.3f}" if elev is not None else "     -99."
    lines.append("*FIELDS")
    lines.append("@L ID_FIELD WSTA....  FLSA  FLOB  FLDT  FLDD  FLDS  "
                 "FLST SLTX  SLDP  ID_SOIL....    FLNAME")
    lines.append(f" 1 {eid[:8]:<8} {wsta}   0.0   0.0 DR000   0.0   0.0 "
                 f"00000 -99    -99  {soil_id}   -99")
    lines.append("@L ...XCRD ...YCRD .....ELEV .............AREA .SLEN "
                 ".FLWR .SLAS FLHST FHDUR")
    lines.append(f" 1{lon_s}{lat_s}{elev_s}         1.000  100. 1.00  1.00   -99  -99")
    lines.append("")

    # INITIAL CONDITIONS
    ic = exp.get("initial_conditions", {})
    sw_layers = ic.get("sw_layers", [])
    no3_layers = ic.get("no3_layers", [])
    nh4_layers = ic.get("nh4_layers", [])
    plt = exp.get("planting", {})
    sim = exp.get("simulation", {})
    icdat = _dd(plt.get("date") or sim.get("start_date"))
    lines.append("*INITIAL CONDITIONS")
    lines.append("@C   PCR ICDAT  ICRT  ICND  ICRN  ICRE  ICWD ICRES ICREN ICREP ICRIP ICRID ICNAME")
    lines.append(f" 1    {cr}{icdat}   300.  100.   .80  1.00   100   -99   -99   -99   -99   -99 -99")
    lines.append("")
    lines.append("@C   SLB  ICBL  SH2O  SNH4  SNO3")
    # Build from soil profile if available; otherwise use initial_conditions arrays
    soil = exp.get("soil", {})
    soil_layers = soil.get("inline", {}).get("layers") or soil.get("layers", [])
    n_layers = max(len(sw_layers), len(soil_layers))
    depths = [l.get("depth_bottom") for l in soil_layers] if soil_layers else []
    for i in range(n_layers):
        depth = depths[i] if i < len(depths) else (i + 1) * 20
        sw = sw_layers[i] if i < len(sw_layers) else -99
        nh4 = nh4_layers[i] if i < len(nh4_layers) else -99
        no3 = no3_layers[i] if i < len(no3_layers) else -99
        depth_s = f"{depth:5.0f}" if depth else "  -99"
        sw_s = f"{sw:6.3f}" if sw != -99 else "  -99."
        nh4_s = f"{nh4:6.3f}" if nh4 != -99 else "  -99."
        no3_s = f"{no3:6.3f}" if no3 != -99 else "  -99."
        lines.append(f" 1  {depth_s}   .050{sw_s}{nh4_s}{no3_s}")
    lines.append("")

    # PLANTING DETAILS
    pdate = _dd(plt.get("date"))
    ppop = _fld(plt.get("pltpop"))
    sdepth = _fld(plt.get("sdepth"))
    rowspc = _fld(plt.get("rowspc"))
    lines.append("*PLANTING DETAILS")
    lines.append("@P PDATE EDATE  PPOP  PPOE  PLME  PLDS  PLRS  PLRD  PLDP  "
                 "PLWT  PAGE  PENV  PLPH  SPRL                        PLNAME")
    lines.append(f" 1{pdate}   -99{ppop}{ppop}   S     R{rowspc}   0.{sdepth}"
                 f"   -99   -99   -99   -99   -99                         -99")
    lines.append("")

    # FERTILIZERS
    ferts = exp.get("fertilizer", [])
    if ferts:
        lines.append("*FERTILIZERS (INORGANIC)")
        lines.append("@F FDATE  FMCD  FACD  FDEP  FAMN  FAMP  FAMK  FAMC  FAMO  FOCD FERNAME")
        for fe in ferts:
            fdate = _dd(fe.get("date"))
            fmcd = (fe.get("type") or "FE005")[:5]
            famn = _fld(fe.get("amount", fe.get("n_pct", 0)))
            lines.append(f" 1{fdate} {fmcd} AP001   0.0{famn}   0.0   0.0   0.0   0.0   -99 -99")
        lines.append("")

    # IRRIGATION
    irrs = exp.get("irrigation", [])
    if irrs:
        lines.append("*IRRIGATION AND WATER MANAGEMENT")
        lines.append("@I  EFIR  IDEP  ITHR  IEPT  IOFF  IAME  IAMT IRNAME")
        lines.append(" 1  1.00    30    50   100 GS000 IR001  -99 -99")
        lines.append("@I IDATE  IROP IRVAL")
        for ir in irrs:
            idate = _dd(ir.get("date"))
            irop = (ir.get("method") or "IR001")[:5]
            irval = _fld(ir.get("amount"))
            lines.append(f" 1{idate} {irop}{irval}")
        lines.append("")

    # HARVEST DETAILS
    harv = exp.get("harvest", {})
    hmethod = (harv.get("method") or "M")[:5]
    hdate = _dd(harv.get("date"))
    hstg = "GS000" if hmethod.upper() in ("M", "A") else "GS000"
    hcom = "HI1" if hmethod.upper() == "M" else "HI1"
    lines.append("*HARVEST DETAILS")
    lines.append("@H HDATE  HSTG  HCOM HSIZE   HPC  HBPC  HNAME")
    lines.append(f" 1{hdate}  {hstg} {hcom}  S       100.   0.  -99")
    lines.append("")

    # SIMULATION CONTROLS
    sdate = _dd(sim.get("start_date"))
    nyrs = sim.get("nyrs", 1)
    water = sim.get("water", "Y")
    nitro = sim.get("nitrogen", "Y")
    phosp = sim.get("phosphorus", "N")
    potas = sim.get("potassium", "N")
    et_me = sim.get("et_method", "R")
    som_me = sim.get("som_method", "G")
    co2_me = "M"
    lines.append("*SIMULATION CONTROLS")
    lines.append("@N GENERAL     NYERS NREPS START SDATE RSEED SNAME.................... SMODEL")
    sname = f"{eid}-1"[:25].ljust(25)
    lines.append(f" 1 GE          {nyrs:5d}     1     S{sdate}  2150 {sname} {model}")
    lines.append("@N OPTIONS     WATER NITRO SYMBI PHOSP POTAS DISES  CHEM  TILL   CO2")
    lines.append(f" 1 OP              {water}     {nitro}     N     {phosp}     {potas}     N     N     N     {co2_me}")
    lines.append("@N METHODS     WTHER INCON LIGHT EVAPO INFIL PHOTO HYDRO NSWIT MESOM MESEV MESOL METMP MEHYD")
    lines.append(f" 1 ME              M     M     E     {et_me}     S     R     R     1     {som_me}     S     2     R     D")
    lines.append("@N MANAGEMENT  PLANT IRRIG FERTI RESID HARVS")
    lines.append(" 1 MA              R     N     N     N     M")
    lines.append("@N OUTPUTS     FNAME OVVEW SUMRY FROPT GROUT CAOUT WAOUT NIOUT MIOUT DIOUT VBOSE CHOUT OPOUT FMOPT")
    lines.append(" 1 OU              N     Y     Y     1     Y     Y     Y     Y     Y     N     Y     N     N     A")
    lines.append("")
    lines.append("@  AUTOMATIC MANAGEMENT")
    lines.append("@N PLANTING    PFRST PLAST PH2OL PH2OU PH2OD PSTMX PSTMN")
    lines.append(" 1 PL          82050 82064    40   100    30    40    10")
    lines.append("@N IRRIGATION  IMDEP ITHRL ITHRU IROFF IMETH IRAMT IREFF")
    lines.append(" 1 IR             30    50   100 GS000 IR001    10  1.00")
    lines.append("@N NITROGEN    NMDEP NMTHR NAMNT NCODE NAOFF")
    lines.append(" 1 NI             30    50    25 FE001 GS000")
    lines.append("@N RESIDUES    RIPCN RTIME RIDEP")
    lines.append(" 1 RE            100     1    20")
    lines.append("@N HARVEST     HFRST HLAST HPCNP HPCNR")
    lines.append(" 1 HA              0 83365   100     0")

    return lines


def xfile_to_json_file(
    xfile_path: str | Path,
    json_path: str | Path | None = None,
) -> Path:
    """Convert an X-file to a JSON experiment file.

    Args:
        xfile_path: Source experiment file (``.MZX``, ``.WHX``, etc.).
        json_path: Destination JSON file (default: same stem + ``.json``).

    Returns:
        Path to the written JSON file.
    """
    xfile_path = Path(xfile_path)
    json_path = Path(json_path) if json_path else xfile_path.with_suffix(".json")
    exp = read_xfile(xfile_path)
    # Don't serialize internal tables
    out = {k: v for k, v in exp.items() if not k.startswith("_")}
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return json_path


def json_to_xfile(
    json_path: str | Path,
    xfile_path: str | Path | None = None,
    *,
    crop_code: str | None = None,
) -> Path:
    """Convert a JSON experiment file to a DSSAT X-file.

    Args:
        json_path: Source experiment JSON file.
        xfile_path: Destination X-file (default: same stem + ``.MZX``
            if crop code is ``"MZ"``).
        crop_code: Two-letter crop code (inferred from JSON if omitted).

    Returns:
        Path to the written X-file.
    """
    json_path = Path(json_path)
    with json_path.open(encoding="utf-8") as f:
        exp = json.load(f)
    cr = _infer_crop_code(exp, crop_code)
    xfile_path = Path(xfile_path) if xfile_path else json_path.with_suffix(f".{cr}X")
    write_xfile(exp, xfile_path, crop_code=cr)
    return xfile_path
