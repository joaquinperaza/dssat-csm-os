"""DSSAT ``.SOL`` soil file ↔ JSON converter.

The DSSAT soil file format contains one or more soil profiles.  Each profile
begins with a line of the form::

    *PEDON_ID  SOURCE      TEXTURE  DEPTH  DESCRIPTION

followed by up to three ``@``-prefixed header sections:

1. ``@SITE``   — geographic metadata
2. ``@SCOM``   — profile-level hydraulic/chemical properties
3. ``@  SLB``  — one line per soil layer

Missing/undefined values are represented by ``-99``.

The JSON schema produced / consumed here matches the ``soil_profile``
``$def`` in ``experiment.schema.json``::

    {
        "id": "IBMZ910014",
        "source": "IBSNAT",
        "texture": "SIC",
        "depth": 210,
        "description": "Millhopper Fine Sand",
        "site": "Gainesville",
        "country": "USA",
        "lat": 29.63,
        "lon": -82.37,
        "family": "Loamy, siliceous, ...",
        "salb": 0.18,
        "slpf": 0.92,
        ...
        "layers": [
            {"depth_bottom": 5, "ll": 0.023, "dul": 0.086, "sat": 0.230,
             "swcn": 7.40, "bd": 1.36, "oc": 0.90, "clay": 2, "silt": 16,
             "ph": 6.0, "shf": 1.0, ...},
            ...
        ]
    }

A ``.SOL`` file typically contains many profiles; :func:`read_sol` returns a
list of dicts, one per profile.  Use :func:`read_sol_profile` to find a
specific profile by ID.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _val(tok: str) -> float | None:
    tok = tok.strip()
    if not tok or tok in ("-99", "-9", "-99.", "-9.", ".", "-99.0"):
        return None
    try:
        return float(tok)
    except ValueError:
        return None


def _parse_header_positions(line: str) -> tuple[list[str], list[tuple[int, int]]]:
    """Return (names, col-spans) from a whitespace-delimited ``@``-header line.

    Column spans are the character ranges where each header's data value
    appears on the *following* data line.
    """
    raw = line.lstrip("@ \t")
    names: list[str] = []
    starts: list[int] = []

    i = 0
    while i < len(raw):
        if raw[i] == " ":
            i += 1
            continue
        j = i
        while j < len(raw) and raw[j] != " ":
            j += 1
        # Offset by length of stripped prefix to get position in original line
        prefix_len = len(line) - len(line.lstrip("@ \t")) + len(line.lstrip()) - len(raw)
        col_start = i + (len(line) - len(line.lstrip("@ \t")))
        names.append(raw[i:j].strip())
        starts.append(col_start)
        i = j

    # Build spans: each spans from its start to next header's start
    spans: list[tuple[int, int]] = []
    for k in range(len(starts)):
        end = starts[k + 1] if k + 1 < len(starts) else starts[k] + 15
        spans.append((starts[k], end))

    return names, spans


def _tokens_from_line(line: str, names: list[str]) -> dict[str, str]:
    """Split a data line on whitespace and map by position to header names."""
    parts = line.split()
    result: dict[str, str] = {}
    for i, (name, tok) in enumerate(zip(names, parts)):
        result[name.upper()] = tok
    return result


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def read_sol(path: str | Path) -> list[dict[str, Any]]:
    """Read all soil profiles from a DSSAT ``.SOL`` file.

    Args:
        path: Path to the ``.SOL`` file.

    Returns:
        List of soil profile dicts.  The list preserves file order.

    Raises:
        FileNotFoundError: If *path* does not exist.

    Example::

        profiles = read_sol("SOIL.SOL")
        print(profiles[0]["id"], profiles[0]["description"])
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SOL file not found: {path}")

    text = path.read_text(encoding="latin-1")
    return _parse_sol_text(text)


def read_sol_profile(
    path: str | Path, profile_id: str
) -> dict[str, Any] | None:
    """Find and return a single profile by ID from a ``.SOL`` file.

    Args:
        path: Path to the ``.SOL`` file.
        profile_id: The 10-character soil profile ID (e.g. ``"IBMZ910014"``).

    Returns:
        Soil profile dict, or ``None`` if not found.
    """
    for profile in read_sol(path):
        if profile.get("id", "").strip().upper() == profile_id.strip().upper():
            return profile
    return None


def _parse_sol_text(text: str) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    mode = ""  # "site", "scom", "layers"
    layer_names: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()

        # Skip blanks and comments
        if not line.strip() or line.strip().startswith("!"):
            continue

        # New profile block
        if line.startswith("*") and not line.startswith("*SOIL"):
            if current is not None:
                profiles.append(current)
            current = _parse_profile_header(line)
            mode = ""
            layer_names = []
            continue

        if current is None:
            continue

        stripped = line.strip()

        # @-header lines
        if stripped.startswith("@"):
            upper = stripped.upper()
            if "SITE" in upper or "COUNTRY" in upper:
                mode = "site"
                # Store the header for potential column parsing; here we
                # just use split-based approach for the next data line
                current["_site_header"] = stripped
            elif "SCOM" in upper or "SALB" in upper:
                mode = "scom"
                current["_scom_header"] = stripped
            elif "SLB" in upper:
                mode = "layers"
                layer_names = [t.upper() for t in stripped.lstrip("@ ").split()]
                current.setdefault("layers", [])
            continue

        # Data lines per mode
        if mode == "site":
            _parse_site_line(line, current)
            mode = ""
        elif mode == "scom":
            _parse_scom_line(line, current)
            mode = ""
        elif mode == "layers":
            layer = _parse_layer_line(line, layer_names)
            if layer:
                current["layers"].append(layer)

    if current is not None:
        profiles.append(current)

    # Clean up internal keys
    for p in profiles:
        p.pop("_site_header", None)
        p.pop("_scom_header", None)

    return profiles


def _parse_profile_header(line: str) -> dict[str, Any]:
    """Parse the ``*PEDON_ID  SOURCE  TEXTURE  DEPTH  DESCRIPTION`` line.

    Fortran format: ``1X, A10, 2X, A11, 1X, A5, 1X, F5.0, 1X, A50``.
    We use whitespace splitting for the first 4 tokens (ID, source, texture,
    depth) which never contain spaces, then reconstruct the free-text
    description from the remainder.  This is robust to minor column-width
    variations across real ``.SOL`` files.
    """
    content = line[1:].rstrip()  # strip leading '*'
    profile: dict[str, Any] = {}

    # First 4 whitespace tokens are always: ID  SOURCE  TEXTURE  DEPTH
    tokens = content.split()
    if not tokens:
        return {"id": "", "description": "", "layers": []}

    profile["id"] = tokens[0]
    profile["source"] = tokens[1] if len(tokens) > 1 else ""
    profile["texture"] = tokens[2] if len(tokens) > 2 else ""

    depth = _val(tokens[3]) if len(tokens) > 3 else None
    if depth is not None:
        profile["depth"] = depth

    # Description: everything after the first 4 tokens.
    # Reconstruct from the original string by skipping 4 whitespace-delimited
    # fields to preserve multi-word descriptions faithfully.
    idx = 0
    for _ in range(4):
        while idx < len(content) and content[idx] == " ":
            idx += 1
        while idx < len(content) and content[idx] != " ":
            idx += 1
    profile["description"] = content[idx:].strip()
    profile["layers"] = []
    return profile


def _parse_site_line(line: str, profile: dict) -> None:
    """Parse the SITE / COUNTRY / LAT / LONG / FAMILY data line."""
    # Typical format (free-field after header):
    # SITE(15) COUNTRY(15) LAT(8.3) LONG(8.3) FAMILY(rest)
    parts = line.split()
    if not parts:
        return
    # Try to find numeric lat/lon within the tokens
    found_nums: list[tuple[int, float]] = []
    for i, tok in enumerate(parts):
        try:
            v = float(tok)
            found_nums.append((i, v))
        except ValueError:
            pass

    if len(found_nums) >= 2:
        lat_idx, lat = found_nums[0]
        lon_idx, lon = found_nums[1]
        profile["lat"] = lat
        profile["lon"] = lon
        profile["site"] = " ".join(parts[:lat_idx]).strip() or parts[0]
        family_start = lon_idx + 1
        if family_start < len(parts):
            profile["family"] = " ".join(parts[family_start:]).strip()
    elif len(parts) >= 2:
        profile["site"] = parts[0]
        profile["country"] = parts[1] if len(parts) > 1 else ""


def _parse_scom_line(line: str, profile: dict) -> None:
    """Parse the SCOM / SALB / SLU1 / SLDR / … data line."""
    _SCOM_KEYS = [
        "scom", "salb", "slu1", "sldr", "slro", "slnf", "slpf",
        "smhb", "smpx", "smke",
    ]
    parts = line.split()
    for key, tok in zip(_SCOM_KEYS, parts):
        if key in ("scom", "smhb", "smpx", "smke"):
            if tok not in ("-99", ".", ""):
                profile[key] = tok
        else:
            v = _val(tok)
            if v is not None:
                profile[key] = v


def _parse_layer_line(line: str, names: list[str]) -> dict[str, Any] | None:
    """Parse one soil layer data line."""
    _LAYER_MAP = {
        "SLB":  "depth_bottom",
        "SLMH": "horizon",
        "SLLL": "ll",
        "SDUL": "dul",
        "SSAT": "sat",
        "SRGF": "shf",
        "SSKS": "swcn",
        "SBDM": "bd",
        "SLOC": "oc",
        "SLCL": "clay",
        "SLSI": "silt",
        "SLCF": "cf",
        "SLNI": "tn",
        "SLHW": "ph",
        "SLHB": "phkcl",
        "SCEC": "cec",
        "SADC": "adc",
        # Initial inorganic N (kg N ha⁻¹ per layer)
        "SNH4": "snh4",
        "SNO3": "sno3",
        # Van Genuchten / optional tier 3
        "SLPX": "pox",
        "SLPT": "ptot",
        "SLPO": "porg",
        "CACO3": "caco3",
        "SLAL": "al",
        "SLFE": "fe",
        "SLMN": "mn",
        "SLBS": "bs",
        "SLPA": "pa",
        "SLPB": "pb",
        "SLKE": "ke",
        "SLMG": "mg",
        "SLNA": "na",
        "SLSU": "su",
        "SLEC": "ec",
        "SLCA": "ca",
    }
    parts = line.split()
    if not parts:
        return None
    layer: dict[str, Any] = {}
    for name, tok in zip(names, parts):
        json_key = _LAYER_MAP.get(name.upper())
        if json_key is None:
            continue
        if name.upper() in ("SLMH",):
            if tok not in ("-99", "."):
                layer[json_key] = tok
        else:
            v = _val(tok)
            if v is not None:
                layer[json_key] = v

    if "depth_bottom" not in layer:
        return None
    return layer


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------

def write_sol(
    profiles: list[dict[str, Any]] | dict[str, Any],
    path: str | Path,
) -> None:
    """Write one or more soil profiles to a DSSAT ``.SOL`` file.

    Args:
        profiles: A single soil profile dict or a list of them.
        path: Destination ``.SOL`` file path.

    Example::

        write_sol(my_profile, "MYSOILS.SOL")
        write_sol([profile1, profile2], "ALLSOILS.SOL")
    """
    if isinstance(profiles, dict):
        profiles = [profiles]

    path = Path(path)
    lines: list[str] = []
    lines.append("*SOILS: Written by DSSAT Python package")
    lines.append("")

    for profile in profiles:
        lines.extend(_format_profile(profile))
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fv(val, width: int = 5, decimals: int = 2) -> str:
    """Format a numeric value for SOL output."""
    if val is None:
        return f"{'  -99':{width}}"[:width]
    try:
        return f"{float(val):{width}.{decimals}f}"
    except (TypeError, ValueError):
        return f"{'  -99':{width}}"[:width]


def _format_profile(p: dict[str, Any]) -> list[str]:
    lines: list[str] = []

    pid = (p.get("id") or "UNKNOWN   ")[:10].ljust(10)
    source = (p.get("source") or "-99        ")[:11].ljust(11)
    texture = (p.get("texture") or "-99  ")[:5].ljust(5)
    depth = f"{float(p.get('depth', -99)):5.0f}" if p.get("depth") else "  -99"
    desc = (p.get("description") or "-99")[:50]
    lines.append(f"*{pid}  {source} {texture} {depth} {desc}")

    # SITE section
    site = p.get("site") or p.get("country") or "-99"
    country = p.get("country", "-99")
    lat = p.get("lat")
    lon = p.get("lon")
    family = p.get("family", "-99")
    lines.append("@SITE        COUNTRY          LAT     LONG SCS FAMILY")
    lat_s = f"{lat:8.2f}" if lat is not None else "   -99.0"
    lon_s = f"{lon:8.2f}" if lon is not None else "   -99.0"
    lines.append(f" {site[:12]:<12} {country[:15]:<15} {lat_s} {lon_s} {family[:50]}")

    # SCOM section
    scom = p.get("scom", "  -99")
    salb = _fv(p.get("salb"), 5, 2)
    slu1 = _fv(p.get("slu1", -99), 5, 1)
    sldr = _fv(p.get("sldr", -99), 5, 2)
    slro = _fv(p.get("slro", -99), 5, 0)
    slnf = _fv(p.get("slnf", 1.0), 5, 2)
    slpf = _fv(p.get("slpf", 1.0), 5, 2)
    smhb = p.get("smhb", "IB001")[:5]
    smpx = p.get("smpx", "IB001")[:5]
    smke = p.get("smke", "IB001")[:5]
    lines.append("@SCOM  SALB  SLU1 SLDR SLRO SLNF SLPF SMHB SMPX SMKE")
    lines.append(
        f"  {scom:<4} {salb} {slu1} {sldr} {slro} {slnf} {slpf} {smhb} {smpx} {smke}"
    )

    # Layer section — detect whether any layer carries initial N data
    layers = p.get("layers", [])
    has_n_init = any(
        layer.get("snh4") is not None or layer.get("sno3") is not None
        for layer in layers
    )

    layer_header = (
        "@  SLB  SLMH  SLLL  SDUL  SSAT  SRGF  SSKS  SBDM  SLOC  "
        "SLCL  SLSI  SLCF  SLNI  SLHW  SLHB  SCEC  SADC"
    )
    if has_n_init:
        layer_header += "  SNH4  SNO3"
    lines.append(layer_header)

    for layer in layers:
        slb = _fv(layer.get("depth_bottom"), 5, 0)
        slmh = (layer.get("horizon") or "  -99")[:5].rjust(5)
        slll = _fv(layer.get("ll"), 6, 3)
        sdul = _fv(layer.get("dul"), 6, 3)
        ssat = _fv(layer.get("sat"), 6, 3)
        srgf = _fv(layer.get("shf", 1.0), 6, 2)
        ssks = _fv(layer.get("swcn"), 6, 2)
        sbdm = _fv(layer.get("bd"), 6, 2)
        sloc = _fv(layer.get("oc"), 6, 3)
        slcl = _fv(layer.get("clay"), 6, 0)
        slsi = _fv(layer.get("silt", -99), 6, 0)
        slcf = _fv(layer.get("cf", 0), 6, 0)
        slni = _fv(layer.get("tn", -99), 6, 2)
        slhw = _fv(layer.get("ph"), 6, 1)
        slhb = _fv(layer.get("phkcl", -99), 6, 1)
        scec = _fv(layer.get("cec", -99), 6, 1)
        sadc = _fv(layer.get("adc", -99), 6, 1)
        row = (
            f"  {slb}{slmh}{slll}{sdul}{ssat}{srgf}{ssks}"
            f"{sbdm}{sloc}{slcl}{slsi}{slcf}{slni}{slhw}{slhb}{scec}{sadc}"
        )
        if has_n_init:
            snh4 = _fv(layer.get("snh4", -99), 6, 2)
            sno3 = _fv(layer.get("sno3", -99), 6, 2)
            row += f"{snh4}{sno3}"
        lines.append(row)

    return lines


# ---------------------------------------------------------------------------
# convenience wrappers
# ---------------------------------------------------------------------------

def sol_to_json_file(
    sol_path: str | Path,
    json_path: str | Path | None = None,
) -> Path:
    """Convert all profiles in a ``.SOL`` file to a JSON array file.

    Args:
        sol_path: Source ``.SOL`` file.
        json_path: Destination JSON file (default: same stem + ``.json``).

    Returns:
        Path to the written JSON file.
    """
    sol_path = Path(sol_path)
    json_path = Path(json_path) if json_path else sol_path.with_suffix(".json")
    profiles = read_sol(sol_path)
    json_path.write_text(json.dumps(profiles, indent=2), encoding="utf-8")
    return json_path


def json_to_sol_file(
    json_path: str | Path,
    sol_path: str | Path | None = None,
) -> Path:
    """Convert a JSON file containing one or more soil profiles to a ``.SOL`` file.

    Args:
        json_path: Source JSON file (list of profiles or single profile).
        sol_path: Destination ``.SOL`` file (default: same stem + ``.SOL``).

    Returns:
        Path to the written ``.SOL`` file.
    """
    json_path = Path(json_path)
    sol_path = Path(sol_path) if sol_path else json_path.with_suffix(".SOL")
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    write_sol(data, sol_path)
    return sol_path
