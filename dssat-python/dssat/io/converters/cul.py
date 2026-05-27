"""DSSAT cultivar file (.CUL) ↔ JSON converter.

Handles the fixed-format DSSAT cultivar coefficient files used by
CERES-Maize (MZCER048.CUL), CERES-Sorghum (SGCER048.CUL),
CERES-Millet (MLCER048.CUL), etc.

The ``.CUL`` file format::

    *MZCER048.CUL

    @VAR#  VRNAME.......... EXPNO   ECO#    P1    P2    P5    G2    G3 PHINT
    IB0001 DEKALB XL71A      . IB0001   220  0.52   730   823  8.80  38.9

Usage::

    from dssat.io.converters.cul import read_cul, write_cul, cul_to_json_file

    cultivars = read_cul("MZCER048.CUL", "MZ")
    write_cul(cultivars, "MZCER048_out.CUL", "MZ")
    json_path = cul_to_json_file("MZCER048.CUL")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _val(tok: str) -> float | str | None:
    """Parse a token as float, or return as string if not numeric."""
    tok = tok.strip()
    if not tok or tok in ("-99", "-9", ".", "-99.", "-99.0"):
        return None
    try:
        return float(tok)
    except ValueError:
        return tok


def _parse_header(line: str) -> list[str]:
    """Parse an ``@``-prefixed header line, returning column names.

    Trailing dots used as DSSAT visual padding (``VRNAME..........``) are
    stripped so the name normalises to ``VRNAME``.
    """
    return [t.upper().rstrip(".") for t in line.lstrip("@ ").split()]


def _parse_data_line(line: str, headers: list[str]) -> dict[str, Any] | None:
    """Parse a cultivar data line into a dict keyed by header names.

    For standard DSSAT CUL layout the first two columns are fixed-width:
    * ``VAR#`` — 6 characters (cols 0-5)
    * ``VRNAME`` — 16 characters (cols 7-22), may contain spaces

    When those two headers are present they are extracted by slice; remaining
    tokens are split on whitespace.
    """
    if not line.strip():
        return None

    result: dict[str, Any] = {}

    if len(headers) >= 2 and headers[0] in ("VAR#", "VARNO") and headers[1] == "VRNAME":
        if len(line) < 8:
            return None
        varno = line[0:6].strip()
        vrname = line[7:27].strip() if len(line) > 27 else line[7:].strip()
        if not varno:
            return None
        result[headers[0]] = varno
        result["VRNAME"] = vrname
        # Remaining tokens parsed as plain whitespace-separated values
        rest_tokens = line[27:].split() if len(line) > 27 else []
        for name, tok in zip(headers[2:], rest_tokens):
            v = _val(tok)
            if v is not None:
                result[name] = v
    else:
        parts = line.split()
        if not parts or len(parts) < 2:
            return None
        for name, tok in zip(headers, parts):
            v = _val(tok)
            if v is not None:
                result[name] = v

    return result if result else None


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def read_cul(path: str | Path, model_code: str = "") -> list[dict[str, Any]]:
    """Read all cultivar records from a DSSAT ``.CUL`` file.

    Args:
        path: Path to the ``.CUL`` file.
        model_code: Optional model/crop code for metadata (e.g. ``"MZ"``).

    Returns:
        List of cultivar dicts.  Each dict has keys matching the column
        headers in the file (uppercased), plus an ``"_model"`` key if
        *model_code* is supplied.

    Raises:
        FileNotFoundError: If *path* does not exist.

    Example::

        cultivars = read_cul("MZCER048.CUL", "MZ")
        print(cultivars[0]["VAR#"], cultivars[0]["VRNAME"])
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CUL file not found: {path}")

    text = path.read_text(encoding="latin-1")
    cultivars = _parse_cul_text(text)

    if model_code:
        for cv in cultivars:
            cv["_model"] = model_code

    return cultivars


def write_cul(
    cultivars: list[dict[str, Any]],
    path: str | Path,
    model_code: str = "",
) -> None:
    """Write cultivar records to a DSSAT ``.CUL`` file.

    Args:
        cultivars: List of cultivar dicts (as returned by :func:`read_cul`).
        path: Destination file path.
        model_code: Model/crop code written in the file header comment.

    Example::

        write_cul(cultivars, "MZCER048_new.CUL", "MZ")
    """
    path = Path(path)
    if not cultivars:
        path.write_text("", encoding="utf-8")
        return

    # Determine headers from union of all keys (excluding private keys)
    all_keys: list[str] = []
    seen: set[str] = set()
    for cv in cultivars:
        for k in cv:
            if not k.startswith("_") and k not in seen:
                all_keys.append(k)
                seen.add(k)

    lines: list[str] = []
    label = model_code.upper() if model_code else "CROP"
    lines.append(f"*{label}.CUL")
    lines.append("")

    # Detect whether we have the standard VAR# / VRNAME first two columns.
    # If so we use the DSSAT fixed-width layout; otherwise plain space-sep.
    _standard = (
        len(all_keys) >= 2
        and all_keys[0] in ("VAR#", "VARNO")
        and all_keys[1] == "VRNAME"
    )

    if _standard:
        # Fixed-width header:  @VAR#  VRNAME..........  <numeric cols>
        numeric_keys = all_keys[2:]
        header_parts = ["@" + all_keys[0].ljust(5), "VRNAME" + "." * 14]
        for k in numeric_keys:
            header_parts.append(f"{k:>7}")
        lines.append("  ".join(header_parts))

        for cv in cultivars:
            varno  = str(cv.get(all_keys[0], "-99"))[:6].ljust(6)
            vrname = str(cv.get("VRNAME", "-99"))[:20].ljust(20)
            num_parts: list[str] = []
            for k in numeric_keys:
                v = cv.get(k)
                if v is None:
                    num_parts.append("  -99")
                elif isinstance(v, float):
                    if abs(v) < 1.0 and v != 0.0:
                        num_parts.append(f"{v:7.4f}")
                    else:
                        num_parts.append(f"{v:7.2f}")
                else:
                    num_parts.append(f"{str(v):>7}")
            lines.append(f"{varno} {vrname}  {'  '.join(num_parts)}")
    else:
        # Fallback: plain space-separated (no multi-word string fields)
        header_line = "@" + "  ".join(all_keys)
        lines.append(header_line)
        for cv in cultivars:
            parts: list[str] = []
            for k in all_keys:
                v = cv.get(k)
                if v is None:
                    parts.append("-99")
                elif isinstance(v, float):
                    parts.append(f"{v:.2f}" if abs(v) >= 1 else f"{v:.4f}")
                else:
                    parts.append(str(v))
            lines.append("  ".join(parts))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cul_to_json_file(
    path: str | Path,
    json_path: str | Path | None = None,
    model_code: str = "",
) -> Path:
    """Convert a DSSAT ``.CUL`` file to a JSON array file.

    Args:
        path: Source ``.CUL`` file.
        json_path: Destination JSON file (default: same stem + ``.json``).
        model_code: Optional model code for metadata.

    Returns:
        Path to the written JSON file.

    Example::

        json_path = cul_to_json_file("MZCER048.CUL", model_code="MZ")
    """
    path = Path(path)
    json_path = Path(json_path) if json_path else path.with_suffix(".json")
    cultivars = read_cul(path, model_code)
    json_path.write_text(json.dumps(cultivars, indent=2), encoding="utf-8")
    return json_path


def json_to_cul_file(
    json_path: str | Path,
    path: str | Path | None = None,
    model_code: str = "",
) -> Path:
    """Convert a JSON cultivar file to a DSSAT ``.CUL`` file.

    Args:
        json_path: Source JSON file (list of cultivar dicts).
        path: Destination ``.CUL`` file (default: same stem + ``.CUL``).
        model_code: Optional model code for the file header.

    Returns:
        Path to the written ``.CUL`` file.

    Example::

        cul_path = json_to_cul_file("MZCER048.json", model_code="MZ")
    """
    json_path = Path(json_path)
    path = Path(path) if path else json_path.with_suffix(".CUL")
    with json_path.open(encoding="utf-8") as f:
        cultivars = json.load(f)
    if isinstance(cultivars, dict):
        cultivars = [cultivars]
    write_cul(cultivars, path, model_code)
    return path


# ---------------------------------------------------------------------------
# internal parser
# ---------------------------------------------------------------------------

def _parse_cul_text(text: str) -> list[dict[str, Any]]:
    """Parse the text content of a ``.CUL`` file.

    Handles:
    - Lines starting with ``*`` — file header / section marker (skip)
    - Lines starting with ``!`` — comments (skip)
    - Lines starting with ``@`` — column header
    - All other non-blank lines — cultivar data rows
    """
    cultivars: list[dict[str, Any]] = []
    headers: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if not stripped:
            continue
        if stripped.startswith("!"):
            continue
        if stripped.startswith("*"):
            continue
        if stripped.startswith("@"):
            headers = _parse_header(stripped)
            continue
        if not headers:
            continue

        record = _parse_data_line(line, headers)
        if record:
            cultivars.append(record)

    return cultivars
