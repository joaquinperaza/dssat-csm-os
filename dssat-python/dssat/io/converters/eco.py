"""DSSAT ecotype file (.ECO) ↔ JSON converter.

Handles the fixed-format DSSAT ecotype coefficient files used by
CERES-Maize (MZCER048.ECO), CERES-Sorghum (SGCER048.ECO),
CERES-Millet (MLCER048.ECO), CERES-Wheat (WHCER048.ECO), etc.

The ``.ECO`` file format mirrors the ``.CUL`` format::

    *MAIZE ECOTYPE COEFFICIENTS: MZCER048 MODEL
    !
    ! COEFF DEFINITIONS ...
    !
    @ECO#  ECONAME.........  TBASE  TOPT ROPT ...
    IB0001 GENERIC MIDWEST1    8.0 34.0  34.0 ...

The first two columns are fixed-width:
* ``ECO#`` — 6 characters
* ``ECONAME`` — 16 characters (may contain spaces)
Remaining columns are space-separated numeric / string tokens.

Usage::

    from dssat.io.converters.eco import read_eco, write_eco, eco_to_json, json_to_eco

    # ECO file → list of dicts
    ecotypes = read_eco("MZCER048.ECO")

    # list of dicts → ECO file
    write_eco(ecotypes, "MZCER048_out.ECO", file_header="*MAIZE ECOTYPE COEFFICIENTS: MZCER048 MODEL")

    # ECO file → JSON string
    json_str = eco_to_json("MZCER048.ECO")

    # JSON string → ECO file
    json_to_eco(json_str, "MZCER048_out.ECO")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _val(tok: str) -> float | str | None:
    """Parse a token as float, or return as string/None."""
    tok = tok.strip()
    if not tok or tok in ("-99", "-9", ".", "-99.", "-99.0"):
        return None
    try:
        return float(tok)
    except ValueError:
        return tok


def _parse_header(line: str) -> list[str]:
    """Parse an ``@``-prefixed header line into column names."""
    return [t.upper().rstrip(".") for t in line.lstrip("@ ").split()]


def _parse_data_line(line: str, headers: list[str]) -> dict[str, Any] | None:
    """Parse an ECO data row (same fixed-width convention as CUL rows).

    * Column 0–5:   ECO# (6 chars)
    * Column 7–22:  ECONAME (16 chars, may contain spaces)
    * Column 23+:   space-separated numeric/string tokens
    """
    if not line.strip() or line.startswith("!") or line.startswith("*"):
        return None

    eco_id = line[0:6].strip()
    if not eco_id:
        return None

    if len(line) > 7:
        eco_name = line[7:23].strip()
        rest = line[23:]
    else:
        eco_name = ""
        rest = ""

    tokens = rest.split()
    numeric_headers = [h for h in headers if h not in ("ECO", "ECO#", "ECONAME")]
    row: dict[str, Any] = {"ECO#": eco_id, "ECONAME": eco_name}

    for i, h in enumerate(numeric_headers):
        row[h] = _val(tokens[i]) if i < len(tokens) else None

    return row


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_eco(path: str | Path) -> list[dict[str, Any]]:
    """Read a DSSAT ``.ECO`` file and return a list of ecotype dicts.

    Parameters
    ----------
    path:
        Path to the ``.ECO`` file.

    Returns
    -------
    list[dict]
        One dict per ecotype row.  Keys are the column names from the
        ``@ECO#`` header row.
    """
    path = Path(path)
    records: list[dict[str, Any]] = []
    headers: list[str] = []

    with path.open(encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith("@"):
                headers = _parse_header(line)
            elif headers and not line.startswith("!") and not line.startswith("*"):
                rec = _parse_data_line(line, headers)
                if rec:
                    records.append(rec)

    return records


def write_eco(
    records: list[dict[str, Any]],
    path: str | Path,
    *,
    file_header: str = "*ECOTYPE COEFFICIENTS",
    comments: str = "",
    float_fmt: str = "{:6.1f}",
) -> None:
    """Write ecotype dicts back to a ``.ECO`` file.

    Parameters
    ----------
    records:
        List of ecotype dicts (as returned by :func:`read_eco`).
    path:
        Output file path.
    file_header:
        First line of the file (the ``*`` line).
    comments:
        Multi-line comment block inserted after the header.
    float_fmt:
        Format string for numeric columns.
    """
    if not records:
        raise ValueError("records must not be empty")

    # Collect all column names, preserving ECO#/ECONAME first
    all_keys = list(records[0].keys())
    numeric_keys = [k for k in all_keys if k not in ("ECO#", "ECONAME")]

    path = Path(path)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(file_header + "\n")
        if comments:
            for c_line in comments.splitlines():
                fh.write(f"! {c_line}\n")
        fh.write("!\n")

        # Header line
        header_parts = ["@ECO#", " ECONAME......... "]
        for k in numeric_keys:
            header_parts.append(f"{k:>6}")
        fh.write(" ".join(header_parts).rstrip() + "\n")

        # Data lines
        for rec in records:
            eco_id = str(rec.get("ECO#", "")).ljust(6)[:6]
            eco_name = str(rec.get("ECONAME", "")).ljust(16)[:16]
            line = f"{eco_id} {eco_name}"
            for k in numeric_keys:
                v = rec.get(k)
                if v is None:
                    line += "  -99."
                elif isinstance(v, float):
                    line += " " + float_fmt.format(v)
                else:
                    line += f" {v:>6}"
            fh.write(line.rstrip() + "\n")


def eco_to_json(path: str | Path, indent: int = 2) -> str:
    """Convert a ``.ECO`` file to a JSON string.

    Parameters
    ----------
    path:
        Path to the ``.ECO`` file.
    indent:
        JSON indent level.

    Returns
    -------
    str
        JSON representation of all ecotype records.
    """
    return json.dumps(read_eco(path), indent=indent)


def eco_to_json_file(path: str | Path, out_path: str | Path | None = None) -> Path:
    """Convert a ``.ECO`` file to a ``.json`` file alongside it.

    Parameters
    ----------
    path:
        Path to the ``.ECO`` file.
    out_path:
        Output JSON path; defaults to *path* with ``.json`` extension.

    Returns
    -------
    Path
        Path of the written JSON file.
    """
    path = Path(path)
    if out_path is None:
        out_path = path.with_suffix(".json")
    out_path = Path(out_path)
    out_path.write_text(eco_to_json(path), encoding="utf-8")
    return out_path


def json_to_eco(
    json_data: str | dict | list,
    path: str | Path,
    *,
    file_header: str = "*ECOTYPE COEFFICIENTS",
    comments: str = "",
) -> Path:
    """Convert a JSON string/object back to a ``.ECO`` file.

    Parameters
    ----------
    json_data:
        JSON string, or already-parsed list/dict.
    path:
        Output ``.ECO`` file path.
    file_header:
        First line of the file.
    comments:
        Optional comment block.

    Returns
    -------
    Path
        Path of the written file.
    """
    if isinstance(json_data, str):
        records = json.loads(json_data)
    elif isinstance(json_data, dict):
        records = [json_data]
    else:
        records = list(json_data)

    write_eco(records, path, file_header=file_header, comments=comments)
    return Path(path)


def eco_from_json_file(
    json_path: str | Path,
    eco_path: str | Path,
    *,
    file_header: str = "*ECOTYPE COEFFICIENTS",
) -> Path:
    """Load a JSON file and write it back as a ``.ECO`` file.

    Parameters
    ----------
    json_path:
        Input JSON file path.
    eco_path:
        Output ``.ECO`` file path.
    file_header:
        First line of the ``.ECO`` file.

    Returns
    -------
    Path
        Path of the written ``.ECO`` file.
    """
    records = json.loads(Path(json_path).read_text(encoding="utf-8"))
    write_eco(records, eco_path, file_header=file_header)
    return Path(eco_path)
