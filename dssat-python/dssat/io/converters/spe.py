"""DSSAT species file (.SPE) ↔ JSON converter.

Handles the keyword/section-based DSSAT species parameter files used by
CERES-Maize (MZCER048.SPE), CERES-Sorghum (SGCER048.SPE),
CERES-Millet (MLCER048.SPE), CERES-Wheat (WHCER048.SPE), etc.

The ``.SPE`` file format uses:

* Section headers prefixed with ``*`` (e.g. ``*TEMPERATURE EFFECTS``)
* Keyword-value pairs: ``  KEYWORD  value [value2 ...]  ! comment``
* Array rows: ``  CO2X   0   220   280 ...``

This converter preserves the full structure including:
- Section names
- All keyword-value pairs (scalars and arrays)
- Comments are stripped when converting to JSON but preserved when reading
  the raw structure

Usage::

    from dssat.io.converters.spe import read_spe, write_spe, spe_to_json, json_to_spe

    # SPE file → nested dict
    data = read_spe("MZCER048.SPE")

    # nested dict → SPE file
    write_spe(data, "MZCER048_out.SPE", file_header="*MAIZE SPECIES COEFFICIENTS: MZCER048 MODEL")

    # SPE file → JSON string
    json_str = spe_to_json("MZCER048.SPE")

    # JSON string → SPE file
    json_to_spe(json_str, "MZCER048_out.SPE")
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_comment(line: str) -> str:
    """Remove inline ``!``-comment from a line."""
    idx = line.find("!")
    return line[:idx].strip() if idx >= 0 else line.strip()


def _parse_value(tokens: list[str]) -> Any:
    """Convert a list of string tokens to a scalar or list of numbers/strings."""
    parsed = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        try:
            # Try int first for cleaner JSON
            v = int(tok)
        except ValueError:
            try:
                v = float(tok)
            except ValueError:
                v = tok
        parsed.append(v)

    if not parsed:
        return None
    if len(parsed) == 1:
        return parsed[0]
    return parsed


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def read_spe(path: str | Path) -> dict[str, Any]:
    """Read a DSSAT ``.SPE`` file and return a structured dict.

    The returned dict has the following shape::

        {
            "_file_header": "...",
            "sections": {
                "TEMPERATURE EFFECTS": {
                    "PRFTC": [6.2, 16.5, 33.0, 44.0],
                    "RGFIL": [5.5, 16.0, 27.0, 35.0],
                },
                "PHOTOSYNTHESIS PARAMETERS": {
                    "PARSR": 0.5,
                    "CO2X": [0, 220, 280, 330, 400, 490, 570, 750, 990, 9999],
                    "CO2Y": [0.0, 0.85, 0.95, ...],
                },
                ...
            }
        }

    Parameters
    ----------
    path:
        Path to the ``.SPE`` file.

    Returns
    -------
    dict
        Nested dict keyed by section then keyword.
    """
    path = Path(path)
    file_header = ""
    sections: dict[str, dict[str, Any]] = {}
    current_section: str | None = None

    with path.open(encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            stripped = line.strip()

            # Skip blank lines and pure comment lines
            if not stripped or stripped.startswith("!"):
                continue

            # Section header
            if stripped.startswith("*"):
                name = stripped.lstrip("*").strip()
                if not file_header:
                    file_header = stripped
                    continue
                current_section = name
                if current_section not in sections:
                    sections[current_section] = {}
                continue

            if current_section is None:
                # Before the first section — skip
                continue

            # Data line: remove comment, split by whitespace
            content = _strip_comment(line)
            if not content:
                continue

            tokens = content.split()
            if not tokens:
                continue

            keyword = tokens[0].upper()
            value_tokens = tokens[1:]
            value = _parse_value(value_tokens)
            if value is not None:
                sections[current_section][keyword] = value

    return {
        "_file_header": file_header,
        "sections": sections,
    }


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def write_spe(
    data: dict[str, Any],
    path: str | Path,
    *,
    file_header: str | None = None,
) -> None:
    """Write a SPE structure dict back to a ``.SPE`` file.

    Parameters
    ----------
    data:
        Dict as returned by :func:`read_spe`.
    path:
        Output file path.
    file_header:
        First ``*``-line of the file.  Defaults to ``data["_file_header"]``
        if present, otherwise ``"*SPECIES COEFFICIENTS"``.
    """
    header = file_header or data.get("_file_header") or "*SPECIES COEFFICIENTS"
    sections = data.get("sections", {})

    path = Path(path)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(header + "\n")

        for section_name, params in sections.items():
            fh.write("\n")
            fh.write(f"*{section_name}\n")
            for keyword, value in params.items():
                if isinstance(value, list):
                    vals_str = "  ".join(
                        (str(int(v)) if isinstance(v, (int, float)) and float(v) == int(v) and abs(v) < 1e10
                         else f"{v:g}" if isinstance(v, float)
                         else str(v))
                        for v in value
                    )
                    fh.write(f"  {keyword:<8}{vals_str}\n")
                elif isinstance(value, float):
                    fh.write(f"  {keyword:<8}{value:g}\n")
                else:
                    fh.write(f"  {keyword:<8}{value}\n")


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------

def spe_to_json(path: str | Path, indent: int = 2) -> str:
    """Convert a ``.SPE`` file to a JSON string.

    Parameters
    ----------
    path:
        Path to the ``.SPE`` file.
    indent:
        JSON indent level.

    Returns
    -------
    str
        JSON representation of the SPE structure.
    """
    return json.dumps(read_spe(path), indent=indent)


def spe_to_json_file(path: str | Path, out_path: str | Path | None = None) -> Path:
    """Convert a ``.SPE`` file to a ``.json`` file alongside it.

    Parameters
    ----------
    path:
        Path to the ``.SPE`` file.
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
    out_path.write_text(spe_to_json(path), encoding="utf-8")
    return out_path


def json_to_spe(
    json_data: str | dict,
    path: str | Path,
    *,
    file_header: str | None = None,
) -> Path:
    """Convert a JSON string/dict back to a ``.SPE`` file.

    Parameters
    ----------
    json_data:
        JSON string, or already-parsed dict.
    path:
        Output ``.SPE`` file path.
    file_header:
        Override the ``*``-line file header.

    Returns
    -------
    Path
        Path of the written file.
    """
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data

    write_spe(data, path, file_header=file_header)
    return Path(path)


def spe_from_json_file(
    json_path: str | Path,
    spe_path: str | Path,
    *,
    file_header: str | None = None,
) -> Path:
    """Load a JSON file and write it back as a ``.SPE`` file.

    Parameters
    ----------
    json_path:
        Input JSON file path.
    spe_path:
        Output ``.SPE`` file path.
    file_header:
        Override the ``*``-line file header.

    Returns
    -------
    Path
        Path of the written ``.SPE`` file.
    """
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    write_spe(data, spe_path, file_header=file_header)
    return Path(spe_path)


# ---------------------------------------------------------------------------
# Convenience: get a specific parameter from a SPE file
# ---------------------------------------------------------------------------

def get_spe_param(
    path: str | Path,
    keyword: str,
    section: str | None = None,
) -> Any:
    """Extract a single parameter value from a ``.SPE`` file.

    Parameters
    ----------
    path:
        Path to the ``.SPE`` file.
    keyword:
        Parameter name (case-insensitive).
    section:
        If given, search only within this section name.

    Returns
    -------
    scalar or list or None
        The parameter value, or None if not found.
    """
    data = read_spe(path)
    keyword_upper = keyword.upper()
    sections = data.get("sections", {})

    if section is not None:
        return sections.get(section, {}).get(keyword_upper)

    # Search all sections
    for params in sections.values():
        if keyword_upper in params:
            return params[keyword_upper]
    return None
