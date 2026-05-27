"""Generic DSSAT genotype file readers.

Handles the two file formats used across DSSAT genotype files:

* **Column-header format** — used by all CUL and ECO files.  A line starting
  with ``@`` defines column names; subsequent non-comment, non-blank lines are
  data rows.  This is the format read by the Fortran ``CUREADR`` / ``ECREADR``
  subroutines in ``CSREADS.for``.

* **Inline keyword-value format** — used by CERES-Maize / Millet / Sorghum
  SPE files.  Each data line has the parameter name as its first whitespace-
  separated token followed by one or more values.  The Fortran code reads
  these with fixed-format ``READ`` statements after scanning for the keyword.

* **@-header + data-rows format** — used by CERES-Wheat SPE files (also read
  via ``SPREADR`` / ``SPREADT`` in ``CSREADS.for``).  A ``@HEADER1 HEADER2``
  line defines column names; one or more data rows follow.  Single-row
  sections produce scalar values; multi-row sections produce arrays.

All three parsers normalise column/keyword names to upper-case.

No values are invented here — defaults must come from ``DFAULT`` rows in the
actual genotype files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _is_comment(line: str) -> bool:
    s = line.strip()
    return not s or s[0] in ("!", "*", "$")


def _parse_header_tokens(header_line: str) -> List[str]:
    """Strip leading ``@`` and trailing ``.`` padding, upper-case each token."""
    return [t.upper().rstrip(".") for t in header_line.lstrip("@ \t").split()]


def _to_float_or_str(tok: str) -> Any:
    """Convert token to float if possible, else return as string."""
    tok = tok.strip()
    if not tok or tok in ("-99", "-9", ".", "-99.", "-99.0"):
        return None
    try:
        return float(tok)
    except ValueError:
        return tok


# ---------------------------------------------------------------------------
# 1. Column-header format (CUL / ECO)
# ---------------------------------------------------------------------------

def read_genotype_row(
    path: Path | str,
    row_id: str,
    *,
    id_col: str = "VAR#",
    fallback_id: str = "DFAULT",
    vrname_col: Optional[str] = "VRNAME",
    econame_col: Optional[str] = "ECONAME",
) -> Dict[str, Any]:
    """Return a dict of column-name → value for *row_id* in a CUL/ECO file.

    The function searches the file for the first ``@``-prefixed header line,
    parses column names from it, then scans data rows for the one whose
    first column matches *row_id* (case-insensitive).  If not found it
    falls back to *fallback_id* (``"DFAULT"``).

    Parameters
    ----------
    path:
        Path to the .CUL or .ECO file.
    row_id:
        The cultivar/ecotype code to look up (e.g. ``"IB0001"``).
    id_col:
        Name of the identifier column (default ``"VAR#"``).
    fallback_id:
        Identifier of the default/fallback row (default ``"DFAULT"``).
    vrname_col:
        If present, column 1 (16-char fixed-width) is extracted as a name.
        Pass ``None`` to skip the fixed-width name extraction.
    econame_col:
        Like *vrname_col* but for ECO files.

    Returns
    -------
    dict
        Column-name → value mapping.  Numeric tokens are stored as ``float``;
        non-numeric tokens (e.g. ECO#) as ``str``.  Missing/``-99`` tokens
        are stored as ``None``.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If no header line (``@``) is found.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Genotype file not found: {path}")

    text = path.read_text(encoding="latin-1", errors="replace")
    lines = text.splitlines()

    headers: List[str] = []
    match_row: Optional[str] = None
    fallback_row: Optional[str] = None

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        if _is_comment(stripped):
            continue
        if stripped.startswith("@"):
            headers = _parse_header_tokens(stripped)
            continue
        if not headers:
            continue
        # Sentinel rows (min/max)
        if raw[:6].strip() in ("999991", "999992"):
            continue

        row_code = raw[:6].strip().upper()
        target = row_id.strip().upper()

        if row_code == target:
            match_row = raw
            break
        if row_code == fallback_id.upper() and fallback_row is None:
            fallback_row = raw

    raw_row = match_row or fallback_row
    if raw_row is None:
        return {id_col: row_id}

    return _parse_row(raw_row, headers, vrname_col=vrname_col, econame_col=econame_col)


def _parse_row(
    raw: str,
    headers: List[str],
    *,
    vrname_col: Optional[str],
    econame_col: Optional[str],
) -> Dict[str, Any]:
    """Parse one data row from a CUL/ECO file given its column headers."""
    result: Dict[str, Any] = {}

    # Detect fixed-width name field (cols 7-22 = 16 chars)
    has_vrname = vrname_col and vrname_col in headers
    has_econame = econame_col and econame_col in headers

    # Column 0: 6-char ID
    result[headers[0]] = raw[:6].strip()

    # Fixed-width name (either VRNAME or ECONAME in position 1)
    name_col: Optional[str] = None
    if has_vrname:
        name_col = vrname_col
    elif has_econame:
        name_col = econame_col

    if name_col and len(headers) >= 2 and headers[1] == name_col:
        result[name_col] = raw[7:23].strip() if len(raw) > 7 else ""
        rest = raw[23:] if len(raw) > 23 else ""
    else:
        rest = raw[6:]

    # Remaining tokens (free-format)
    remaining_headers = headers[1:] if name_col is None else headers[2:]
    tokens = rest.split()
    for hdr, tok in zip(remaining_headers, tokens):
        result[hdr] = _to_float_or_str(tok)

    return result


# ---------------------------------------------------------------------------
# 2. Inline keyword-value format (CERES Maize/Millet/Sorghum SPE)
# ---------------------------------------------------------------------------

def parse_spe_inline(path: Path | str) -> Dict[str, List[float]]:
    """Parse a CERES SPE file using the inline keyword-value format.

    Each non-comment, non-section, non-``@`` line has the parameter name as
    its **first whitespace-separated token** followed by numeric values.
    Section markers (``*``) and comment lines (``!``) are skipped.

    Parameters
    ----------
    path:
        Path to the SPE file (e.g. ``MZCER048.SPE``).

    Returns
    -------
    dict
        Keyword (upper-case) → list of floats.  Type tokens (e.g. ``LIN``)
        are silently skipped.

    Example
    -------
    ``{'PRFTC': [6.2, 16.5, 33.0, 44.0], 'PARSR': [0.5], ...}``
    """
    path = Path(path)
    result: Dict[str, List[float]] = {}

    for raw in path.read_text(encoding="latin-1", errors="replace").splitlines():
        s = raw.strip()
        if not s or s[0] in ("!", "*", "$", "@"):
            continue
        tokens = s.split()
        if not tokens:
            continue
        keyword = tokens[0].upper()
        # Collect numeric values; stop at non-numeric (type code etc.)
        vals: List[float] = []
        for t in tokens[1:]:
            try:
                vals.append(float(t))
            except ValueError:
                pass  # skip type codes like LIN, QDR
        result[keyword] = vals

    return result


# ---------------------------------------------------------------------------
# 3. @-header + data-rows format (CERES-Wheat SPE via CSREADS SPREADT)
# ---------------------------------------------------------------------------

def parse_spe_csreads(path: Path | str) -> Dict[str, Any]:
    """Parse a CERES-Wheat SPE file (CSREADS ``SPREADT``/``SPREADR`` format).

    A ``@HEADER1 HEADER2 ...`` line defines column names for the data rows
    that follow it (until the next ``@`` header).

    * **Single data row**: column values are stored as scalars.
    * **Multiple data rows**: column values are stored as lists.

    Parameters
    ----------
    path:
        Path to the SPE file (e.g. ``WHCER048.SPE``).

    Returns
    -------
    dict
        Column-name (upper-case) → scalar float or list of floats.

    Example
    -------
    ``{'PGERM': 10.0, 'PEMRG': 8.0, 'TRGEM': [1, 26, 50, 60], ...}``
    """
    path = Path(path)
    result: Dict[str, Any] = {}

    current_headers: List[str] = []
    current_data: List[List[float]] = []

    def _flush(hdrs: List[str], data: List[List[float]]) -> None:
        if not hdrs or not data:
            return
        for col_i, hdr in enumerate(hdrs):
            col_vals: List[float] = []
            for row in data:
                if col_i < len(row):
                    col_vals.append(row[col_i])
            if not col_vals:
                continue
            result[hdr] = col_vals[0] if len(col_vals) == 1 else col_vals

    for raw in path.read_text(encoding="latin-1", errors="replace").splitlines():
        s = raw.strip()
        if not s or s[0] in ("*", "$"):
            continue
        if s[0] == "!":
            continue

        if s[0] == "@":
            # New header: flush previous section, start new
            _flush(current_headers, current_data)
            current_headers = _parse_header_tokens(s)
            current_data = []
            continue

        # Data row — parse numeric values
        vals: List[float] = []
        for t in s.split():
            try:
                vals.append(float(t))
            except ValueError:
                pass  # skip non-numeric annotations
        if vals:
            current_data.append(vals)

    # Flush last section
    _flush(current_headers, current_data)

    return result
