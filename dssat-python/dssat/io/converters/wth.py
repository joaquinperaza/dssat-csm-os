"""DSSAT ``.WTH`` weather file ↔ JSON converter.

The DSSAT weather file format consists of:

1. An optional ``$`` title line.
2. A station header line beginning with ``@``, listing column names
   (``INSI``, ``LAT``, ``LONG``, ``ELEV``, ``TAV``, ``AMP``, ``REFHT``,
   ``WNDHT``) followed by a single data line.
3. A daily-data header line beginning with ``@`` (``DATE``, ``SRAD``,
   ``TMAX``, ``TMIN``, ``RAIN``, ``DEWP``, ``WIND``, ``PAR``, ``EVAP``,
   ``RHUM``), followed by one data line per day.

Dates in legacy files use ``YYDDD`` (5 chars, 2-digit year).  Post-2000
files may use ``YYYYDDD`` (7 chars, 4-digit year).  Missing values are
represented by ``-99`` or ``.``.

The JSON schema produced / consumed here matches the weather station format
used by :mod:`dssat.io.readers.load_weather`::

    {
        "insi": "UFGA",
        "lat": 29.63,
        "lon": -82.37,
        "elev": 30.0,
        "tav": 22.5,
        "tamp": 7.8,
        "refht": 2.0,
        "wndht": 2.0,
        "records": [
            {"date": 1982100, "srad": 12.5, "tmax": 28.0,
             "tmin": 16.0, "rain": 0.0, ...},
            ...
        ]
    }
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _parse_headers(line: str) -> tuple[list[str], list[tuple[int, int]]]:
    """Return (names, column-spans) from a DSSAT ``@``-prefixed header line.

    Column spans are (start, end) inclusive indices into the original line
    (0-based), matching the PARSE_HEADERS logic in the Fortran code.
    """
    # Strip leading '@' or '@ '
    raw = line.lstrip("@").rstrip("\n")
    names: list[str] = []
    spans: list[tuple[int, int]] = []

    i = 0
    n = len(raw)
    while i < n:
        if raw[i] == " ":
            i += 1
            continue
        j = i
        while j < n and raw[j] != " ":
            j += 1
        # name occupies raw[i:j], which is line[i+1:j+1] (accounting for '@')
        col_start = i + 1  # +1 because raw = line[1:]
        col_end = j        # exclusive in raw → inclusive last char at j
        # Extend span to include trailing spaces up to next token
        # (DSSAT aligns data values under the header name's rightmost char)
        names.append(raw[i:j].strip())
        spans.append((col_start, col_end))
        i = j

    return names, spans


def _val(tok: str) -> float | None:
    """Parse a DSSAT token; return None for missing values (``-99``, ``-9``, ``.``)."""
    tok = tok.strip()
    if tok in (".", "", "-99", "-9", "-99.0", "-9.0"):
        return None
    try:
        return float(tok)
    except ValueError:
        return None


def _fmt(val: float | None, width: int = 6, decimals: int = 1) -> str:
    """Format a value for DSSAT output, using ``-99.`` for missing."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return f"{'  -99.':{width}}"[:width]
    fmt = f"{{:{width}.{decimals}f}}"
    return fmt.format(val)


def _yyddd_to_yyyyddd(yyddd: int) -> int:
    """Convert DSSAT 5-digit YYDDD date to 7-digit YYYYDDD.

    Years 00–30 are treated as 2000–2030; 31–99 as 1931–1999.
    """
    yy = yyddd // 1000
    ddd = yyddd % 1000
    if yy <= 30:
        yyyy = 2000 + yy
    else:
        yyyy = 1900 + yy
    return yyyy * 1000 + ddd


def _yyyyddd_to_yyddd(yyyyddd: int) -> int:
    """Convert 7-digit YYYYDDD to legacy 5-digit YYDDD."""
    yyyy = yyyyddd // 1000
    ddd = yyyyddd % 1000
    yy = yyyy % 100
    return yy * 1000 + ddd


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def read_wth(path: str | Path) -> dict[str, Any]:
    """Parse a DSSAT ``.WTH`` file into a JSON-compatible dict.

    Args:
        path: Path to the ``.WTH`` file.

    Returns:
        Weather dict with ``insi``, ``lat``, ``lon``, ``elev``, ``tav``,
        ``tamp``, ``refht``, ``wndht``, and ``records`` keys.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be parsed.

    Example::

        data = read_wth("UFGA8201.WTH")
        print(data["lat"], data["tav"])
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"WTH file not found: {path}")

    lines = path.read_text(encoding="latin-1").splitlines()

    station: dict[str, Any] = {
        "insi": path.stem[:4].upper(),
        "lat": None, "lon": None, "elev": None,
        "tav": None, "tamp": None,
        "refht": 2.0, "wndht": 2.0,
    }
    records: list[dict[str, Any]] = []

    # State: 0 = before station header, 1 = after station header,
    #        2 = reading daily data
    state = 0
    daily_names: list[str] = []
    daily_spans: list[tuple[int, int]] = []

    for line in lines:
        if not line.strip() or line.startswith("!"):
            continue
        if line.startswith("$"):
            continue  # title line

        if line.startswith("@") or line.startswith(" @"):
            header_line = line.lstrip()
            names, spans = _parse_headers(header_line)
            if not names:
                continue
            first = names[0].upper()
            if first == "INSI":
                state = 1
                sta_names, sta_spans = names, spans
            elif first in ("DATE", "YEAR", "YRDOY"):
                state = 2
                daily_names = [n.upper() for n in names]
                daily_spans = spans
            continue

        if state == 1:
            # Station data line — parse by column spans
            _read_station_line(line, sta_names, sta_spans, station)
            state = 0  # wait for daily header
            continue

        if state == 2:
            # Daily data line
            rec = _read_daily_line(line, daily_names)
            if rec is not None:
                records.append(rec)

    station["records"] = records
    # Clean None → omit from output for optional fields
    return {k: v for k, v in station.items() if v is not None or k == "records"}


def _read_station_line(
    line: str,
    names: list[str],
    spans: list[tuple[int, int]],
    out: dict,
) -> None:
    """Parse a station header data line using whitespace tokenisation.

    The DSSAT station line is always whitespace-delimited; column spans from
    PARSE_HEADERS would need to perfectly match field widths, which varies
    across real files.  Simple positional zip is more robust.
    """
    _FIELD_MAP = {
        "INSI": "insi",
        "LAT": "lat", "WTHLAT": "lat",
        "LONG": "lon", "WTHLONG": "lon",
        "ELEV": "elev", "WELEV": "elev",
        "TAV": "tav",
        "AMP": "tamp",
        "REFHT": "refht",
        "WNDHT": "wndht",
        "CCO2": "co2", "CO2": "co2",
    }
    tokens = line.split()
    for name, tok in zip(names, tokens):
        key_up = name.upper()
        json_key = _FIELD_MAP.get(key_up)
        if json_key is None:
            continue
        if json_key == "insi":
            out["insi"] = tok.strip()
        else:
            v = _val(tok)
            if v is not None:
                out[json_key] = v


def _read_daily_line(line: str, names: list[str]) -> dict[str, Any] | None:
    """Parse one daily weather data line (whitespace-split)."""
    tokens = line.split()
    if not tokens:
        return None

    _DAILY_MAP = {
        "DATE": "date", "YRDOY": "date",
        "SRAD": "srad",
        "TMAX": "tmax",
        "TMIN": "tmin",
        "RAIN": "rain",
        "DEWP": "dewp",
        "WIND": "windsp",
        "PAR": "par",
        "EVAP": "evap",
        "RHUM": "rhum",
        "TDEW": "dewp",
        "VAPR": "vapr",
        "CO2": "co2",
    }

    rec: dict[str, Any] = {}
    for i, (name, tok) in enumerate(zip(names, tokens)):
        key = _DAILY_MAP.get(name.upper())
        if key is None:
            continue
        if key == "date":
            try:
                raw = int(tok)
                # 5-digit → YYDDD, 7-digit → YYYYDDD
                if raw < 100_000:
                    rec["date"] = _yyddd_to_yyyyddd(raw)
                else:
                    rec["date"] = raw
            except ValueError:
                return None
        else:
            v = _val(tok)
            if v is not None:
                rec[key] = v

    if "date" not in rec:
        return None
    return rec


# ---------------------------------------------------------------------------

def write_wth(
    data: dict[str, Any],
    path: str | Path,
    *,
    station_id: str | None = None,
    title: str | None = None,
    use_4digit_year: bool = True,
) -> None:
    """Write a JSON weather dict to a DSSAT ``.WTH`` file.

    Args:
        data: Weather dict (output of :func:`read_wth` or a compatible JSON).
        path: Destination file path.
        station_id: Override the 4-char station code (default: from data or
            derived from the filename stem).
        title: Optional title string for the ``$`` line.
        use_4digit_year: If ``True`` (default) write 7-digit ``YYYYDDD`` dates;
            if ``False`` write legacy 5-digit ``YYDDD`` dates.

    Example::

        write_wth(data, "UFGA8201.WTH", title="Gainesville, FL")
    """
    path = Path(path)
    insi = (station_id or data.get("insi") or path.stem[:4]).upper()
    title_str = title or data.get("title") or ""

    lines: list[str] = []
    lines.append(f"${insi}  {title_str}")
    lines.append("@ INSI      LAT     LONG  ELEV   TAV   AMP REFHT WNDHT")

    lat = data.get("lat")
    lon = data.get("lon")
    elev = data.get("elev")
    tav = data.get("tav")
    tamp = data.get("tamp")
    refht = data.get("refht", 2.0)
    wndht = data.get("wndht", 2.0)

    def _fmtf(v, w, d):
        if v is None:
            return f"{'  -99.':{w}}"[:w]
        return f"{v:{w}.{d}f}"

    sta_line = (
        f"  {insi:<4}"
        f"  {_fmtf(lat,  7, 3)}"
        f"  {_fmtf(lon,  7, 3)}"
        f"  {_fmtf(elev, 5, 0)}"
        f"  {_fmtf(tav,  4, 1)}"
        f"  {_fmtf(tamp, 5, 1)}"
        f"   {_fmtf(refht, 3, 1)}"
        f"   {_fmtf(wndht, 3, 1)}"
    )
    lines.append(sta_line)

    # Determine which optional columns to include
    records = data.get("records", [])
    has_dewp = any("dewp" in r for r in records)
    has_wind = any("windsp" in r for r in records)
    has_par = any("par" in r for r in records)
    has_evap = any("evap" in r for r in records)
    has_rhum = any("rhum" in r for r in records)

    header_cols = ["DATE", "SRAD", "TMAX", "TMIN", "RAIN"]
    if has_dewp:
        header_cols.append("DEWP")
    if has_wind:
        header_cols.append("WIND")
    if has_par:
        header_cols.append("PAR")
    if has_evap:
        header_cols.append("EVAP")
    if has_rhum:
        header_cols.append("RHUM")

    def _hdr(col):
        widths = {
            "DATE": 5, "SRAD": 6, "TMAX": 6, "TMIN": 6, "RAIN": 6,
            "DEWP": 6, "WIND": 6, "PAR": 6, "EVAP": 6, "RHUM": 6,
        }
        w = widths.get(col, 6)
        return f"{col:>{w}}"

    lines.append("@" + "".join(_hdr(c) for c in header_cols))

    for rec in records:
        date_int = int(rec["date"])
        if use_4digit_year:
            date_str = f"{date_int:07d}"
        else:
            date_str = f"{_yyyyddd_to_yyddd(date_int):05d}"

        row = date_str
        for col in header_cols[1:]:
            key_map = {
                "SRAD": "srad", "TMAX": "tmax", "TMIN": "tmin",
                "RAIN": "rain", "DEWP": "dewp", "WIND": "windsp",
                "PAR": "par", "EVAP": "evap", "RHUM": "rhum",
            }
            v = rec.get(key_map[col])
            row += _fmtf(v, 6, 1)

        lines.append(row)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def wth_to_json_file(
    wth_path: str | Path,
    json_path: str | Path | None = None,
) -> Path:
    """Convert a ``.WTH`` file to a ``.json`` file.

    Args:
        wth_path: Source ``.WTH`` file.
        json_path: Destination ``.json`` file.  Defaults to same stem as
            *wth_path* with ``.json`` extension.

    Returns:
        Path to the written JSON file.
    """
    wth_path = Path(wth_path)
    json_path = Path(json_path) if json_path else wth_path.with_suffix(".json")
    data = read_wth(wth_path)
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return json_path


def json_to_wth_file(
    json_path: str | Path,
    wth_path: str | Path | None = None,
    **kwargs: Any,
) -> Path:
    """Convert a weather JSON file to a ``.WTH`` file.

    Args:
        json_path: Source ``.json`` file.
        wth_path: Destination ``.WTH`` file.  Defaults to same stem as
            *json_path* with ``.WTH`` extension.
        **kwargs: Forwarded to :func:`write_wth`.

    Returns:
        Path to the written ``.WTH`` file.
    """
    json_path = Path(json_path)
    wth_path = Path(wth_path) if wth_path else json_path.with_suffix(".WTH")
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    write_wth(data, wth_path, **kwargs)
    return wth_path
