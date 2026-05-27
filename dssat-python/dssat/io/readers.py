"""JSON input readers for the DSSAT Python package.

These replace the legacy fixed-format text file parsers (``IPSIM.for``,
``IPSOIL_Inp.for``, ``WEATHR_Inp.for``, etc.) with clean JSON-based I/O.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from dssat.core.types import SoilType, WeatherType
from dssat.core.constants import NL


def load_experiment(path: str | Path) -> dict:
    """Load and parse an experiment JSON file.

    The JSON must conform to the ``experiment.schema.json`` schema.

    Args:
        path: Path to the experiment ``.json`` file.

    Returns:
        Parsed experiment dictionary.

    Raises:
        FileNotFoundError: If *path* does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Experiment file not found: {path}")
    with path.open() as f:
        data = json.load(f)
    return data


def load_weather(source: str | Path | dict) -> tuple[dict[int, dict], dict]:
    """Load weather records from a JSON file or inline dict.

    Args:
        source: Either a path to a weather JSON file or an inline dict
            with the weather station data.

    Returns:
        Tuple ``(records_by_date, station_meta)`` where *records_by_date*
        maps YYYYDDD integers to daily record dicts and *station_meta*
        contains station-level metadata (lat, lon, tav, tamp, …).
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Weather file not found: {path}")
        with path.open() as f:
            data = json.load(f)
    else:
        data = source

    records: dict[int, dict] = {}
    for rec in data.get("records", []):
        records[int(rec["date"])] = rec

    meta = {k: v for k, v in data.items() if k != "records"}
    return records, meta


def load_soil(source: str | Path | dict) -> SoilType:
    """Load a soil profile from a JSON file or inline dict.

    Args:
        source: Path to a soil JSON file or an inline profile dict
            (matching the ``soil_profile`` JSON schema definition).

    Returns:
        Populated :class:`~dssat.core.types.SoilType` instance.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Soil file not found: {path}")
        with path.open() as f:
            data = json.load(f)
    else:
        data = source

    soil = SoilType()
    soil.slno = data.get("id", "")
    soil.sldesc = data.get("description", "")
    soil.taxon = data.get("taxon", "")
    soil.salb = float(data.get("salb", 0.13))
    soil.slpf = float(data.get("slpf", 1.0))
    soil.dmod = float(data.get("dmod", 1.0))

    layers = data.get("layers", [])
    nlayr = min(len(layers), NL)
    soil.nlayr = nlayr

    prev_depth = 0.0
    for i, lyr in enumerate(layers[:NL]):
        ds = float(lyr["depth_bottom"])
        soil.ds[i] = ds
        soil.dlayr[i] = ds - prev_depth
        prev_depth = ds
        soil.bd[i] = float(lyr.get("bd", 1.3))
        soil.ll[i] = float(lyr["ll"])
        soil.dul[i] = float(lyr["dul"])
        soil.sat[i] = float(lyr["sat"])
        soil.swcn[i] = float(lyr.get("swcn", 0.5))
        soil.oc[i] = float(lyr.get("oc", 0.5))
        soil.clay[i] = float(lyr.get("clay", 20.0))
        soil.sand[i] = float(lyr.get("sand", 40.0))
        soil.silt[i] = 100.0 - soil.clay[i] - soil.sand[i]
        soil.ph[i] = float(lyr.get("ph", 6.5))
        soil.poros[i] = 1.0 - soil.bd[i] / 2.65
        # Compute kg2ppm (conversion: 1 kg ha-1 = 1/(BD*dlayr*10) µg g-1)
        soil.kg2ppm[i] = (
            1.0 / (soil.bd[i] * soil.dlayr[i] * 10.0) if soil.dlayr[i] > 0 else 0.0
        )
        soil.shf[i] = float(lyr.get("shf", 1.0))

    return soil


def build_initial_sw(
    exp_data: dict, soil: SoilType
) -> np.ndarray:
    """Extract initial soil water content from experiment data.

    Args:
        exp_data: Parsed experiment dictionary.
        soil: Soil profile (used for lower-limit bounds).

    Returns:
        Array of initial SW values (cm³ cm⁻³), shape ``(NL,)``.
    """
    sw = np.zeros(NL)
    ic = exp_data.get("initial_conditions", {})
    sw_list = ic.get("sw_layers", [])
    for i, val in enumerate(sw_list[: soil.nlayr]):
        sw[i] = max(float(val), soil.ll[i] * 0.30)
    # Fill any missing layers with DUL
    for i in range(len(sw_list), soil.nlayr):
        sw[i] = soil.dul[i]
    return sw


def build_initial_nutrients(
    exp_data: dict, soil: SoilType
) -> tuple[np.ndarray, np.ndarray]:
    """Extract initial NO3 and NH4 from experiment data.

    Args:
        exp_data: Parsed experiment dictionary.
        soil: Soil profile.

    Returns:
        Tuple ``(no3, nh4)`` each shape ``(NL,)`` in µg g⁻¹.
    """
    no3 = np.zeros(NL)
    nh4 = np.zeros(NL)
    ic = exp_data.get("initial_conditions", {})
    for i, val in enumerate(ic.get("no3_layers", [])[:soil.nlayr]):
        no3[i] = float(val)
    for i, val in enumerate(ic.get("nh4_layers", [])[:soil.nlayr]):
        nh4[i] = float(val)
    return no3, nh4
