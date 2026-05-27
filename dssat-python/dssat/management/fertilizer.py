"""Fertilizer placement module.

Translates fertilizer event amounts to per-layer N additions (kg N/ha).
Mirrors the DSSAT Fert_Place.for logic.
"""

from __future__ import annotations

import numpy as np

from dssat.core.constants import NL
from dssat.core.types import FertilizerEvent, SoilType


def place_fertilizer(event: FertilizerEvent, soilprop: SoilType) -> dict:
    """Translate a fertilizer event to per-layer N additions.

    Args:
        event: The fertilizer application event.
        soilprop: Soil profile properties (used for layer depths).

    Returns:
        Dictionary with keys ``"sno3"``, ``"snh4"``, and ``"urea"``, each
        an ndarray of shape ``(NL,)`` with N additions in kg N/ha.
    """
    sno3 = np.zeros(NL)
    snh4 = np.zeros(NL)
    urea = np.zeros(NL)

    # Determine fertilizer type — support both 'type' and 'fert_type' attributes
    ftype = getattr(event, "type", None) or getattr(event, "fert_type", "FE005")

    # Total N applied (kg N/ha)
    n_pct = getattr(event, "n_pct", 46.0)
    amount = getattr(event, "amount", 0.0)
    n_kgha = amount * n_pct / 100.0

    if n_kgha <= 0.0:
        return {"sno3": sno3, "snh4": snh4, "urea": urea}

    # Find the target layer
    depth = getattr(event, "depth", 0.0)
    target_layer = 0  # default: surface layer
    if depth is not None and depth > 0.0:
        cum_depth = 0.0
        for L in range(soilprop.nlayr):
            cum_depth += soilprop.dlayr[L]
            if depth <= cum_depth:
                target_layer = L
                break
        else:
            target_layer = soilprop.nlayr - 1  # below all layers → bottom layer

    # Allocate N to pools based on fertilizer type
    if ftype == "FE005":
        # Urea: all to urea pool
        urea[target_layer] = n_kgha
    elif ftype == "FE004":
        # Ammonium nitrate: 50/50 NH4 / NO3
        snh4[target_layer] = n_kgha * 0.5
        sno3[target_layer] = n_kgha * 0.5
    elif ftype == "FE001":
        # Ammonium sulfate: all to NH4
        snh4[target_layer] = n_kgha
    elif ftype == "FE002":
        # Sodium nitrate: all to NO3
        sno3[target_layer] = n_kgha
    else:
        # Default: 50% NH4, 50% NO3
        snh4[target_layer] = n_kgha * 0.5
        sno3[target_layer] = n_kgha * 0.5

    return {"sno3": sno3, "snh4": snh4, "urea": urea}
