"""CERES-Maize root growth and water uptake.

Translated from ``MZ_ROOTS.for`` and the root-extraction calls in
``MZ_NUPTAK.for``.

Root growth advances the rooting front by a daily depth increment
proportional to thermal time and is limited by the soil hospitality
factor (SHF).  Root length density is distributed exponentially from
the surface to the rooting front.
"""

from __future__ import annotations

import numpy as np

from dssat.core.constants import NL, SEASINIT, INTEGR, RATE


def grow_roots(
    grort: float,
    dtt: float,
    pltpop: float,
    sw: np.ndarray,
    ll: np.ndarray,
    dul: np.ndarray,
    sat: np.ndarray,
    dlayr: np.ndarray,
    ds: np.ndarray,
    shf: np.ndarray,
    nlayr: int,
    rlwr: float,
    rlv: np.ndarray,
    rtdep: float,
    rwumx: float,
) -> tuple[np.ndarray, float]:
    """Extend rooting depth and distribute root length density.

    Args:
        grort: Daily root growth rate (g plant⁻¹ d⁻¹).
        dtt: Daily thermal time (°C d).
        pltpop: Plant population (plants m⁻²).
        sw: Soil water content by layer.
        ll: Lower limit by layer.
        dul: Drained upper limit by layer.
        sat: Saturation water content by layer.
        dlayr: Layer thickness (cm).
        ds: Cumulative depth to bottom of layer (cm).
        shf: Soil hospitality factor by layer (0–1).
        nlayr: Number of active layers.
        rlwr: Root length per unit weight (cm g⁻¹).
        rlv: Root length density by layer (cm cm⁻³) — updated in-place.
        rtdep: Current rooting depth (cm).
        rwumx: Maximum root water uptake per unit RLV (cm³ cm⁻³ d⁻¹).

    Returns:
        Tuple ``(rlv_new, rtdep_new)`` — updated root length density array
        and updated rooting depth (cm).
    """
    # Root depth extension
    # 4.5 cm per °C·d is typical for maize (Borg & Grimes, 1986)
    drtdep = dtt * 4.5 / 100.0  # cm d⁻¹ per unit DTT
    for L in range(nlayr):
        if ds[L] > rtdep:
            drtdep *= shf[L]
            break
    rtdep_new = min(rtdep + drtdep, ds[nlayr - 1])

    # Root length density — exponential distribution from surface to front
    total_rl = grort * pltpop * 10.0 * rlwr  # cm root m⁻²
    rlv_new = rlv.copy()
    total_depth = rtdep_new
    if total_depth > 0.0:
        for L in range(nlayr):
            mid = ds[L] - dlayr[L] / 2.0
            if mid <= total_depth:
                rlv_new[L] = (rlv[L] + total_rl * shf[L] * dlayr[L]
                               / total_depth / 100.0)
                rlv_new[L] = min(rlv_new[L], 10.0)  # cap at 10 cm cm⁻³
            else:
                rlv_new[L] = 0.0

    return rlv_new, rtdep_new


def root_water_uptake(
    rlv: np.ndarray,
    sw: np.ndarray,
    ll: np.ndarray,
    dlayr: np.ndarray,
    nlayr: int,
    rwumx: float,
    rwuep1: float,
    eop: float,
) -> tuple[np.ndarray, float]:
    """Compute potential root water uptake by layer.

    Based on the Ritchie root water extraction function.

    Args:
        rlv: Root length density by layer (cm cm⁻³).
        sw: Soil water content by layer.
        ll: Lower limit by layer.
        dlayr: Layer thickness (cm).
        nlayr: Number of active layers.
        rwumx: Maximum water uptake per unit RLV (cm³ cm⁻³ d⁻¹).
        rwuep1: Threshold ratio for turgor stress.
        eop: Potential transpiration (mm d⁻¹).

    Returns:
        Tuple ``(unh4_equiv, trwup)`` where *unh4_equiv* is a per-layer
        extraction array (reused as RWU for N calculations) and *trwup* is
        the total potential root water uptake (cm d⁻¹).
    """
    rwu = np.zeros(NL)
    trwup = 0.0
    ep1 = eop * 0.1  # mm → cm

    for L in range(nlayr):
        avail = max(sw[L] - ll[L], 0.0) * dlayr[L]
        if rlv[L] > 0.0 and avail > 0.0:
            rwu[L] = min(rwumx * rlv[L] * dlayr[L], avail)
            trwup += rwu[L]

    trwup = min(trwup, ep1)
    return rwu, trwup
