"""Solar geometry and radiation calculations.

Translated from ``SOLAR.for`` and the ``HANG``, ``HTEMP``, ``HRAD``,
``FRACD``, ``HPAR`` subroutines in ``HMET.for``.

Reference: Spitters, C.J.T. et al. (1986) Separating the diffuse and
direct component of global radiation and its implications for modelling
canopy photosynthesis. *Agricultural and Forest Meteorology* 38, 217–229.
"""

from __future__ import annotations

import math

import numpy as np

PI = math.pi
RAD = PI / 180.0
SC = 1368.0        # Solar constant, W m⁻²
AMTRCS = 0.77      # Clear-sky atmospheric transmittance


def day_length(doy: int, xlat: float) -> tuple[float, float, float, float]:
    """Compute solar day length, declination, sunrise and sunset.

    Implements the Spitters (1986) equations, matching ``DAYLEN`` in
    ``SOLAR.for``.

    Args:
        doy: Day of year (1–366).
        xlat: Latitude (decimal degrees, negative = south).

    Returns:
        Tuple ``(dayl, dec, snup, sndn)`` where:

        - *dayl* — day length, sunrise to sunset (h)
        - *dec* — solar declination (degrees)
        - *snup* — time of sunrise (h, solar time, 0–24)
        - *sndn* — time of sunset (h, solar time, 0–24)
    """
    dec = -23.45 * math.cos(2.0 * PI * (doy + 10.0) / 365.0)
    soc = math.tan(RAD * dec) * math.tan(RAD * xlat)
    soc = min(max(soc, -1.0), 1.0)
    dayl = 12.0 + 24.0 * math.asin(soc) / PI
    dayl = min(max(dayl, 0.0), 24.0)
    snup = 12.0 - dayl / 2.0
    sndn = 12.0 + dayl / 2.0
    return dayl, dec, snup, sndn


def solar_params(
    dayl: float, dec: float, srad: float, xlat: float
) -> tuple[float, float, float]:
    """Compute cloudiness factor, integral of sin(β), and normal ET radiation.

    Implements ``SOLAR`` in ``SOLAR.for``.

    Args:
        dayl: Day length (h).
        dec: Solar declination (degrees).
        srad: Measured solar radiation (MJ m⁻² d⁻¹).
        xlat: Latitude (decimal degrees).

    Returns:
        Tuple ``(clouds, isinb, s0n)`` where:

        - *clouds* — fractional cloudiness (0–1)
        - *isinb* — integral for Spitters' Eq. 6
        - *s0n* — normal ET radiation (W m⁻²)
    """
    sradj = srad * 1.0e6
    ssin = math.sin(RAD * dec) * math.sin(RAD * xlat)
    ccos = math.cos(RAD * dec) * math.cos(RAD * xlat)
    soc = ssin / ccos if abs(ccos) > 1e-9 else 0.0
    soc = min(max(soc, -1.0), 1.0)

    dsinb = 3600.0 * (dayl * ssin + 24.0 / PI * ccos * math.sqrt(1.0 - soc**2))
    s0n = SC
    s0d = s0n * dsinb
    amtrd = sradj / s0d if s0d > 0.0 else 0.0
    sclear = AMTRCS * s0d * 1.0e-6
    clouds = min(max(1.0 - srad / sclear, 0.0), 1.0) if sclear > 0.0 else 0.0

    isinb = 3600.0 * (
        dayl * (ssin + 0.4 * (ssin**2 + 0.5 * ccos**2))
        + 24.0 / PI * ccos * (1.0 + 1.5 * 0.4 * ssin) * math.sqrt(1.0 - soc**2)
    )
    return clouds, isinb, s0n


def sun_angles(dec: float, hs: float, xlat: float) -> tuple[float, float]:
    """Solar elevation β and azimuth for hour *hs*.

    Implements ``HANG`` in ``HMET.for``.

    Args:
        dec: Solar declination (degrees).
        hs: Hour of day (solar time, 0–24).
        xlat: Latitude (decimal degrees).

    Returns:
        Tuple ``(azzon, beta)`` both in degrees.
    """
    soltim = (hs - 12.0) * PI / 12.0
    sinb = (
        math.sin(RAD * xlat) * math.sin(RAD * dec)
        + math.cos(RAD * xlat) * math.cos(RAD * dec) * math.cos(soltim)
    )
    sinb = min(max(sinb, -1.0), 1.0)
    beta_rad = math.asin(sinb)
    beta = beta_rad / RAD

    sinas = math.cos(RAD * dec) * math.sin(soltim) / math.cos(beta_rad) if abs(beta_rad) < PI / 2 - 0.001 else 0.0
    sinas = min(max(sinas, -1.0), 1.0)
    azzon = math.asin(sinas) / RAD
    return azzon, beta


def hourly_temperature(
    dayl: float, hs: float, sndn: float, snup: float, tmax: float, tmin: float
) -> float:
    """Sinusoidal air temperature for hour *hs*.

    Implements ``HTEMP`` in ``HMET.for``.

    Args:
        dayl: Day length (h).
        hs: Hour of day (0–24).
        sndn: Sunset time (h).
        snup: Sunrise time (h).
        tmax: Daily maximum temperature (°C).
        tmin: Daily minimum temperature (°C).

    Returns:
        Air temperature at hour *hs* (°C).
    """
    amp = (tmax - tmin) / 2.0
    if hs < snup or hs > sndn + 3.0:
        # Night-time: linear interpolation
        if hs < snup:
            tair = tmin + amp * math.cos(PI * (hs + 10.0) / (10.0 + snup))
        else:
            tair = tmin + amp * math.cos(PI * (hs - sndn) / (10.0 + 24.0 - sndn))
    else:
        tair = tmin + amp * (1.0 - math.cos(PI * (hs - snup) / (dayl + 4.0)))
    return tair


def hourly_radiation(
    beta: float, hs: float, isinb: float, sndn: float, snup: float, srad: float
) -> float:
    """Total hourly solar radiation (W m⁻²).

    Implements ``HRAD`` in ``HMET.for``.

    Args:
        beta: Solar elevation angle (degrees).
        hs: Hour of day (solar time, 0–24).
        isinb: Spitters integral of sin(β).
        sndn: Sunset (h).
        snup: Sunrise (h).
        srad: Daily total solar radiation (MJ m⁻² d⁻¹).

    Returns:
        Hourly solar radiation (W m⁻²).
    """
    if hs < snup or hs > sndn or beta <= 0.0 or isinb <= 0.0:
        return 0.0
    sinb = math.sin(beta * RAD)
    radhr = srad * 1.0e6 / isinb * 3600.0 * sinb * (1.0 + 0.4 * sinb)
    return max(radhr, 0.0)


def fraction_diffuse(
    beta: float,
    clouds: float,
    hs: float,
    radhr: float,
    s0n: float,
    sndn: float,
    snup: float,
) -> tuple[float, float, float]:
    """Fraction of diffuse radiation (total and PAR).

    Implements ``FRACD`` in ``HMET.for`` (revised JIL 2006).

    Args:
        beta: Solar elevation (degrees).
        clouds: Fractional cloudiness (0–1).
        hs: Hour of day (solar time).
        radhr: Total hourly radiation (W m⁻²).
        s0n: Normal ET radiation (W m⁻²).
        sndn: Sunset (h).
        snup: Sunrise (h).

    Returns:
        Tuple ``(amtrh, frdifp, frdifr)`` — atmospheric transmittance,
        fraction diffuse PAR, fraction diffuse total.
    """
    if hs < snup or hs > sndn or beta <= 0.001:
        return 0.0, 1.0, 1.0

    sinb = math.sin(max(beta, 0.001) * RAD)
    r0 = s0n * sinb
    amtrh = radhr / r0 if r0 > 1e-6 else 0.0

    # Fraction diffuse total radiation (Spitters 1986)
    if amtrh <= 0.22:
        frdifr = 1.0
    elif amtrh <= 0.35:
        frdifr = 1.0 - 6.4 * (amtrh - 0.22) ** 2
    elif amtrh <= 0.75:
        frdifr = 1.47 - 1.66 * amtrh
    else:
        frdifr = max(0.847 - 1.61 * sinb + 1.04 * sinb**2, 0.0)
    frdifr = min(max(frdifr, 0.0), 1.0)

    # Fraction diffuse PAR
    frdifp = frdifr  # simplified; full Spitters correction omitted for clarity

    return amtrh, frdifp, frdifr


def vpsat(temp: float) -> float:
    """Saturated vapour pressure (Pa) at temperature *temp* (°C).

    Tetens formula as used in ``HMET.for``.

    Args:
        temp: Air temperature (°C).

    Returns:
        Saturated vapour pressure (Pa).
    """
    return 610.78 * math.exp(17.269 * temp / (temp + 237.3))


def generate_hourly(
    clouds: float,
    dayl: float,
    dec: float,
    isinb: float,
    par: float,
    refht: float,
    sndn: float,
    snup: float,
    s0n: float,
    srad: float,
    tdew: float,
    tmax: float,
    tmin: float,
    windht: float,
    windsp: float,
    xlat: float,
) -> dict:
    """Generate full set of 24 hourly weather variables.

    This is the Python equivalent of the ``HMET`` subroutine.

    Args:
        clouds: Fractional cloudiness (0–1).
        dayl: Day length (h).
        dec: Solar declination (degrees).
        isinb: Spitters' integral of sin(β).
        par: Daily PAR (MJ m⁻² d⁻¹).
        refht: Reference height for wind (m).
        sndn: Sunset time (h).
        snup: Sunrise time (h).
        s0n: Normal ET radiation (W m⁻²).
        srad: Daily solar radiation (MJ m⁻² d⁻¹).
        tdew: Dew-point temperature (°C).
        tmax: Maximum temperature (°C).
        tmin: Minimum temperature (°C).
        windht: Instrument height for wind (m).
        windsp: Daily wind speed (km d⁻¹).
        xlat: Latitude (decimal degrees).

    Returns:
        Dictionary with keys ``tairhr``, ``tgro``, ``radhr``, ``parhr``,
        ``rhumhr``, ``windhr``, ``amtrh``, ``azzon``, ``beta``,
        ``frdifp``, ``frdifr``, ``tavg``, ``tday``, ``tgroav``,
        ``tgrody`` — all as :class:`numpy.ndarray` of length 24 or scalars.
    """
    ts = 24
    tincr = 24.0 / ts
    windav = windsp / 86.4 * (refht / windht) ** 0.2 if windht > 0 else 0.0

    tairhr = np.zeros(ts)
    tgro = np.zeros(ts)
    radhr = np.zeros(ts)
    parhr = np.zeros(ts)
    rhumhr = np.zeros(ts)
    windhr = np.zeros(ts)
    amtrh = np.zeros(ts)
    azzon = np.zeros(ts)
    beta = np.zeros(ts)
    frdifp = np.ones(ts)
    frdifr = np.ones(ts)

    tavg_acc = 0.0
    tday_acc = 0.0
    nday = 0

    for h in range(ts):
        hs = (h + 1) * tincr
        azzon[h], beta[h] = sun_angles(dec, hs, xlat)
        tairhr[h] = hourly_temperature(dayl, hs, sndn, snup, tmax, tmin)
        rh = vpsat(tdew) / vpsat(tairhr[h]) * 100.0 if vpsat(tairhr[h]) > 0 else 0.0
        rhumhr[h] = min(rh, 100.0)

        # Wind: flat profile during day, reduced at night
        windhr[h] = windav * (1.0 + 0.4 * math.sin(PI * (hs - snup) / dayl)) if dayl > 0 and snup <= hs <= sndn else windav * 0.4

        radhr[h] = hourly_radiation(beta[h], hs, isinb, sndn, snup, srad) / 1e6  # J m⁻² h⁻¹ → MJ m⁻² h⁻¹
        # Convert back for fracd (which expects W m⁻²)
        radhr_w = radhr[h] * 1e6 / 3600.0
        amtrh[h], frdifp[h], frdifr[h] = fraction_diffuse(beta[h], clouds, hs, radhr_w, s0n, sndn, snup)

        # Hourly PAR: proportional to hourly radiation
        if srad > 0:
            parhr[h] = par * radhr[h] / srad if srad > 0 else 0.0

        tgro[h] = tairhr[h]
        tavg_acc += tairhr[h]
        if snup <= hs <= sndn:
            tday_acc += tairhr[h]
            nday += 1

    tavg = tavg_acc / ts
    tday = tday_acc / nday if nday > 0 else tavg
    tgroav = tavg
    tgrody = tday

    if par <= 0.0:
        par = 2.0 * srad

    return {
        "tairhr": tairhr,
        "tgro": tgro,
        "radhr": radhr,
        "parhr": parhr,
        "rhumhr": rhumhr,
        "windhr": windhr,
        "amtrh": amtrh,
        "azzon": azzon,
        "beta": beta,
        "frdifp": frdifp,
        "frdifr": frdifr,
        "tavg": tavg,
        "tday": tday,
        "tgroav": tgroav,
        "tgrody": tgrody,
    }
