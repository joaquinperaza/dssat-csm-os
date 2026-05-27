"""Core data types for the DSSAT Cropping System Model.

These dataclasses are the Python equivalents of the Fortran derived types
declared in ``ModuleDefs.for``.  Every module receives and returns instances
of these types rather than long positional argument lists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from dssat.core.constants import NL, TS, NELEM, NAPPL


# ---------------------------------------------------------------------------
# Control / simulation bookkeeping
# ---------------------------------------------------------------------------

@dataclass
class ControlType:
    """Simulation control metadata.

    Passed to every module on every time step so that any sub-routine can
    query the current simulation date, run mode, and configuration without
    having to read files again.

    Attributes:
        crop: Two-letter crop code (e.g. ``"MZ"``, ``"WH"``, ``"SB"``).
        model: Eight-character model name (e.g. ``"MZCER048"``).
        run: Sequential run number within a batch.
        trtnum: Treatment number being simulated.
        yrdoy: Current date as YYYYDDD (year + day-of-year, 7 digits).
        yrsim: Simulation start date, YYYYDDD.
        das: Days after sowing.
        dynamic: Current simulation phase (RUNINIT, SEASINIT, RATE, …).
        nyrs: Number of years to simulate.
        multi: Sequence number for multiple runs.
        frop: Frequency of output (days).
        n_elems: Number of nutrient elements active (1=N, 2=N+P, 3=N+P+K).
        rnmode: Run mode character (``"N"``=normal, ``"S"``=sensitivity, …).
        fileio: Path to the experiment file.
        dssatp: Path to the DSSAT data directory.
        crop_status: Current crop status code.
    """

    crop: str = "XX"
    model: str = "XXXXXXXX"
    run: int = 1
    trtnum: int = 1
    yrdoy: int = 1001001
    yrsim: int = 1001001
    das: int = 0
    dynamic: int = 1
    nyrs: int = 1
    multi: int = 1
    frop: int = 1
    n_elems: int = 1
    rnmode: str = "N"
    fileio: str = ""
    dssatp: str = ""
    crop_status: int = 0


@dataclass
class SwitchType:
    """Simulation method and output switches.

    Single-character switches control which physical processes are active and
    whether specific output files are written.  The names match the Fortran
    variable names in ``ModuleDefs.for``.

    Attributes:
        iswwat: Water balance switch (``"Y"``/``"N"``).
        iswnit: Nitrogen switch (``"Y"``/``"N"``).
        iswpho: Phosphorus switch (``"Y"``/``"N"``).
        iswpot: Potassium switch (``"Y"``/``"N"``).
        iswsym: Symbiotic N fixation switch.
        iswtil: Tillage switch.
        meevp: Evapotranspiration method (``"R"``=Ritchie, ``"F"``=FAO-56, …).
        mehyd: Hydrology method.
        meinf: Infiltration / runoff method.
        mesom: Soil organic matter method (``"G"``=CERES, ``"P"``=Century).
        mewth: Weather method (``"M"``=measured, ``"G"``=generated).
        ideto: Output observed vs. simulated flag.
        idetg: Growth output flag.
        idetn: Nitrogen output flag.
        idetp: Phosphorus output flag.
        idetw: Water output flag.
        iplti: Planting method (``"A"``=automatic, ``"R"``=on reported date).
        iirri: Irrigation method.
        ihari: Harvest method.
        iferi: Fertiliser application method.
        ico2: CO2 method.
    """

    iswwat: str = "Y"
    iswnit: str = "Y"
    iswpho: str = "N"
    iswpot: str = "N"
    iswsym: str = "Y"
    iswtil: str = "N"
    meevp: str = "R"
    mehyd: str = "R"
    meinf: str = "S"
    mesom: str = "G"
    mewth: str = "M"
    meghg: str = "N"
    mesol: str = "D"
    mesev: str = "S"
    meli:  str = "E"
    mepho: str = "C"
    metmp: str = "E"
    ideto: str = "Y"
    idetg: str = "Y"
    idetn: str = "Y"
    idetp: str = "Y"
    idetw: str = "Y"
    idetc: str = "Y"
    idetd: str = "Y"
    idetr: str = "Y"
    idets: str = "Y"
    ideth: str = "N"
    idetl: str = "0"
    iplti: str = "R"
    iirri: str = "N"
    ihari: str = "M"
    iferi: str = "R"
    iresi: str = "N"
    ico2:  str = "D"
    fname: str = "N"
    fmopt: str = "A"
    attp:  str = "A"


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------

@dataclass
class WeatherType:
    """Daily and hourly meteorological variables.

    Attributes:
        xlat: Latitude (decimal degrees, negative = south).
        xlong: Longitude (decimal degrees, negative = west).
        xelev: Elevation (m).
        refht: Reference height for wind measurement (m).
        windht: Instrument height for wind (m).
        tmax: Daily maximum air temperature (°C).
        tmin: Daily minimum air temperature (°C).
        tavg: Daily mean air temperature (°C).
        tday: Mean temperature during daylight hours (°C).
        srad: Solar radiation (MJ m⁻² d⁻¹).
        rain: Precipitation (mm d⁻¹).
        rhum: Relative humidity (%).
        tdew: Dew-point temperature (°C).
        windsp: Wind speed (km d⁻¹).
        windrun: Daily wind run (km).
        co2: Atmospheric CO₂ concentration (µmol mol⁻¹ = ppm).
        dayl: Day length, sunrise to sunset (h).
        par: Photosynthetically active radiation (MJ m⁻² d⁻¹).
        clouds: Fractional cloud cover (0–1).
        snup: Sunrise time (h).
        sndn: Sunset time (h).
        tgroav: Average canopy/soil temperature (°C).
        tgrody: Yesterday's canopy/soil temperature (°C).
        vapr: Vapour pressure (kPa).
        vpdf: Vapour pressure deficit (kPa).
        ozon7: 7-day mean ozone (Dobson).
        tairhr: Hourly air temperature (°C), shape ``(TS,)``.
        tgro: Hourly canopy/soil surface temperature (°C), shape ``(TS,)``.
        radhr: Hourly solar radiation (MJ m⁻² h⁻¹), shape ``(TS,)``.
        parhr: Hourly PAR (µmol m⁻² s⁻¹), shape ``(TS,)``.
        rhumhr: Hourly relative humidity (%), shape ``(TS,)``.
        windhr: Hourly wind speed (km h⁻¹), shape ``(TS,)``.
        amtrh: Hourly diffuse fraction, shape ``(TS,)``.
        azzon: Hourly solar azimuth (radians), shape ``(TS,)``.
        beta: Hourly solar elevation (radians), shape ``(TS,)``.
        frdifp: Hourly fraction diffuse PAR, shape ``(TS,)``.
        frdifr: Hourly fraction diffuse total radiation, shape ``(TS,)``.
    """

    xlat: float = 0.0
    xlong: float = 0.0
    xelev: float = 0.0
    refht: float = 2.0
    windht: float = 10.0

    tmax: float = 25.0
    tmin: float = 15.0
    tavg: float = 20.0
    tday: float = 22.0
    tavgd: float = 20.0
    srad: float = 15.0
    rain: float = 0.0
    rhum: float = 70.0
    tdew: float = 10.0
    windsp: float = 150.0
    windrun: float = 150.0
    co2: float = 380.0
    dco2: float = 0.0
    dayl: float = 12.0
    par: float = 7.0
    clouds: float = 0.0
    snup: float = 6.0
    sndn: float = 18.0
    tgroav: float = 20.0
    tgrody: float = 20.0
    vapr: float = 1.5
    vpdf: float = 0.5
    vpd_transp: float = 0.5
    tamp: float = 10.0
    tav: float = 20.0
    ta: float = 20.0
    ozon7: float = 0.0
    notdew: bool = False
    nowind: bool = False

    tairhr: np.ndarray = field(default_factory=lambda: np.zeros(TS))
    tgro: np.ndarray = field(default_factory=lambda: np.zeros(TS))
    radhr: np.ndarray = field(default_factory=lambda: np.zeros(TS))
    parhr: np.ndarray = field(default_factory=lambda: np.zeros(TS))
    rhumhr: np.ndarray = field(default_factory=lambda: np.zeros(TS))
    windhr: np.ndarray = field(default_factory=lambda: np.zeros(TS))
    amtrh: np.ndarray = field(default_factory=lambda: np.zeros(TS))
    azzon: np.ndarray = field(default_factory=lambda: np.zeros(TS))
    beta: np.ndarray = field(default_factory=lambda: np.zeros(TS))
    frdifp: np.ndarray = field(default_factory=lambda: np.zeros(TS))
    frdifr: np.ndarray = field(default_factory=lambda: np.zeros(TS))


# ---------------------------------------------------------------------------
# Soil
# ---------------------------------------------------------------------------

@dataclass
class SoilType:
    """Soil profile properties.

    All layer-indexed arrays have length ``NL`` (20).  Only the first
    ``nlayr`` elements contain meaningful data; the rest are zero-padded.

    Attributes:
        nlayr: Actual number of layers in this profile.
        slno: Soil profile ID string.
        sldesc: Textual description.
        taxon: Soil taxonomic classification.
        salb: Dry soil albedo (fraction).
        msalb: Mulch/soil albedo.
        cmsalb: Canopy/mulch/soil albedo.
        swalb: Wet soil albedo.
        slpf: Soil fertility factor (0–1).
        dmod: Organic matter decomposition modifier.
        dlayr: Layer thickness (cm).
        ds: Cumulative depth to bottom of layer (cm).
        bd: Bulk density (g cm⁻³).
        ll: Volumetric water content at lower limit (cm³ cm⁻³).
        dul: Volumetric water content at drained upper limit (cm³ cm⁻³).
        sat: Volumetric water content at saturation (cm³ cm⁻³).
        swcn: Saturated hydraulic conductivity (cm h⁻¹).
        wcr: Residual water content (cm³ cm⁻³).
        oc: Organic carbon (g 100g⁻¹).
        clay: Clay content (%).
        sand: Sand content (%).
        silt: Silt content (%).
        ph: Soil pH in water.
        phkcl: Soil pH in KCl.
        cec: Cation exchange capacity (cmol⁺ kg⁻¹).
        stones: Stone content (vol %).
        kg2ppm: Conversion factor kg ha⁻¹ → µg g⁻¹ by layer.
        shf: Soil hospitality factor for root growth (0–1).
        poros: Total porosity (cm³ cm⁻³).
        alpha_vg: van Genuchten α parameter (cm⁻¹).
        n_vg: van Genuchten n parameter.
        m_vg: van Genuchten m parameter (= 1 - 1/n).
    """

    nlayr: int = 0
    slno: str = ""
    sldesc: str = ""
    taxon: str = ""
    smpx: str = ""

    salb: float = 0.13
    msalb: float = 0.13
    cmsalb: float = 0.13
    swalb: float = 0.08
    slpf: float = 1.0
    dmod: float = 1.0
    ales: float = 0.0

    dlayr: np.ndarray = field(default_factory=lambda: np.zeros(NL))
    ds: np.ndarray = field(default_factory=lambda: np.zeros(NL))
    bd: np.ndarray = field(default_factory=lambda: np.ones(NL) * 1.3)
    ll: np.ndarray = field(default_factory=lambda: np.zeros(NL))
    dul: np.ndarray = field(default_factory=lambda: np.zeros(NL))
    sat: np.ndarray = field(default_factory=lambda: np.zeros(NL))
    swcn: np.ndarray = field(default_factory=lambda: np.zeros(NL))
    wcr: np.ndarray = field(default_factory=lambda: np.zeros(NL))
    oc: np.ndarray = field(default_factory=lambda: np.zeros(NL))
    clay: np.ndarray = field(default_factory=lambda: np.zeros(NL))
    sand: np.ndarray = field(default_factory=lambda: np.zeros(NL))
    silt: np.ndarray = field(default_factory=lambda: np.zeros(NL))
    ph: np.ndarray = field(default_factory=lambda: np.ones(NL) * 6.5)
    phkcl: np.ndarray = field(default_factory=lambda: np.ones(NL) * 5.5)
    cec: np.ndarray = field(default_factory=lambda: np.zeros(NL))
    stones: np.ndarray = field(default_factory=lambda: np.zeros(NL))
    kg2ppm: np.ndarray = field(default_factory=lambda: np.zeros(NL))
    shf: np.ndarray = field(default_factory=lambda: np.ones(NL))
    poros: np.ndarray = field(default_factory=lambda: np.zeros(NL))
    alpha_vg: np.ndarray = field(default_factory=lambda: np.zeros(NL))
    n_vg: np.ndarray = field(default_factory=lambda: np.zeros(NL))
    m_vg: np.ndarray = field(default_factory=lambda: np.zeros(NL))
    extp: np.ndarray = field(default_factory=lambda: np.zeros(NL))
    orgp: np.ndarray = field(default_factory=lambda: np.zeros(NL))
    totp: np.ndarray = field(default_factory=lambda: np.zeros(NL))
    totbas: np.ndarray = field(default_factory=lambda: np.zeros(NL))
    exca: np.ndarray = field(default_factory=lambda: np.zeros(NL))
    exk: np.ndarray = field(default_factory=lambda: np.zeros(NL))
    exna: np.ndarray = field(default_factory=lambda: np.zeros(NL))
    caco3: np.ndarray = field(default_factory=lambda: np.zeros(NL))
    sasc: np.ndarray = field(default_factory=lambda: np.zeros(NL))
    saea: np.ndarray = field(default_factory=lambda: np.zeros(NL))
    texture: list = field(default_factory=lambda: [""] * NL)
    coarse: np.ndarray = field(default_factory=lambda: np.zeros(NL, dtype=bool))


# ---------------------------------------------------------------------------
# Residue / senescence
# ---------------------------------------------------------------------------

@dataclass
class ResidueType:
    """Above- and below-ground plant residue.

    Attributes:
        res_wt: Dry weight of residue (kg ha⁻¹).
        res_lig: Lignin fraction of residue (kg ha⁻¹).
        res_e: Elemental content array, shape ``(NELEM,)`` (kg ha⁻¹).
        res_wt_layer: Residue weight by soil layer (kg ha⁻¹), shape ``(NL,)``.
    """

    res_wt: float = 0.0
    res_lig: float = 0.0
    res_e: np.ndarray = field(default_factory=lambda: np.zeros(NELEM))
    res_wt_layer: np.ndarray = field(default_factory=lambda: np.zeros(NL))


# ---------------------------------------------------------------------------
# Fertiliser
# ---------------------------------------------------------------------------

@dataclass
class FertilizerEvent:
    """A single fertiliser application event.

    Attributes:
        date: Application date, YYYYDDD.
        amount: Amount applied (kg ha⁻¹).
        n_pct: N percentage in material (%).
        p_pct: P₂O₅ percentage in material (%).
        k_pct: K₂O percentage in material (%).
        fert_type: Fertiliser type code (e.g. ``"FE005"`` = urea).
        depth: Incorporation depth (cm).
        placement: Placement code (``"AP"``=broadcast, ``"BD"``=banded, …).
    """

    date: int = 0
    amount: float = 0.0
    n_pct: float = 0.0
    p_pct: float = 0.0
    k_pct: float = 0.0
    fert_type: str = "FE001"
    type: str = "FE005"
    depth: float = 0.0
    placement: str = "AP"


@dataclass
class FertType:
    """Accumulated fertiliser data passed to soil modules on a given day.

    Attributes:
        fday: Application date for today's events, YYYYDDD.
        fertdata: Per-layer N/P/K additions (kg ha⁻¹), shape ``(NL, NELEM)``.
    """

    fday: int = 0
    fertdata: np.ndarray = field(
        default_factory=lambda: np.zeros((NL, NELEM))
    )


# ---------------------------------------------------------------------------
# Irrigation
# ---------------------------------------------------------------------------

@dataclass
class IrrigationEvent:
    """A single irrigation event.

    Attributes:
        date: Application date, YYYYDDD.
        amount: Water applied (mm).
        method: Method code (``"IR001"``=furrow, ``"IR004"``=sprinkler, …).
        depth: Depth at which water enters soil (cm).
    """

    date: int = 0
    amount: float = 0.0
    method: str = "IR001"
    depth: float = 0.0


# ---------------------------------------------------------------------------
# Tillage
# ---------------------------------------------------------------------------

@dataclass
class TillageEvent:
    """A single tillage operation.

    Attributes:
        date: Operation date, YYYYDDD.
        implement: Implement code (e.g. ``"TI001"``).
        depth: Tillage depth (cm).
        mix_pct: Fraction of soil mixed (0–1).
    """

    date: int = 0
    implement: str = "TI001"
    depth: float = 10.0
    mix_pct: float = 0.5


@dataclass
class TillType:
    """Accumulated tillage state passed to soil modules.

    Attributes:
        tildate: Most recent tillage date, YYYYDDD.
        tildep: Tillage depth (cm).
        tilmix: Mixing fraction (0–1).
        til_bd: Adjusted bulk densities after tillage, shape ``(NL,)``.
    """

    tildate: int = 0
    tildep: float = 0.0
    tilmix: float = 0.0
    til_bd: np.ndarray = field(default_factory=lambda: np.zeros(NL))


# ---------------------------------------------------------------------------
# Mulch
# ---------------------------------------------------------------------------

@dataclass
class MulchType:
    """Mulch layer state variables.

    Attributes:
        mulch_wt: Mulch dry weight (kg ha⁻¹).
        mulch_cover: Fraction of soil surface covered (0–1).
        mulch_depth: Mulch layer depth (cm).
        mulch_water: Water held in mulch (mm).
        mulch_n: N content of mulch (kg ha⁻¹).
    """

    mulch_wt: float = 0.0
    mulch_cover: float = 0.0
    mulch_depth: float = 0.0
    mulch_water: float = 0.0
    mulch_n: float = 0.0


# ---------------------------------------------------------------------------
# Flood / paddy
# ---------------------------------------------------------------------------

@dataclass
class FloodWatType:
    """Flooded-field water balance state.

    Attributes:
        flood: Current flood depth (mm).
        infilt: Infiltration from flood (mm d⁻¹).
        puddled: Whether the soil has been puddled.
        bunded: Whether the field is bunded (levee present).
    """

    flood: float = 0.0
    infilt: float = 0.0
    puddled: bool = False
    bunded: bool = False


@dataclass
class FloodNType:
    """Flooded-field nitrogen state.

    Attributes:
        floodnh4: NH₄ concentration in flood water (mg L⁻¹).
        floodno3: NO₃ concentration in flood water (mg L⁻¹).
    """

    floodnh4: float = 0.0
    floodno3: float = 0.0


# ---------------------------------------------------------------------------
# Organic matter application
# ---------------------------------------------------------------------------

@dataclass
class OrgMatAppType:
    """Organic material application event state.

    Attributes:
        om_date: Application date, YYYYDDD.
        om_wt: Material dry weight (kg ha⁻¹).
        om_n: N content (kg ha⁻¹).
        om_p: P content (kg ha⁻¹).
        om_depth: Incorporation depth (cm).
    """

    om_date: int = 0
    om_wt: float = 0.0
    om_n: float = 0.0
    om_p: float = 0.0
    om_depth: float = 0.0


# ---------------------------------------------------------------------------
# Plant stress
# ---------------------------------------------------------------------------

@dataclass
class PlantStresType:
    """Stress indices experienced by the crop.

    All indices are 0 (maximum stress) to 1 (no stress) unless noted.

    Attributes:
        turgor: Turgor (expansion) water stress index.
        photo: Photosynthesis water stress index.
        n_new: N stress for new growth.
        n_phot: N stress for photosynthesis.
        p_stress1: P stress index 1.
        p_stress2: P stress index 2.
        k_stress: K stress index.
        temp_excess: High-temperature stress index (0–1).
        temp_deficit: Low-temperature stress index (0–1).
    """

    turgor: float = 1.0
    photo: float = 1.0
    n_new: float = 1.0
    n_phot: float = 1.0
    p_stress1: float = 1.0
    p_stress2: float = 1.0
    k_stress: float = 1.0
    temp_excess: float = 1.0
    temp_deficit: float = 1.0
