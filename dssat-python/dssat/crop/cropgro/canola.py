"""CROPGRO-Canola crop model.

Provides:
- ``CanolaCultivar`` — cultivar/ecotype coefficients for spring canola.
- ``CropGROCanola``  — full CROPGRO-Canola model class.

Canola (Brassica napus L.) is a long-day obligate plant.  It does NOT fix
atmospheric N2, so the NFix sub-model is inactive.

Usage::

    from dssat.crop.cropgro import CropGROCanola, CanolaCultivar

    cv = CanolaCultivar()   # generic spring canola defaults
    model = CropGROCanola(cultivar=cv, pltpop=80.0, sdepth=2.0, yrsim=2020090)

    # Inside daily loop:
    model.run(control, iswitch, soilprop, weather, sw, nh4, no3, es, eop)

References:
    Robertson, M.J., Holland, J.F., & Bambach, R. (1999). Response of canola
    and Indian mustard to sowing date in the grain belt of north-eastern
    Australia. Aust. J. Exp. Agric. 39, 31–41.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np

from dssat.core.constants import NL, RUNINIT, SEASINIT, RATE, INTEGR
from dssat.core.types import ControlType, SwitchType, SoilType, WeatherType, ResidueType
from dssat.crop.base import CropModel
from dssat.crop.cropgro.phenology import CropGROPhenology
from dssat.crop.cropgro.growth import CropGROGrowth
from dssat.crop.cropgro.nfix import CropGRONFix
from dssat.crop.ceres_maize.roots import grow_roots, root_water_uptake


# ---------------------------------------------------------------------------
# Cultivar dataclass
# ---------------------------------------------------------------------------

@dataclass
class CanolaCultivar:
    """CROPGRO-Canola cultivar and ecotype coefficients.

    Defaults represent a generic spring canola.

    Attributes:
        varno:   Cultivar identifier (6-char).
        vrname:  Cultivar name.
        econo:   Ecotype identifier (6-char).
        csdvar:  Short-day threshold below which there is no photoperiod
                 acceleration (h). For long-day canola, development is slow
                 when days are shorter than this.
        cldvar:  Long-day threshold above which maximum photoperiod
                 acceleration is reached (h).
        ppsen:   Photoperiod sensitivity (informational, d h^-1).
        em_fl:   Photothermal days from emergence to R1 (flowering).
        fl_sh:   Photothermal days from R1 to R3 (first silique/pod).
        fl_sd:   Photothermal days from R1 to R5 (seed fill start).
        sd_pm:   Photothermal days from R5 to R7 (physiological maturity).
        fl_lf:   Days from R1 to last leaf appearance.
        lfmax:   Maximum leaf photosynthesis rate (mg CO2 dm^-2 s^-1).
        slavr:   Specific leaf area reference (cm^2 g^-1).
        sizlf:   Maximum leaf size (cm^2).
        xfruit:  Maximum growth fraction to fruit.
        wtpsd:   Maximum seed weight (g seed^-1). Canola seeds ~4.5 mg each.
        sfdur:   Seed fill duration (d under optimal conditions).
        sdpdv:   Seeds per silique (pod equivalent). ~18–20 for canola.
        podur:   Pod (silique) detachment duration (d).
        sdpro:   Protein fraction in seed (g g^-1).
        sdlip:   Lipid fraction in seed (g g^-1) — canola is a high-oil crop.
        tbase:   Base temperature for development (°C).
        topt1:   Lower optimum temperature (°C).
        topt2:   Upper optimum temperature (°C).
        tmax:    Maximum temperature (°C).
        rue:     Radiation use efficiency (informational, g DM MJ^-1 PAR).
        kcan:    Light extinction coefficient.
        slpf:    Genetic soil fertility factor (0–1).
        gdde:    GDD per cm for emergence depth.
        fix_n:   Canola does NOT fix N; always False.
    """

    varno: str = "IB0001"
    vrname: str = "Generic Spring Canola"
    econo: str = "CN0001"

    # Phenology (long-day plant)
    csdvar: float = 8.0        # Short-day threshold (no acceleration below)
    cldvar: float = 16.0       # Long-day threshold (full acceleration above)
    ppsen: float = 0.8
    em_fl: float = 30.0
    fl_sh: float = 12.0
    fl_sd: float = 20.0
    sd_pm: float = 35.0
    fl_lf: float = 15.0

    # Growth
    lfmax: float = 1.0
    slavr: float = 200.0
    sizlf: float = 120.0
    xfruit: float = 1.0
    wtpsd: float = 0.0045      # 4.5 mg (canola seeds are very small)
    sfdur: float = 30.0
    sdpdv: float = 18.0        # Seeds per silique
    podur: float = 10.0

    # Composition
    sdpro: float = 0.24
    sdlip: float = 0.42        # High-oil crop

    # Temperature (development)
    tbase: float = 0.0         # Cold-tolerant
    topt1: float = 20.0
    topt2: float = 25.0
    tmax: float = 35.0

    # Temperature response for photosynthesis (tb, to1, to2, tm)
    # Canola photosynthesis peaks at cooler temperatures than soybean
    fnpgt: tuple = (0.0, 15.0, 25.0, 35.0)

    # Canopy / RUE
    rue: float = 1.4
    kcan: float = 0.80
    slpf: float = 1.0

    # Misc
    gdde: float = 6.0
    fix_n: bool = False        # Canola does NOT fix N

    # Ecotype parameters (equivalent to SBGRO048/CNGRO048 ECO defaults)
    pl_em:  float = 2.0       # PL-EM: photothermal days sowing→emergence
    em_v1:  float = 5.0       # EM-V1: photothermal days emergence→V1
    v1_ju:  float = 0.0       # V1-JU: photothermal days V1→juvenile end
    ju_r0:  float = 5.0       # JU-R0: photothermal days juvenile→R0
    pm06:   float = 0.0       # PM06: proportion of FL-SH to FL-SD
    pm09:   float = 0.35      # PM09: proportion of SD-PM for NDSET
    r7_r8:  float = 14.0      # R7-R8: photothermal days R7→R8
    fl_vs:  float = 20.0      # FL-VS: photothermal days R1→end veg growth
    trifol: float = 0.40      # TRIFL: leaf appearance rate (leaves per thermal day)
    thvar:  float = 0.0       # THVAR: minimum rate under unfavorable daylength
    r1ppo:  float = 0.0       # R1PPO: post-R1 photoperiod sensitivity offset
    csdvrr: float = 0.0       # CSDVRR (auto-computed)
    cldvrr: float = 0.0       # CLDVRR (auto-computed)
    optbi:  float = 5.0       # OPTBI: optimum min-T for canola dev (cooler than soybean)
    slobi:  float = 0.010     # SLOBI: sensitivity of dev to min-T

    def __post_init__(self) -> None:
        """Derive CSDVRR/CLDVRR from CSDVAR/CLDVAR and R1PPO."""
        if self.csdvrr == 0.0:
            self.csdvrr = self.csdvar - self.r1ppo
        if self.cldvrr == 0.0:
            self.cldvrr = self.cldvar - self.r1ppo


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class CropGROCanola(CropModel):
    """CROPGRO-Canola crop model.

    Implements the CROPGRO framework for spring canola with:
    - Long-day photoperiod response (flowering accelerated by long days)
    - No biological N fixation
    - Same photosynthesis and C partitioning logic as soybean

    Args:
        cultivar:  CanolaCultivar coefficients.
        pltpop:    Plant population (plants m^-2).
        sdepth:    Sowing depth (cm).
        yrsim:     Simulation start date (YYYYDDD).
        rowspc:    Row spacing (cm).
        slpf:      Soil fertility factor (0–1).
        yrplt:     Planting date (YYYYDDD); -99 uses yrsim.
    """

    crop_code: str = "CN"
    model_name: str = "CNGRO048"

    def __init__(
        self,
        cultivar: CanolaCultivar,
        pltpop: float,
        sdepth: float,
        yrsim: int,
        rowspc: float = 25.0,
        slpf: float = 1.0,
        yrplt: int = -99,
    ) -> None:
        self.cultivar = cultivar
        self.pltpop = pltpop
        self.sdepth = sdepth
        self.yrsim = yrsim
        self.rowspc = rowspc
        self.slpf = slpf
        self.yrplt = yrplt if yrplt > 0 else yrsim

        # Sub-model instances
        self._pheno = CropGROPhenology(
            cultivar=cultivar,
            pltpop=pltpop,
            sdepth=sdepth,
            yrsim=yrsim,
            short_day=False,   # canola = long-day
        )
        self._growth = CropGROGrowth(
            cultivar=cultivar,
            pltpop=pltpop,
            rowspc=rowspc,
            slpf=slpf,
        )
        # NFix is inactive for canola (fix_n=False)
        self._nfix = CropGRONFix(fix_n=False, pltpop=pltpop)

        # Root state
        self.rlv: np.ndarray = np.zeros(NL)
        self.rtdep: float = 0.0
        self.trwup: float = 0.0
        self.uno3: np.ndarray = np.zeros(NL)
        self.unh4: np.ndarray = np.zeros(NL)
        self.senesce: ResidueType = ResidueType()

        # N stress (external, set by engine)
        self._nstres_external: float = 1.0
        self._n_demand: float = 0.0

    # ------------------------------------------------------------------
    @classmethod
    def from_cultivar_dict(cls, d: dict, **kwargs: Any) -> "CropGROCanola":
        """Construct a model from a cultivar parameter dictionary.

        Allows JSON-based experiment loading.

        Args:
            d:      Dictionary of cultivar attribute name → value.
            kwargs: Additional keyword arguments forwarded to ``__init__``.

        Returns:
            CropGROCanola instance.

        Example::

            model = CropGROCanola.from_cultivar_dict(
                {"varno": "IB0001", "em_fl": 28.0, "sd_pm": 40.0},
                pltpop=80.0, sdepth=2.0, yrsim=2020090,
            )
        """
        valid_fields = {f.name for f in dataclasses.fields(CanolaCultivar)}
        cv = CanolaCultivar(**{k: v for k, v in d.items() if k in valid_fields})
        return cls(cultivar=cv, **kwargs)

    # ------------------------------------------------------------------
    @property
    def pheno(self):
        """Current phenology state."""
        return self._pheno.state

    @property
    def growth(self):
        """Current growth state."""
        return self._growth.state

    # ------------------------------------------------------------------
    def set_nstres(self, v: float) -> None:
        """Set external N stress factor (0–1).

        Called by the engine after computing N supply/demand ratio.

        Args:
            v: N stress factor (0 = full stress, 1 = no stress).
        """
        self._nstres_external = float(max(0.0, min(1.0, v)))

    # ------------------------------------------------------------------
    def run(
        self,
        control: ControlType,
        iswitch: SwitchType,
        soilprop: SoilType,
        weather: WeatherType,
        sw: np.ndarray,
        nh4: np.ndarray,
        no3: np.ndarray,
        es: float,
        eop: float,
    ) -> None:
        """Run one simulation step.

        Args:
            control:   Simulation control block.
            iswitch:   Simulation switches.
            soilprop:  Soil profile properties.
            weather:   Today's weather data.
            sw:        Volumetric soil water content by layer.
            nh4:       Ammonium by layer (µg g^-1).
            no3:       Nitrate by layer (µg g^-1).
            es:        Potential soil evaporation (mm d^-1) — not used directly.
            eop:       Potential plant transpiration (mm d^-1).
        """
        dynamic = control.dynamic
        yrdoy = control.yrdoy
        nlayr = soilprop.nlayr

        # -- Phenology
        self._pheno.run(
            dynamic=dynamic,
            yrdoy=yrdoy,
            tmax=weather.tmax,
            tmin=weather.tmin,
            dayl=weather.dayl,
            sw=sw,
            ll=soilprop.ll,
            dlayr=soilprop.dlayr,
            nlayr=nlayr,
            swfac=self._growth.state.swfac,
            nstres=self._growth.state.nstres,
            iswwat=iswitch.iswwat,
        )

        if dynamic in (RUNINIT, SEASINIT):
            self._init_roots(soilprop, nlayr)
            self._growth.run(
                dynamic=dynamic,
                pheno=self.pheno,
                yrdoy=yrdoy,
                tmax=weather.tmax,
                tmin=weather.tmin,
                tday=weather.tday,
                srad=weather.srad,
                co2=weather.co2,
                sw=sw,
                dul=soilprop.dul,
                ll=soilprop.ll,
                dlayr=soilprop.dlayr,
                nlayr=nlayr,
                eop=eop,
                trwup=self.trwup,
                iswwat=iswitch.iswwat,
                iswnit=iswitch.iswnit,
            )
            self._nfix.run(
                dynamic=dynamic,
                pheno=self.pheno,
                tmax=weather.tmax,
                tmin=weather.tmin,
                sw=sw,
                ll=soilprop.ll,
                dul=soilprop.dul,
                dlayr=soilprop.dlayr,
                nlayr=nlayr,
                wrt=self._growth.state.wrt,
                growth_avail=self._growth.state.growth,
                iswwat=iswitch.iswwat,
            )
            return

        # -- Skip if not yet emerged or already mature
        if not self.pheno.emerged:
            return

        if (
            self.pheno.mdate > 0
            and self.pheno.mdate <= yrdoy
            and self.pheno.crop_status == 1
        ):
            return

        # -- Root water uptake
        if iswitch.iswwat != "N":
            _, self.trwup = root_water_uptake(
                rlv=self.rlv,
                sw=sw,
                ll=soilprop.ll,
                dlayr=soilprop.dlayr,
                nlayr=nlayr,
                rwumx=self._growth._RWUMX,
                rwuep1=self._growth._RWUEP1,
                eop=eop,
            )

        # -- Growth
        self._growth.run(
            dynamic=dynamic,
            pheno=self.pheno,
            yrdoy=yrdoy,
            tmax=weather.tmax,
            tmin=weather.tmin,
            tday=weather.tday,
            srad=weather.srad,
            co2=weather.co2,
            sw=sw,
            dul=soilprop.dul,
            ll=soilprop.ll,
            dlayr=soilprop.dlayr,
            nlayr=nlayr,
            eop=eop,
            trwup=self.trwup,
            iswwat=iswitch.iswwat,
            iswnit=iswitch.iswnit,
        )

        # Apply external N stress from engine
        if dynamic == RATE and iswitch.iswnit == "Y":
            self._growth.state.nstres = min(
                self._growth.state.nstres, self._nstres_external
            )
            self.uno3, self.unh4 = self._compute_n_uptake(soilprop, no3, nh4, sw)

        # -- No N fixation for canola (nfix is a no-op, fix_n=False)
        self._nfix.run(
            dynamic=dynamic,
            pheno=self.pheno,
            tmax=weather.tmax,
            tmin=weather.tmin,
            sw=sw,
            ll=soilprop.ll,
            dul=soilprop.dul,
            dlayr=soilprop.dlayr,
            nlayr=nlayr,
            wrt=self._growth.state.wrt,
            growth_avail=self._growth.state.growth,
            iswwat=iswitch.iswwat,
        )

        # -- Root growth (INTEGR phase)
        if dynamic == INTEGR:
            self.rlv, self.rtdep = grow_roots(
                grort=self._growth.state.grort,
                dtt=self.pheno.dtx,
                pltpop=self.pltpop,
                sw=sw,
                ll=soilprop.ll,
                dul=soilprop.dul,
                sat=soilprop.sat,
                dlayr=soilprop.dlayr,
                ds=soilprop.ds,
                shf=soilprop.shf,
                nlayr=nlayr,
                rlwr=self._growth._RLWR,
                rlv=self.rlv,
                rtdep=self.rtdep,
                rwumx=self._growth._RWUMX,
            )

    # ------------------------------------------------------------------
    def _init_roots(self, soilprop: SoilType, nlayr: int) -> None:
        """Initialise root arrays at start of season.

        Rooting depth is set to the full first-layer thickness so the
        first-layer midpoint is always within the root zone on day 1.
        """
        self.rlv[:] = 0.0
        self.rtdep = soilprop.dlayr[0]  # full first-layer depth
        self.trwup = 0.0
        self.rlv[0] = 0.5 / max(soilprop.dlayr[0], 1.0)

    # ------------------------------------------------------------------
    def _compute_n_uptake(
        self,
        soilprop: SoilType,
        no3: np.ndarray,
        nh4: np.ndarray,
        sw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute daily N uptake per soil layer (kg N ha^-1 d^-1).

        Simplified version of NUPTAK.for for CROPGRO (canola variant).

        Args:
            soilprop: Soil properties.
            no3:      Nitrate by layer (µg g^-1).
            nh4:      Ammonium by layer (µg g^-1).
            sw:       Soil water by layer (cm^3 cm^-3).

        Returns:
            Tuple ``(uno3, unh4)`` — NO3 and NH4 uptake per layer
            (kg N ha^-1 d^-1).
        """
        g = self._growth.state
        nlayr = soilprop.nlayr
        kg2ppm = soilprop.kg2ppm
        dlayr = soilprop.dlayr
        rlv = self.rlv
        shf = soilprop.shf
        ll = soilprop.ll
        sat = soilprop.sat

        total_wt = g.wlf + g.wst + g.wsh + g.wsd
        tcnp = 0.040  # canola has higher vegetative N demand
        tanc = g.stovn / max(total_wt, 0.001)
        ndem = total_wt * max(0.0, tcnp - tanc)
        andem = ndem * self.pltpop * 10.0
        self._n_demand = andem

        uno3 = np.zeros(NL)
        unh4 = np.zeros(NL)
        if andem <= 0.0:
            return uno3, unh4

        trnu = 0.0
        rno3u = np.zeros(NL)
        rnh4u = np.zeros(NL)
        for L in range(nlayr):
            if rlv[L] > 0.0 and kg2ppm[L] > 0.0:
                sw_frac = (sw[L] - ll[L]) / max(0.001, sat[L] - ll[L])
                smdfr = max(0.0, min(1.0, 1.5 - 6.0 * (sw_frac - 0.5) ** 2))
                rfac = 1.0 - np.exp(-8.0 * rlv[L])
                rno3u[L] = max(0.0, rfac * smdfr * shf[L] * 0.075 * no3[L] * dlayr[L])
                rnh4u[L] = max(0.0, rfac * smdfr * shf[L] * 0.075 * (nh4[L] - 0.5) * dlayr[L])
                trnu += rno3u[L] + rnh4u[L]

        if trnu < 1e-6:
            return uno3, unh4

        andem = min(andem, trnu)
        nuf = andem / trnu
        for L in range(nlayr):
            if kg2ppm[L] > 0.0:
                xmin_no3 = 0.25 / kg2ppm[L]
                xmin_nh4 = 0.5 / kg2ppm[L]
                sno3 = no3[L] / kg2ppm[L]
                snh4 = nh4[L] / kg2ppm[L]
                uno3[L] = min(rno3u[L] * nuf, max(0.0, sno3 - xmin_no3))
                unh4[L] = min(rnh4u[L] * nuf, max(0.0, snh4 - xmin_nh4))

        return uno3, unh4

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """Return end-of-season summary variables.

        Returns:
            Dictionary with keys matching DSSAT Summary.OUT column names.
        """
        g = self.growth
        p = self.pheno

        from dssat.core.date_utils import yrdoy_to_date

        def _safe_date(yrdoy: int) -> str:
            try:
                return str(yrdoy_to_date(yrdoy)) if yrdoy > 0 else "NA"
            except Exception:
                return "NA"

        seedno_m2 = g.seedno * self.pltpop * 10.0
        grain_wt_mg = (
            g.wsd / max(g.seedno, 0.001) * 1000.0
            if g.seedno > 0
            else 0.0
        )
        hi = g.wsd / max(g.wlf + g.wst + g.wsh + g.wsd, 0.001)

        return {
            "crop": self.crop_code,
            "model": self.model_name,
            "cultivar": self.cultivar.varno,
            "plant_date": _safe_date(self.yrplt),
            "r1_date": _safe_date(p.r1_date),
            "r3_date": _safe_date(p.r3_date),
            "r5_date": _safe_date(p.r5_date),
            "r7_date": _safe_date(p.r7_date),
            "maturity_date": _safe_date(p.mdate),
            "yield_kg_ha": round(g.yield_, 1),
            "biomass_kg_ha": round(g.biomas * 10.0, 1),
            "leaf_area_index": round(g.lai, 3),
            "grain_number_m2": round(seedno_m2, 0),
            "grain_wt_mg": round(grain_wt_mg, 2),
            "n_uptake_kg_ha": round(g.wtnup * self.pltpop * 10.0, 2),
            "harvest_index": round(hi, 3),
            "crop_status": p.crop_status,
            "rstage_final": p.rstage,
        }
