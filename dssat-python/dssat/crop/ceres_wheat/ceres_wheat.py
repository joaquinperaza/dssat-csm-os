"""CERES-Wheat crop model — top-level coordinator.

Assembles the phenology, growth, and root sub-models into the single
interface that the main :class:`~dssat.core.engine.CropModel` API expects.

Usage::

    from dssat.crop.ceres_wheat import CeresWheat, WheatCultivar

    cv = WheatCultivar(p1v=0.0, p1d=3.675, p5=500, g1=6.24, g2=32.0,
                       g3=1.15, phint=95.0)
    model = CeresWheat(cultivar=cv, pltpop=300.0, sdepth=3.0, rowspc=20.0)

    # inside the daily loop:
    model.run(control, iswitch, soilprop, weather, ...)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dssat.core.constants import NL, RUNINIT, SEASINIT, RATE, INTEGR, OUTPUT
from dssat.core.types import (
    ControlType, SwitchType, SoilType, WeatherType, ResidueType,
)
from dssat.crop.base import CropModel
from dssat.crop.ceres_wheat.phenology import (
    WheatCultivar, WheatPhenology, WheatPhenologyState,
)
from dssat.crop.ceres_wheat.growth import WheatGrowth, WheatGrowthState
from dssat.crop.ceres_wheat.roots import grow_roots, root_water_uptake


class CeresWheat(CropModel):
    """CERES-Wheat crop model.

    Implements the full CERES-Wheat simulation cycle: phenology
    (including vernalization and photoperiod responses), photosynthesis,
    organ growth, root growth, and N uptake.

    Args:
        cultivar: Cultivar and ecotype coefficients.
        pltpop: Plant population (plants m⁻²).
        sdepth: Sowing depth (cm).
        rowspc: Row spacing (cm).
        slpf: Soil fertility factor (0–1).
        yrsim: Simulation start date (YYYYDDD).  Used for state
            initialisation; defaults to a generic value if not supplied.
        yrplt: Planting date (YYYYDDD); if -99 uses first sowing trigger.
    """

    crop_code: str = "WH"
    model_name: str = "WHCER048"

    def __init__(
        self,
        cultivar: WheatCultivar,
        pltpop: float,
        sdepth: float,
        rowspc: float = 20.0,
        slpf: float = 1.0,
        yrsim: int = 2020001,
        yrplt: int = -99,
    ) -> None:
        self.cultivar = cultivar
        self.pltpop = pltpop
        self.sdepth = sdepth
        self.rowspc = rowspc
        self.slpf = slpf
        self.yrsim = yrsim
        self.yrplt = yrplt

        self._pheno = WheatPhenology(cultivar, pltpop, sdepth, yrsim)
        self._growth = WheatGrowth(cultivar, pltpop, rowspc, slpf)

        # Root state
        self.rlv: np.ndarray = np.zeros(NL)
        self.rtdep: float = 0.0
        self.trwup: float = 0.0
        self.unh4: np.ndarray = np.zeros(NL)
        self.uno3: np.ndarray = np.zeros(NL)
        self.senesce: ResidueType = ResidueType()

        # Heading date (stage 3 reached)
        self._das_heading: int = -99

    # ------------------------------------------------------------------
    @property
    def pheno(self) -> WheatPhenologyState:
        """Current phenology state."""
        return self._pheno.state

    @property
    def growth(self) -> WheatGrowthState:
        """Current growth state."""
        return self._growth.state

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
            control: Simulation control block.
            iswitch: Simulation switches.
            soilprop: Soil profile properties.
            weather: Today's weather data.
            sw: Soil water content by layer.
            nh4: Ammonium by layer (µg g⁻¹).
            no3: Nitrate by layer (µg g⁻¹).
            es: Potential soil evaporation (mm d⁻¹).
            eop: Potential transpiration (mm d⁻¹).
        """
        dynamic = control.dynamic
        yrdoy = control.yrdoy
        nlayr = soilprop.nlayr

        # ---- Phenology
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
                srad=weather.srad,
                co2=weather.co2,
                sw=sw,
                dul=soilprop.dul,
                sat=soilprop.sat,
                ll=soilprop.ll,
                dlayr=soilprop.dlayr,
                nlayr=nlayr,
                nh4=nh4,
                no3=no3,
                kg2ppm=soilprop.kg2ppm,
                shf=soilprop.shf,
                rlv=self.rlv,
                rtdep=self.rtdep,
                trwup=self.trwup,
                eop=eop,
                iswwat=iswitch.iswwat,
                iswnit=iswitch.iswnit,
            )
            return

        if self.pheno.istage in (7, 8):
            return

        # Track heading date (stage 3 = anthesis initiation)
        if self.pheno.istage >= 3 and self._das_heading < 0:
            self._das_heading = control.das

        # ---- Root water uptake
        _, self.trwup = root_water_uptake(
            rlv=self.rlv,
            sw=sw,
            ll=soilprop.ll,
            dlayr=soilprop.dlayr,
            nlayr=nlayr,
            rwumx=self._growth._rwumx,
            rwuep1=self._growth._rwuep1,
            eop=eop,
        )

        # ---- Growth
        self._growth.run(
            dynamic=dynamic,
            pheno=self.pheno,
            yrdoy=yrdoy,
            tmax=weather.tmax,
            tmin=weather.tmin,
            srad=weather.srad,
            co2=weather.co2,
            sw=sw,
            dul=soilprop.dul,
            sat=soilprop.sat,
            ll=soilprop.ll,
            dlayr=soilprop.dlayr,
            nlayr=nlayr,
            nh4=nh4,
            no3=no3,
            kg2ppm=soilprop.kg2ppm,
            shf=soilprop.shf,
            rlv=self.rlv,
            rtdep=self.rtdep,
            trwup=self.trwup,
            eop=eop,
            iswwat=iswitch.iswwat,
            iswnit=iswitch.iswnit,
        )

        # ---- Root growth
        if dynamic == INTEGR:
            g = self.growth
            self.rlv, self.rtdep = grow_roots(
                grort=g.grort,
                dtt=self.pheno.dtt,
                pltpop=self.pltpop,
                sw=sw,
                ll=soilprop.ll,
                dul=soilprop.dul,
                sat=soilprop.sat,
                dlayr=soilprop.dlayr,
                ds=soilprop.ds,
                shf=soilprop.shf,
                nlayr=nlayr,
                rlwr=self._growth._rlwr,
                rlv=self.rlv,
                rtdep=self.rtdep,
                rwumx=self._growth._rwumx,
            )

    # ------------------------------------------------------------------
    def _init_roots(self, soilprop: SoilType, nlayr: int) -> None:
        """Initialise root state at start of season."""
        self.rlv[:] = 0.0
        self.rtdep = self.sdepth
        self.trwup = 0.0
        self._das_heading = -99
        cumdep = 0.0
        for L in range(nlayr):
            cumdep += soilprop.dlayr[L]
            if self.sdepth <= cumdep:
                self.rlv[L] = 0.01
                break

    # ------------------------------------------------------------------
    @classmethod
    def from_cultivar_dict(cls, d: dict) -> "CeresWheat":
        """Construct a CeresWheat model from an experiment coefficient dict.

        The dictionary is expected to contain cultivar coefficients plus
        management fields ``pltpop``, ``sdepth``, ``rowspc``, and
        optionally ``slpf`` and ``yrsim``.

        This matches the pattern used by the JSON experiment loader.

        Args:
            d: Flat dictionary of cultivar + management parameters.

        Returns:
            Configured :class:`CeresWheat` instance.
        """
        cv = WheatCultivar.from_dict(d)
        pltpop = float(d.get("pltpop", 300.0))
        sdepth = float(d.get("sdepth", 3.0))
        rowspc = float(d.get("rowspc", 20.0))
        slpf = float(d.get("slpf", 1.0))
        yrsim = int(d.get("yrsim", 2020001))
        yrplt = int(d.get("yrplt", -99))
        return cls(
            cultivar=cv,
            pltpop=pltpop,
            sdepth=sdepth,
            rowspc=rowspc,
            slpf=slpf,
            yrsim=yrsim,
            yrplt=yrplt,
        )

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """Return a dictionary of end-of-season output variables.

        Returns:
            Dictionary with keys matching standard DSSAT Summary.OUT
            column names, plus wheat-specific additions.
        """
        g = self.growth
        p = self.pheno
        from dssat.core.date_utils import yrdoy_to_date

        def _safe_date(yrdoy: int) -> str:
            try:
                return str(yrdoy_to_date(yrdoy)) if yrdoy > 0 else "NA"
            except Exception:
                return "NA"

        return {
            "crop": self.crop_code,
            "model": self.model_name,
            "cultivar": self.cultivar.varno,
            "plant_date": _safe_date(self.yrplt),
            "heading_date": _safe_date(p.isdate),
            "anthesis_date": _safe_date(p.isdate),
            "maturity_date": _safe_date(p.mdate),
            "yield_kg_ha": round(g.yield_, 1),
            "biomass_kg_ha": round(g.biomas * 10.0, 1),
            "leaf_area_index": round(g.lai, 3),
            "grain_number_m2": round(g.seedno * self.pltpop * 10.0, 0),
            "grain_wt_mg": round(
                g.grnwt / max(g.seedno, 0.001) * 1000.0, 2
            ),
            "n_uptake_kg_ha": round(g.wtnup * self.pltpop * 10.0, 2),
            "harvest_index": round(
                g.gpsm / max(g.biomas, 0.001), 3
            ),
            "crop_status": p.crop_status,
            "istage_final": p.istage,
            "das_heading": self._das_heading,
        }
