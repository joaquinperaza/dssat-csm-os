"""CERES-Sorghum crop model — top-level coordinator.

This module assembles the phenology, growth, and root sub-models into a
single interface that the main :class:`~dssat.core.engine.CropModel` API
expects.

Usage::

    from dssat.crop.ceres_sorghum import CeresSorghum, SorghumCultivar

    cv = SorghumCultivar(p1=450, p2o=12.5, p2r=180, p3=100, p4=50, p5=480,
                         g1=6.5, g2=27.0, phint=38.9, panth=550)
    model = CeresSorghum(cultivar=cv, pltpop=15.0, sdepth=4.0, yrsim=2020100)

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
from dssat.crop.ceres_sorghum.phenology import SorghumCultivar, SorghumPhenology, PhenologyState
from dssat.crop.ceres_sorghum.growth import SorghumGrowth, GrowthState
from dssat.crop.ceres_sorghum.roots import grow_roots, root_water_uptake


class CeresSorghum(CropModel):
    """CERES-Sorghum crop model.

    Implements the full CERES-Sorghum simulation cycle: phenology,
    photosynthesis, organ growth, root growth, and N uptake.

    Args:
        cultivar: Cultivar and ecotype coefficients.
        pltpop: Plant population (plants m⁻²).
        sdepth: Sowing depth (cm).
        yrsim: Simulation start date (YYYYDDD).
        rowspc: Row spacing (cm).
        slpf: Soil fertility factor (0–1).
        yrplt: Planting date (YYYYDDD); if -99 uses first sowing trigger.
    """

    crop_code: str = "SG"
    model_name: str = "SGCER048"

    def __init__(
        self,
        cultivar: SorghumCultivar,
        pltpop: float,
        sdepth: float,
        yrsim: int,
        rowspc: float = 75.0,
        slpf: float = 1.0,
        yrplt: int = -99,
    ) -> None:
        self.cultivar = cultivar
        self.pltpop = pltpop
        self.sdepth = sdepth
        self.yrsim = yrsim
        self.rowspc = rowspc
        self.slpf = slpf
        self.yrplt = yrplt

        self._pheno = SorghumPhenology(cultivar, pltpop, sdepth, yrsim)
        self._growth = SorghumGrowth(cultivar, pltpop, rowspc, slpf)

        # Root state
        self.rlv: np.ndarray = np.zeros(NL)
        self.rtdep: float = 0.0
        self.trwup: float = 0.0
        self.unh4: np.ndarray = np.zeros(NL)
        self.uno3: np.ndarray = np.zeros(NL)
        self.fracfts: np.ndarray = np.zeros(NL)
        self.senesce: ResidueType = ResidueType()

    # ------------------------------------------------------------------
    @property
    def pheno(self) -> PhenologyState:
        """Current phenology state."""
        return self._pheno.state

    @property
    def growth(self) -> GrowthState:
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
            srad=weather.srad,
            dayl=weather.dayl,
            snow=0.0,
            sw=sw,
            ll=soilprop.ll,
            dlayr=soilprop.dlayr,
            nlayr=nlayr,
            iswwat=iswitch.iswwat,
            twilen=getattr(weather, "twilen", weather.dayl),
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
        cumdep = 0.0
        for L in range(nlayr):
            cumdep += soilprop.dlayr[L]
            if self.sdepth <= cumdep:
                self.rlv[L] = 0.01
                break

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """Return a dictionary of end-of-season output variables.

        Returns:
            Dictionary with keys matching standard DSSAT Summary.OUT
            column names.
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
            "anthesis_date": _safe_date(p.isdate),
            "maturity_date": _safe_date(p.mdate),
            "yield_kg_ha": round(g.yield_, 1),
            "biomass_kg_ha": round(g.biomas * 10.0, 1),
            "leaf_area_index": round(g.lai, 3),
            "grain_number_m2": round(g.seedno * self.pltpop * 10.0, 0),
            "grain_wt_mg": round(g.grnwt / max(g.seedno, 0.001) * 1000.0, 2),
            "n_uptake_kg_ha": round(g.wtnup * self.pltpop * 10.0, 2),
            "harvest_index": round(
                g.gpsm / max(g.biomas, 0.001), 3
            ),
            "crop_status": p.crop_status,
            "istage_final": p.istage,
        }
